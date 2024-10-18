"""
    Train a text-to-image diffusion prior using lucidrain's dalle2 implementation.
    
    Example Run:
    
    python train_diffusion_prior.py \
        --data.train_dataset=/n/holyscratch01/alvarez_lab/Lab/datasets/imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
        --data.val_dataset=/n/holyscratch01/alvarez_lab/Lab/datasets/imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
        --data.num_workers=24 \
        --logging.folder=/n/holystore01/LABS/alvarez_lab/Lab/Projects/dnn_feedback_dev/logs/clip_diffusion/
    
"""

import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import sys
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from types import SimpleNamespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from dalle2_pytorch import OpenAIClipAdapter
from torch.cuda.amp import autocast

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from pdb import set_trace

import models
from models import weights as Weights
from models import load_model_from_weights
from models.prompt_ensemble import PromptEnsemble
from models.text_to_image_diffusion import TextToImageDiffusion

from lib.dataloaders import get_val_loader, get_train_loader

from fastprogress.fastprogress import master_bar, progress_bar

weight_map = dict(
    set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_63a=Weights.Set15_Weights.set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_63a
)

Section('model', 'model details').params(
    arch=Param(str, default='set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_63a'),
    clip_name=Param(str, default='ViT-B/16'),
    image_embed_scale=Param(float, default=1.0),
    condition_on_text_encodings=Param(bool, default=False)
)

Section('opt', 'training details').params(
    optimizer=Param(str, default='SGD'),
)

Section('sched', 'learning rate schedule').params(
    lr_schedule_type=Param(str, default='cyclic'),
    lr=Param(float, default=.001),
    epochs=Param(int, default=10),
    lr_peak_epoch=Param(int, default=2)
)

Section('training', 'training details').params(
    device1=Param(str, default='cuda:0'),
    device2=Param(str, default='cuda:1'),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.ffcv file to use for training', required=True),
    val_dataset=Param(str, '.ffcv file to use for validation', required=True),
    crop_size=Param(int, 'image crop size', default=256),
    batch_size=Param(int, 'batch size', default=512),
    num_workers=Param(int, 'number of loader workers', default=24),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
)

@param('sched.lr')
@param('sched.epochs')
@param('sched.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch, lr_start=1e-4, lr_end=1e-8):
    xs = [0, lr_peak_epoch, epochs]
    ys = [lr_start * lr, lr, lr_end]
    return np.interp([epoch], xs, ys)[0]

@param('sched.lr_schedule_type')
def get_lr(self, epoch, lr_schedule_type):
    lr_schedules = {
        'cyclic': get_cyclic_lr,
        #'cyclic_plateau': get_cyclic_lr_with_plateau,
        #'step': get_step_lr
    }

    return lr_schedules[lr_schedule_type](epoch)
    
@param('model.clip_name')
@param('training.device1')
@param('training.device2')
def get_text_label_embeddings(clip_name, device1, device2):
    clip_model = OpenAIClipAdapter(name=clip_name)
    label_encoder = PromptEnsemble(clip_model = clip_model)
    label_encoder.to(device1);
    
    label_idxs = torch.arange(1000).to(device1)
    all_text_embeds, all_text_encodings = label_encoder(label_idxs)
    
    return all_text_embeds.to(device2), all_text_encodings.to(device2)

@param('model.arch')
@param('training.device1')
def load_vision_model(arch, device1):
    weights = weight_map[arch]
    vision_model = load_model_from_weights(weights)
    vision_model.to(device1)
    
    return vision_model

@param('model.image_embed_scale')
@param('model.condition_on_text_encodings')
@param('training.device2')
def load_diffusion_model(image_embed_scale, condition_on_text_encodings, device2):
    
    model = TextToImageDiffusion(image_embed_scale=image_embed_scale, condition_on_text_encodings=condition_on_text_encodings)
    model.to(device2)
    
    return model

@param('data.train_dataset')
@param('data.val_dataset')
@param('data.crop_size')
@param('data.batch_size')
@param('data.num_workers')
@param('training.device1')
def get_dataloaders(train_dataset, val_dataset, crop_size, batch_size, num_workers, device1):
    
    # using the val_loader for both the train and val dataste
    # so we're just center cropping
    train_loader, inv_transform = get_val_loader(train_dataset, shuffle=True, crop_size=crop_size, 
                                                 batch_size=batch_size, num_workers=num_workers, device=device1)
    
    
    val_loader, _ = get_val_loader(val_dataset, shuffle=False, crop_size=crop_size, 
                                   batch_size=batch_size, num_workers=num_workers, device=device1)
    
    return train_loader, val_loader

Section('opt', 'training details').params(
    optimizer=Param(str, default='SGD'),
)

@param('opt.optimizer')
@param('sched.lr')
def get_optimizer(model, optimizer, lr):
    
    if optimizer=='SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return opt

@param('model.condition_on_text_encodings')
@param('training.device1')
def train_one_epoch(epoch, all_text_embeds, all_text_encodings, vision_model, 
                    model, train_loader, optimizer, mb, condition_on_text_encodings, device1):
    
    # get lr for each batch in this epoch
    iters = len(train_loader)
    lrs = []
    lr_start, lr_end = get_lr(epoch), get_lr(epoch+1)
    lrm_lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])
    lrs.append(lrm_lrs) # only one param group
    
    vision_model.eval()
    model.train()
    train_loss_total = 0.0
    pb = progress_bar(train_loader, parent=mb)
    for batch_num,(imgs,labels) in enumerate(pb):
        imgs = imgs.to(device1, non_blocking=True)

        ### Training start
        for idx,param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lrs[idx][batch_num]

        with torch.no_grad():
            with autocast():                
                image_embeds = vision_model(imgs).to(device2)
                text_embed = all_text_embeds.index_select(dim=0, index=labels)
                text_encodings = all_text_encodings.index_select(dim=0, index=labels)
        
        optimizer.zero_grad()
        
        with autocast():
            train_loss = model(
                text_embeds = text_embed.half(),
                image_embeds = image_embeds.half(),
                text_encodings = text_encodings if condition_on_text_encodings else None
            )
                
        if torch.isnan(train_loss).any():
            set_trace()
        train_loss_total += train_loss.item()
        
        train_loss.backward()
        optimizer.step()

        if batch_num % 25 == 0:
            avg_train_loss = train_loss_total / (batch_num + 1)
            pb.comment = f"Epoch {epoch}, Batch {batch_num}, Train Loss: {train_loss.item():.4f}"
            break
            
    train_loss_epoch = train_loss_total/(batch_num+1)

    return train_loss_epoch

@param('model.condition_on_text_encodings')
@param('training.device1')
@param('training.device2')
def validate(all_text_embeds, all_text_encodings, vision_model, model, val_loader, mb, 
             condition_on_text_encodings, device1, device2):
    
    vision_model.eval()
    model.eval()
    for batch_num,(imgs,labels) in enumerate(progress_bar(val_loader, parent=mb)):
        imgs = imgs.to(device1, non_blocking=True)

        with torch.no_grad():
            image_embeds = vision_model(imgs).to(device2)
            text_embed = all_text_embeds.index_select(dim=0, index=labels)
            text_encodings = all_text_encodings.index_select(dim=0, index=labels) if condition_on_text_encodings else None
            val_loss = model(
                text_embeds = text_embed,
                image_embeds = image_embeds,
                text_encodings = text_encodings
            )
            pred_image_embeddings = model.sample(text_embed, text_encodings)
            val_cos_sim = F.cosine_similarity(pred_image_embeddings, image_embeds).mean(dim=0)
        val_loss_total += val_loss.item()
        val_cos_sim_total += val_cos_sim.item()
        
    val_loss_epoch = val_loss_total/(batch_num+1)
    val_cos_sim_epoch = val_cos_sim_total/(batch_num+1)
    
    return val_loss_epoch, val_cos_sim_epoch

@param('model.arch')
@param('model.clip_name')
@param('logging.folder')
def save_model(model, results, arch, clip_name, folder):
    folder_name = f"{clip_name.replace('-','_').replace('/','_')}_{arch}"
    log_folder = os.path.join(folder, folder_name)
    filename = os.path.join(log_folder, 'final_weights.pth')
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    all_params = get_current_config()
    params = {
        '.'.join(k): all_params[k] for k in all_params.entries.keys()
    }
    print(f"saving results: {filename}")
    torch.save(dict(
        state_dict=model.state_dict(),
        results=results,
        params=params,
    ), filename)    

@param('sched.epochs')
def train_loop(all_text_embeds, all_text_encodings, vision_model, model, train_loader, val_loader, optimizer,
               start_epoch, epochs):
    
    print(f"Begin training loop: epochs={epochs}")
    results = defaultdict(list)
    mb = master_bar(range(start_epoch, epochs))
    for epoch in mb:
        train_loss_epoch = train_one_epoch(epoch, all_text_embeds, all_text_encodings, vision_model, 
                                           model, train_loader, optimizer, mb)
        
        val_loss_epoch, val_cos_sim_epoch = validate(all_text_embeds, all_text_encodings, vision_model, 
                                                     model, val_loader, mb)
        
        results['epoch'].append(epoch)
        results['train_loss_avg'].append(train_loss_epoch)
        results['val_loss_avg'].append(val_loss_epoch)
        results['val_cossim_avg'].append(val_cos_sim_epoch)

        mb.write(f'Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val CosSim: {val_cos_sim_epoch:.4f}') 
        
        break
    
    save_model(model, results)
    
def main():
    
    print("==> Get prompt ensemble embeddings")
    all_text_embeds, all_text_encodings = get_text_label_embeddings()
    print(all_text_embeds.shape, all_text_encodings.shape)
    
    print("==> Loading lrm vision model")
    vision_model = load_vision_model()
    
    print("==> Loading diffusion model")
    model = load_diffusion_model()
    
    print("==> Get data loaders")
    train_loader, val_loader = get_dataloaders()

    print("==> Get optimizer")
    optimizer = get_optimizer(model)
    print(optimizer)
    
    start_epoch = 0
    train_loop(all_text_embeds, all_text_encodings, vision_model, model, train_loader, val_loader, optimizer, start_epoch)
    
    print("Done!")
    
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    
    # collect_argparse_args doesn't add config_file to args,
    # so we do it here:
    # args = vars(parser.parse_args(sys.argv[1:]))
    # config.collect({
    #     'config.file': args['config_file'][0]
    # })
    
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    main()