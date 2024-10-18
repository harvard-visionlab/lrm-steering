import os
import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

__all__ = ['Stats', 'PrototypeActivationMeter', 'compute_prototypes', 'load_vision_prototypes']

vision_prototypes = dict(
    alexnet_lrm1="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm1_40b29a3427-e1282174.pth",
    alexnet_lrm2="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm2_17b4229a30-83b9be6e.pth",
    alexnet_lrm3="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm3_63ab1b3b06-d59f87e7.pth",
)

clip_prototypes = dict(
    alexnet_lrm1="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_prototypes_ViT_B16_to_alexnet_lrm1_40b29a3427-8729ac14.pth",
    alexnet_lrm2="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_prototypes_ViT_B16_to_alexnet_lrm2_17b4229a30-97b050f0.pth",
    alexnet_lrm3="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_prototypes_ViT_B16_to_alexnet_lrm3_63ab1b3b06-b710a264.pth",
)

@torch.no_grad()
def compute_prototypes(model, dataset, batch_size=250, num_workers=len(os.sched_getaffinity(0)), device=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True)
    
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==> computing prototypes using device: {device}")
    prototype_meter = PrototypeActivationMeter()
    
    model.eval()
    model.to(device)
    for imgs,labels,indexes in tqdm(dataloader):
        output = model(imgs.to(device, non_blocking=True))

        for label,activation in zip(labels,output):
            prototype_meter.update([label.item()], [activation])

    return prototype_meter.state_dict()
    
class Stats:
    def __init__(self):
        self.count = None
        self.sum = None
        self.sumsq = None
        self.mean = None
        self.std = None

    def update(self, activation):
        if self.count is None:
            # This is the first activation tensor we've seen for this key,
            # so we initialize sums, sumsq, means, and std based on its shape.
            self.count = 0
            self.sum = torch.zeros_like(activation)
            self.sumsq = torch.zeros_like(activation)
            self.mean = torch.zeros_like(activation)
            self.std = torch.zeros_like(activation)

        # Update running stats here...
        self.count += 1
        self.sum += activation
        self.sumsq += activation**2
        self.mean = self.sum / self.count
        self.std = torch.sqrt(self.sumsq / self.count - self.mean**2)
        self.std[torch.isnan(self.std)] = 0

    def state_dict(self):
        return dict(
            count=self.count,
            sum=self.sum,
            sumsq=self.sumsq,
            mean=self.mean,
            std=self.std,
        )

class PrototypeActivationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.prototypes = defaultdict(Stats)

    def update(self, indices, activations):
        for index,activation in zip(indices, activations):
            self.prototypes[index].update(activation)

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.prototypes.items()}

    def __str__(self):
        fmtstr = 'PrototypeActivationMeter:\n'
        for i in range(self.num_classes):
            fmtstr += f'Class {i}: '
            for j in range(self.num_units):
                fmtstr += f'Unit {j}: Mean={self.means[i,j]:.4f}, Std={self.std[i,j]:.4f}; '
            fmtstr += '\n'
        return fmtstr

def load_vision_prototypes(model_name, map_location='cpu', check_hash=True, **kwargs):
    url = vision_prototypes[model_name]
    print(f"==> Loading vision_prototypes: {url}")
    return torch.hub.load_state_dict_from_url(url, 
                                              map_location=map_location, 
                                              check_hash=check_hash,
                                              **kwargs)

def load_clip_prototypes(model_name, map_location='cpu', check_hash=True, **kwargs):
    url = clip_prototypes[model_name]
    print(f"==> Loading clip_prototypes: {url}")
    return torch.hub.load_state_dict_from_url(url, 
                                              map_location=map_location, 
                                              check_hash=check_hash,
                                              **kwargs)
    