'''
    These are the main lrms from our neurips2023 paper, "Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections", ported to use the new lrm API which makes it easier to specify connections between layers.
    
    Models using the original LRMNet class, as well as other LRM models reported in the paper, can be found in the neurips2023 folder. 
'''
import os
import re
import torch
import inspect
import torchvision.models as tv_models
from pathlib import Path
from urllib.parse import urlparse
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
from .lrmnet import LRMNet, SteerableLRM

weight_urls = {
    "alexnet_lrm1": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set14/set14_alexnet_lrm_cls6to8_9to0_2steps_stepwise/df5a9767-1046-4d0f-9f82-649c0e5c7881/set14_alexnet_lrm_cls6to8_9to0_2steps_stepwise_final_weights-40b29a3427.pth",
    "alexnet_lrm2": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set15/set15_alexnet_torchvision_imagenet1k_lrm_2back_2steps/84bdc4f4-1de0-4438-941b-43e574298694/set15_alexnet_torchvision_imagenet1k_lrm_2back_2steps_final_weights-17b4229a30.pth",
    "alexnet_lrm3": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set15/set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps/28453e80-c5e5-4d76-bc81-99c5fade39ff/set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_final_weights-63ab1b3b06.pth",
}

def get_state_dict(url, model_dir=torch.hub.get_dir(), progress=True, check_hash=True):
        
    cache_filename = os.path.basename(urlparse(url).path)

    checkpoint = load_state_dict_from_url(
        url = url,
        model_dir = model_dir,
        map_location = 'cpu',
        progress = progress,
        check_hash = check_hash,
        # file_name = cache_filename
    )

    print(f"local_filename: {os.path.join(model_dir,cache_filename)}")

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
    
    # we renamed "backbone" to be "feedforward"
    state_dict = {k.replace("backbone.", "feedforward."):v for k,v in state_dict.items()}

    # we adjusted the lrm module nameing to replace "." with "_" in layer names (instead of just trimming the .)
    pattern = re.compile(r"features(\d+)")
    state_dict = {pattern.sub(lambda m: f"features_{m.group(1)}", k):v for k,v in state_dict.items()}

    return state_dict

def load_pretrained(model, model_name): 
    url = weight_urls[model_name]
    hash_id = Path(url).stem.split("-")[-1]
    print(f"==> Loading weights for {model_name}, hash_id={hash_id}")
    print(url)
    state_dict = get_state_dict(url)
    msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    model.hash_id = hash_id
    
    return model

def alexnet_lrm1(pretrained=True, steering=True):

    # instantiate alexnet baseline model
    backbone = tv_models.alexnet(weights=None)

    # specify modulatory connections
    mod_connections = [
        # 1
        dict(source="classifier.6", destination="features.8"),             # output => conv4.conv
        dict(source="features.9", destination="features.0"),               # conv4.ReLU => conv1.conv
    ]

    if steering:
        model = SteerableLRM(backbone, mod_connections, forward_passes=2, img_size=224)
    else:
        model = LRMNet(backbone, mod_connections, forward_passes=2, img_size=224)
    
    if pretrained:  
        model_name = inspect.currentframe().f_code.co_name
        model = load_pretrained(model, model_name)
        
    return model

def alexnet_lrm2(pretrained=True, steering=True):

    # instantiate alexnet baseline model
    backbone = tv_models.alexnet(weights=None)

    # specify modulatory connections
    mod_connections = [
        # 1
        dict(source="classifier.6", destination="features.8"),             # output => conv4.conv
        dict(source="features.9", destination="features.0"),               # conv4.ReLU => conv1.conv
        
        # 2
        dict(source="classifier.6", destination="features.10"),            # output => conv5.conv
        dict(source="features.12", destination="features.3"),              # conv5.ReLU => conv2.conv        
    ]

    if steering:
        model = SteerableLRM(backbone, mod_connections, forward_passes=2, img_size=224)
    else:
        model = LRMNet(backbone, mod_connections, forward_passes=2, img_size=224)
    
    if pretrained:  
        model_name = inspect.currentframe().f_code.co_name
        model = load_pretrained(model, model_name)
        
    return model

def alexnet_lrm3(pretrained=True, steering=True):

    # instantiate alexnet baseline model
    backbone = tv_models.alexnet(weights=None)

    # specify modulatory connections
    mod_connections = [
        # 1
        dict(source="classifier.6", destination="features.8"),             # output => conv4.conv
        dict(source="features.9", destination="features.0"),               # conv4.ReLU => conv1.conv
        
        # 2
        dict(source="classifier.6", destination="features.10"),            # output => conv5.conv
        dict(source="features.12", destination="features.3"),              # conv5.ReLU => conv2.conv
        
        # 3
        dict(source="classifier.2", destination="features.6"),             # fc6.ReLU => conv3.conv
    ]
    
    if steering:
        model = SteerableLRM(backbone, mod_connections, forward_passes=2, img_size=224)
    else:
        model = LRMNet(backbone, mod_connections, forward_passes=2, img_size=224)
    
    if pretrained:  
        model_name = inspect.currentframe().f_code.co_name
        model = load_pretrained(model, model_name)
        
    return model
