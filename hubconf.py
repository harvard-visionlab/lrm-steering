import torch, torchvision
import lrm_models as _lrm_models
import lrm_datasets as _lrm_datasets
import lrm_utils as _lrm_utils

dependencies = ['torch', 'torchvision']

# model, transforms = torch.hub.load("harvard-visionlab/lrm-steering-dev", "alexnet_lrm1")

prototype_urls = dict(
    alexnet_lrm1="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm1_40b29a3427-e1282174.pth",
    alexnet_lrm2="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm2_17b4229a30-83b9be6e.pth",
    alexnet_lrm3="https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/visual_prototypes_alexnet_lrm3_63ab1b3b06-d59f87e7.pth",
)

def get_standard_transforms(img_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # DEFAULT_CROP_RATIO = 224/256
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomResizedCrop(crop_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    inv_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
        ),
        torchvision.transforms.ToPILImage(),
    ])

    return dict(train_transform=train_transform, val_transform=val_transform,
                test_transform=test_transform, inv_transform=inv_transform)

def alexnet_lrm1(**kwargs):
    model = _lrm_models.alexnet_lrm1(**kwargs)
    transforms = get_standard_transforms()
    return model, transforms

def alexnet_lrm2(**kwargs):
    model = _lrm_models.alexnet_lrm2(**kwargs)
    transforms = get_standard_transforms()
    return model, transforms

def alexnet_lrm3(**kwargs):
    model = _lrm_models.alexnet_lrm3(**kwargs)
    transforms = get_standard_transforms()
    return model, transforms

def LRMNet(**kwargs):
    return _lrm_models.LRMNet

def SteerableLRM(**kwargs):
    return _lrm_models.SteerableLRM

def datasets(dataset, split, *args, **kwargs):
    return _lrm_datasets.__dict__[dataset](split, *args, **kwargs)

def compute_prototypes():
    return _lrm_utils.compute_prototypes

def load_prototypes():    
    return lambda model_name: torch.hub.load_state_dict_from_url(prototype_urls[model_name], map_location='cpu', check_hash=True)
