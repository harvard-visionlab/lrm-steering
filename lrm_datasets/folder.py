from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from pathlib import Path

__all__ = ['ImageFolderIndex', 'ImagenetteDatasetRemapIN1K']

class ImageFolderIndex(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

class ImagenetteDatasetRemapIN1K(object):
    wordnet_to_label = {
      'n03028079': 497,
      'n03445777': 574,
      'n01440764': 0,
      'n03425413': 571,
      'n02102040': 217,
      'n02979186': 482,
      'n03394916': 566,
      'n03417042': 569,
      'n03000684': 491,
      'n03888257': 701
    }

    def __init__(self, root_dir, transform=None, loader=default_loader):
        self.root_dir = root_dir
        self.image_files = sorted([str(path) for path in Path(root_dir).rglob('*') if path.suffix.lower() == '.jpeg'])
        self.samples = [(img, Path(img).parent.stem, self.wordnet_to_label[Path(img).parent.stem])
                        for img in self.image_files]
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename,wordnet_id,label = self.samples[index]
        img = self.loader(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img,label,index  

    def __repr__(self) -> str:
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root_dir is not None:
            body.append(f"Root location: {self.root_dir}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""