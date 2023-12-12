import os
from .download import get_remote_data_file
from .folder import ImagenetteDatasetRemapIN1K

__all__ = ['imagenette2']

def imagenette2(split, transform=None, cache_dir=None):
  print("imagenette2", split, transform, cache_dir)
  url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
  cached_filename, extracted_folder = get_remote_data_file(url, cache_dir=cache_dir)
  dataset = ImagenetteDatasetRemapIN1K(os.path.join(extracted_folder, split), transform=transform)
  return dataset

  
