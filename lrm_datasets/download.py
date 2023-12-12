import os
import sys
import re
import errno
import warnings
import hashlib
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

import torch
import tarfile
from torch.hub import download_url_to_file, get_dir
from pdb import set_trace

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def download_data_from_url(
    url: str,
    data_dir: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None
) -> Dict[str, Any]:
    r"""Downloads the object at the given URL.

    If downloaded file is a .tar file or .tar.gz file, it will be automatically
    decompressed.

    If the object is already present in `data_dir`, it's deserialized and
    returned.

    The default value of ``data_dir`` is ``<hub_dir>/../data`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        data_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if data_dir is None:
        hub_dir = torch.hub.get_dir()
        data_dir = hub_dir.replace("/hub", "/data")

    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(data_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            #r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            #hash_prefix = r.group(1) if r else None
            matches = HASH_REGEX.findall(filename) # matches is Optional[Match[str]]
            hash_prefix = matches[-1] if matches else None

        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file

def get_remote_data_file(url, cache_dir=torch.hub.get_dir().replace("/hub", "/data"),
                         progress=True, check_hash=False, expected_hash=None) -> Mapping[str, Any]:

    if cache_dir is None:
      cache_dir = torch.hub.get_dir().replace("/hub", "/data")

    cached_filename = download_data_from_url(
        url = url,
        data_dir = cache_dir,
        progress = progress,
        check_hash = check_hash,
    )

    print(f"cached_filename: {cached_filename}")
    extracted_folder = decompress_tarfile_if_needed(cached_filename)

    return cached_filename, extracted_folder

def decompress_tarfile_if_needed(file_path, output_dir=None):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Determine the directory of the tar file
    dir_name = os.path.dirname(file_path) or '.'

    # Set the output directory. If none is provided, default to the tar file's location
    if output_dir is None:
        output_dir = dir_name

    # If the output directory doesn't exist, create it
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the first folder inside the tar file is the root folder of its contents
    # This will be used to check if the contents have already been extracted
    with tarfile.open(file_path, 'r:*') as tar:
        top_folder_name = os.path.commonprefix(tar.getnames())
        expected_extracted_folder = os.path.join(output_dir, top_folder_name)

    # Check if the contents have already been extracted
    if os.path.exists(expected_extracted_folder):
        print(f"Contents have already been extracted to {expected_extracted_folder}.")
    else:
        # Contents have not been extracted; proceed with extraction
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=output_dir)
            print(f"File {file_path} has been decompressed to {output_dir}.")

    return expected_extracted_folder

def calculate_sha256(file_path, block_size=8192):
    """
    Calculate the SHA256 hash of a file.

    :param file_path: path to the file being read.
    :param block_size: size of each read block. A value used for memory efficiency.
                       Default is 8192 bytes.
    :return: the SHA256 hash.
    """
    sha256 = hashlib.sha256()

    # Open the file in binary mode and read it in chunks.
    with open(file_path, 'rb') as f:
        while True:
            # Read a block from the file
            data = f.read(block_size)
            if not data:
                break  # Reached end of file

            # Update the hash
            sha256.update(data)

    # Return the hexadecimal representation of the digest
    return sha256.hexdigest()
