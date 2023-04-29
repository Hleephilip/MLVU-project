#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install git+https://github.com/openai/CLIP.git')


# In[ ]:


get_ipython().run_line_magic('cd', "'/home/n0/mlvu028/cc3m'")


# In[ ]:


import functools
import io
import json
import os
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import clip
import click
import numpy as np
import PIL.Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import cv2
from collections import OrderedDict
import os.path as op
import random


# In[ ]:


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def open_dataset(source: str):
  images = []
  captions = []

  max_idx = maybe_min(3318333, None)


  for tar_idx in range(301, 306):
    try:
      tar_idx_str = f'{tar_idx:05d}'
      path = os.path.join(source, f'{tar_idx:05d}.tar')
  
      with tarfile.open(path) as tar:
        for img_idx in range(0, 10000):
          if img_idx % 1000 == 0:
            print(f'tar {tar_idx} / img {img_idx}')
          try:
            idx_str = f'{img_idx:04d}'
            if f'{tar_idx_str}{idx_str}.jpg' in tar.getnames():
              image = tar.getmember(f'{tar_idx_str}{idx_str}.jpg')
              with tar.extractfile(image) as file:
                image = np.asarray(PIL.Image.open(file))
                if image.shape != (256, 256, 3):
                  continue
                images.append(image)
                #images.append(image.reshape(-1, 3, 256, 256))

              text = tar.getmember(f'{tar_idx_str}{idx_str}.txt')
              with tar.extractfile(text) as binary, io.TextIOWrapper(binary) as content:
                text = content.readlines()
              #with tar.extractfile(text) as file:
                #text =s file.read()
                captions.append(text)

          except:
            print(f'{tar_idx_str}{idx_str}.jpg failed')
    except:
      print(f'{tar_idx_str}.tar failed')

  images = np.asarray(images)
  # print('before conc',len(images))
  # images = np.concatenate(images)
  # print('after conc',images.shape)
  # images = images.transpose([0, 2, 3, 1]) 
  # print('after transp',images.shape)

  def iterate_images():
    for idx, img in enumerate(images): 
      yield dict(img=img, txt=captions[idx])  

  return max_idx, iterate_images()


# source = '.'
# open_dataset(source)

def custom_reshape(img, mode='bicubic', ratio=0.99):   # more to be implemented here
    full_size = img.shape[-2]
    prob = torch.rand(())

    if full_size < 224:
        pad_1 = torch.randint(0, 224-full_size, ())
        pad_2 = torch.randint(0, 224-full_size, ())
        m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
        reshaped_img = m(img)
    else:
        cut_size = torch.randint(int(ratio*full_size), full_size, ())
        left = torch.randint(0, full_size-cut_size, ())
        top = torch.randint(0, full_size-cut_size, ())
        cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img , (224, 224), mode=mode, align_corners=False)
    return  reshaped_img


def clip_preprocess():
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def error(msg):
    print('Error: ' + msg)
    sys.exit(1)


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        try:
            img = PIL.Image.fromarray(img, 'RGB')
        except:
            img = PIL.Image.fromarray(img)
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]


def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None


source = '/content/drive/MyDrive/Colab Notebooks/MLVU Team Project/sample_data1'
dest = '/content/drive/MyDrive/Colab Notebooks/MLVU Team Project/sample_data1_results'
resize_filter = 'lanczos'
transform = 'center-crop'
width = 256
height = 256


"""Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.
The input dataset format is guessed from the --source argument:
\b
--source *_lmdb/                    Load LSUN dataset
--source cifar-10-python.tar.gz     Load CIFAR-10 dataset
--source train-images-idx3-ubyte.gz Load MNIST dataset
--source path/                      Recursively load all images from path/
--source dataset.zip                Recursively load all images from dataset.zip
Specifying the output format and path:
\b
--dest /path/to/dir                 Save output files under /path/to/dir
--dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip
The output dataset format can be either an image folder or an uncompressed zip archive.
Zip archives makes it easier to move datasets around file servers and clusters, and may
offer better training performance on network file systems.
Images within the dataset archive will be stored as uncompressed PNG.
Uncompresed PNGs can be efficiently decoded in the training loop.
Class labels are stored in a file called 'dataset.json' that is stored at the
dataset root folder.  This file has the following structure:
\b
{
    "labels": [
        ["00000/img00000000.png",6],
        ["00000/img00000001.png",9],
        ... repeated for every image in the datase
        ["00049/img00049999.png",1]
    ]
}
If the 'dataset.json' file cannot be found, the dataset is interpreted as
not containing class labels.
Image scale/crop and resolution requirements:
Output images must be square-shaped and they must all have the same power-of-two
dimensions.
To scale arbitrary input image size to a specific width and height, use the
--width and --height options.  Output resolution will be either the original
input resolution (if --width/--height was not specified) or the one specified with
--width/height.
Use the --transform=center-crop or --transform=center-crop-wide options to apply a
center crop transform on the input image.  These options should be used with the
--width and --height options.  For example:
\b
python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
    --transform=center-crop-wide --width 512 --height=384
"""

PIL.Image.init() # type: ignore
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-L/14", device=device)  # for image embedding
clip_model_txt, _ = clip.load("ViT-L/14", device=device) # for text embedding
clip_model.to(device).eval()
clip_model_txt.to(device).eval()

print('start')
if dest == '':
    print('--dest output filename or directory must not be an empty string')
num_files, input_iter = open_dataset(source)
print('source ready')
archive_root_dir, save_bytes, close_dest = open_dest(dest)
print('target ready')


transform_image = make_transform(transform, width, height, resize_filter)

dataset_attrs = None

labels = []
clip_img_features = []
clip_txt_features = []
s_count = 0
f_count = 0

for idx, image in tqdm(enumerate(input_iter), total=num_files):
#for filename in sorted(glob.glob(source + '/*.jpg')):
  #with open(filename) as f:
      idx_str = f'{idx:09d}'
      #archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
      #idx = os.path.basename(filename).split('.')[0]
      #idx = int(idx)
      #idx_str = f'{idx:09d}'
      #print(idx_str[:5])
      archive_fname = f'{idx_str[:5]}/img{idx_str}.jpg'
      try:
          # Apply crop and resize.

          # a = PIL.Image.fromarray(image['img'], {1: 'L', 3: 'RGB'}[channels])
          # print('before transforming')
          # a.show()

          img = transform_image(image['img'])

          # Transform may drop images.
          if img is None:
              continue

          # Error check to require uniform image attributes across
          # the whole dataset.
          channels = img.shape[2] if img.ndim == 3 else 1
          cur_image_attrs = {
              'width': img.shape[1],
              'height': img.shape[0],
              'channels': channels
          }
          if dataset_attrs is None:
              dataset_attrs = cur_image_attrs
              width = dataset_attrs['width']
              height = dataset_attrs['height']
              if width != height:
                  error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
              if dataset_attrs['channels'] not in [1, 3]:
                  error('Input images must be stored as RGB or grayscale')
              if width != 2 ** int(np.floor(np.log2(width))):
                  error('Image width/height after scale and crop are required to be power-of-two')
          if dataset_attrs == cur_image_attrs:
              #         elif dataset_attrs != cur_image_attrs:
              #             err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
              #             error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))
              with torch.no_grad():
                  # Save the image as an uncompressed PNG.
                  img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
                  # print('uncompressed')
                  # img.show()
                  feature = torch.zeros(1, 768).to(device)
                  cut_num_ = 1
                  for _ in range(cut_num_):  # random crop and resize to get the average feature of image
                      reshaped_img = custom_reshape(T.ToTensor()(img).unsqueeze(0))
                      normed_img = clip_preprocess()(reshaped_img).to(device)
                      with torch.no_grad():
                          feature += clip_model.encode_image(normed_img)
                          # print(normed_img.shape) # torch.Size([1, 3, 224, 224])
                  feature = feature / cut_num_

                  text = image['txt']
                  # print(len(text))
                  #text_filename = idx_str + '.txt'
                  #if os.path.isfile(text_filename):
                  #  text_path = os.path.join(source, text_filename)
                  #  with open(text_path) as t:
                  #    text = t.read()
                
                  text_feature_list = []
                  for text_line in text:
                      if text_line != '' and not text_line.isspace():
                          try:
                              tokenized_text = clip.tokenize([text_line]).to(device)
                              text_feature = clip_model_txt.encode_text(tokenized_text)
                              text_feature_list.append(text_feature.view(-1).cpu().numpy().tolist())
                          except:
                              # if the text is too long, we heuristically split and average the features
                              split_text = text_line.split('.')
                              split_text_list = []
                              for te in split_text:
                                  if te != '.' and te != '' and not te.isspace():
                                      split_text_list += te.split(',')
                              tokenized_text = []
                              for te in split_text_list:
                                  tokenized_text.append(clip.tokenize([te]).cuda())

                              text_feature = 0.
                              for te in tokenized_text:
                                  text_feature += clip_model_txt.encode_text(te) / len(tokenized_text)
                              text_feature_list.append(text_feature.view(-1).cpu().numpy().tolist())
                              print('text too long')

                  clip_img_features.append([archive_fname, feature.view(-1).cpu().numpy().tolist()])
                  clip_txt_features.append([archive_fname, text_feature_list])

          image_bits = io.BytesIO()
          img.save(image_bits, format='png', compress_level=0, optimize=False)
          save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
          #labels.append([archive_fname, image['label']] if image['label'] is not None else None)
          s_count += 1
      
      except:
          print(f'{archive_fname} failed')
          f_count += 1
      
      # if s_count == 10 or f_count == 10 : break

metadata = {
    'labels': labels if all(x is not None for x in labels) else None,
    'clip_img_features': clip_img_features if all(x is not None for x in clip_img_features) else None,
    'clip_txt_features': clip_txt_features if all(x is not None for x in clip_txt_features) else None,

}
save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
print(f'{s_count} {f_count}')
close_dest()
