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
import webdataset as wds
from torch.utils.data import Dataset
from typing import List

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
#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------


# ####### Custom Implementation ######
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name

    return img_id_to_img_path


def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []

        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)

    return img_id_to_captions

# ####### Custom Implementation ######

def open_image_folder(source_dir, annotation_dir, *, max_images: Optional[int]):

    annotation_file = annotation_dir
    # print("annotation_file : ", annotation_file)
    annotations = read_json(annotation_file)

    img_id_to_filename = get_img_id_to_img_path(annotations)
    img_id_to_captions = get_img_id_to_captions(annotations)

    img_ids = list(img_id_to_filename.keys())

    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    print(f'image number {len(img_ids)}') # train : 82783

    # Load labels. (not used) 
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(img_ids), max_images)
    # print(max_idx) # train : 82783
    def iterate_images():
        for idx, fname in enumerate(img_ids):
            img_filename = img_id_to_filename[fname]
            img_path = op.join(source_dir, img_filename)

            arch_fname = os.path.relpath(img_path, source_dir)
            arch_fname = arch_fname.replace('\\', '/')

            try:
                img = np.array(PIL.Image.open(img_path))

                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                try:
                    text_line = img_id_to_captions[fname]
                    txt = text_line
                except:
                    txt = ''
            except:
                print(f'{img_path} failed')
            # exit(0)
            yield dict(img=img, label=labels.get(arch_fname), txt=txt)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()


#----------------------------------------------------------------------------
def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

def open_dataset(source, annotation, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, annotation, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

# class CC3M(Dataset):
#     def __init__(self, root, file_name_range: List):
#         self.data = []
#         self.data_len = 0

#         for n in tqdm(file_name_range):
#             url = os.path.join(root, f"{n}.tar")
#             dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg", "json")
#             for image, json in dataset:
#                 caption = json["caption"]
#                 fname = f"{json['key']}.png"
#                 self.data.append((fname, image, caption))
#                 self.data_len += 1
#             del dataset
    
#     def __len__(self):
#         return self.data_len

#     def __getitem__(self, idx):
#         return self.data[idx]
    

# def generate_file_range(num_files: int) -> List:
#     file_name_range = []
#     for i in range(num_files):
#         file_name_range.append(str(i).zfill(5))
#     return file_name_range

class CC3M(Dataset):
    def __init__(self, root, n: str):
        self.data = []
        self.data_len = 0

        url = os.path.join(root, f"{n}.tar")
        dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg", "json")
        for image, json in dataset:
            caption = json["caption"]
            fname = f"{json['key']}.png"
            self.data.append((fname, image, caption))
            self.data_len += 1
        del dataset
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]
    

def generate_file_range(num_files: int) -> List:
    file_name_range = []
    for i in range(num_files):
        file_name_range.append(str(i).zfill(5))
    return file_name_range
#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
@click.option('--emb_dim', help='CLIP embedding dim', type=int)
@click.option('--max_images', help='Output only up to `max-images` images', type=int, default=None)

def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int],
    emb_dim: Optional[int],
    max_images: Optional[int]
):
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
    clip_model, _ = clip.load("ViT-L/14")  # for image embedding
    clip_model_txt, _ = clip.load("ViT-L/14") # for text embedding
    clip_model.cuda().eval()
    clip_model_txt.cuda().eval()
    num_files = 332
    file_name_range = generate_file_range(num_files)

    print('start')
    # whole_dataset = 
    # print('source ready, total', len(whole_dataset), 'files')
    print('source ready')
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    print('target ready')

    transform_image = make_transform(transform, width, height, resize_filter)
    dataset_attrs = None

    # labels = []
    clip_img_features = []
    clip_txt_features = []
    s_count = 0
    f_count = 0

    for n in file_name_range:
        print(f"Processing {n}.tar")
        whole_dataset = CC3M(source, n)
        for idx, data in tqdm(enumerate(whole_dataset), total=len(whole_dataset)):
            idx_str = f'{(idx + s_count + f_count):08d}'
            fname, image, caption = data
            archive_fname = f"{idx_str[:5]}/img{fname}"

            try:
                image = np.array(image)
                # Apply crop and resize.
                # print(type(image), np.max(image), np.min(image), image.shape)
                img = transform_image(image)
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
                        feature = torch.zeros(1, emb_dim).cuda()
                        cut_num_ = 1
                        for _ in range(cut_num_):  # random crop and resize to get the average feature of image
                            reshaped_img = custom_reshape(T.ToTensor()(img).unsqueeze(0))
                            normed_img = clip_preprocess()(reshaped_img).cuda()
                            with torch.no_grad():
                                feature += clip_model.encode_image(normed_img)
                                # print(normed_img.shape) # torch.Size([1, 3, 224, 224])
                        feature = feature / cut_num_

                        text = [caption]
                        text_feature_list = []
                        for text_line in text:
                            # print(text_line)
                            if text_line != '' and not text_line.isspace():
                                try:
                                    tokenized_text = clip.tokenize([text_line]).cuda()
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
                # labels.append([archive_fname, image['label']] if image['label'] is not None else None)
                s_count += 1
            
            except:
                print(f'{archive_fname} failed')
                f_count += 1
            
            if max_images is not None and \
                ((s_count > 0 and s_count % max_images == 0) or\
                 (f_count > 0 and f_count % max_images == 0)) : break
        
        del whole_dataset

    metadata = {
        'clip_img_features': clip_img_features if all(x is not None for x in clip_img_features) else None,
        'clip_txt_features': clip_txt_features if all(x is not None for x in clip_txt_features) else None,
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    print(f'{s_count} {f_count}')
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
