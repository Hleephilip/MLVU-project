# MLVU-project
2023-1 Machine Learning for Visual Understanding Team project

work in progress


## Requirements

The implementation is based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [CLIP](https://github.com/openai/CLIP), the required packages can be found in the links.

Or refer to [requirements.txt](https://github.com/frogyunmax/MLVU-project/blob/main/requirements.txt)


## Data Preprocessing

### MS-COCO

Dataset can be downloaded from [here](https://cocodataset.org/#download).

Command for training set preprocessing:

```
python preproc_datasets.py --source=../DATA/mscoco_data/train2014/ \
                           --annotation=../DATA/mscoco_data/annotations/captions_train2014.json \
                           --dest=COCO2014_train_CLIP_ViTL14.zip --width=256 --height=256 \
                           --transform=center-crop --emb_dim 768
```

The files at directory `../DATA/mscoco_data/` is like:

```
../DATA/mscoco_data/
  ├── train2014
  │   ├── COCO_train2014_000000000009.jpg
  │   ├── COCO_train2014_0000000000025.jpg
  │   └── ...
  ├── val2014
  │   ├── COCO_val2014_0000000000042.jpg
  │   ├── COCO_val2014_0000000000073.jpg
  │   └── ...
  └── annotations
      ├── captions_train2014.json
      ├── captions_val2014.json
      └── ...
``` 


### CC3M

Dataset can be downloaded from [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md). Total 77G storage is required.

Command for training set preprocessing:

```
python preproc_datasets_cc3m.py --source=../DATA/cc3m_train/ \
                                --dest=CC3M_train_CLIP_ViTL14.zip --width=256 --height=256 \
                                --transform=center-crop --emb_dim 768
```

There are total 996 files in `../DATA/cc3m_train/`. Directory structure is like:

```
../DATA/cc3m_train/
  ├── 00000_stats.json
  ├── 00000.parquet
  ├── 00000.tar
  ├── ...
  └── 00331.tar
``` 

## Conditional Encoder-Decoder

Implementation based on [Diffusion Autoencoders](https://github.com/phizaz/diffae).

Command for training :

```
python train_latent_ddim.py --data_path=../DATA/COCO2014_train_CLIP_ViTL14_v2.zip \
                            --epochs=100 --batch_size=256 --learning_rate=1e-4 \
                            --log_name='exp1' --log_version='0' --gpus=[1] --cfg_prob=0.1
```

Using only 1 gpu is highly recommended.
