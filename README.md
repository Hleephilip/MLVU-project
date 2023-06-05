# MLVU-project
2023-1 Machine Learning for Visual Understanding Team project

work in progress


## Requirements

See [requirements.txt](https://github.com/frogyunmax/MLVU-project/blob/main/requirements.txt)

```
pip install -r requirements.txt
```

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

### Training

Implementation based on [Diffusion Autoencoders](https://github.com/phizaz/diffae).

Submit a job for training :

```
sbatch run.sh
```

Command for python (inside `run.sh`) : 

```
python train_latent_ddim.py --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                            --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                            --epochs 100 --batch_size 256 --learning_rate 1e-4 \
                            --log_name "exp1" --log_version "v0" --target "txt" --use_default_init
```

Using only 1 gpu is highly recommended.


### Evaluation

Command for python (inside `run.sh`) : 

```
python eval_latent_ddim.py --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                           --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                           --checkpoint_path <YOUR_CHECKPOINT_PATH> \
                           --batch_size 2048 --target "txt" --use_default_init
```

### Inference model using text prompt (Generate image embedding)

Command for python (inside `run.sh`) : 

```
python sample_z.py  --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                    --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                    --cfg_prob 0.05 --use_default_init \
                    --target "img" --checkpoint_path <YOUR_CHECKPOINT_PATH> \
                    --text_query "A white boat floating on a lake under mountains"
```

## Downstream tasks

TBD

## Experiments

Check results in [here](https://docs.google.com/spreadsheets/d/1iorV8BHk1StLiq6OIMCb_GdRo_Lg9sVqjwcGnNep-6A/edit#gid=1980661715).

## Paper

[Overleaf Link](https://www.overleaf.com/project/6453de916a02aa1601de1b36).

