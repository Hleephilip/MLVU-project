# Modality Translation through Conditional Encoder-Decoder

Machine Learning for Visual Understanding (M3224.000100, Spring 2023) Team project

Hyunsoo Lee, Yoonsang Lee, Maria Pak, Jinri Kim

Seoul National University

## Abstract

With the recent rise of multi-modal learning, many of the developed models are task-specific, and, as a consequence, lack generalizability when applied to other types of downstream tasks. One of the representative models that overcomes this issue of generalizability is CLIP, which attacks downstream tasks using cosine similarity metric. However, CLIP has shown relatively low cosine similarity between text and image vector representations. For this reason, we aim to develop a new approach that more accurately maps the hyperplanes of text and image embeddings, and thus, achieves a high-quality text-image modality translation. To this end, we propose a new conditional encoder-decoder model that maps a latent space of one modality given another modality as a condition. We observe that our model is a general method that can be used with various latent encoders and decoders, which are not limited to multi-modal models. Experiments show that conditional encoder-decoder achieves comparable results with the previous state-of-the-art on several downstream tasks. 

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

Dataset can be downloaded from [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md). 

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

Command for training conditional encoder-decoder using MS-COCO dataset: 

```
python train_latent_ddim.py --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                            --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                            --epochs 100 --batch_size 256 --learning_rate 1e-4 \
                            --log_name "exp1" --target "txt" --use_default_init \
                            --x_dim 512 --condition_dim 512 --lambda_2 2.0 --cfg_prob 0.05 \
                            --checkpoint_path <YOUR CHECKPOINT PATH>
```

Command for training conditional encoder-decoder using CC3M dataset: 

```
python train_latent_ddim_cc3m.py --train_data_path "../DATA/CC3M_processed/CC3M_train_CLIP_ViTL14" \
                                 --val_data_path "../DATA/CC3M_processed/CC3M_val_CLIP_ViTL14.zip" \
                                 --epochs 100 --batch_size 256 --learning_rate 1e-4 \
                                 --log_name "exp1" --target "txt" --use_default_init \
                                 --x_dim 768 --condition_dim 768 --lambda_2 2.0 --cfg_prob 0.05 \
                                 --checkpoint_path <YOUR CHECKPOINT PATH>
```

DDP plugin is implemented, however using a single gpu is highly recommended.

### Evaluation

Command for evaluating trained conditional encoder-decoder: 

```
python eval_latent_ddim.py --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                           --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                           --saved_checkpoint_path <YOUR_CHECKPOINT_PATH> \
                           --batch_size 2048 --target "txt" --use_default_init \
                           --x_dim 768 --condition_dim 768 --lambda_2 2.0 --cfg_prob 0.05
```

### Inference model using text prompt (Generate image embedding)

Command for python : 

```
python sample_z.py  --train_data_path "../DATA/COCO2014_train_CLIP_ViTB32.zip" \
                    --val_data_path "../DATA/COCO2014_val_CLIP_ViTB32.zip" \
                    --cfg_prob 0.05 --use_default_init \
                    --target "img" --checkpoint_path <YOUR_CHECKPOINT_PATH> \
                    --text_query "A white boat floating on a lake under mountains"
```

## Downstream tasks

- Text-to-Image Generation
- Image Retrieval
- Image Captioning
- Image Classification

## Requirements

See [requirements.txt](https://github.com/frogyunmax/MLVU-project/blob/main/requirements.txt)

```
pip install -r requirements.txt
```

## Paper

Check in [here](https://drive.google.com/file/d/1nXQzt6FHOkRugbepxe6ukWz76u_oP-Ln/view?usp=sharing).

## Presentation

TBD


## Acknowledgments

This repository is implementation based on [Diffusion Autoencoders](https://github.com/phizaz/diffae).
