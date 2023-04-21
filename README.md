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
python preproc_datasets.py --source=../mscoco_data/train2014/ \
                           --annotation=../mscoco_data/annotations/captions_train2014.json \
                           --dest=COCO2014_train_CLIP_ViTL14.zip --width=256 --height=256 \
                           --transform=center-crop --emb_dim 768
```

the files at ../mscoco_data/ is like:

```
../mscoco/data
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

For preprocessing, refer to [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py) and [Hugging Face](https://huggingface.co/datasets/conceptual_captions/tree/main).
