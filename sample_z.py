import argparse
import warnings
import clip
import os.path as op
import numpy as np
import PIL
import PIL.Image
from templates import *
from templates_latent import *
from typing import List
from train_loop import LatentModel
from dataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='256, 512, 2048 available')
    parser.add_argument('--train_data_path', type=str, default='../DATA/COCO2014_train_CLIP_ViTB32.zip')
    parser.add_argument('--val_data_path', type=str, default='../DATA/COCO2014_val_CLIP_ViTB32.zip')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gpus', type=List[int], default=[0], help='recommendation: use 1 gpu')
    parser.add_argument('--target', type=str, default="img", help='candidates: img, txt, wav')
    parser.add_argument('--cfg_prob', type=float, default=0.1)
    parser.add_argument('--cfg_guidance', type=float, default=5.0)
    parser.add_argument('--lambda_1', type=float, default=1.0)
    parser.add_argument('--lambda_2', type=float, default=2.0)
    parser.add_argument('--lambda_3', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--use_default_init', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/predict_img_B32_epoch=67-step=21963-v1.ckpt")
    parser.add_argument('--text_query', type=str, default="a picture of a car")
    args = parser.parse_args()

    warnings.filterwarnings(action='ignore')
    conf = latent_conditional_ddim(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LatentModel(conf, args.train_data_path, args.val_data_path, args.cfg_prob, args.cfg_guidance, 
                        args.lambda_1, args.lambda_2, args.lambda_3, args.gamma, args.use_default_init, args.target)
    eval_path = args.checkpoint_path
    print('loading from:', eval_path)
    state = torch.load(eval_path, map_location='cpu')
    print('step:', state['global_step'])
    model.load_state_dict(state['state_dict'])
    print('finished loading checkpoint')
    model = model.to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device) # NOTE : please modify pre-trained encoder
    text = clip.tokenize([args.text_query]).to(device)
    text_features = clip_model.encode_text(text)

    out_feature = model.run_sample(text_features)
    