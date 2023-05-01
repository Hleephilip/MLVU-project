import argparse
import warnings
from templates import *
from templates_latent import *
from typing import List

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256, help='256, 512, 2048 available')
    parser.add_argument('--train_data_path', type=str, default='../DATA/COCO2014_train_CLIP_ViTL14_v2.zip')
    parser.add_argument('--val_data_path', type=str, default='../DATA/COCO2014_val_CLIP_ViTL14_v2.zip')
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--checkpoint_name', type=str, default='last')
    parser.add_argument('--checkpoint_top_k', type=int, default=1)
    parser.add_argument('--use_pretrained', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_name', type=str, default='test_1')
    parser.add_argument('--log_version', type=str, default='')
    parser.add_argument('--log_print_interval', type=int, default=1)
    parser.add_argument('--gpus', type=List[int], default=[0], help='recommendation: use 1 gpu')
    parser.add_argument('--target', type=str, default='txt', help='candidates: img, txt, wav')
    parser.add_argument('--cond', type=str, default='img', help='candidates: img, txt, wav')
    parser.add_argument('--cfg_prob', type=float, default=0.1)
    parser.add_argument('--cfg_guidance', type=float, default=5.0)
    
    args = parser.parse_args()
    
    warnings.filterwarnings(action='ignore')
    assert args.batch_size in [256, 512, 2048]
    conf = latent_conditional_ddim(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(conf, device=device, args=args)
    
