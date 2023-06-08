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
from collections import OrderedDict

def print_and_save(ret_ours: dict, ret_baseline: dict, idx_dict: dict):
    os.makedirs('retrieval_result/ours', exist_ok=True)
    os.makedirs('retrieval_result/baseline', exist_ok=True)
    print("[Result] Ours")
    for i, (k, v) in enumerate(ret_ours):
        print(f"Top {i + 1} : {idx_dict[k]} ({100 *v:.4f}%)")
        img = Image.fromarray(np.array(PIL.Image.open(idx_dict[k])))
        img.save(os.path.join('retrieval_result/ours', f"top_{i+1}.png"))
    
    print("\n[Result] Baseline")
    for i, (k, v) in enumerate(ret_baseline):
        print(f"Top {i + 1} : {idx_dict[k]} ({100 *v:.4f}%)")
        img = Image.fromarray(np.array(PIL.Image.open(idx_dict[k])))
        img.save(os.path.join('retrieval_result/baseline', f"top_{i+1}.png"))
    return

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

def open_image_folder(source_dir, annotation_dir):

    annotation_file = annotation_dir
    # print("annotation_file : ", annotation_file)
    annotations = read_json(annotation_file)

    img_id_to_filename = get_img_id_to_img_path(annotations)
    img_id_to_captions = get_img_id_to_captions(annotations)

    img_ids = list(img_id_to_filename.keys())

    # input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    # print(f'image number {len(img_ids)}') # train : 82783

    idx_dict = {}
    for idx, fname in enumerate(img_ids):
        img_filename = img_id_to_filename[fname]
        img_path = op.join(source_dir, img_filename)
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        idx_dict[archive_fname] = img_path
    return idx_dict


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
    parser.add_argument('--num_images_sample', type=int, default=10)
    parser.add_argument('--condition_dim', type=int, default=512)
    parser.add_argument('--x_dim', type=int, default=512)
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

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([args.text_query]).to(device)
    text_features = clip_model.encode_text(text)

    out_feature = model.run_sample(text_features)

    print("Loading datasets ...")
    if args.target == 'img':
        dataset_for_retrieval = ZipDataset_img(args.val_data_path, return_fname=True)
    elif args.target == 'txt':
        dataset_for_retrieval = ZipDataset_txt(args.val_data_path, return_fname=True)
    else:
        raise NotImplementedError()

    cos_sim = torch.nn.CosineSimilarity()

    print("Run retrieval ... Our model")
    dataloader = torch.utils.data.DataLoader(dataset_for_retrieval, batch_size=1, shuffle=False)
    ret_ours = {}
    min_sim_file = None
    min_sim = 1.0
    for batch in tqdm(dataloader, total=len(dataset_for_retrieval)):
        fname, cond, tgt = batch
        cond, tgt = cond.to(device), tgt.to(device)
        fname = fname[0]
        similarity = cos_sim(tgt, out_feature).item()
        if len(ret_ours) < args.num_images_sample: 
            ret_ours[fname] = similarity
            if similarity < min_sim:
                min_sim_file = fname
                min_sim = similarity
        
        else:
            if similarity <= min_sim : pass
            else:
                del ret_ours[min_sim_file]
                ret_ours[fname] = similarity
                min_sim_file = [k for k, v in ret_ours.items() if min(ret_ours.values()) == v][0]
                min_sim = ret_ours[min_sim_file]

    print("Run retrieval ... Baseline")
    dataloader = torch.utils.data.DataLoader(dataset_for_retrieval, batch_size=1, shuffle=False)
    ret_baseline = {}
    min_sim_file = None
    min_sim = 1.0
    for batch in tqdm(dataloader, total=len(dataset_for_retrieval)):
        fname, cond, tgt = batch
        cond, tgt = cond.to(device), tgt.to(device)
        fname = fname[0]
        similarity = cos_sim(tgt, text_features).item()
        if len(ret_baseline) < args.num_images_sample: 
            ret_baseline[fname] = similarity
            if similarity < min_sim:
                min_sim_file = fname
                min_sim = similarity
        
        else:
            if similarity <= min_sim : pass
            else:
                del ret_baseline[min_sim_file]
                ret_baseline[fname] = similarity
                min_sim_file = [k for k, v in ret_baseline.items() if min(ret_baseline.values()) == v][0]
                min_sim = ret_baseline[min_sim_file]

    # idx_dict = open_image_folder("../DATA/mscoco_data/val2014/", "../DATA/mscoco_data/annotations/captions_val2014.json")
    # torch.save(idx_dict, 'assets/COCO_val_img_idx.pt')
    idx_dict = torch.load('assets/COCO_val_img_idx.pt')
    print_and_save(sorted(ret_ours.items(), reverse=True, key = lambda item: item[1]),
                   sorted(ret_baseline.items(), reverse=True, key = lambda item: item[1]),
                   idx_dict)