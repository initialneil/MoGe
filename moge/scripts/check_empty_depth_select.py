import argparse
import os
import os.path as osp
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
from PIL import Image
from pathlib import Path
import json
import wandb
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from collov.jsonl import read, write

IMAGE_SUFFIX = ("png", "jpg", "jpeg", "webp")
ADE20K_WALL = 0
ADE20K_FLOOR = 3
ADE20K_WIN = 8
ADE20K_DOOR = 14
ADE20K_FIXED = [
    ADE20K_WALL, ADE20K_FLOOR, ADE20K_WIN,
    ADE20K_DOOR,
]

def calc_fixed_mask(mask, fixed_ids):
    m = np.zeros_like(mask)
    for id in fixed_ids:
        m = m | (mask == id)
    return m

def calc_mask_info(mask, empty_mask):
    # fixed areas
    mask_fixed = calc_fixed_mask(mask, ADE20K_FIXED)
    empty_mask_fixed = calc_fixed_mask(empty_mask, ADE20K_FIXED)
    both_fixed = mask_fixed & empty_mask_fixed

    fix_but_changed = both_fixed & (mask != empty_mask)
    div_fix_but_changed = fix_but_changed.sum() / (both_fixed.sum() + 1e-3)

    info = {
        'div_fix_but_changed': div_fix_but_changed.item(),
        'fix_but_changed': fix_but_changed.sum().item(),
        'both_fixed': both_fixed.sum().item(),
    }
    return info

class JsonlDataset(Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # image = Image.open(osp.join(self.data_root, item['image'])).convert('RGB')
        try:
            image_depth = Image.open(osp.join(self.data_root, item['image_depth']))
            empty_depth = Image.open(osp.join(self.data_root, item['empty_depth']))
            image_mask = Image.open(osp.join(self.data_root, item['image_mask']))
            empty_mask = Image.open(osp.join(self.data_root, item['empty_mask']))
        except:
            print(f"Error loading image: {item['image']}")
            return {
                'image_depth': None,
                'empty_depth': None,
                'image_mask': None,
                'empty_mask': None,
                'item': item,
                'fn': Path(item['image']).stem,
            }

        image_mask = np.array(image_mask)
        empty_mask = np.array(empty_mask)
        image_depth = np.array(image_depth).astype(float) / 4000.0
        empty_depth = np.array(empty_depth).astype(float) / 4000.0

        return {
                'image_depth': image_depth,
                'empty_depth': empty_depth,
                'image_mask': image_mask,
                'empty_mask': empty_mask,
                'item': item,
                'fn': Path(item['image']).stem,
        }
    
def check_empty_depth(data_root, input_json_fn):
    # dataset
    data_list = read(input_json_fn)
    dataset = JsonlDataset(data_root, data_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True,
                            collate_fn=lambda x: x)

    # Initialize info dict for gathering results from all processes
    info_list = []
    
    for batches in tqdm(dataloader):
        for batch in batches:
            key = batch['item']['image']
            image_depth = batch['image_depth']
            if image_depth is None:
                # print(f'Save error image: {img_path}')
                # err_fn = img_path.replace(image_root, err_root)
                # print(f'To {err_fn}')
                # os.makedirs(osp.dirname(err_fn), exist_ok=True)
                # shutil.copyfile(img_path, err_fn)
                continue

            try:
                image_mask, image_depth = batch['image_mask'], batch['image_depth']
                empty_mask, empty_depth = batch['empty_mask'], batch['empty_depth']
                
                trans_fn = A.Compose([
                    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
                    # A.CenterCrop(256, 480, p=1.0),
                ])
                image_mask = trans_fn(image=image_mask)['image']
                image_depth = trans_fn(image=image_depth)['image']
                empty_mask = trans_fn(image=empty_mask)['image']
                empty_depth = trans_fn(image=empty_depth)['image']

                m = (image_mask == ADE20K_WALL)
                m_image_depth = torch.tensor(image_depth[m])
                m_empty_depth = torch.tensor(empty_depth[m])
                count = m.sum()
                div_median = (m_empty_depth / m_image_depth).median()
                div_mean = (m_empty_depth / m_image_depth).mean()
                diff10 = ((m_empty_depth / m_image_depth - 1.0).abs() > 0.1).sum() / count
                diff20 = ((m_empty_depth / m_image_depth - 1.0).abs() > 0.2).sum() / count
                diff50 = ((m_empty_depth / m_image_depth - 1.0).abs() > 0.5).sum() / count
                diff100 = ((m_empty_depth / m_image_depth - 1.0).abs() > 1.0).sum() / count
                diff200 = ((m_empty_depth / m_image_depth - 1.0).abs() > 2.0).sum() / count
                diff_median = (m_empty_depth - m_image_depth).median()
                diff_mean = (m_empty_depth - m_image_depth).mean()

                info = {
                    'key': key,
                    'empty_depth: median': m_empty_depth.median().item(),
                    'empty_depth: mean': m_empty_depth.mean().item(),
                    'depth: median': m_image_depth.median().item(),
                    'depth: mean': m_image_depth.mean().item(),
                    'empty_depth/depth: median': div_median.item(),
                    'empty_depth/depth: mean': div_mean.item(),
                    'empty_depth/depth: diff10': diff10.item(),
                    'empty_depth/depth: diff20': diff20.item(),
                    'empty_depth/depth: diff50': diff50.item(),
                    'empty_depth/depth: diff100': diff100.item(),
                    'empty_depth/depth: diff200': diff200.item(),
                    'empty_depth - depth: median': diff_median.item(),
                    'empty_depth - depth: mean': diff_mean.item(),
                }

                mask_info = calc_mask_info(image_mask, empty_mask)
                info.update(mask_info)
                # print(f'[{key}]')
                # print(f'empty_depth/depth: median = {median:.2f}, diff10 = {diff10:.2f}, diff20 = {diff20:.2f}')
                
                # # Convert mask to uint8 and add channel dimension
                # m_vis = (m.astype(np.uint8) * 255)[..., np.newaxis].repeat(3, axis=2)
                # # Convert depth arrays to uint8 and add channel dimension
                # depth_vis = (np.clip(image_depth / 10.0, 0, 1) * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=2)
                # empty_depth_vis = (np.clip(empty_depth / 10.0, 0, 1) * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=2)
                
                # canvas = np.concatenate([
                #     image,
                #     empty_img,
                #     m_vis,
                #     depth_vis,
                #     empty_depth_vis,
                # ], axis=1)
                # side_by_sidy = np.concatenate([
                #     batch['image'],
                #     batch['empty_img'],
                # ], axis=1)
                
                if div_median < 0.7 or div_median > 1.4:
                    print(f'[{key}]')
                    print(f'empty_depth/depth: median = {div_median:.2f}, diff10 = {diff10:.2f}, diff20 = {diff20:.2f}')
                    
                    """
                    wandb.log({
                        "side_by_side":
                            wandb.Image(side_by_sidy, caption=f"{key}"),
                        f"empty_depth/depth":
                            wandb.Image(canvas, caption=f"empty_depth/depth: median = {div_median:.2f}, diff10 = {diff10:.2f}, diff20 = {diff20:.2f}"),
                    })
                    """

                # image_pil = np.array(batch['image_pil'])
                info_list.append(info)
                pass

            except:
                print('Process crashed')
                # print(f'Save error image: {img_path}')
                # err_fn = img_path.replace(image_root, err_root)
                # print(f'To {err_fn}')
                # os.makedirs(osp.dirname(err_fn), exist_ok=True)
                # shutil.copyfile(img_path, err_fn)
                continue

            pass
    return info_list

def parse_args():
    parser = argparse.ArgumentParser(description="Check empty depth directories")
    parser.add_argument("--data_root", required=True, help="Root data directory")
    parser.add_argument("--input_json_fn", help="Input JSON file name")
    parser.add_argument("--output_json_fn", help="Output JSON file name")
    return parser.parse_args()

def main():
    args = parse_args()
    def to_abspath(p, root):
        if osp.isabs(p):
            return p
        return osp.join(root, p)

    input_json_fn = to_abspath(args.input_json_fn, args.data_root)
    output_json_fn = to_abspath(args.output_json_fn, args.data_root)

    check_dir = osp.join(args.data_root, "check")
    os.makedirs(check_dir, exist_ok=True)
    check_fn = osp.join(check_dir, f"{Path(args.input_json_fn).stem}[check_empty_depth].json")
    if osp.exists(check_fn):
        with open(check_fn, "r") as fp:
            info_list = json.load(fp)
    else:
        info_list = check_empty_depth(args.data_root, input_json_fn)
        with open(check_fn, "w") as fp:
            json.dump(info_list, fp)


    ########################################
    # check depth results
    depth_median = np.array([c["depth: median"] for c in info_list if c is not None])
    depth_mean = np.array([c["depth: mean"] for c in info_list if c is not None])
    div_median = np.array([c["empty_depth/depth: median"] for c in info_list if c is not None])
    div_mean = np.array([c["empty_depth/depth: mean"] for c in info_list if c is not None])
    print(f"{len(depth_median)} valid frames")

    div_fix_but_changed = np.array([c["div_fix_but_changed"] for c in info_list if c is not None])
    fix_but_changed = np.array([c["fix_but_changed"] for c in info_list if c is not None])
    both_fixed = np.array([c["both_fixed"] for c in info_list if c is not None])
    print(f"{len(div_fix_but_changed)} valid frames")

    ########################################
    # check select
    low_thresh, high_thresh = 0.95, 1.05
    select_idxs = (div_median > low_thresh) & (div_median < high_thresh)
    select_idxs &= (div_mean > low_thresh) & (div_mean < high_thresh)
    print(f"select in ({low_thresh}, {high_thresh}): {select_idxs.sum()} frames")

    change_thresh = 0.05
    select_idxs = (div_fix_but_changed < change_thresh)
    print(f"select mask {select_idxs.sum()} frames")

    # select files
    selected_info = [
        c for c in info_list if (
            c is not None and (c["empty_depth/depth: median"] > low_thresh) and (
                c["empty_depth/depth: median"] < high_thresh) and (
                    c["empty_depth/depth: mean"] > low_thresh) and (
                        c["empty_depth/depth: mean"] < high_thresh) and (
                            c["div_fix_but_changed"] < change_thresh)
        )
    ]
    print(f"select both depth and mask: {len(selected_info)} frames")

    selected_keys = set([c['key'] for c in selected_info])

    ########################################
    # load jsonl
    all_data = read(input_json_fn)

    select_data = [d for d in all_data if d['image'] in selected_keys]
    write(select_data, output_json_fn)
    print(select_data[:1])

    pass


if __name__ == "__main__":
    main()


