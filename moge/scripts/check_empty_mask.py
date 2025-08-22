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
from PIL import Image
from pathlib import Path
import json
import wandb
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from collov.ade20k_indoor import KEEP_LABEL2ID
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

class ImageFileDataset(Dataset):
    def __init__(self, image_root, mask_root,
                 empty_img_root, empty_mask_root,
                 path_list, transform=tvt.ToTensor()):
        self.image_root = image_root
        self.mask_root = mask_root
        self.empty_img_root = empty_img_root
        self.empty_mask_root = empty_mask_root
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        ext = Path(path).suffix
        mask_path = osp.join(self.mask_root, self.path_list[idx])
        empty_mask_path = osp.join(self.empty_mask_root, self.path_list[idx]).replace(ext, '.png')
        img_path = osp.join(self.image_root, self.path_list[idx])
        empty_img_path = osp.join(self.empty_img_root, self.path_list[idx])

        try:
            mask = Image.open(mask_path)
            empty_mask = Image.open(empty_mask_path)
            mask = np.array(mask)
            empty_mask = np.array(empty_mask)

            # image_pil = Image.open(img_path).convert('RGB')
            # empty_img_pil = Image.open(empty_img_path)
            # image = np.array(image_pil)
            # empty_img = np.array(empty_img_pil)

        except:
            print(f"Error on image: {mask_path}")
            return None

        return {
            # 'image_pil': image_pil,
            # 'empty_img_pil': empty_img_pil,
            'mask': mask,
            'empty_mask': empty_mask,
            # 'image': image,
            # 'empty_img': empty_img,
            'path': self.path_list[idx],
        }
    
########################################
def check_empty_mask(func_args):
    idx, dataset = func_args

    batch = dataset[idx]
    # image = batch['image']
    if batch is None:
        return None

    try:
        path = str(Path(batch['path']))
        fn = Path(path).stem
        mask = batch['mask']
        empty_mask = batch['empty_mask']
        
        trans_fn = A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            # A.CenterCrop(256, 480, p=1.0),
        ])
        # image = trans_fn(image=image)['image']
        mask = trans_fn(image=mask)['image']
        empty_mask = trans_fn(image=empty_mask)['image']

        info = {
            'path': path,
            'fn': fn,
        }
        mask_info = calc_mask_info(mask, empty_mask)
        info.update(mask_info)

        # key = batch['fn']
        # print(f'[{key}]')
        # print(f'div_fix_but_changed = {div_fix_but_changed:.2f}')
        
        # Convert mask to uint8 and add channel dimension
        # m_vis = (fix_but_changed.astype(np.uint8) * 255)[..., np.newaxis].repeat(3, axis=2)
        
        # image = batch['image']
        # empty_img = batch['empty_img']
        # image = trans_fn(image=image)['image']
        # empty_img = trans_fn(image=empty_img)['image']
        # canvas = np.concatenate([
        #     image,
        #     empty_img,
        #     m_vis,
        # ], axis=1)
        # side_by_sidy = np.concatenate([
        #     batch['image'],
        #     batch['empty_img'],
        # ], axis=1)
        
        # if div_median < 0.7 or div_median > 1.4:
        #     print(f'[{key}]')
        #     print(f'empty_depth/depth: median = {div_median:.2f}, diff10 = {diff10:.2f}, diff20 = {diff20:.2f}')
            
        """
        wandb.log({
            f"div_fix_but_changed: {div_fix_but_changed}":
                wandb.Image(canvas, caption=f"{key}"),
        })
        """

        # image_pil = np.array(batch['image_pil'])
        return info

    except:
        print('Process crashed')
        # print(f'Save error image: {img_path}')
        # err_fn = img_path.replace(image_root, err_root)
        # print(f'To {err_fn}')
        # os.makedirs(osp.dirname(err_fn), exist_ok=True)
        # shutil.copyfile(img_path, err_fn)
        return None

    pass

########################################
import concurrent.futures

def parallel_foreach(_func, _args_list, max_workers=8, show_tqdm=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        res_list = []
        if show_tqdm:
            from tqdm import tqdm
            for res in tqdm(executrer.map(_func, _args_list), total=len(_args_list)):
                res_list.append(res)
        else:
            for res in executrer.map(_func, _args_list):
                res_list.append(res)
    return res_list


########################################
def parse_args():
    parser = argparse.ArgumentParser(description="Check empty depth directories")
    parser.add_argument("--data_root", required=True, help="Root data directory")
    parser.add_argument("--image_root", default="image", help="Image root directory")
    parser.add_argument("--mask_root", default="mask", help="Mask root directory")
    parser.add_argument("--depth_root", default="depth_moge2", help="Depth root directory")
    parser.add_argument("--empty_img_root", default="empty_syn", help="Empty image root directory")
    parser.add_argument("--empty_mask_root", default="mask_empty_syn", help="Empty mask root directory")
    parser.add_argument("--empty_depth_root", default="depth_moge2_empty_syn", help="Empty depth root directory")
    parser.add_argument("--err_root", default="err_empty_syn", help="Error root directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check and convert paths to absolute
    image_root = osp.join(args.data_root, args.image_root) if not osp.isabs(args.image_root) else args.image_root
    mask_root = osp.join(args.data_root, args.mask_root) if not osp.isabs(args.mask_root) else args.mask_root
    depth_root = osp.join(args.data_root, args.depth_root) if not osp.isabs(args.depth_root) else args.depth_root
    empty_img_root = osp.join(args.data_root, args.empty_img_root) if not osp.isabs(args.empty_img_root) else args.empty_img_root
    empty_mask_root = osp.join(args.data_root, args.empty_mask_root) if not osp.isabs(args.empty_mask_root) else args.empty_mask_root
    empty_depth_root = osp.join(args.data_root, args.empty_depth_root) if not osp.isabs(args.empty_depth_root) else args.empty_depth_root
    err_root = osp.join(args.data_root, args.err_root) if not osp.isabs(args.err_root) else args.err_root

    print(f"image_root: {image_root}")
    print(f"mask_root: {mask_root}")
    print(f"empty_img_root: {empty_img_root}")
    print(f"empty_mask_root: {empty_mask_root}")
    print(f"err_root: {err_root}")

    """
    wandb.init(project="MoGe", name="check_empty_mask")
    """

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_name = "MoGe"
    #     accelerator.init_trackers(tracker_name, config=vars(args))

    # scandir by the main thread
    image_paths = scandir(mask_root, suffix=IMAGE_SUFFIX, recursive=True)
    exist_paths = []
    for path in tqdm(image_paths, desc='scandir ...'):
        img_path = osp.join(mask_root, path)
        ext = Path(path).suffix
        empty_img_path = osp.join(empty_mask_root, path).replace(ext, '.png')
        if osp.exists(img_path) and osp.exists(empty_img_path):
            exist_paths.append(path)

        # if len(exist_paths) > 10000:
        #     break

    exist_paths = list(set(exist_paths))
    print(f'scan {len(exist_paths)} existing paths with empty mask images')
    path_list = exist_paths

    # dataset
    dataset = ImageFileDataset(image_root, mask_root,
                               empty_img_root, empty_mask_root,
                               path_list)
    check_empty_mask((0, dataset))
    all_infos = []

    num_workers = 64
    print(f"Processing {len(dataset)} items with {num_workers} workers")

    all_infos = parallel_foreach(check_empty_mask, [(idx, dataset) for idx in range(len(dataset))],
                                 max_workers=num_workers, show_tqdm=True)

    # # Use a global variable for dataset
    # global_dataset = dataset

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(check_empty_mask, idx, global_dataset) for idx in range(len(dataset))]
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
    #         result = future.result()
    #         if result is not None:
    #             all_infos.append(result)
    print(f"Processed {len(all_infos)} valid results out of {len(dataset)} total items")

    with open(f'{err_root}/check_empty_mask.json', 'w') as fp:
        json.dump(all_infos, fp)

    pass


if __name__ == "__main__":
    main()


