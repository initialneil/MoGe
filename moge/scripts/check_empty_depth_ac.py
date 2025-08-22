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
from moge.collov.files import scandir

IMAGE_SUFFIX = ("png", "jpg", "jpeg", "webp")
ADE20K_WALL = 0

class ImageFileDataset(Dataset):
    def __init__(self, image_root, mask_root, depth_root,
                 empty_img_root, empty_depth_root,
                 path_list, transform=tvt.ToTensor()):
        self.image_root = image_root
        self.mask_root = mask_root
        self.depth_root = depth_root
        self.empty_img_root = empty_img_root
        self.empty_depth_root = empty_depth_root
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = osp.join(self.image_root, self.path_list[idx])
        ext = Path(img_path).suffix
        mask_path = osp.join(self.mask_root, self.path_list[idx])
        depth_path = osp.join(self.depth_root, self.path_list[idx]).replace(ext, '.png')
        empty_img_path = osp.join(self.empty_img_root, self.path_list[idx])
        empty_depth_path = osp.join(self.empty_depth_root, self.path_list[idx]).replace(ext, '.png')

        try:
            image_pil = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)
            depth = Image.open(depth_path)
            empty_img_pil = Image.open(empty_img_path)
            empty_depth = Image.open(empty_depth_path)
        except:
            print(f"Error loading image: {img_path}")
            return {
                'image': None,
                'image_tensor': None,
                'fn': self.file_list[idx],
                'img_path': img_path,
            }

        image = np.array(image_pil)
        empty_img = np.array(empty_img_pil)
        mask = np.array(mask)
        depth = np.array(depth).astype(float) / 4000.0
        empty_depth = np.array(empty_depth).astype(float) / 4000.0

        return {
            'image_pil': image_pil,
            'empty_img_pil': empty_img_pil,
            'image': image,
            'mask': mask,
            'depth': depth,
            'empty_img': empty_img,
            'empty_depth': empty_depth,
            'fn': self.path_list[idx],
            'img_path': img_path,
            'mask_path': mask_path,
            'depth_path': depth_path,
            'empty_img_path': empty_img_path,
            'empty_depth_path': empty_depth_path,
        }
    
def parse_args():
    parser = argparse.ArgumentParser(description="Check empty depth directories")
    parser.add_argument("--data_root", required=True, help="Root data directory")
    parser.add_argument("--image_root", default="image", help="Image root directory")
    parser.add_argument("--mask_root", default="mask", help="Mask root directory")
    parser.add_argument("--depth_root", default="depth_moge2", help="Depth root directory")
    parser.add_argument("--empty_img_root", default="empty_syn", help="Empty image root directory")
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
    empty_depth_root = osp.join(args.data_root, args.empty_depth_root) if not osp.isabs(args.empty_depth_root) else args.empty_depth_root
    err_root = osp.join(args.data_root, args.err_root) if not osp.isabs(args.err_root) else args.err_root

    # accelerate
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"image_root: {image_root}")
        print(f"mask_root: {mask_root}")
        print(f"depth_root: {depth_root}")
        print(f"empty_img_root: {empty_img_root}")
        print(f"empty_depth_root: {empty_depth_root}")
        print(f"err_root: {err_root}")

    """
    wandb.init(project="MoGe", name="check_empty_depth")
    """

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_name = "MoGe"
    #     accelerator.init_trackers(tracker_name, config=vars(args))

    # scandir by the main thread
    if accelerator.is_main_process:
        image_paths = scandir(image_root, suffix=IMAGE_SUFFIX, recursive=True)
        exist_paths = []
        for path in tqdm(image_paths, desc='scandir ...'):
            img_path = osp.join(image_root, path)
            ext = Path(img_path).suffix
            empty_img_path = img_path.replace(image_root, empty_img_root)
            if osp.exists(img_path) and osp.exists(empty_img_path):
                exist_paths.append(path)

            # if len(exist_paths) > 10000:
            #     break

        exist_paths = list(set(exist_paths))
        print(f'scan {len(exist_paths)} existing paths with empty images')

    # Broadcast total_folders to all processes
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        exist_paths = None
    path_list = accelerate.utils.broadcast_object_list([exist_paths], from_process=0)[0]

    # dataset
    dataset = ImageFileDataset(image_root, mask_root, depth_root,
                               empty_img_root, empty_depth_root,
                               path_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda x: x)
    dataloader = accelerator.prepare(dataloader)

    # Initialize info dict for gathering results from all processes
    info = {}
    
    for batches in tqdm(dataloader,
                        disable=(not accelerator.is_main_process)):
        for batch in batches:
            fn = Path(batch['fn']).stem
            image = batch['image']
            if image is None:
                # print(f'Save error image: {img_path}')
                # err_fn = img_path.replace(image_root, err_root)
                # print(f'To {err_fn}')
                # os.makedirs(osp.dirname(err_fn), exist_ok=True)
                # shutil.copyfile(img_path, err_fn)
                continue

            try:
                mask, depth = batch['mask'], batch['depth']
                empty_img, empty_depth = batch['empty_img'], batch['empty_depth']
                img_path = batch['img_path']
                empty_img_path = batch['empty_img_path']
                
                trans_fn = A.Compose([
                    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
                    # A.CenterCrop(256, 480, p=1.0),
                ])
                image = trans_fn(image=image)['image']
                mask = trans_fn(image=mask)['image']
                depth = trans_fn(image=depth)['image']
                empty_img = trans_fn(image=empty_img)['image']
                empty_depth = trans_fn(image=empty_depth)['image']

                m = (mask == ADE20K_WALL)
                m_depth = torch.tensor(depth[m])
                m_empty_depth = torch.tensor(empty_depth[m])
                count = m.sum()
                div_median = (m_empty_depth / m_depth).median()
                div_mean = (m_empty_depth / m_depth).mean()
                diff10 = ((m_empty_depth / m_depth - 1.0).abs() > 0.1).sum() / count
                diff20 = ((m_empty_depth / m_depth - 1.0).abs() > 0.2).sum() / count
                diff50 = ((m_empty_depth / m_depth - 1.0).abs() > 0.5).sum() / count
                diff100 = ((m_empty_depth / m_depth - 1.0).abs() > 1.0).sum() / count
                diff200 = ((m_empty_depth / m_depth - 1.0).abs() > 2.0).sum() / count
                diff_median = (m_empty_depth - m_depth).median()
                diff_mean = (m_empty_depth - m_depth).mean()

                key = empty_img_path.replace(empty_img_root, '')
                info[key] = {
                    'empty_depth: median': m_empty_depth.median().item(),
                    'empty_depth: mean': m_empty_depth.mean().item(),
                    'depth: median': m_depth.median().item(),
                    'depth: mean': m_depth.mean().item(),
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
                # print(f'[{key}]')
                # print(f'empty_depth/depth: median = {median:.2f}, diff10 = {diff10:.2f}, diff20 = {diff20:.2f}')
                
                # Convert mask to uint8 and add channel dimension
                m_vis = (m.astype(np.uint8) * 255)[..., np.newaxis].repeat(3, axis=2)
                # Convert depth arrays to uint8 and add channel dimension
                depth_vis = (np.clip(depth / 10.0, 0, 1) * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=2)
                empty_depth_vis = (np.clip(empty_depth / 10.0, 0, 1) * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=2)
                
                canvas = np.concatenate([
                    image,
                    empty_img,
                    m_vis,
                    depth_vis,
                    empty_depth_vis,
                ], axis=1)
                side_by_sidy = np.concatenate([
                    batch['image'],
                    batch['empty_img'],
                ], axis=1)
                
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

    print(f'[gpu {accelerator.process_index}] Processed {len(info)} images')
    with open(f'{err_root}/check_empty_depth_gpu{accelerator.process_index}.json', 'w') as fp:
        json.dump(info, fp)

    if accelerator.is_main_process:
        # Gather info from all processes
        all_infos = accelerate.utils.gather_object([info])
        print(f'len(info) = {len(info)}, len(all_infos) = {len(all_infos)}')
        # Merge all gathered info dictionaries
        merged_info = {}
        for process_info in all_infos:
            merged_info.update(process_info)
        print(f'len(merged_info) = {len(merged_info)}')

        with open(f'{err_root}/check_empty_depth.json', 'w') as fp:
            json.dump(merged_info, fp)

    pass


if __name__ == "__main__":
    main()


