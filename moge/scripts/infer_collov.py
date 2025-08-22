
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
from PIL import Image
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from moge.model import import_model_class_by_version
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
import utils3d
from accelerate import Accelerator
import shutil
from tqdm import tqdm

IMAGE_SUFFIX = ("png", "jpg", "jpeg", "webp")

class ImageFileDataset(Dataset):
    def __init__(self, img_dir, file_list, transform=tvt.ToTensor()):
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.file_list[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            return {
                'image': None,
                'image_tensor': None,
                'fn': self.file_list[idx],
                'img_path': img_path,
            }

        image_tensor = self.transform(image)
        return {
            'image': image,
            'image_tensor': image_tensor,
            'fn': self.file_list[idx],
            'img_path': img_path,
        }
    
def parse_args():
    parser = argparse.ArgumentParser(description="Count image folders and images.")
    parser.add_argument('--image_root', type=str, default='/mnt/sdc/data/image', help='Directory containing images')
    parser.add_argument('--output_root', type=str, default='/mnt/sdc/data/depth_moge2')
    parser.add_argument('--err_root', type=str, default='/mnt/sdc/data/err_image')
    parser.add_argument('--model_version', type=str, default='v2')
    parser.add_argument('--pretrained', type=str, default='checkpoints/moge-2-vitl-normal/model.pt')
    parser.add_argument('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
        Higher value means more tokens and the finer details will be captured, but inference can be slower. \
        Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
        `resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
    parser.add_argument('--fov_x', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
    parser.add_argument('--threshold', type=float, default='0.04', help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')
    parser.add_argument('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
    `resolution_level` will be ignored if `num_tokens` is provided. Default: None')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision for much faster inference.')
    parser.add_argument('--z16_scale', type=float, default='4000', help='depth x4000 -> uint16 -> png')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    image_root = args.image_root.replace('\\', '/')
    output_root = args.output_root.replace('\\', '/')
    err_root = args.err_root.replace('\\', '/')
    resolution_level = args.resolution_level
    fov_x_ = args.fov_x
    num_tokens = args.num_tokens
    threshold = args.threshold
    use_fp16 = args.fp16
    z16_scale = args.z16_scale
    batch_size = args.batch_size

    # model
    model_version = args.model_version
    pretrained_model_name_or_path = args.pretrained
    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).eval()

    # accelerate
    accelerate = Accelerator()
    model = accelerate.prepare(model)
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # walk for image directories
    total_folders = 0
    for img_dir, dirs, files in os.walk(image_root):
        fns = [f for f in files if Path(f).suffix.lstrip('.').lower() in IMAGE_SUFFIX]
        img_count = len(fns)
        if img_count > 0:
            total_folders += 1

    # walk and process
    folder_count = 0
    for img_dir, dirs, files in os.walk(image_root):
        fns = [f for f in files if Path(f).suffix.lstrip('.').lower() in IMAGE_SUFFIX]
        img_count = len(fns)
        if img_count > 0:
            folder_count += 1
            if accelerate.is_main_process:
                print(f"Folder: {img_dir}, Images: {img_count}")

            out_dir = img_dir.replace(image_root, output_root)
            info_fn = osp.join(out_dir, 'info.json')
            if osp.exists(info_fn):
                if accelerate.is_main_process:
                    print(f"Info file already exists: {info_fn}")
                continue

            info = {
                'fns': fns,
                'z16_scale': z16_scale,
                'depths': {}
            }

            os.makedirs(out_dir, exist_ok=True)
            dataset = ImageFileDataset(img_dir, fns)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda x: x)
            dataloader = accelerate.prepare(dataloader)
            accelerate.wait_for_everyone()
            
            for batches in tqdm(dataloader, desc=f'[{folder_count}/{total_folders}]',
                                disable=(not accelerate.is_main_process)):
                for batch in batches:
                    fn = Path(batch['fn']).stem
                    image_tensor = batch['image_tensor']
                    image = np.array(batch['image'])
                    img_path = batch['img_path']
                    if image is None or image_tensor is None:
                        print(f'Save error image: {img_path}')
                        err_fn = img_path.replace(image_root, err_root)
                        print(f'To {err_fn}')
                        os.makedirs(osp.dirname(err_fn), exist_ok=True)
                        shutil.copyfile(img_path, err_fn)
                        continue

                    try:
                        height, width = image.shape[:2]

                        # Inference
                        output = module.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
                        depth = output['depth'].cpu().numpy()
                        intrinsics = output['intrinsics'].cpu().numpy()

                        # points = output['points'].cpu().numpy()
                        # mask = output['mask'].cpu().numpy()
                        # normal = output['normal'].cpu().numpy() if 'normal' in output else None
                                
                        # mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold)
                        # faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                        #     points,
                        #     image.astype(np.float32) / 255,
                        #     utils3d.numpy.image_uv(width=width, height=height),
                        #     normal,
                        #     mask=mask_cleaned,
                        #     tri=True
                        # )
                        # save_ply(f'{out_dir}/{fn}.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

                        info['depths'][fn] = {
                            'min_depth': depth.min().item(),
                            'max_depth': depth.max().item(),
                            'intrinsics': intrinsics.tolist(),
                        }

                        depth_z16 = (depth * z16_scale).clip(0, 65535).astype(np.uint16)
                        cv2.imwrite(f'{out_dir}/{fn}.png', depth_z16)
                    except:
                        print('Process crashed')
                        print(f'Save error image: {img_path}')
                        err_fn = img_path.replace(image_root, err_root)
                        print(f'To {err_fn}')
                        os.makedirs(osp.dirname(err_fn), exist_ok=True)
                        shutil.copyfile(img_path, err_fn)
                        continue


                    pass
                accelerate.wait_for_everyone()
            
            if accelerate.is_main_process:
                with open(info_fn, 'w') as fp:
                    json.dump(info, fp)
            accelerate.wait_for_everyone()



if __name__ == "__main__":
    main()


