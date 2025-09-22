import os
import os.path as osp
import cv2
import numpy as np
import tyro
from dataclasses import dataclass
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

IMAGE_SUFFIX = ("png", "jpg", "jpeg", "webp")

def resize_image_by_longest_side(image, max_length=1024):
    """
    Resize an image such that its longest side does not exceed the specified maximum length.

    Args:
        image (PIL.Image): The input image.
        max_length (int): The maximum allowed length for the longest side.

    Returns:
        PIL.Image: The resized image.
    """
    # Get the original dimensions
    original_width, original_height = image.size

    # Determine the scaling factor
    if original_width > original_height:
        scaling_factor = max_length / original_width
    else:
        scaling_factor = max_length / original_height

    # Calculate new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def predict_semantic_mask(img, processor, model):
    width, height = img.size
    img = resize_image_by_longest_side(img, 1280)
    new_width, new_height = img.size
    inputs = processor(img, task_inputs=["semantic"], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(new_height, new_width)]
    )
    mask = predicted_semantic_map[0].cpu().numpy()
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(mask).convert("L")


class ImageDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        return img_path

@dataclass
class Args:
    image_root: str
    output_root: str

def main():
    args = tyro.cli(Args)
    image_root = args.image_root
    output_root = args.output_root
    accelerator = Accelerator()

    # model
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to("cuda")

    image_paths = scandir(image_root, suffix=IMAGE_SUFFIX, recursive=True)
    img_list = [fn for fn in image_paths]

    dataset = ImageDataset(img_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    processor, model, dataloader = accelerator.prepare(processor, model, dataloader)

    for batch in tqdm(dataloader):
        img_path = osp.join(image_root, batch[0])
        ext = Path(img_path).suffix

        out_path = osp.join(output_root, batch[0]).replace(ext, '.png')
        if osp.exists(out_path):
            continue

        image = Image.open(img_path).convert("RGB")
        mask = predict_semantic_mask(image, processor, model)

        os.makedirs(osp.dirname(out_path), exist_ok=True)
        mask.save(out_path)

    pass

if __name__ == '__main__':
    main()

