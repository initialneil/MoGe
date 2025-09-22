import argparse
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import json
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from collov.jsonl import read, write

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

    dataset = read(input_json_fn)
    print(f"Loaded {len(dataset)} entries from {input_json_fn}")

    for entry in tqdm(dataset):
        image = entry["image"]
        empty = entry["empty"]
        image_depth = image.replace("image/", "depth_moge2/").replace(Path(image).suffix, ".png")
        empty_depth = empty.replace("image/", "depth_moge2/").replace(Path(empty).suffix, ".png")
        image_mask = image.replace("image/", "mask/").replace(Path(image).suffix, ".png")
        empty_mask = empty.replace("image/", "mask/").replace(Path(empty).suffix, ".png")

        entry["image_depth"] = image_depth
        entry["empty_depth"] = empty_depth
        entry["image_mask"] = image_mask
        entry["empty_mask"] = empty_mask

    write(dataset, output_json_fn)
    print(f"Wrote {len(dataset)} entries to {output_json_fn}")

    pass


if __name__ == "__main__":
    main()


