import os
import os.path as osp
from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.jsonl import read, write

########################################
# check depth results
info_fn = "/mnt/sdc/data/err_empty_syn/check_empty_depth_v0.1.1.json"
with open(info_fn, "r") as fp:
    info = json.load(fp)
print(f"Loaded {len(info)} frames from {info_fn}")

depth_median = np.array([c["depth: median"] for c in info if c is not None])
depth_mean = np.array([c["depth: mean"] for c in info if c is not None])
div_median = np.array([c["empty_depth/depth: median"] for c in info if c is not None])
div_mean = np.array([c["empty_depth/depth: mean"] for c in info if c is not None])
print(f"{len(depth_median)} valid frames")

div_fix_but_changed = np.array([c["div_fix_but_changed"] for c in info if c is not None])
fix_but_changed = np.array([c["fix_but_changed"] for c in info if c is not None])
both_fixed = np.array([c["both_fixed"] for c in info if c is not None])
print(f"{len(div_fix_but_changed)} valid frames")

########################################
# check select
low_thresh, high_thresh = 0.9, 1.05
select_idxs = (div_median > low_thresh) & (div_median < high_thresh)
select_idxs &= (div_mean > low_thresh) & (div_mean < high_thresh)
print(f"select in ({low_thresh}, {high_thresh}): {select_idxs.sum()} frames")

change_thresh = 0.05
select_idxs = (div_fix_but_changed < change_thresh)
print(f"select mask {select_idxs.sum()} frames")

# select files
selected_info = [
    c for c in info if (
        c is not None and (c["empty_depth/depth: median"] > low_thresh) and (
            c["empty_depth/depth: median"] < high_thresh) and (
                c["empty_depth/depth: mean"] > low_thresh) and (
                    c["empty_depth/depth: mean"] < high_thresh) and (
                        c["div_fix_but_changed"] < change_thresh)
    )
]
print(f"select both depth and mask: {len(selected_info)} frames")

selected_fns = set([Path(c['path']).stem for c in selected_info])

########################################
# load jsonl
jsonl_fn = "/mnt/sdc/data/metadata_depth_empty_image_caption.jsonl"
all_data = read(jsonl_fn)

select_data = [d for d in all_data if Path(d['image']).stem in selected_fns]
write(select_data, "/mnt/sdc/data/metadata_depth_empty_image_caption_100k.jsonl")
print(select_data[:1])

pass
