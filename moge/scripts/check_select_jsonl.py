import os
import json
from matplotlib import pyplot as plt
import numpy as np

# check depth results
info_fn = "/mnt/sdc/data/err_empty_syn/check_empty_depth_v0.1.0.json"
with open(info_fn, "r") as fp:
    info = json.load(fp)
print(f"Loaded {len(info)} frames from {info_fn}")

depth_median = np.array([c["depth: median"] for c in info if c is not None])
depth_mean = np.array([c["depth: mean"] for c in info if c is not None])
div_median = np.array([c["empty_depth/depth: median"] for c in info if c is not None])
div_mean = np.array([c["empty_depth/depth: mean"] for c in info if c is not None])
print(f"{len(depth_median)} valid frames")
depth_info = info

# check mask results
info_fn = "/mnt/sdc/data/err_empty_syn/check_empty_mask_v0.1.0.json"
with open(info_fn, "r") as fp:
    info = json.load(fp)
print(f"Loaded {len(info)} frames from {info_fn}")

div_fix_but_changed = np.array([c["div_fix_but_changed"] for c in info if c is not None])
fix_but_changed = np.array([c["fix_but_changed"] for c in info if c is not None])
both_fixed = np.array([c["both_fixed"] for c in info if c is not None])
print(f"{len(div_fix_but_changed)} valid frames")
mask_info = info

# check select
low_thresh, high_thresh = 0.9, 1.05
select_idxs = (div_median > low_thresh) & (div_median < high_thresh)
select_idxs &= (div_mean > low_thresh) & (div_mean < high_thresh)
print(f"select in ({low_thresh}, {high_thresh}): {select_idxs.sum()} frames")

high_thresh = 0.05
select_idxs = (div_fix_but_changed < high_thresh)
print(f"select mask {select_idxs.sum()} frames")

# select files
selected_depth = [
    k for k, v in depth_info.items() if (
        v is not None and (v["empty_depth/depth: median"] > low_thresh) and (
            v["empty_depth/depth: median"] < high_thresh) and (
                v["empty_depth/depth: mean"] > low_thresh) and (
                    v["empty_depth/depth: mean"] < high_thresh)
    )
]

pass
