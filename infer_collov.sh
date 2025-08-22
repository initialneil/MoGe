# # infer image
# CUDA_VISIABLE_DEVICES=0,1,2,3,4,5,6 \
#     accelerate launch --main_process_port 29071 \
#     moge/scripts/infer_collov.py \
#     --image_root /mnt/sdc/data/image \
#     --output_root /mnt/sdc/data/depth_moge2 \
#     --pretrained checkpoints/moge-2-vitl-normal/model.pt \
#    --z16_scale 4000

# infer empty_syn
CUDA_VISIABLE_DEVICES=0,1,2,3,4,5,6 \
    accelerate launch --main_process_port 29071 \
    moge/scripts/infer_collov.py \
    --image_root /mnt/sdc/data/empty_syn \
    --err_root /mnt/sdc/data/err_empty_syn \
    --output_root /mnt/sdc/data/depth_moge2_empty_syn \
    --pretrained checkpoints/moge-2-vitl-normal/model.pt \
    --z16_scale 4000
