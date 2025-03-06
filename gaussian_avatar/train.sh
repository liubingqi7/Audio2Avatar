export CUDA_VISIBLE_DEVICES=0
python train_pipe.py --num_epochs 101 --rgb \
    --clip_length 4 \
    --smplx_model_path /home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl \
    --output_dir results_no_deform_len_4_neuman \
    --ckpt_path ckpts_no_deform_len_4_neuman/ \
    --data_folder data/gs_data/data/neuman_bike/ \
    --image_height 711 \
    --image_width 1265 \
    # --net_ckpt_path gaussian_net_1001.pth \
    # --animation_net_ckpt_path animation_net_501.pth \


# python train_lightning.py --num_epochs 50 --rgb \
#     --clip_length 4 \
#     --smplx_model_path /home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl \
#     --output_dir results_3_5_thuman_test_speed \
#     --ckpt_path ckpts_3_5_thuman_test_speed/ \
#     --data_folder data/gs_data/data/neuman_bike/ \
#     --image_height 1024 \
#     --image_width 1024 \
#     --use_wandb \
#     --experiment_name 3_4_thuman_test_speed \
    # --use_ckpt \
