python train_pipe.py --num_epochs 1001 --rgb \
    --clip_length 4 \
    --smplx_model_path smpl/SMPL_NEUTRAL.pkl \
    --output_dir results_213115_length_4 \
    --ckpt_path ckpts_2/ \
    --data_folder data/gs_data/data/neuman_bike \
    --image_height 711 \
    --image_width 1265 \
    # --use_ckpt \
    # --net_ckpt_path gaussian_net_501.pth \
    # --animation_net_ckpt_path animation_net_501.pth \
