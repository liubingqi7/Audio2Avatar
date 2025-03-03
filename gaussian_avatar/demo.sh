python demo.py --rgb \
    --clip_length 4 \
    --output_dir demo_output \
    --ckpt_path ckpts_deform_len_4_neuman/ \
    --data_folder data/gs_data/data/neuman_bike/  \
    --image_height 711 \
    --image_width 1265 \
    --net_ckpt_path gaussian_net_5001.pth \
    --smplx_model_path smpl/SMPL_NEUTRAL.pkl \
    --animation_net_ckpt_path animation_net_5001.pth \
