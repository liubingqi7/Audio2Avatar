export CUDA_VISIBLE_DEVICES=0
python train_pipe.py --num_epochs 5001 --rgb \
    --clip_length 2 \
    --smplx_model_path smpl/SMPL_NEUTRAL.pkl \
    --output_dir results_no_deform_len_4_neuman \
    --ckpt_path ckpts_no_deform_len_4_neuman/ \
    --data_folder data/gs_data/data/neuman_bike/ \
    --image_height 711 \
    --image_width 1265 \
    # --use_ckpt \
    # --net_ckpt_path gaussian_net_1001.pth \
    # --animation_net_ckpt_path animation_net_501.pth \
