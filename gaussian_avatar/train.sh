export CUDA_VISIBLE_DEVICES=2
python train_pipe.py --num_epochs 1001 --rgb \
    --clip_length 4 \
    --smplx_model_path smpl/SMPL_NEUTRAL.pkl \
    --output_dir results_219_dyn_male \
    --ckpt_path ckpts_219_dyn_male_length_4/ \
    --data_folder data/gs_data/data/dynvideo_male/ \
    --image_height 1024 \
    --image_width 1024 \
    # --use_ckpt \
    # --net_ckpt_path gaussian_net_1001.pth \
    # --animation_net_ckpt_path animation_net_501.pth \
