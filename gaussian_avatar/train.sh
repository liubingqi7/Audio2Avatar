export CUDA_VISIBLE_DEVICES=0
python train_pipe.py --num_epochs 101 --rgb \
    --clip_length 4 \
    --smplx_model_path /home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl \
    --output_dir 3_3_1503_thuman_test \
    --ckpt_path 3_3_1503_thuman_test/ \
    --data_folder data/gs_data/data/neuman_bike/ \
    --image_height 1024 \
    --image_width 1024 \
    # --use_ckpt \
    # --net_ckpt_path gaussian_net_1001.pth \
    # --animation_net_ckpt_path animation_net_501.pth \
