export CUDA_VISIBLE_DEVICES=0

python demo.py --rgb \
    --clip_length 4 \
    --smplx_model_path /home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl \
    --output_dir demo_output_zju_no_l2 \
    --ckpt_path /home/liubingqi/work/Audio2Avatar/gaussian_avatar/results_320_spatio_temporal_zjumocap/checkpoints/epoch=153-train_loss=0.00.ckpt\
    --image_height 1024 \
    --image_width 1024 \
    --num_iters 2 \
    --batch_size 1 \
    --deform \
    --mutiview \
    --n_input_frames 4 \
    --n_test_frames 4 \
