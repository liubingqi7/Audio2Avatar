export CUDA_VISIBLE_DEVICES=0

python demo.py --rgb \
    --clip_length 4 \
    --smplx_model_path /home/liubingqi/work/liubingqi/SMPL_SMPLX/SMPL_models/smpl/SMPL_NEUTRAL.pkl \
    --output_dir demo_output_315_fix_6 \
    --ckpt_path /home/liubingqi/work/Audio2Avatar/gaussian_avatar/results_315_iter2_fix/checkpoints/epoch=39-train_loss=0.00.ckpt \
    --image_height 1024 \
    --image_width 1024 \
    --num_iters 2 \
    --batch_size 1 \
    # --deform \
