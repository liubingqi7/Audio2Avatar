import numpy as np

# 加载npz文件
npz_path = "/home/qizhu/Desktop/Work/MotionGeneration/Audio2Avatar/audio2motion/baselines/PantoMatrix/outputs/audio2pose/custom/1226_132423_emage/999/res_2_scott_0_103_103.npz"
data = np.load(npz_path)

# 打印所有属性
print("NPZ文件包含以下属性:")
for key in data.files:
    print(f"属性名: {key}")
    print(f"形状: {data[key].shape}")
    print(f"数据类型: {data[key].dtype}")
    print("---")

data.close()
