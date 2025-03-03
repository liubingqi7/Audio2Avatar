import os
import cv2
import numpy as np
import imageio.v2 as imageio

def images_to_video(image_folder, output_video, fps=10):
    """
    将一个文件夹中的所有图片合成视频（使用 cv2.imread 读取图片）
    :param image_folder: 存放图片的文件夹路径
    :param output_video: 输出视频的路径（.mp4）
    :param fps: 帧率，默认10
    """
    # 获取文件夹中所有图片（忽略大小写）
    images = sorted([img for img in os.listdir(image_folder)
                     if img.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    if not images:
        print("⚠️ 没有找到图片，请检查文件夹！")
        return

    # 使用 cv2.imread 读取第一张图片作为参考
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"⚠️ 读取图片失败：{first_image_path}")
        return

    # 转换颜色格式：
    # cv2.imread 默认读取为 BGR，如果是灰度图（二维），转换为 RGB（三通道）
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        # 如果有 Alpha 通道，则从 BGRA 转换到 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    target_shape = frame.shape  # (height, width, 3)
    print("目标图片尺寸：", target_shape)

    writer = imageio.get_writer(output_video, fps=fps, codec="libx264")

    print(f"📷 发现 {len(images)} 张图片，开始合成视频...")
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"⚠️ 读取图片失败：{image_path}, 跳过！")
            continue

        # 颜色格式转换
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 尺寸不一致则调整到目标尺寸
        if frame.shape != target_shape:
            frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
        
        writer.append_data(frame)

    writer.close()
    print(f"✅ 视频已生成: {output_video}")

# 示例用法
if __name__ == "__main__":
    image_folder = "/home/qizhu/Downloads/demo_output"  # 替换为你的图片文件夹路径
    output_video = "output.mp4"
    images_to_video(image_folder, output_video, fps=10)
