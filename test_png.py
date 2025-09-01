import cv2
import os
import sys
import time

def check_dataset(image_dir, timestamp_file):
    """
    从时间戳文件中读取时间戳，拼接出每个相机的图像路径，并使用 OpenCV 尝试加载。
    """
    if not os.path.isdir(image_dir):
        print(f"错误: 图像目录不存在 -> {image_dir}")
        return False

    if not os.path.isfile(timestamp_file):
        print(f"错误: 时间戳文件不存在 -> {timestamp_file}")
        return False

    # 根据你的文件夹结构定义相机子目录
    camera_folders = ["left", "right", "sideleft", "sideright"]
    # 常见的图像文件扩展名
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

    print(f"正在从 {timestamp_file} 读取时间戳...")
    
    try:
        with open(timestamp_file, 'r') as f:
            timestamps = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取时间戳文件时出错: {e}")
        return False

    print(f"找到 {len(timestamps)} 个时间戳。开始校验所有相机的图像...")

    all_ok = True
    total_images_checked = 0
    
    for ts in timestamps:
        for folder in camera_folders:
            found_image = False
            # 尝试不同的扩展名
            for ext in image_extensions:
                image_filename = ts + ext
                image_path = os.path.join(image_dir, folder, image_filename)

                if os.path.exists(image_path):
                    found_image = True
                    total_images_checked += 1
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION)
                        if img is None:
                            print(f"!!! 读取失败或文件损坏: {image_path}")
                            all_ok = False
                    except Exception as e:
                        print(f"!!! 发生异常，文件可能严重损坏: {image_path}, 错误: {e}")
                        all_ok = False
                    # 找到后就不用再试这个时间戳的其他扩展名了
                    break 
            
            # 如果在所有扩展名中都找不到该时间戳对应的图像
            if not found_image:
                # 打印一个警告，但这不一定是个错误，可能某些相机没有这个时间戳的图像
                # print(f"--- 警告: 在 '{folder}' 目录中未找到时间戳为 {ts} 的图像")
                pass

    print(f"总共检查了 {total_images_checked} 个图像文件。")
    return all_ok

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python check_images.py <图像数据根目录> <时间戳文件路径>")
        print("示例: python check_images.py /home/wk/Datasets/real_re_4 /home/wk/Datasets/real_re_4/timestamp.txt")
        sys.exit(1)

    image_directory = sys.argv[1]
    timestamps_path = sys.argv[2]
    
    num_iterations = 5
    print(f"将对数据集进行 {num_iterations} 轮完整性检查...")
    print("-" * 30)

    for i in range(num_iterations):
        print(f"\n--- 第 {i + 1}/{num_iterations} 轮检查 ---")
        start_time = time.time()
        
        ok = check_dataset(image_directory, timestamps_path)
        
        end_time = time.time()
        print(f"第 {i + 1} 轮检查完成，耗时: {end_time - start_time:.2f} 秒。")

        if ok:
            print("状态: 本轮所有图像均可正常读取。")
        else:
            print("状态: 在本轮检查中发现问题文件！")
            break
    
    print("\n" + "=" * 30)
    print("所有检查已完成。")