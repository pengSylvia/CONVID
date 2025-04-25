import os

def rename_images_in_folder(folder_path, start_number=2000):
    """
    将指定文件夹中的所有图片按数字递增重命名。
    
    参数:
        folder_path: 要处理的文件夹路径
        start_number: 重命名的起始数字，默认为1
    
    注意:
        该函数会处理所有常见的图片格式（.jpg, .jpeg, .png, .gif, .bmp, .webp）
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在")
        return
    
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    
    # 过滤出图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif'}
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # 按文件名排序，以保证文件的处理顺序
    image_files.sort()
    
    # 重命名每个图片文件
    for idx, image_file in enumerate(image_files):
        # 分离文件名和扩展名
        file_base, file_extension = os.path.splitext(image_file)
        
        # 构建新文件名
        new_filename = f"{start_number + idx}{file_extension}"
        
        # 构建完整路径
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"已重命名: {image_file} -> {new_filename}")
    
    print(f"已成功重命名 {len(image_files)} 个图片文件")

# 使用示例
if __name__ == "__main__":
    # 指定要处理的文件夹路径
    folder_path = "/Users/huangtingqiao/Desktop/FaceRecognition 2/X_datasets/train/normal"  # 替换为你的实际文件夹路径
    
    # 调用函数进行重命名（从1开始）
    rename_images_in_folder(folder_path, 5000)