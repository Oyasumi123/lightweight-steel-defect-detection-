# coding:utf-8

import os


def find_duplicate_filenames(folder1, folder2):
    # 获取文件夹1中的所有文件名（不包含扩展名）
    files1 = set(os.path.splitext(f)[0] for f in os.listdir(folder1))
    # 获取文件夹2中的所有文件名（不包含扩展名）
    files2 = set(os.path.splitext(f)[0] for f in os.listdir(folder2))

    # 找到重复的文件名
    duplicates = files1.intersection(files2)

    return duplicates


def main():
    # 在这里直接指定文件夹路径
    folder1 = "F:\model\liyolonet\liyolo\my-data\huafen\9"
    folder2 = "F:\model\liyolonet\liyolo\my-data\GC\labels\\test"

    # 检查文件夹是否存在
    if not os.path.isdir(folder1):
        print(f"Error: Folder {folder1} does not exist.")
        return
    if not os.path.isdir(folder2):
        print(f"Error: Folder {folder2} does not exist.")
        return

    # 查找重复文件
    duplicates = find_duplicate_filenames(folder1, folder2)

    # 打印结果
    if duplicates:
        print("Duplicate filenames found (ignoring extensions):")
        for filename in duplicates:
            print(filename)
    else:
        print("No duplicate filenames found (ignoring extensions).")


if __name__ == "__main__":
    main()