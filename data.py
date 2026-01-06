import kagglehub

# Download latest version
path = kagglehub.dataset_download("borhanitrash/cat-dataset")

print("Path to dataset files:", path)
import shutil
import os

# 想要放的資料夾（自行設定）
target_dir = "/home/nthuuser/Andy/DOE/CDDPM/CAT/data"

# 移動到指定位置
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("下載完成，資料已移動到:", target_dir)
