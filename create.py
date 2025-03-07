import os

# Đường dẫn gốc là thư mục hiện tại (.)
base_path = os.getcwd()

# Danh sách thư mục cần tạo (không cần thêm project_siamese)
directories = [
    "data",
    "data/test_samples",
    "models/siamese",
    "models/resnet_siamese",
    "scripts",
    "utils",
    "outputs/siamese/logs",
    "outputs/siamese/plots",
    "outputs/resnet_siamese/logs",
    "outputs/resnet_siamese/plots",
]

# Danh sách file cần tạo (không cần thêm project_siamese)
files = [
    ("data/train_pairs_part1.h5", ""),
    ("data/valid_pairs_part1.h5", ""),
    ("data/test_pairs.h5", ""),
    ("models/siamese/siamese_model.py", ""),
    ("models/siamese/best_siamese_model.pth", ""),
    ("models/siamese/siamese_checkpoint.pth", ""),
    ("models/resnet_siamese/resnet_siamese_model.py", ""),
    ("models/resnet_siamese/best_resnet_model.pth", ""),
    ("models/resnet_siamese/resnet_checkpoint.pth", ""),
    ("scripts/train_siamese.py", ""),
    ("scripts/train_resnet_siamese.py", ""),
    ("scripts/test_siamese.py", ""),
    ("scripts/test_resnet_siamese.py", ""),
    ("utils/dataset.py", ""),
    ("utils/loss.py", ""),
    ("utils/helpers.py", ""),
    ("outputs/siamese/logs/training_log.txt", ""),
    ("outputs/resnet_siamese/logs/training_log.txt", ""),
    ("requirements.txt", ""),
    ("README.md", ""),
]

# Tạo thư mục
for directory in directories:
    os.makedirs(os.path.join(base_path, directory), exist_ok=True)

# Tạo file rỗng
for file_path, content in files:
    full_path = os.path.join(base_path, file_path)
    if not os.path.exists(full_path):
        with open(full_path, 'w') as f:
            f.write(content)

print(f"Created project structure in current directory: {base_path}")