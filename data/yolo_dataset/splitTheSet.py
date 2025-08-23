import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# Ścieżki
images_path = "images/"
labels_path = "labels/"

# Pobierz wszystkie pliki obrazów
image_files = glob.glob(os.path.join(images_path, "*.jpg"))  # lub .png

# Podziel na train/val (80/20)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Stwórz foldery
os.makedirs("train/images", exist_ok=True)
os.makedirs("train/labels", exist_ok=True)
os.makedirs("val/images", exist_ok=True)
os.makedirs("val/labels", exist_ok=True)

# Przenieś pliki
for img_path in train_images:
    filename = os.path.basename(img_path)
    label_filename = filename.replace('.jpg', '.txt')  # lub .png

    shutil.copy(img_path, f"train/images/{filename}")
    if os.path.exists(f"labels/{label_filename}"):
        shutil.copy(f"labels/{label_filename}", f"train/labels/{label_filename}")

for img_path in val_images:
    filename = os.path.basename(img_path)
    label_filename = filename.replace('.jpg', '.txt')

    shutil.copy(img_path, f"val/images/{filename}")
    if os.path.exists(f"labels/{label_filename}"):
        shutil.copy(f"labels/{label_filename}", f"val/labels/{label_filename}")