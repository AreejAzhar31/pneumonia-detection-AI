import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────
RAW_TRAIN = 'data/raw/chest_xray/train'
RAW_TEST  = 'data/raw/chest_xray/test'
RAW_VAL   = 'data/raw/chest_xray/val'

# ── Step 1: Collect all images with labels ─────────
def collect_images(folder):
    data = []
    
    normal_folder    = os.path.join(folder, 'NORMAL')
    pneumonia_folder = os.path.join(folder, 'PNEUMONIA')
    
    # Normal images
    if os.path.exists(normal_folder):
        for filename in os.listdir(normal_folder):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                full_path = os.path.join(normal_folder, filename)
                data.append((full_path, 'normal', filename))
    
    # Pneumonia images — bacteria = severe, virus = mild
    if os.path.exists(pneumonia_folder):
        for filename in os.listdir(pneumonia_folder):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                full_path = os.path.join(pneumonia_folder, filename)
                if 'bacteria' in filename.lower():
                    label = 'severe'
                elif 'virus' in filename.lower():
                    label = 'mild'
                else:
                    label = 'severe'  # default if unknown
                data.append((full_path, label, filename))
    
    return data

print("Collecting images...")
train_data = collect_images(RAW_TRAIN)
test_data  = collect_images(RAW_TEST)
val_data   = collect_images(RAW_VAL)

# Combine val into train since val is very small
train_data = train_data + val_data

print(f"Train images: {len(train_data)}")
print(f"Test images:  {len(test_data)}")

# ── Step 2: Split train into train and val ─────────
train, val = train_test_split(
    train_data,
    test_size=0.15,
    random_state=42,
    stratify=[x[1] for x in train_data]
)

print(f"\nAfter split:")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test_data)}")

# ── Step 3: Process and save images ───────────────
def save_images(data, split_name):
    print(f"\nSaving {split_name} images...")
    count = 0
    for full_path, label, filename in data:
        save_dir  = os.path.join('data', split_name, label)
        save_path = os.path.join(save_dir, filename)

        os.makedirs(save_dir, exist_ok=True)

        try:
            img = Image.open(full_path).convert('RGB')
            img = img.resize((224, 224))
            img.save(save_path)
            count += 1

            if count % 200 == 0:
                print(f"  {count} images saved...")

        except Exception as e:
            print(f"  Skipped {filename}: {e}")

    print(f"  Done — {count} images saved for {split_name}")

save_images(train,     'train')
save_images(val,       'val')
save_images(test_data, 'test')

print("\nModule 1 Complete! All images sorted and ready.")