import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ─────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-4
NUM_CLASSES   = 2
CLASS_NAMES   = ['NORMAL', 'PNEUMONIA']

# Use the chest_xray folder you already have
TRAIN_DIR = 'chest_xray/train'
VAL_DIR   = 'chest_xray/val'
MODEL_DIR = 'model'
LOG_DIR   = 'logs'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ============================================================
#  STEP 1 — Data Generators
# ============================================================
print("\n" + "="*55)
print("  MODULE 2 — MODEL TRAINING STARTED")
print("="*55)
print("\n[Step 1] Loading data from Module 1 output folders...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"  Class indices : {train_gen.class_indices}")
print(f"  Train samples : {train_gen.samples}")
print(f"  Val samples   : {val_gen.samples}")

# ============================================================
#  STEP 2 — Build Model  (EfficientNetB3 Transfer Learning)
# ============================================================
print("\n[Step 2] Building EfficientNetB3 Transfer Learning model...")

base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # Freeze base for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"  Total parameters: {model.count_params():,}")

# ============================================================
#  STEP 3 — Callbacks
# ============================================================
print("\n[Step 3] Setting up callbacks...")

checkpoint_path = os.path.join(MODEL_DIR, 'best_model.h5')

def get_callbacks(log_file, patience_es=7, patience_lr=3):
    return [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-8, verbose=1),
        CSVLogger(os.path.join(LOG_DIR, log_file))
    ]

# ============================================================
#  STEP 4 — Phase 1: Train head only (base frozen)
# ============================================================
print("\n[Step 4] Phase 1 — Training classification head (base frozen)...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=get_callbacks('training_log.csv'),
    verbose=1
)

# ============================================================
#  STEP 5 — Phase 2: Fine-tune top 30 layers
# ============================================================
print("\n[Step 5] Phase 2 — Fine-tuning top 30 layers of EfficientNetB3...")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_ft = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=get_callbacks('finetune_log.csv', patience_es=5, patience_lr=2),
    verbose=1
)

# ============================================================
#  STEP 6 — Save Final Model
# ============================================================
print("\n[Step 6] Saving final model...")
final_path = os.path.join(MODEL_DIR, 'pneumonia_model_final.h5')
model.save(final_path)
print(f"  ✓ Final model  → {final_path}")
print(f"  ✓ Best model   → {checkpoint_path}")

# ============================================================
#  STEP 7 — Evaluate on Validation Set
# ============================================================
print("\n[Step 7] Evaluating on validation set...")
val_gen.reset()
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"  Validation Accuracy : {val_acc*100:.2f}%")
print(f"  Validation Loss     : {val_loss:.4f}")

val_gen.reset()
y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
y_true = val_gen.classes
print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ============================================================
#  STEP 8 — Save Training Curves & Confusion Matrix
# ============================================================
print("\n[Step 8] Saving plots...")

acc      = history.history['accuracy']     + history_ft.history['accuracy']
val_a    = history.history['val_accuracy'] + history_ft.history['val_accuracy']
loss     = history.history['loss']         + history_ft.history['loss']
val_l    = history.history['val_loss']     + history_ft.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(acc,   label='Train Accuracy', color='royalblue')
axes[0].plot(val_a, label='Val Accuracy',   color='tomato')
axes[0].set_title('Accuracy — Phase 1 + Fine-tune')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(loss,  label='Train Loss', color='royalblue')
axes[1].plot(val_l, label='Val Loss',   color='tomato')
axes[1].set_title('Loss — Phase 1 + Fine-tune')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'), dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Validation Set')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"  ✓ Training curves  → logs/training_curves.png")
print(f"  ✓ Confusion matrix → logs/confusion_matrix.png")

print("\n" + "="*55)
print("  MODULE 2 COMPLETE!")
print("  → Module 3 loads: model/best_model.h5")
print(f"  → Class order: {CLASS_NAMES}")
print("="*55)
