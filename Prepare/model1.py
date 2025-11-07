import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import math, time

# ==== Config ====
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15
SEED = 42


# ===== Utils: Progress callback =====
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, name, steps_per_epoch):
        super().__init__()
        self.name = name
        self.steps = steps_per_epoch
        self.epoch_start = None
        self.batch_start = None

    def on_train_begin(self, logs=None):
        print("\n[Start] Training:", self.name)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start is None:
            return
        dt = time.time() - self.batch_start
        pct = (batch + 1) / self.steps
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        eta = dt * (self.steps - (batch + 1))
        print(
            f"\r[{bar}] {pct:6.2%}  "
            f"batch {batch+1}/{self.steps}  "
            f"{dt*1000:6.0f} ms/batch  ETA {eta:5.1f}s",
            end="", flush=True
        )

    def on_epoch_end(self, epoch, logs=None):
        total = time.time() - self.epoch_start
        print(
            f"\n[Done] epoch {epoch+1} in {total:.1f}s  "
            f"loss={logs.get('loss'):.4f}  acc={logs.get('accuracy'):.4f}  "
            f"val_loss={logs.get('val_loss'):.4f}  val_acc={logs.get('val_accuracy'):.4f}"
        )

# ==== Generators (shared) ====
train_gen = ImageDataGenerator(validation_split=0.2, rescale=1/255.0)

# ==== Model 1: Leaf vs Background ====
print("\nTraining Model 1: Leaf vs Background")
train_data = train_gen.flow_from_directory(
    "binary_leaf_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="training",
    class_mode="binary",
    shuffle=True,
    seed=SEED
)
val_data = train_gen.flow_from_directory(
    "binary_leaf_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="validation",
    class_mode="binary",
    shuffle=False
)

print("Class indices (leaf dataset):", train_data.class_indices)
steps1 = math.ceil(train_data.samples / BATCH)
val_steps1 = math.ceil(val_data.samples / BATCH)
print(f"Samples: train={train_data.samples}, val={val_data.samples} | steps/epoch={steps1}, val_steps={val_steps1}")

model1 = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb1 = [ProgressCallback("Leaf vs Background", steps1)]
model1.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps1,
    validation_steps=val_steps1,
    callbacks=cb1,
    verbose=0
)
model1.save("leaf_or_background.h5")
print("Saved: leaf_or_background.h5")

# ==== Model 2: Healthy vs Diseased ====
print("\nTraining Model 2: Healthy vs Diseased")
train_data2 = train_gen.flow_from_directory(
    "binary_health_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="training",
    class_mode="binary",
    shuffle=True,
    seed=SEED
)
val_data2 = train_gen.flow_from_directory(
    "binary_health_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="validation",
    class_mode="binary",
    shuffle=False
)

print("Class indices (health dataset):", train_data2.class_indices)
steps2 = math.ceil(train_data2.samples / BATCH)
val_steps2 = math.ceil(val_data2.samples / BATCH)
print(f"Samples: train={train_data2.samples}, val={val_data2.samples} | steps/epoch={steps2}, val_steps={val_steps2}")

model2 = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb2 = [ProgressCallback("Healthy vs Diseased", steps2)]
model2.fit(
    train_data2,
    validation_data=val_data2,
    epochs=EPOCHS,
    steps_per_epoch=steps2,
    validation_steps=val_steps2,
    callbacks=cb2,
    verbose=0
)
model2.save("healthy_or_diseased.h5")
print("Saved: healthy_or_diseased.h5")

print("\nDone training both CNNs.")
