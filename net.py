from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import sys


# ============================================================
# PODEŠAVANJA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data" / "train"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 32
TEST_SPLIT = 0.30
RANDOM_SEED = 123
EPOCHS = 50

MODEL_PATH = BASE_DIR / "traffic_sign_cnn.keras"


CLASS_NAMES = [
    "limit_20",
    "limit_30",
    "limit_50",
    "limit_60",
    "limit_70",
    "limit_80",
    "limit_100",
    "limit_120",
    "stop",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# PROVERA PODATAKA
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder ne postoji:\n{DATASET_DIR}"
    )

for class_name in CLASS_NAMES:
    class_directory = DATASET_DIR / class_name

    if not class_directory.exists():
        raise FileNotFoundError(
            f"Nedostaje folder klase:\n{class_directory}"
        )


# ============================================================
# UČITAVANJE I PODELA PODATAKA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=TEST_SPLIT,
    subset="training",
    seed=RANDOM_SEED,
    shuffle=True,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=TEST_SPLIT,
    subset="validation",
    seed=RANDOM_SEED,
    shuffle=False,
)


print("\nKlase:")

for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index}: {class_name}")


# ============================================================
# OPTIMIZACIJA UČITAVANJA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(
    buffer_size=1000,
    seed=RANDOM_SEED
).prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Male promene slike pomažu mreži da nauči znakove pod
# različitim uglovima, položajima i osvetljenjem.
#
# Ne koristimo horizontalno okretanje jer bi ono moglo
# izmeniti izgled cifara i znakova.

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=0.08
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),
        tf.keras.layers.RandomContrast(
            factor=0.15
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# CNN MODEL
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3
            )
        ),

        # Augmentacija se koristi samo tokom treniranja.
        data_augmentation,

        # Pikseli iz opsega 0–255 prelaze u 0–1.
        tf.keras.layers.Rescaling(
            scale=1.0 / 255.0
        ),

        # ----------------------------------------------------
        # PRVI KONVOLUTIVNI BLOK
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Dropout(0.20),

        # ----------------------------------------------------
        # DRUGI KONVOLUTIVNI BLOK
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Dropout(0.25),

        # ----------------------------------------------------
        # TREĆI KONVOLUTIVNI BLOK
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Dropout(0.30),

        # ----------------------------------------------------
        # KLASIFIKACIONI DEO
        # ----------------------------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            units=128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.40),

        # Devet izlaznih neurona, po jedan za svaku klasu.
        tf.keras.layers.Dense(
            units=NUM_CLASSES,
            activation="softmax"
        ),
    ],
    name="traffic_sign_cnn"
)


# ============================================================
# KOMPILACIJA MODELA
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


model.summary()



# ============================================================
# CALLBACK FUNKCIJE
# ============================================================

callbacks = [
    # Zaustavlja treniranje ako se test/validation loss
    # ne poboljšava sedam epoha.
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    # Čuva najbolju verziju modela.
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Smanjuje learning rate kada model prestane
    # značajno da napreduje.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]


# ============================================================
# TRENIRANJE
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# EVALUACIJA
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nRezultati modela:")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# ČUVANJE ZAVRŠNOG MODELA
# ============================================================

model.save(MODEL_PATH)

print(f"\nModel je sačuvan u:\n{MODEL_PATH}")


# ============================================================
# GRAFICI TRENIRANJA
# ============================================================

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

epochs_range = range(
    1,
    len(training_accuracy) + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    training_accuracy,
    label="Trening accuracy"
)

plt.plot(
    epochs_range,
    validation_accuracy,
    label="Test/validation accuracy"
)

plt.xlabel("Epoha")
plt.ylabel("Accuracy")
plt.title("Tačnost CNN modela")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    BASE_DIR / "accuracy_graph.png"
)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    training_loss,
    label="Trening loss"
)

plt.plot(
    epochs_range,
    validation_loss,
    label="Test/validation loss"
)

plt.xlabel("Epoha")
plt.ylabel("Loss")
plt.title("Greška CNN modela")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    BASE_DIR / "loss_graph.png"
)

plt.show()
