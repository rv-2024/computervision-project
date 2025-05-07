


# ─── Standard imports ────────────────────────────────────────────────────────
import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from PIL import Image

from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROJECT_DIR   = "/content/drive/MyDrive/CV/Brain-Tumor-Classification"
TRAIN_FOLDER  = "Training"
TEST_FOLDER   = "Testing"
CLASSES       = ["glioma_tumor","no_tumor","meningioma_tumor","pituitary_tumor"]
IMG_SIZE      = 224
BATCH_SIZE    = 32
VAL_SPLIT     = 0.1
BASE_EPOCHS   = 10
FINE_EPOCHS   = 5
LR_HEAD       = 1e-3
LR_FINE       = 1e-4
RANDOM_SEED   = 101

PALETTE_DARK  = ["#1F1F1F","#313131","#636363","#AEAEAE","#DADADA"]
PALETTE_WARM  = ["#330000","#582626","#9E1717","#D35151","#E9B4B4"]
PALETTE_GREEN = ["#01411C","#4B6F44","#4F7942","#74C365","#D0F0C0"]

# ─── PATH UTILITIES ──────────────────────────────────────────────────────────
def get_paths(base):
    tr = os.path.join(base, TRAIN_FOLDER)
    te = os.path.join(base, TEST_FOLDER)
    for p in (tr, te):
        if not os.path.isdir(p):
            raise IOError(f"Folder not found: {p}")
    return tr, te

# ─── VISUALIZATION HELPERS ───────────────────────────────────────────────────
def display_palettes():
    fig, axs = plt.subplots(1, 3, figsize=(12, 2))
    for ax, pal in zip(axs, [PALETTE_DARK, PALETTE_WARM, PALETTE_GREEN]):
        ax.bar(range(len(pal)), [1]*len(pal), color=pal)
        ax.axis("off")
    plt.show()

def plot_class_distribution(folder, title):
    counts = {cls: len(os.listdir(os.path.join(folder, cls)))
              for cls in CLASSES
              if os.path.isdir(os.path.join(folder, cls))}
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="pastel")
    plt.xticks(rotation=30); plt.title(title); plt.tight_layout(); plt.show()

def display_samples_grid(folder, per_class=3):
    fig = plt.figure(figsize=(per_class*2, len(CLASSES)*2)); idx = 1
    for cls in CLASSES:
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir): continue
        for f in random.sample(os.listdir(cls_dir), per_class):
            img = Image.open(os.path.join(cls_dir, f)).resize((IMG_SIZE, IMG_SIZE))
            ax = fig.add_subplot(len(CLASSES), per_class, idx)
            ax.imshow(img); ax.axis("off")
            if idx % per_class == 1:
                ax.set_ylabel(cls, rotation=0, labelpad=40, va='center')
            idx += 1
    plt.tight_layout(); plt.show()

def preview_basic_augmentation(folder, class_name):
    cls_dir = os.path.join(folder, class_name)
    f = random.choice(os.listdir(cls_dir))
    img = cv2.resize(cv2.imread(os.path.join(cls_dir, f)), (IMG_SIZE, IMG_SIZE)) / 255.0
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax in axes.flatten():
        aug = tf.image.random_flip_left_right(img)
        aug = tf.image.random_brightness(aug, 0.1)
        ax.imshow(aug); ax.axis("off")
    plt.show()

# ─── DATA UTILITIES ───────────────────────────────────────────────────────────
def load_resize(folder):
    X, y = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir): continue
        for f in os.listdir(cls_dir):
            im = cv2.imread(os.path.join(cls_dir, f))
            if im is None: continue
            X.append(cv2.resize(im, (IMG_SIZE, IMG_SIZE)))
            y.append(cls)
    return np.array(X), np.array(y)

def oversample_to_max(X, y):
    counts = {cls: np.sum(y==cls) for cls in CLASSES}
    max_c = max(counts.values())
    X_list, y_list = [X], [y]
    for cls, c in counts.items():
        if c < max_c:
            idxs = np.where(y==cls)[0]
            extra = np.random.choice(idxs, max_c - c, replace=True)
            X_list.append(X[extra]); y_list.append(y[extra])
    Xb = np.concatenate(X_list); yb = np.concatenate(y_list)
    return sklearn_shuffle(Xb, yb, random_state=RANDOM_SEED)

def encode_and_weights(labels):
    idxs = [CLASSES.index(l) for l in labels]
    Y = to_categorical(idxs, num_classes=len(CLASSES))
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=idxs)
    return Y, dict(enumerate(cw))

# ─── CLASS-SPECIFIC FOCAL LOSS ───────────────────────────────────────────────
def focal_loss(alpha, gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        ce = -y_true * tf.math.log(tf.clip_by_value(y_pred,1e-8,1.0))
        w  = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(w * ce, axis=-1)
    return loss_fn

# give glioma double weight
LOSS = focal_loss(alpha=[0.5,0.1667,0.1667,0.1667], gamma=2.0)

# ─── MIXUP GENERATOR ─────────────────────────────────────────────────────────
class MixUpGenerator(Sequence):
    def __init__(self, X, Y, batch_size=BATCH_SIZE, alpha=0.2, shuffle=True):
        self.X, self.Y = X, Y
        self.bs, self.alpha, self.shuffle = batch_size, alpha, shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def on_epoch_end(self):
        self.idxs = np.arange(len(self.X))
        if self.shuffle: np.random.shuffle(self.idxs)

    def __getitem__(self, i):
        b = self.idxs[i*self.bs:(i+1)*self.bs]
        X1, Y1 = self.X[b], self.Y[b]
        m = np.random.beta(self.alpha, self.alpha, size=(len(b),1,1,1))
        m_y = m.reshape(-1,1)
        b2 = np.random.choice(self.idxs, size=len(b), replace=False)
        X2, Y2 = self.X[b2], self.Y[b2]
        X_mix = m*X1 + (1-m)*X2
        Y_mix = m_y*Y1 + (1-m_y)*Y2
        return X_mix, Y_mix

# ─── MODEL & TRAINING ────────────────────────────────────────────────────────
def build_model():
    try:
        b = EfficientNetB3(include_top=False, weights="imagenet",
                           input_shape=(IMG_SIZE,IMG_SIZE,3))
    except:
        b = EfficientNetB3(include_top=False, weights=None,
                           input_shape=(IMG_SIZE,IMG_SIZE,3))
    b.trainable = False
    x = GlobalAveragePooling2D()(b.output)
    x = Dropout(0.5)(x)
    out = Dense(len(CLASSES), activation="softmax")(x)
    m = Model(b.input, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(LR_HEAD),
              loss=LOSS, metrics=["accuracy"])
    return m

def get_callbacks(stage):
    return [
        ModelCheckpoint(f"{stage}.h5", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=2, verbose=1),
        TensorBoard(log_dir="logs")
    ]

def train_two_stage_with_mixup(model, X_tr, Y_tr, X_val, Y_val, cw):
    gen = MixUpGenerator(X_tr, Y_tr)
    h1 = model.fit(gen, steps_per_epoch=len(gen),
                   validation_data=(X_val, Y_val),
                   epochs=BASE_EPOCHS, class_weight=cw,
                   callbacks=get_callbacks("stage1"))
    for layer in model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_FINE),
                  loss=LOSS, metrics=["accuracy"])
    h2 = model.fit(gen, steps_per_epoch=len(gen),
                   validation_data=(X_val, Y_val),
                   epochs=FINE_EPOCHS, class_weight=cw,
                   callbacks=get_callbacks("stage2"))
    return h1, h2

def plot_history(h, tag):
    e = range(1, len(h.history["accuracy"])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(e, h.history["accuracy"], label="train")
    plt.plot(e, h.history["val_accuracy"], label="val")
    plt.title(f"Acc – {tag}"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(e, h.history["loss"], label="train")
    plt.plot(e, h.history["val_loss"], label="val")
    plt.title(f"Loss – {tag}"); plt.legend()
    plt.show()

def final_eval(model, X_test, Y_test):
    preds = model.predict(X_test)
    p = np.argmax(preds, axis=1)
    t = np.argmax(Y_test, axis=1)
    print(classification_report(t, p, target_names=CLASSES))
    cm = confusion_matrix(t, p)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.show()

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────
train_p, test_p = get_paths(PROJECT_DIR)
display_palettes()
plot_class_distribution(train_p, "Train Counts")
X_traw, y_traw = load_resize(train_p)
X_test,  y_test = load_resize(test_p)
X_traw, y_traw = sklearn_shuffle(X_traw, y_traw, random_state=RANDOM_SEED)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_traw, y_traw, test_size=VAL_SPLIT,
    stratify=y_traw, random_state=RANDOM_SEED
)
X_tr, y_tr = oversample_to_max(X_tr, y_tr)
Y_tr, cw  = encode_and_weights(y_tr)
Y_val, _  = encode_and_weights(y_val)
Y_test, _ = encode_and_weights(y_test)
model = build_model()
h1, h2 = train_two_stage_with_mixup(model, X_tr, Y_tr, X_val, Y_val, cw)
plot_history(h1, "Stage 1")
plot_history(h2, "Stage 2")
final_eval(model, X_test, Y_test)