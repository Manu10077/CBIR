import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, hog

DATASET_PATH = "dataset/"
SAVE_PATH = "features/"
IMG_SIZE = 256

# ---------- COLOR HISTOGRAM ----------
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-6)
    return hist

# ---------- LBP ----------
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-6)
    return hist

# ---------- LDiP ----------
def extract_ldip(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(gray)
    diag2 = np.diag(np.fliplr(gray))
    hist, _ = np.histogram(np.hstack((diag1, diag2)), bins=32, range=(0,255))
    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-6)
    return hist

# ---------- HOG ----------
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False)
    fd = fd / (np.linalg.norm(fd) + 1e-6)
    return fd

# ---------- COLOR MOMENTS ----------
def extract_color_moments(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chans = cv2.split(hsv)
    feats = []
    for c in chans:
        feats.append(np.mean(c))
        feats.append(np.std(c))
        feats.append(np.mean(np.abs(c - np.mean(c))))
    return np.array(feats)

# ---------- EDGE HIST ----------
def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist, _ = np.histogram(edges, bins=32, range=(0,255))
    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-6)
    return hist

# ---------- FEATURE FOR ANY IMAGE ----------
def extract_features_of(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fv = np.concatenate([
        extract_color_histogram(img),
        extract_lbp(img),
        extract_ldip(img),
        extract_hog(img),
        extract_color_moments(img),
        extract_edges(img)
    ])

    fv = np.nan_to_num(fv)
    fv = fv.astype(np.float32)
    return fv

# ---------- EXTRACT DATASET ----------
def extract_features():
    features = {}
    image_files = []

    # Read subfolders
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                image_files.append(os.path.join(root, file))

    print("Total images found:", len(image_files))

    for i, file in enumerate(image_files):
        img = cv2.imread(file)
        if img is None:
            continue
        fv = extract_features_of(img)
        features[file] = fv
        print(f"[OK] {i+1}/{len(image_files)}")

    np.save(os.path.join(SAVE_PATH, "features.npy"), features)
    print("✅ Features saved successfully!")

if __name__ == "__main__":
    extract_features()
