import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cosine
from cnn_extract import extract_features_of

# ---------- LOAD SAVED FEATURES ----------
features = np.load("features/cnn_features.npy", allow_pickle=True).item()
print("✅ Loaded dataset features:", len(features))
print("Feature vector shape:", next(iter(features.values())).shape)

# ---------- DUPLICATE REMOVAL ----------
def remove_duplicates(results, threshold=0.995):
    filtered = []
    seen = []

    for sim, fname in results:
        if len(seen) == 0:
            filtered.append((sim, fname))
            seen.append(fname)
            continue

        too_close = False
        for s, f in filtered:
            if abs(sim - s) < (1 - threshold):
                too_close = True
                break
        if not too_close:
            filtered.append((sim, fname))

    return filtered

# ---------- RECURSIVE GROUPING ----------
def recursive_grouping(results):
    groups = {}
    for sim, fname in results:
        bucket = round(sim, 2)
        if bucket not in groups:
            groups[bucket] = []
        groups[bucket].append((sim, fname))
    sorted_keys = sorted(groups.keys(), reverse=True)

    final_list = []
    for key in sorted_keys:
        final_list.extend(groups[key])
        if len(final_list) >= 10:
            break
    return final_list[:10]

# ---------- IMAGE RETRIEVAL ----------
def retrieve_similar_images(query_path, top_k=10):
    query_img = cv2.imread(query_path)
    if query_img is None:
        raise FileNotFoundError(f"❌ Query image not found: {query_path}")

    query_fv = extract_features_of(query_img)
    print("Query feature shape:", query_fv.shape)

    results = []
    for key, fv in features.items():
        if fv.shape != query_fv.shape:
            print(f"⚠ Skipped (shape mismatch): {key}")
            continue
        sim = 1 - cosine(query_fv, fv)
        results.append((sim, key))

    print("Total matches before filtering:", len(results))
    if not results:
        print("❌ No valid matches found. Please recheck dataset features.")
        return []

    results = sorted(results, key=lambda x: x[0], reverse=True)
    results = remove_duplicates(results)
    results = recursive_grouping(results)

    print("Final matches after grouping:", len(results))
    return results[:top_k]

# ---------- DISPLAY RESULTS ----------
def display_results(query_path, results):
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    n = len(results)
    cols = 5
    rows = math.ceil(n / cols)

    plt.figure(figsize=(14, 8))
    plt.subplot(rows + 1, cols, 1)
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis("off")

    for i, (sim, fname) in enumerate(results):
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows + 1, cols, i + cols + 1)
        plt.imshow(img)
        plt.title(f"{sim:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---------- RUN ----------
if __name__ == "__main__":
    query = "query.jpg"  # replace with your query image path
    results = retrieve_similar_images(query, top_k=10)
    if results:
        display_results(query, results)