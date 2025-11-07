import os
import numpy as np
import faiss

MATRIX_PATH = "features/cnn_matrix.npy"
INDEX_PATH = "features/cnn.index"

def build_index():
    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError("cnn_matrix.npy not found. Run cnn_extract.py first.")

    mat = np.load(MATRIX_PATH)  # shape: (N, 4096)
    if mat.ndim != 2:
        raise ValueError("Feature matrix must be 2-D.")

    d = mat.shape[1]
    # Inner Product index (works as cosine because vectors are L2-normalized)
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    faiss.write_index(index, INDEX_PATH)

    print("✅ FAISS index built and saved:", INDEX_PATH)
    print("Vectors indexed:", index.ntotal)
    print("Dimensionality:", d)

if __name__ == "__main__":
    build_index()