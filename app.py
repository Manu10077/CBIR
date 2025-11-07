import os
import cv2
import faiss
import numpy as np
import streamlit as st
from cnn_extract import extract_features_of
from PIL import Image

# ---- Load FAISS index + filename map ----
INDEX_PATH = "features/cnn.index"
FNAMES_PATH = "features/cnn_filenames.npy"

st.set_page_config(page_title="CBIR Web App", layout="wide")
st.title("🔎 Content-Based Image Retrieval (CNN + FAISS)")
st.caption("Upload an image to find visually similar images from the dataset.")

if not (os.path.exists(INDEX_PATH) and os.path.exists(FNAMES_PATH)):
    st.error("FAISS index or filenames not found. Run:\n"
             "1) python cnn_extract.py\n2) python faiss_index.py")
    st.stop()

index = faiss.read_index(INDEX_PATH)
filenames = np.load(FNAMES_PATH, allow_pickle=True)
dim = index.d
st.write(f"Indexed images: **{len(filenames)}** | Embedding dim: **{dim}**")

# ---- Sidebar controls ----
top_k = st.sidebar.slider("Top-K results", min_value=5, max_value=30, value=10, step=1)

# ---- File uploader ----
uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])
if uploaded:
    # Show query
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Query Image", use_container_width=False)

    # Convert to OpenCV BGR for our extractor
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    qv = extract_features_of(img_bgr).astype(np.float32)  # (4096,)
    q = np.expand_dims(qv, axis=0)                        # (1, 4096)

    # FAISS search (inner product ~ cosine)
    scores, idx = index.search(q, top_k)
    scores = scores[0]; idx = idx[0]

    # Collate results
    results = [(float(scores[i]), str(filenames[idx[i]])) for i in range(len(idx))]
    results = [(s, p) for (s, p) in results if os.path.exists(p)]

    st.subheader("Top Matches")
    if not results:
        st.warning("No matches found (paths may be invalid). Rebuild features/index.")
    else:
        cols = st.columns(5)
        labels = []
        for i, (sim, path) in enumerate(results):
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = os.path.basename(os.path.dirname(path))  # robust label
                labels.append(label)
                cols[i % 5].image(img, caption=f"{label}  (sim: {sim:.2f})")
            except Exception:
                continue

        if labels:
            # Majority-vote predicted category
            pred = max(set(labels), key=labels.count)
            st.success(f"Predicted Category: **{pred}**")

    with st.expander("Paths & Scores (debug)"):
        st.write(results)
