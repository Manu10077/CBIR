from flask import Flask, render_template, request, send_from_directory
import os, cv2, numpy as np, faiss, random
from werkzeug.utils import secure_filename
from cnn_extract import extract_features_of

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load FAISS index + filenames
index = faiss.read_index("features/cnn.index")
filenames = np.load("features/cnn_filenames.npy", allow_pickle=True)
DATASET_PATH = "dataset"

def get_random_dataset_images(limit=12):
    all_imgs = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_imgs)
    return all_imgs[:limit]

@app.route("/", methods=["GET", "POST"])
def home():
    dataset_images = get_random_dataset_images()
    if request.method == "POST":
        file = request.files.get("query_image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please upload an image.", dataset_images=dataset_images)

        filename = secure_filename(file.filename)
        query_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(query_path)

        img = cv2.imread(query_path)
        query_fv = extract_features_of(img).astype(np.float32)
        q = np.expand_dims(query_fv, axis=0)

        # FAISS search
        k = 12
        scores, idx = index.search(q, k)
        results = []
        for i in range(k):
            sim = float(scores[0][i])
            abs_path = filenames[idx[0][i]]
            if os.path.exists(abs_path):
                fname = os.path.basename(abs_path)
                results.append((sim, f"/dataset/{fname}"))

        predicted = "Dataset"
        if results:
            predicted = os.path.basename(os.path.dirname(filenames[idx[0][0]])) or "Dataset"

        return render_template(
            "index.html",
            query_img=filename,
            results=results,
            predicted=predicted,
            dataset_images=dataset_images,
        )
    return render_template("index.html", dataset_images=dataset_images)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/dataset/<filename>")
def dataset_file(filename):
    return send_from_directory("dataset", filename)

if __name__ == "__main__":
    from vercel_python_wsgi import serverless_wsgi
    app = serverless_wsgi(app)
