from flask import Flask, render_template, request
import numpy as np
import cv2
import faiss

app = Flask(__name__)

# Load FAISS features once
features = np.load("cnn_features.npy", allow_pickle=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_file = request.files['query_image']
        if query_file:
            query_path = "static/query.jpg"
            query_file.save(query_path)

            # Import TensorFlow inside the route to avoid memory bloat
            import tensorflow as tf
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
            from tensorflow.keras.preprocessing import image

            model = VGG16(weights='imagenet', include_top=False)
            img = image.load_img(query_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            query_features = model.predict(x).flatten()

            index = faiss.IndexFlatL2(query_features.shape[0])
            index.add(features)
            D, I = index.search(np.array([query_features]), 10)
            return render_template('results.html', results=I[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run()


