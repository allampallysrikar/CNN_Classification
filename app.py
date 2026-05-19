import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import urllib.request

# Load EfficientNetB0 pre-trained on ImageNet (full classification model)
model = tf.keras.applications.EfficientNetB0(weights='imagenet')

# Fetch ImageNet class labels
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_index = json.loads(response.read().decode())

# Build labels list of 1000 names (index -> human-readable name)
labels = [None] * 1000
for idx, (synset, name) in class_index.items():
    labels[int(idx)] = name.replace("_", " ")


def classify_image(img):
    if img is None:
        return {}

    # Resize to (224, 224)
    img_pil = Image.fromarray(img).resize((224, 224))
    img_array = np.array(img_pil)

    # Expand dims to create batch of 1
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Preprocess for EfficientNetB0
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)

    # Predict
    predictions = model.predict(img_preprocessed, verbose=0)
    probs = predictions[0]

    # Get top 5 indices
    top5_indices = np.argsort(probs)[::-1][:5]

    # Build result dict {label: confidence_float}
    result = {labels[i]: float(probs[i]) for i in top5_indices}
    return result


# Build Gradio app
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🐾 CNN Animal Classifier\n"
        "### Upload any animal photo — the model will identify it using deep learning\n"
        "Built with TensorFlow & EfficientNetB0 by [Srikar Allampally](https://github.com/allampallysrikar)"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload an animal image", type="numpy")
            classify_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top Predictions")

    classify_btn.click(fn=classify_image, inputs=image_input, outputs=label_output)

    gr.Markdown("""
---
*Powered by EfficientNetB0 trained on ImageNet (1000 classes including cats, dogs, birds, and hundreds more animals).*
""")

demo.launch()
