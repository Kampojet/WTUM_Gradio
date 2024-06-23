import gradio as gr

import tensorflow as tf
from PIL import Image
import numpy as np

model1 = tf.keras.models.load_model("InceptionV3.keras")
model2 = tf.keras.models.load_model("ResNet50V2.keras")
classes = ['angry', 'happy', 'relaxed', 'sad']

def predict_model1(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = np.argmax(model1.predict(img), axis=1)[0]
    return f"Your dog is {classes[prediction]}"

def predict_model2(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = np.argmax(model2.predict(img), axis=1)[0]
    return f"Your dog is {classes[prediction]}"

def predict_selected_model(image, model_name):
    if model_name == "Model 1" and image is not None:
        return predict_model1(image)
    elif model_name == "Model 2" and image is not None:
        return predict_model2(image)
    elif image is None:
        return "Choose an image"
    else:
        return "Choose a model"

inputs = [
    gr.Image(type="pil", label="Upload an image"),
    gr.Radio(["Model 1", "Model 2"], label="Select Model", value="Model 1")
]

output = gr.Text()

gr.Interface(
    fn=predict_selected_model,
    inputs=inputs,
    outputs=output,
    title="Dog Emotion Predictor",
    description="Select a model and upload an image to predict the class."
).launch(share=True, server_name='0.0.0.0', server_port=7860)
