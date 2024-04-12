import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
import cv2
from keras.models import load_model

model = load_model('Model.keras')

def predict_tumor(image):
    bytes_data = image.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((150, 150))
    image = np.array(image)
    input_img = np.expand_dims(image, axis = 0)
    result = model.predict(input_img)
    result = int(result[0][0])
    return result

def main():
    st.title("Brain Tumor Detection")
    image = st.file_uploader("Upload Image: ", type = ["csv", "png", "jpg"])
    show_file = st.empty()
    # if isinstance(image, BytesIO):
    #     show_file.image(img)
    if st.button("Predict"):
        result = predict_tumor(image)
        if (result == 1):
            st.error("Tumor detected in brain.")
        else:
            st.success("No tumor detected.")
        
if __name__ == '__main__':
    main()
