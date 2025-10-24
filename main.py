import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import zipfile
import os


# Function to unzip the models if not already extracted
def extract_zip(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


# Function to generate caption
def generate_caption(image_path, model, tokenizer, feature_extractor, max_length=34, img_size=224):
    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Generate the caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


# Streamlit app interface
def main():
    st.title("Image Caption Generator")
    st.write("Upload one or more images and generate captions using the trained model.")

    # Sidebar for image upload
    st.sidebar.title("Upload Images")
    uploaded_images = st.sidebar.file_uploader(
        "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    # Paths for the saved models and tokenizer zip files
    feature_zip_path = "feature.zip"  # Path to the feature.zip file
    model_zip_path = "model.zip"      # Path to the model.zip file
    tokenizer_zip_path = "tokenizer.zip"  # Path to the tokenizer.zip file

    # Folder to extract models
    extract_folder = "models/"
    
    # Check if models are already extracted, if not, extract them
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Extract zip files
    extract_zip(feature_zip_path, extract_folder)
    extract_zip(model_zip_path, extract_folder)
    extract_zip(tokenizer_zip_path, extract_folder)

    # Paths to the extracted models and tokenizer
    feature_extractor_path = os.path.join(extract_folder, 'feature_extractor.keras')
    model_path = os.path.join(extract_folder, 'model.keras')
    tokenizer_path = os.path.join(extract_folder, 'tokenizer.pkl')

    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    if uploaded_images:
        cols = st.columns(2)  # Two columns for image and caption display

        for idx, uploaded_image in enumerate(uploaded_images):
            # Save each uploaded image temporarily
            image_path = f"uploaded_image_{idx}.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Generate caption for the image
            caption = generate_caption(image_path, caption_model, tokenizer, feature_extractor)

            # Display the image and its caption with styled text
            with cols[idx % 2]:  # Alternate between the two columns
                st.image(image_path, use_container_width=True)
                st.markdown(
                    f"""<p style="font-size:18px;"><b style="color:black;">Caption:</b> <span style="color:red;">{caption}</span></p>""",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
