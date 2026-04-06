import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

st.header('🍎 Image Classification Model')

# Load model
model = load_model('Image_Classify.keras')

img_width = 180
img_height = 180

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum',
'carrot','cauliflower','chilli pepper','corn','cucumber','eggplant',
'garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango',
'onion','orange','paprika','pear','peas','pineapple','pomegranate',
'potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
'tomato','turnip','watermelon']

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", width=200)

    # Resize image
    image = image.resize((img_width, img_height))

    # Convert to array
    img_arr = tf.keras.utils.img_to_array(image)

    # Expand dimensions
    img_bat = tf.expand_dims(img_arr, 0)

    # Prediction
    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)

    # Output result
    st.write("🥕 Prediction: {} ".format(data_cat[np.argmax(score)]))
    st.write("📊 Accuracy: {:0.2f}%".format(np.max(score)*100))