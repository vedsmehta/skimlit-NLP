
# import data app builder
import streamlit as st
# Import numpy and pandas as well as tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from utils import get_classes
import app2

def main():

	st.title("Food Vision Big™")

	# Sidebar

	food_items = pd.DataFrame({'classes':get_classes()})

	st.sidebar.markdown("# Food Vision 🍔👁")

	st.sidebar.markdown("""
						### Food Vision is a data app that uses a **CNN Deep Learning Model** to predict what food is in your image. The data comes from the **Food101** Dataset  
						#### Cool Fact: The CNN Model (EfficientNetB0) used in this application beats the **DeepFood** paper which had a top-1 accuracy of 74%. This model has a 78% top-1 accuracy.  

						#### To get started upload an image file from your computer. This file should be a food image of the following classes:  
						## 
						""")
	st.sidebar.write(food_items)

	st.markdown('### A data app that identifies the food in your photo')

	# Model and file data
	model = load_model('food_vision_big_efficient_net_b0_big_boi.h5')
	image = st.file_uploader("Choose a PNG or JPG file with your food item", type=['jpeg', 'jpg', 'png'])

	if image is not None:

		read_img = image.read()

		ready_image = load_and_prepare_image(read_img, 224)

		predicted_data = predict(model, ready_image)

		image_class ,image_class_name = get_image_class(predicted_data, get_classes())

		st.image(image, caption=image.name)
		st.markdown(f"## Your Image Is **{image_class_name.capitalize().replace('_', ' ')}**")

		st.button("Was the model correct?", on_click=lambda: correct(model, ready_image, image_class))
		st.button("Was the model incorrect?", on_click=lambda: incorrect(model, ready_image))
def get_image_class(prediction, class_names):
	pred_class = prediction.argmax()
	print(prediction)
	class_name = get_classes()[pred_class]
	return pred_class, class_name

def predict(model, data, is_singular=True):
	if is_singular:
		return model.predict(tf.expand_dims(data, axis=0))
	else:
		return model.predict(data)

def load_and_prepare_image(image, size, scale=False):
	image = tf.io.decode_image(image)
	image = tf.image.resize(image, [size, size])
	image = tf.cast(image, tf.float32)
	if scale:
		return image/255.
	else:
		return image
def load_model(filepath):
	return tf.keras.models.load_model(filepath=filepath)

def correct(model, image, image_class):
    image_label = image_class
    model.fit(tf.expand_dims(image, axis=0), image_label)
    model.save('food_vision_big_efficient_net_b0_big_boi.h5')

def incorrect(model, image):
    app2.main(model, image)
    main()

if __name__ == '__main__':
	main()
