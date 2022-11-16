
# import data app builder
import streamlit as st
# Import numpy and pandas as well as tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

def main():

	st.markdown("""
	# Food Vision Big™
	### A data app that identifies the food in your photo
				""")
	# Sidebar
	class_df = pd.DataFrame({'classes':get_classes()})

	st.sidebar.markdown("# Food Vision 🍔👁")

	st.sidebar.markdown("""
						### Food Vision is a data app that uses a **CNN Deep Learning Model** to predict what food is in your image. The data comes from the **Food101** Dataset  
						#### Cool Fact: The CNN Model (EfficientNetB0) used in this application beats the **DeepFood** paper which had a top-1 accuracy of 74%. This model has a 78% top-1 accuracy.  

						#### To get started upload an image file from your computer. This file should be a food image of the following classes:  
						## 
						""")
	st.sidebar.write(class_df)

	# Model and file data
	model = load_model('food_vision_big_efficient_net_b0_big_boi.h5')
	img = st.file_uploader("Choose a PNG or JPG file with your food item",
    					   type=['jpeg', 'jpg', 'png'])

	if img is not None:

		read_img = img.read()

		ready_image = load_and_prepare_image(read_img, 224)

		predicted_data = predict(model, ready_image)

		_ ,image_class = get_image_class(predicted_data, get_classes())

		st.image(img, caption=img.name)
		st.markdown(f"## Your Image Is {image_class.upper().replace('_', ' ')}")

def get_image_class(prediction, class_names):
	pred_class = prediction.argmax()
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
	if scale:
		return image/255.
	else:
		return image
def load_model(filepath):
	return tf.keras.models.load_model(filepath=filepath)

def get_classes():
	class_names = ['apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']
	return class_names


if __name__ == '__main__':
	main()
