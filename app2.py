import time
from utils import get_classes
import tensorflow as tf
import streamlit as st
import numpy as np

def main(model, image):
    correct_option = st.selectbox("What was the correct food?", options=get_classes())
    if correct_option is not None:
        for i in get_classes():
            if correct_option == i:
                model.fit(x=tf.expand_dims(image, axis=0), y=np.array([get_classes().index(i)]))
                model.save('food_vision_big_efficient_net_b0_big_boi.h5')
                st.write("### Thanks For Your Feedback!")
                st.write("Redirecting...")

                time.sleep(2)
