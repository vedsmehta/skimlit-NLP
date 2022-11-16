# Imports
import time

import streamlit as st
import tensorflow as tf
import app_utils

# Header

st.title("SkimLit 📄🔥")
st.markdown("## A data app for making RCT Medical Abstracts easier to read")

before_example = """\n
Currently, there is confusion about the value of using nutritional support to
treat malnutrition and improve functional outcomes in chronic obstructive pulmonary disease (COPD).
This systematic review and meta-analysis of randomized, controlled trials...
"""

after_example = """\n
**BACKGROUND**:  Currently, there is confusion about the value of using nutritional support to treat malnutrition and improve functional outcomes in chronic obstructive pulmonary disease (COPD).\n
**OBJECTIVE**:  This systematic review and meta-analysis of randomized, controlled trials (RCT) aimed to clarify the effectiveness of nutritional support in improving functional outcomes in COPD.\n
**METHODS**:  A systematic review identified 12 RCT (n = 448) in stable COPD patients investigating the effects of nutritional support (dietary advice (1 RCT), oral nutritional supplements (10 RCT), enteral tube feeding (1 RCT)) versus control on functional outcomes.\n
**RESULTS**:  Meta-analysis of the changes induced by intervention found that while respiratory function (forced expiratory volume in 1 s, lung capacity, blood gases) was unresponsive to nutritional support, both inspiratory and expiratory muscle strength (maximal inspiratory mouth pressure +3.86 standard error (SE) 1.89 cm H2 O, P = 0.041; maximal expiratory mouth pressure +11.85 SE 5.54 cm H2 O, P = 0.032) and handgrip strength (+1.35 SE 0.69 kg, P = 0.05) were significantly improved and associated with weight gains of ≥2 kg.\n
**CONCLUSIONS**:  Nutritional support produced significant improvements in quality of life in some trials, although meta-analysis was not possible. It also led to improved exercise performance and enhancement of exercise rehabilitation programmes. This systematic review and meta-analysis demonstrates that nutritional support in COPD results in significant improvements in a number of clinically relevant functional outcomes, complementing a previous review showing improvements in nutritional intake and weight.\n
"""

st.markdown("Before: "+before_example)
st.markdown("After: "+after_example)

# Model and Predictions
class StreamlitModel(object):
    """Base Model Class in Streamlit """
    def __init__(self):
        st.markdown("## Your Turn!")
    def display_text_input(self, label, default_value=""):
        self.input_text = st.text_area(label, default_value, height=50)
        st.markdown("**Your text before SkimLit**: \n\n"+self.input_text)
    def predict_and_display(self, model, abstract_data):
        self.skimlitted_text = app_utils.preprocess_and_predict_abstract(abstract_data, model)
        st.markdown("**Your text after SkimLit!**: \n"+self.skimlitted_text)
    def __call__(self, filepath, custom_objects=None):

        self.display_text_input("Enter Your Abstract Data (string format)")
        with st.spinner("Loading Model... (This May Take a While)"):
            # Load model
            self.tf_model = tf.keras.models.load_model(filepath,
                                                        custom_objects=custom_objects).assert_existing_objects_matched()
        st.success("Model Loaded!")
        
        with st.spinner("Making Abstract Easier to Read..."):
            # Display Abstractified data
            self.predict_and_display(self.tf_model, self.input_text)
        st.success("Done!")
# initialized st model
my_st_model = StreamlitModel()
# Call the model
my_st_model = my_st_model("skimlit_tribrid_model")


