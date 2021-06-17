# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:34:57 2021

@author: Aitor
"""
#import the model
from tensorflow.keras.models import load_model
model = load_model('my_model.hdf5', compile = False)

#Create the app in streamlit
import streamlit as st
st.write("""
         # Pavement distress Prediction
         """
         )
st.write("This is a simple image classification web app to predict pavement distress")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

#Resize the image uploaded by the user
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a corner break!")
    elif np.argmax(prediction) == 1:
        st.write("It is a longitudinal cracking!")
    elif np.argmax(prediction) == 2:
        st.write("It is a map cracking!")
    else:
        st.write("It is a transverse cracking!")
    
    #st.text("Probability (0: corner break, 1: longitudinal cracking, 2: map cracking, 3:transverse cracking")
    st.text("Probability:")
    st.text("0: corner break")
    st.text("1: longitudinal cracking")
    st.text("2: map cracking")
    st.text("3: transverse cracking")
    st.write(prediction)