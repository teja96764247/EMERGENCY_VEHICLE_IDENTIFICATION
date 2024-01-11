import streamlit as st
import tensorflow as tf





@st.cache_data
def load_model():
    model=tf.keras.models.load_model('emergency.hdf5')
    return model
model=load_model()

st.write("""
       #EMERGENCY VEHICLE IDENTIFICATION
""")

file=st.file_uploader("Please upload an vehicle image",type=['jpg','png'])

import cv2
from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(64,64)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,64,64,3])
    prediction=model.predict(img_reshape)
    result=np.argmax(prediction)
    if(result==0):
        return "The Vehicle is NOT an Emergeny Vehicle"
    elif(result==1):
        return "The Vehicle is an Emergency Vehicle"
    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    output=predict_function(image,model)
    st.success(output)