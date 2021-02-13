import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import streamlit as st 
from PIL import Image
import detect1
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from io import BytesIO, StringIO

#image path finding function
#def file_selector(uploaded_file, folder_path='./data/images/'):
    #filenames = os.listdir(folder_path)
#   return os.path.join(folder_path, uploaded_file)

#main program
if __name__ == '__main__':
    end = False    
    st.title('My first app')
    st.title("Upload + Classification Example")
    count_image = 2
    #upload prompt 
    uploaded_file = st.file_uploader("Choose an image...", type="jpg") 
    if uploaded_file is not None:
        #show an image
        image = Image.open(uploaded_file)
        st.write(type(image))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        #get a file name
        file_names = uploaded_file.name
        #store in img_array
        #img_array = file_selector(file_names)
        #st.write(img_array)
        #pass that image path to detect1
        label, info = detect1.main(image)
        st.title(info)
        st.image(label, caption='Detected Image.', use_column_width=True)
        cv2.imwrite('./detections/' + str(file_names), label)
        st.title("checked")
            
            
