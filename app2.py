import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
import math
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r'C:\Users\DELL\traffic sign -streamlit\densenet-best-model-mixed-data-01.hdf5')
  return model


with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Traffic Sign Classification 🚦
         """
         )

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png","jpeg"])

import cv2
from PIL import Image
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
col1, col2 = st.columns(2)

if file is None:
    st.text("Please upload an image file")
else:
    with col1:
      image = np.array(Image.open(file))
      IMG_SIZE = (224,224)
      img = cv2.resize(image,IMG_SIZE)
      img=img/255.0
      img = np.expand_dims(img, axis=0)
      st.image(img,width = 300)
      predict_x= model.predict(img)
      classes_x=np.argmax(predict_x,axis=1)
      numbers = [0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,6,7,8,9]
      #classes = [' Speed limit (5km/h)','Speed limit (15km/h)','Speed limit (30km/h)','Speed limit (40km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)','Dont Go straight or left','Dont Go straight or Right','Dont Go straight',' Dont Go Left','Dont Go Left or Right','Dont Go Right','Dont overtake from Left','No Uturn','No Car',' No horn','Speed limit (40km/h)','Speed limit (50km/h)','Go straight or right','Go straight','Go Left','Go Left or right','Go Right','keep Left','Roundabout mandatory','watch out for cars','Horn','Bicycles crossing','Uturn','Road Divider','Traffic signals','Danger Ahead','Zebra Crossing','Bicycles crossing','Children crossing',' Dangerous curve to the left',' Dangerous curve to the right','Unknown1','Unknown2','Unknown3','Go right or straight','Go left or straight','Unknown4','ZigZag Curve','Train Crossing','Under Construction','Unknown5','Fences','Heavy Vehicle Accidents','Unknown6','Give Way','No stopping','No entry','Unknown7','Unknown8']
      classes = [' Speed limit (5km/h)','Speed limit (15km/h)','Dont Go straigh','Dont Go Left','Dont Go Left or Right',' Dont Go Right','Dont overtake from Left','No Uturn','No Car','No horn','Speed limit (40km/h)',' Speed limit (50km/h)','Speed limit (30km/h)','Go straight or right','Go straight','Go Left','Go Left or right','Go Right',' keep Left','keep Right','Roundabout mandatory','watch out for cars','Horn','Speed limit (40km/h)','Bicycles crossing','Uturn','Road Divider','Traffic signals','Danger Ahead','Zebra Crossing','Bicycles crossing','Children crossing','Dangerous curve to the left','Dangerous curve to the right','Speed limit (50km/h)','Unknown1','Unknown2','Unknown3',' Go right or straight',' Go left or straight',' Unknown4','ZigZag Curve','Train Crossing','Under Construction','Unknown5','Speed limit (60km/h)','Fences','Heavy Vehicle Accidents','Unknown6','Give Way','No stopping','No entry','Unknown7','Unknown8','Speed limit (70km/h)','speed limit (80km/h)','Dont Go straight or left','Dont Go straight or Right']
      dir_clases = dict(zip(numbers,classes))
    with col2:
      #st.write(f'Model is **{predict_x[0][classes_x[0]]*100:.2f}%** sure that it is ** {dir_clases[classes_x[0]]} **')
      dir_clases = dict(zip(predict_x[0],classes))
      import collections
      od = collections.OrderedDict(sorted(dir_clases.items(),reverse=True))

      for key,values in od.items():
        st.write(f'{key*100:.2f}% - {values}')