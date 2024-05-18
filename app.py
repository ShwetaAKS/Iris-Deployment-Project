#Import all require Packages
import numpy as np
import pandas as pd
import pickle
import streamlit as st

#create user interface using streamlit app
st.set_page_config(page_title="Iris Deployment Project",layout='wide')

#Show the title of app in the body
st.title("Iris Deployment Project - Shweta Kolapkar")

#taking inputs from users
sep_len = st.number_input('Sepal Length: ',min_value=0.00,step=0.01)
sep_wid = st.number_input('Sepal Width: ',min_value=0.00,step=0.01)
pet_len = st.number_input('Petal Length: ',min_value=0.00,step=0.01)
pet_wid = st.number_input('Petal Width: ',min_value=0.00,step=0.01)

#Adding a submit button to the page
button = st.button('Predict')

#After pressing submit button what should happen
if button:
    #Load preprocessor
    with open('/workspaces/Iris-Deployment-Project/Notebook/pre.pkl','rb') as file1:
        pre = pickle.load(file1)

    #Load the model
    with open('/workspaces/Iris-Deployment-Project/Notebook/model.pkl','rb') as file2:
        model = pickle.load(file2)

    #if submit button is pressed
    dct = {'sepal_length':[sep_len],'sepal_width':[sep_wid],'petal_length':[pet_len],'petal_width':[pet_wid]}
    xnew = pd.DataFrame(dct)

    #Preprocess xnew
    xnew_pre = pre.transform(xnew)

    #Get predictions along with probability
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)

    #Show above results in streamlit
    st.subheader(f'Predicted Result : ')
    st.subheader(f'Predicted Species : {pred[0]}')
    st.subheader(f'Probability : {max_prob*100:.2f}%')
    st.progress(max_prob)
