# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:49:59 2021

@author: sachin sharma
"""
from PIL import Image

import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt

from sklearn.externals import joblib
import os


@st.cache
def load_dataset(path):
    df=pd.read_csv(path)
    return df
@st.cache
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

@st.cache
def main():
    st.title("Mobile Price Prediction App")
    st.subheader("Predicts the price range of smartphones using Python and Streamlit")
    img=Image.open('is.jfif')
    st.image(img)
    
    menu=['EDA','Prediction']
    choices=st.sidebar.selectbox("Select Activities",menu)
    
    if choices =='EDA':
        st.subheader('EDA')
        data=load_dataset('train.csv')
        st.dataframe(data)
    
        
       
        
        if st.checkbox('show summary'):
            st.write(data.describe())
        if st.checkbox('columns'):
            st.write(data.columns)
            
        if st.checkbox('columns information'):
            
            
            st.write({'battery_power': 'Total energy a battery can store in one time measured in mAh',

'blue': 'Has bluetooth or not',

'clock_speed': 'speed at which microprocessor executes instructions',

'dual_sim': 'Has dual sim support or not',


'fc': 'Front Camera mega pixels',

'four_g': 'Has 4G or not',

'int_memory': 'Internal Memory in Gigabytes',

'm_dep': 'Mobile Depth in cm',

'mobile_wt': 'Weight of mobile phone',

'n_cores': 'Number of cores of processor',

'pc_Primary':'Camera mega pixels',

'px_height': 'Pixel Resolution Height',

'px_width': 'Pixel Resolution Width',

'ram': 'Random Access Memory in Mega Bytes',

'sc_h': 'Screen Height of mobile in cm',

'sc_w': 'Screen Width of mobile in cm',

'talk_time': 'longest time that a single battery charge will last when you are',

'three_g': 'Has 3G or not'})
        if st.checkbox('Dataset shape'):
            st.write(data.shape)
        
        if st.checkbox('class distribution of target variable'):
            var=data['price_range'].value_counts()
            list_labels=['Low Cost','Medium Cost','High Cost','Very High Cost']
        
            colors=['r','g','b','m']
            fig,ax=plt.subplots()
            ax.pie(var,labels=list_labels,colors=colors,autopct='%1.2f%%')
            st.pyplot(fig)
        
            
        
    if choices=='Prediction':
        
        st.subheader("Fetures to be selected for prediction")
        st.write(['battery_power',
 'clock_speed',
 'int_memory',
 'mobile_wt',
 'pc',
 'px_height',
 'px_width',
 'ram',
 'sc_w',
 'sc_4'
 'n_cores'])
        battery_power=st.sidebar.slider("Total energy a battery can store in one time measured in mAh",501,1998)
        
        fc=st.sidebar.slider("Front Camera mega pixels",0,19,2)
        m_dep=st.sidebar.slider('Mobile Depth in cm',0.1,1.0,0.2)
        
        int_memory=st.sidebar.slider('Inter Menmory In Gegabytes',2,64,2)
        mobile_wt=st.sidebar.slider("Weight Of Mobile(In Grams)",80,200,100)
        n_cores=st.sidebar.slider('Number of cores of processor',1,8,1)
        
        px_height=st.sidebar.slider("Pixel Resolution Height",0,1960,5)
        px_width=st.sidebar.slider("Pixel Resolution Width",500,1998,500)
        ram=st.sidebar.slider("RAM(In Megabytes)",256,3998)
        sc_h=st.sidebar.slider("Screen Height of mobile in cm",5,19,5)
        sc_w=st.sidebar.slider("Screen Width of mobile in cm",0,18,1)
        
        
        dict_1={'Total energy a battery can store in one time measured in mAh':battery_power,
                'Front Camera mega pixels':fc,
                
                'Mobile Depth in cm':m_dep,
                'Inter Menmory In Gegabytes':int_memory,
                "Weight Of Mobile(In Grams)":mobile_wt,
                'Number of cores of processor':n_cores,
                
                'Pixel Resolution Height':px_height,
                'Pixel Resolution Width':px_width,
                "RAM(In Megabytes)":ram,
                "Screen Height of mobile in cm":sc_h,
                "Screen Width of mobile in cm":sc_w}
                
                
            
                
                
            
            
            
            

            
            
        st.subheader("User Inputs")
        st.write(dict_1)
        
        values=np.array([battery_power,fc,int_memory,m_dep,mobile_wt,n_cores,px_height,px_width,ram,sc_h,sc_w,]).reshape(1,-1)
        predictor=load_prediction_models('mobile_prcie_prediction.pkl')
        preds=predictor.predict(values)
        st.subheader("Predicted Results")
        list_labels=['Low Cost','Medium Cost','High Cost','Very High Cost']
        results=np.argmax(preds)
        st.write("The predicted price range of the mobile is ",str(list_labels[results])+" category")
        

    
        
        
if __name__ == '__main__':
	main()
    
