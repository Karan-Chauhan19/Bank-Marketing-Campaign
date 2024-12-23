'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''
import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib


class TestData :
    def __init__(self) :
        pass

    def testing(self) :

        st.sidebar.title("Enter customer data :") 

        left_col, _ = st.columns([1, 3])  # 1:3 ratio for left column vs right empty space

        with left_col :
            Job = st.sidebar.selectbox("Job :", ['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'])
            Marital = st.sidebar.selectbox("Marital Status :", ['married','single','divorced'])
            Education = st.sidebar.selectbox("Education :", ['primary','secondary','tertiary','unknown'])
            House_loan = st.sidebar.selectbox("House loan :", ['yes','no'])
            Personal_loan = st.sidebar.selectbox("Personal loan :", ['yes','no'])
            Contact = st.sidebar.selectbox("Contact method :", ['cellular','telephone','unknown'])
            Month = st.sidebar.selectbox("Month of last contact :", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
            Poutcome = st.sidebar.selectbox("Previous Outcome :", ['failure','success','unknown','other'])
            Duration = st.sidebar.number_input("Duration : ", min_value=0, max_value=5000, value=320)
            Campaign = st.sidebar.number_input("Campaign :", min_value=0, max_value=100, value=3)
            pdays = st.sidebar.number_input("Pdays :", min_value=-1, max_value=900, value=-1)
            pre_campaign = st.sidebar.number_input("Previous :", min_value=0, max_value=300, value=0)

            st.markdown("""
            <style>
            /* Change font size for all labels */
            .stSelectbox {
                font-size: 50px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            user_input = {'job':Job,'marital':Marital,'education':Education,'house_loan':House_loan,'personal_loan':Personal_loan,'contact':Contact,'month':Month,'duration':Duration,'campaign':Campaign,'pdays':pdays,'pre_campaign':pre_campaign,'poutcome':Poutcome}

            if st.sidebar.button("Submit") :

                user_df = pd.DataFrame([user_input])
                model_pipeline1 = joblib.load('preprocessing_pipeline.pkl')
                user_input_transform = model_pipeline1.transform(user_df)
                model1 = load_model('model.h5')
                user_prediction = model1.predict(user_input_transform)
                output = 1 if user_prediction[0][0] > 0.5 else 0

                if output == 1 :
                    output_message = '<p style="font-size:20px; font-weight:bold; white-space:nowrap;">Customer is subscribed to a term deposit !!!</p>'
                else :
                    output_message = '<p style="font-size:20px; font-weight:bold; white-space:nowrap;">Customer is not subscribed to a term deposit !!!</p>'

                st.markdown(f"{output_message}", unsafe_allow_html=True)




            
            

            


