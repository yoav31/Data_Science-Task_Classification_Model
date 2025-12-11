import streamlit as st
import pandas as pd
import os
from src.data_preprocessing import load_data
from EDA.sur_by_gen import survival_by_gender
from EDA.sur_by_class import survival_by_class
from EDA.sur_by_age import survival_by_age
from models.trained_model import train_model_download

st.title("Titanic Survival Prediction Model Training")
st.write("This app trains a neural network model to predict survival on the Titanic dataset.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    file_path = os.path.join("data", uploaded_file.name)
    df.to_csv(file_path, index=False)
    
    st.write("Sample data:", df.head())
    st.title("Exploratory Data Analysis")
    survival_by_gender(df)
    survival_by_class(df)
    survival_by_age(df)
    st.button("click to Load Data and Train Model", on_click=load_data)
    model_path = "models/titanic_model.pth"

    with open(model_path, "rb") as f:  
        model_bytes = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="trained_titanic_model.pth",
        mime="application/octet-stream")
    st.title("")

    