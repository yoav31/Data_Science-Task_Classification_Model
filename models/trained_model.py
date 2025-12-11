def train_model_download():
    import streamlit as st
    from models.trained_model import train_and_save_model
    model_path = train_and_save_model()
    with open(model_path, "rb") as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name="trained_titanic_model.h5",
            mime="application/octet-stream"
        )
    