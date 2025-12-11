import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def survival_by_class(df):
    
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Pclass", y="Survived", data=df, ax=ax)
    ax.set_title("Survival Rate by Passenger Class")
    st.pyplot(fig)