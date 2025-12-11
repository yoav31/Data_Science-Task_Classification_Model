import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def survival_by_age(df):
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    survival_rate_by_age = df.groupby('AgeGroup')['Survived'].mean().reset_index()
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='AgeGroup', y='Survived', data=survival_rate_by_age, palette="viridis")
    plt.title("Survival Probability by Age Group")
    plt.ylabel("Survival Probability")
    plt.xlabel("Age Group")
    plt.ylim(0,1)
    
    st.pyplot(plt.gcf())
