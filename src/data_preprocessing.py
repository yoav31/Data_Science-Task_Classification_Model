import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.train import train_model
import streamlit as st

def load_data(path="Data/train.csv"):
    
    st.write("Loading and preprocessing data...")
    path="Data/train.csv"
    df = pd.read_csv(path)

    print(df.isnull().sum())
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df.replace({True: 1, False: 0})

    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    train_model(df)

