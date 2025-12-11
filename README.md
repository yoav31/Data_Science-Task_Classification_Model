# Titanic Survival Prediction

Predict whether a passenger survived the Titanic disaster using a PyTorch neural network and visualize results in a Streamlit app.


## Description
This project demonstrates an end-to-end machine learning pipeline:
- Load and preprocess the Titanic dataset (train.csv from Kaggle)
- Perform exploratory data analysis (EDA) with plots
- Train a PyTorch classification model
- Evaluate the model on a held-out validation set
- Visualize results and run inference in Streamlit

## Architecture & Design Choices
- Model: PyTorch feedforward neural network with 3 layers
- Loss function: BCEWithLogitsLoss for binary classification
- Optimizer: Adam
- Data splitting: 70% train, 15% validation, 15% test
- Evaluation: Accuracy on validation and test sets
- EDA: Visualizations using Seaborn and Matplotlib, integrated into Streamlit

## Installation
- git clone https://github.com/yoav31/Data_Science-Task_Classification_Model 
- cd Data_Science-Task_Classification_Model
- pip install -r requirements.txt

## Run the app:
- streamlit run ds_app.py
