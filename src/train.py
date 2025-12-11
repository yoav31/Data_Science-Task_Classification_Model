import streamlit as st
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from src.model import TitanicModel
import os

def train_model(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

   
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

    input_dim = X_train.shape[1]
    model = TitanicModel(input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 25
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total

        st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # שמירת המודל
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "titanic_model.pth")
    torch.save(model.state_dict(), model_path)
    st.write("Model saved to", model_path)
    st.write("Training complete.")
    st.write("Final Validation Accuracy: {:.4f}".format(val_acc))

    # --- Evaluation on test set ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    test_acc = correct / total
    st.write("Test Accuracy: {:.4f}".format(test_acc))
