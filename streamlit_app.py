import streamlit as st
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load vocabulary mappings from JSON files
with open(r"stoi.json", "r") as f:
    stoi = json.load(f)

with open(r"itos.json", "r") as f:
    itos = json.load(f)
    
def preprocess_input(input_text, stoi, block_size=15):
    input_words = input_text.strip().lower().split()
    input_indices = [stoi.get(word, stoi['.']) for word in input_words]
    input_indices = input_indices[-block_size:]
    input_tensor = input_indices + [0] * (block_size - len(input_indices))
    return torch.tensor(input_tensor[-block_size:], dtype=torch.long).unsqueeze(0)

def decode_index(indices):
    return [itos[str(i.item())] for i in indices]

def predict_next_words(input_text, k):
    input_tensor = preprocess_input(input_text, stoi, block_size)
    model.eval()
    predicted_words = []

    for _ in range(k):
        with torch.no_grad():
            output = model(input_tensor)  # Output shape should match vocab size
            top_k_indices = output.topk(1).indices[0]  # Get the top prediction
            predicted_word = decode_index(top_k_indices)
            predicted_words.append(predicted_word[0])  # Append the predicted word
            
            # Update the input_tensor with the new predicted word
            new_input = input_tensor[0].tolist()[1:] + top_k_indices.tolist()  # Shift left and add new prediction
            input_tensor = torch.tensor(new_input, dtype=torch.long).unsqueeze(0).to(input_tensor.device)  # Maintain the same device

    return predicted_words

# Define the NextWord model class
class NextWord(torch.nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, activation_function='relu', hidden_size=1024):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.lin1 = torch.nn.Linear(block_size * emb_dim, hidden_size)
        
        # Choose activation function
        if activation_function == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_function == 'tanh':
            self.activation = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.lin2(x)
        return x

# Function to load the model based on user selections
def load_model(emb_dim, block_size, activation_function):
    model = NextWord(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, activation_function=activation_function)
    model_path = f"{block_size}_{emb_dim}_{activation_function}.pth"
 
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    return model

# Streamlit UI
st.title("Next Word Prediction App")

# Model selection UI
emb_dim_options = [32, 64]  # Available embedding sizes
block_size_options = [5, 10, 15]  # Available context lengths
activation_options = ['relu', 'tanh']  # Available activation functions

emb_dim = st.selectbox("Select Embedding Size:", emb_dim_options)
block_size = st.selectbox("Select Context Length:", block_size_options)
activation_function = st.selectbox("Select Activation Function:", activation_options)

# Load the selected model
model = load_model(emb_dim, block_size, activation_function)

user_input = st.text_input("Enter your input text:")
k = st.number_input("Enter the number of predictions (k):", min_value=1, value=5)

if st.button("Predict"):
    complete_sentence = user_input
    for _ in range(k):
        predictions = predict_next_words(complete_sentence, 1)
        complete_sentence += " " + " ".join(predictions)
    st.write("Predicted complete sentence:")
    st.write(complete_sentence)

