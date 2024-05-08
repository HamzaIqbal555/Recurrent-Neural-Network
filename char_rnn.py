#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def one_hot_encode(file_name):
    # Read the file
    with open(file_name, 'r') as file:
        data = file.read()

    # Get unique characters
    unique_chars = sorted(set(data))
    vocab_size = len(unique_chars)

    # Create dictionaries for character-to-integer and integer-to-character mappings
    char_to_int = {c: i for i, c in enumerate(unique_chars)}
    int_to_char = {i: c for i, c in enumerate(unique_chars)}

    # Convert characters to integers
    int_data = [char_to_int[char] for char in data]

    # Convert integers to one-hot encoding
    one_hot_data = np.zeros((len(data), vocab_size), dtype=np.int32)
    for i, integer in enumerate(int_data):
        one_hot_data[i, integer] = 1

    return one_hot_data, vocab_size, char_to_int, int_to_char

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model(X, Y, epochs=10):
    model = build_model((X.shape[1],), Y.shape[1])
    model.summary()
    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        history = model.fit(X, Y, epochs=1, verbose=2)
        print("Loss:", history.history['loss'][0])

def evaluate_model(model, X, int_to_char):
    start_index = np.random.randint(0, len(X)-1)
    sentence = ''
    current_char = X[start_index]
    for _ in range(50):
        prediction = model.predict(np.array([current_char]))[0]
        next_index = np.argmax(prediction)
        next_char = int_to_char[next_index]
        sentence += next_char
        current_char = np.eye(len(prediction))[next_index]  # Convert next_char to one-hot
    print("Generated Sequence:", sentence)

file_name = 'abcde.txt'
one_hot_data, vocab_size, char_to_int, int_to_char = one_hot_encode(file_name)

# Prepare training data
X_train = one_hot_data[:-1]
Y_train = one_hot_data[1:]

# Train the model
train_model(X_train, Y_train, epochs=10)

# Evaluate the model
model = build_model((X_train.shape[1],), vocab_size)
model.fit(X_train, Y_train, epochs=10, verbose=0)  # Train a bit more for better evaluation
evaluate_model(model, X_train, int_to_char)

