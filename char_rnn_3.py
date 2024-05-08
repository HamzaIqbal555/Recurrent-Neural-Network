#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)

def letterEncoding(fileName):
    rawText = open(fileName, 'r', encoding='utf-8').read()
    rawText = rawText.lower()

    chars = sorted(list(set(rawText)))
    charToInt = dict((c, i) for i, c in enumerate(chars))
    intToChar = dict((i, c) for i, c in enumerate(chars))

    nChars = len(rawText)
    nVocab = len(chars)

    integers = [charToInt[char] for char in rawText]
    oneHotMatrix = np.eye(nVocab)[integers]

    return oneHotMatrix, nVocab, charToInt, intToChar

def buildTrainingData(oneHotMatrix):
    X = []
    Y = []

    for i in range((len(oneHotMatrix) - 1)):
        X.append(oneHotMatrix[i])
        Y.append(oneHotMatrix[i + 1])
    X = torch.Tensor(np.array(X))
    Y = torch.Tensor(np.array(Y))
    return X, Y

class Net(nn.Module):
    def __init__(self, vocabSize, hiddenSize):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(vocabSize + hiddenSize, 4)
        self.dense2 = nn.Linear(4, hiddenSize)
        self.dense3 = nn.Linear(hiddenSize, vocabSize)

    def forward(self, x, prevState):
        concatenatedInput = torch.cat((x, prevState), dim=1)
        x = self.dense1(concatenatedInput)
        hiddenState = self.dense2(x)
        output = self.dense3(hiddenState)
        return output, hiddenState

def evaluateFunc(seq, model, state, vocabSize, intToChar):
    seqArray = seq.numpy()
    charIndex = np.argmax(seqArray)
    inputChar = intToChar[charIndex]
    
    # Initializing the prediction with the input sequence
    predictions = inputChar
    if inputChar == ' ':
        inputChar = " (space)"
    
    print(f"Randomly selected starting character: {inputChar}")

    modelInput = seq.unsqueeze(0)
    for i in range(50):
        output, state = model(modelInput, state)
        probabilities = F.softmax(output, dim=1)
        predictedIndex = torch.argmax(probabilities, dim=1)
        predictedChar = intToChar[predictedIndex.item()]
        
        predictions += predictedChar
        newInput = torch.zeros_like(modelInput)
        newInput[0, predictedIndex] = 1
        modelInput = newInput 
    return predictions

def evaluateStaticSequenceFunc(sequence, model, state, vocabSize, intToChar, charToInt):
    predictions = sequence[0]
    seqArray = np.array([charToInt[c] for c in sequence])

    for i in range(len(sequence) - 1):
        oneHotInput = np.zeros(vocabSize, dtype=np.float32)
        oneHotInput[seqArray[i]] = 1
        oneHotInputTensor = torch.tensor(oneHotInput).unsqueeze(0)
        output, state = model(oneHotInputTensor, state)
        probabilities = F.softmax(output, dim=1)
        predictedIndex = torch.argmax(probabilities, dim=1)
        predictedChar = intToChar[predictedIndex.item()]

        predictions += predictedChar

    return predictions

def trainAndEvaluate(fileName, sequence):
    oneHotMatrix, vocabSize, charToInt, intToChar = letterEncoding(fileName)
    X, Y = buildTrainingData(oneHotMatrix)
    hiddenSize = 2

    model = Net(vocabSize, hiddenSize)
    epochs = 10
    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lenX = len(X)

    print("Training ......")
    for epoch in range(epochs):
        state = torch.zeros(1, hiddenSize)
        lossE = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            xT = X[i].unsqueeze(0)
            yT = Y[i].unsqueeze(0)
            yPred, newState = model(xT, state)
            loss = lossFn(yPred, yT)
            loss.backward(retain_graph=True)
            optimizer.step()
            lossE += loss.item()
            state = torch.clone(newState.detach())
            epochLoss = lossE / lenX
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss:.4f}")

    randomInput = X[random.randint(0, lenX)]
    state = torch.zeros(1, hiddenSize)
    
    generatedSequence = evaluateStaticSequenceFunc(sequence, model, state, vocabSize, intToChar, charToInt)
    print(f"Generated sequence for seed: {generatedSequence}")

    generatedSequence = evaluateFunc(randomInput, model, state, vocabSize, intToChar)
    print("Generated sequence:")
    print(generatedSequence)


trainAndEvaluate('abcde.txt', "abcde")
# Uncomment the following line to run the function
#trainAndEvaluate('abcde_edcba.txt', "abcde edcba")


# In[3]:


trainAndEvaluate('abcde_edcba.txt', "abcde edcba")

