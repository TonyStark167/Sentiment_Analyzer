import numpy as np
import pandas as pd

def tokenizer(char):
    return ord(char) * 0.01

def relu(x):
    return np.maximum(0,x)

df = pd.read_csv("data.csv")

x_train = df["Text"]
y_train = df.drop(columns=["Text"])

x_raw = input("Prompt : ")

x = np.array([])

for token in list(x_raw):
    x = np.append(x, token)

print(len(x))

input_neuron_count = 100
hidden_neuron_count = 30
output_neuron_count = 2

input_weights = np.ones((input_neuron_count,hidden_neuron_count))
hidden_weights = np.ones((hidden_neuron_count,output_neuron_count))

def NN(user_input):
    input_neurons = np.zeros(input_neuron_count)
    for i in range(len(user_input)):
        input_neurons[i] = tokenizer(user_input[i])

    hidden_neurons = relu(np.dot(input_neurons, input_weights))

    output = relu(np.dot(hidden_neurons, hidden_weights))
    return np.argmax(output)

def MSE():
    error = 0
    for i in range(len(x_train)):
        error += (y_train["Sentiment"][i] - NN(list(x_train[i]))) ** 2

    return error / len(x_train)

print(MSE())
