import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0,x)

df = pd.read_csv("data.csv")

x_train = df["Text"]
y_train = df.drop(columns=["Text"])

x_raw = input("Prompt : ")

x = np.array([])

for token in list(x_raw):
    x = np.append(x, ord(token) * 0.01)

print(len(x))

input_neuron_count = 100
hidden_neuron_count = 30
output_neuron_count = 2

input_neurons = np.zeros(input_neuron_count)
input_weights = np.ones((input_neuron_count,hidden_neuron_count))

hidden_neurons = relu(np.dot(input_neurons, input_weights))

hidden_weights = np.ones((hidden_neuron_count,output_neuron_count))

output = relu(np.dot(hidden_neurons, hidden_weights))
print(output)
