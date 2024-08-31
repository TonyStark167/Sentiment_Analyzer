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


for i in range(input_neuron_count):
    for j in range(hidden_neuron_count):
        input_weights[i, j] = np.random.rand()

for i in range(hidden_neuron_count):
    for j in range(output_neuron_count):
        hidden_weights[i, j] = np.random.rand()


def NN(user_input):
    input_neurons = np.zeros(input_neuron_count)
    for i in range(len(user_input)):
        input_neurons[i] = tokenizer(user_input[i])

    hidden_neurons = relu(np.dot(input_neurons, input_weights))

    output = np.dot(hidden_neurons, hidden_weights)
    print(output)
    if output[0] < output[1]:
        return 1
    elif output[0] > output[1]:
        return -1
    else:
        return 0

def MSE():
    error = 0
    for i in range(len(x_train)):
        error += (y_train["Sentiment"][i] - NN(list(x_train[i]))) ** 2

    return error / len(x_train)

epoches = 30
learning_rate = 0.001

def train():
    global input_weights, hidden_weights
    for _ in range(epoches):
        for i in range(len(x_train)):
            input_weights -= (-2 * y_train["Sentiment"][i] * (y_train["Sentiment"][i] - NN(x_train[i]))**2) / len(x_train) * learning_rate
            hidden_weights -= (-2 * y_train["Sentiment"][i] * (y_train["Sentiment"][i] - NN(x_train[i]))**2) / len(x_train) * learning_rate
        print(MSE())

train()
print(NN(x))
