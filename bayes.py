import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

X = pd.read_csv('samples.csv')
y = pd.read_csv('labels.csv')

def convert_labels(original):
    result = []
    for i in range(len(original)):
        label = np.argmax(original[i])
        result.append(label)
    return result

y_converted = convert_labels(y.to_numpy())
x_train, x_test, y_train, y_test = train_test_split(X, y_converted)

print(np.shape(x_train))
print(np.shape(y_train))

def calc_accuracy(predicted, actual):
    if len(predicted) != len(actual):
        return print("ERROR: predicted labels and actual labels are not the same length")
    else:
        score = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                score+= 1
        return score / len(predicted)


model = BernoulliNB(alpha=0.1)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))