import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
labels = []
samples = []

def get_data(theLabels, theSamples):
    with open("samples.csv") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            theSamples.append(row)
    theSamples = np.array(theSamples)

    with open("labels.csv") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            theLabels.append(row)
    theLabels = np.array(theLabels)

    return theLabels, theSamples

labels, samples = get_data(labels, samples)
x_train, x_test, y_train, y_test = train_test_split(samples, labels)

def predict(model, testdata):
    classes = [
    "Depressant",
    "Psychedelics",
    "Stimulants",
    "Dissociatives",
    "Nootropic",
    "Entactogens",
    "Cannabinoid",
    "Opioids",
    "Deliriant",
    "Sedative",
    "Antipsychotic",
    "Oneirogen",
    "Hallucinogens",
    "Eugeroic",
    "Antidepressant",
]
    print(testdata)
    predictions = model.predict(testdata)
    index = np.argmax(predictions)
    prediction = classes[index]
    return prediction, predictions


#begin model stuff
def run_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=221))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=15, activation='softmax'))

    sgd = optimizers.SGD(learning_rate=0.06, momentum=0.5, nesterov=True)

    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, batch_size=8)

    #evaluate model on test data
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    #by experimenting with different effects we can see how they affect the prediction
    #create a fake input to see what class the model predicts it belongs to
    test = np.zeros((1, 221))
    test[0][107]  = 1
    prediction, predictions = predict(model, test)
    print(predictions)
    print('I think this drug is of class: ' + prediction)
run_model()




