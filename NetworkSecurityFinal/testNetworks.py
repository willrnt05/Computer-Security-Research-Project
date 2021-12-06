from math import *
import math
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv


def WriteDatasetToCSV(nameOfCSV, dataSet):
    with open(nameOfCSV + ".csv", 'w', newline='') as csvFile:
        fieldNames = ['x', 'y(x)']
        writer = csv.DictWriter(csvFile, fieldnames = fieldNames)
        for k, v in dataSet.items():
            writer.writerow({'x': k, 'y(x)': v})

#ay' + by + c = 0
#y(0) = z
def firstOrderDataset(a, b, c, z, startX, endX, stepX):
    constant = z + (c/b)
    x = startX
    dataSetDict = {}
    while(x <= endX):
        y = constant*exp((-b/a) * x) - (c/b)
        dataSetDict[x] = y
        x += stepX
    print(dataSetDict)

def firstOrderLinearEquationOne(startX, endX, stepX):
    def y(x):
        return 5*exp((-3/2) * x) - (2)

    dataSetDict = {}
    x = startX
    while (x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict

def firstOrderLinearEquationTwo(startX, endX, stepX):
    def y(x):
        return 9*exp((3/2) * x) + (2)

    dataSetDict = {}
    x = startX
    while (x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict


#ay'' + by' + cy + d = 0
#This will require building a specific equation for efficiency and accuracy
def secondOrderLinearEquationOne(startX, endX, stepX):
    #Temporary Equation Chosen, one Real and one Complex Roots
    #a = 1, b=-4, c=9, d=0
    def y(x):
        return 5*exp(2*x)*cos(sqrt(5)*x) - 3*exp(2*x)*sin(sqrt(5)*x)
    dataSetDict = {}
    x = startX
    while(x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict

def secondOrderLinearEquationTwo(startX, endX, stepX):
    def y(x):
        return 7*exp(2*x) - 4*exp(-2*x)
    dataSetDict = {}
    x = startX
    while(x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict


#a(y')^2 + by + c = 0

#5(y')^2 + 6y - 2 = 0, let C = 10
def firstOrderNonLinearEquationOne(startX, endX, stepX):
    def y(x):
        return (1/60) * (-18*10*math.sqrt(10)*x - 45*(math.pow(10, 2)) - 18*math.pow(x, 2) + 20)

    dataSetDict = {}
    x = startX
    while (x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict

#-7(y')2 - 4y + 3 = 0
def firstOrderNonLinearEquationTwo(startX, endX, stepX):
    def y(x):
        return (1/28) * (-8*10*math.sqrt(7)*x - 28*(math.pow(10, 2)) - 4*math.pow(x, 2) + 21)

    dataSetDict = {}
    x = startX
    while (x <= endX):
        dataSetDict[x] = y(x)
        x += stepX
    return dataSetDict

#startX, endX, Step, dataSet, [layer sizes], epochs, batchSize
"""
Dataset Variables
1. First Order Linear Eq 1
2. First Order Linear Eq 2
3. Second Order Linear Eq 1
4. Second Order Linear Eq 2
5. First Order Non-Linear Eq 1
6. First Order Non-Linear Eq 2
"""
def TestSetup(startX, endX, step, dataset, testSize, layers, epochs, batchSize):

    datasetNumber = dataset

    if(dataset == 1):
        WriteDatasetToCSV("TestFile", firstOrderLinearEquationOne(startX, endX, step))
    elif(dataset == 2):
        WriteDatasetToCSV("TestFile", firstOrderLinearEquationTwo(startX, endX, step))
    elif (dataset == 3):
        WriteDatasetToCSV("TestFile", secondOrderLinearEquationOne(startX, endX, step))
    elif (dataset == 4):
        WriteDatasetToCSV("TestFile", secondOrderLinearEquationTwo(startX, endX, step))
    elif (dataset == 5):
        WriteDatasetToCSV("TestFile", firstOrderNonLinearEquationOne(startX, endX, step))
    elif (dataset == 6):
        WriteDatasetToCSV("TestFile", firstOrderNonLinearEquationTwo(startX, endX, step))

    dataset = loadtxt('TestFile.csv', delimiter=',')

    X = dataset[:, 0]
    Y = dataset[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize)

    model = Sequential()

    model.add(Dense(layers[0], input_dim=1, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    for i in layers[1:]:
        model.add(Dense(i, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, verbose=0)

    test_mse = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.4f' % test_mse)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_test, model.predict(X_test), '.')
    plt.xlabel('x')
    plt.ylabel('prediction')
    plt.title('Model Prediction for Eq. ' + str(datasetNumber))

    plt.subplot(1, 2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X, Y, '.')
    plt.title('Original data')

    plt.show()

def main():

    TestSetup(-5, 5, 0.00001, 1, 0.2, [16,24,16], 150, 5000)
    TestSetup(-5, 5, 0.00001, 2, 0.2, [16, 24, 16], 150, 5000)
    TestSetup(-5, 5, 0.00001, 3, 0.2, [16, 24, 32, 64, 32, 24, 16], 150, 5000)
    TestSetup(-5, 5, 0.00001, 4, 0.2, [16, 24, 32, 64, 32, 24, 16], 150, 5000)
    #TestSetup(-5, 5, 0.00001, 5, 0.2, [16, 24, 32, 64, 128, 256, 128, 32, 24, 16], 10, 5000)
    #TestSetup(-5, 5, 0.00001, 6, 0.2, [16, 24, 32, 64, 128, 256, 128, 32, 24, 16], 10, 5000)

main()