# Numeric Python Library.
import numpy
import pandas
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('spending.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

def create_dataset(dataset):
	dataX, dataY = [], []
	for i in range(0,len(dataset)):
		dataX.append(i % 7)
		dataY.append(dataset[i])
	return numpy.array(dataX), numpy.array(dataY)

datasetX, datasetY = create_dataset(dataset)

datasetX = datasetX.reshape(-1,1)
datasetY = datasetY.reshape(-1,1)
# normalize the dataset
Xscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler = MinMaxScaler(feature_range=(0, 1))
datasetX = Xscaler.fit_transform(datasetX)
datasetY = Yscaler.fit_transform(datasetY)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
trainX, testX = datasetX[0:train_size], datasetX[train_size:len(dataset)]
trainY, testY = datasetY[0:train_size], datasetY[train_size:len(dataset)]

model = Sequential()

model.add(Dense(64,input_dim=1, activation='relu'))
model.add(Dropout(.2))
model.add(Activation("linear"))
model.add(Dense(32,activation='relu'))
model.add(Activation("linear"))
model.add(Dense(32,activation='relu'))
model.add(Activation("linear"))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
model.fit(trainX, trainY, nb_epoch=256, batch_size=2, verbose=2)
# make predictions
trainPredict = model.predict(datasetX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = Yscaler.inverse_transform(trainPredict)
trainY = Yscaler.inverse_transform(trainY)
testPredict = Yscaler.inverse_transform(testPredict)
testY = Yscaler.inverse_transform(testY)

plt.plot(testY)
plt.plot(testPredict)
plt.show()





