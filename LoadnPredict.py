import numpy
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import datetime
import seaborn as sns
import pandas
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

lstmbr = "LSTM_breakfast.h5"
lstmlu = "LSTM_lunch.h5"
nlbr = 'NonLinear_breakfast.h5'
nllu = "NonLinear_lunch.h5"

csvbr = "spending_breakfast.csv"
csvlu = "spending_lunch.csv"

def loadModel(modelname):
    model = load_model(modelname)
    if (modelname[:4] == "LSTM"):
        return model, True
    else:
        return model, False

model, isLSTM = loadModel(nllu)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv(csvlu, index_col=0, engine='python',parse_dates=True)
dataset = dataframe.values
dataset = dataset.astype('float32')

if (isLSTM):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # reshape into X=t and Y=t+1
    look_back = 6
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
else:
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

if isLSTM:

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions

    ogdf = dataframe
    #convert from 2d to 1d nparray
    trainPredictCost = trainPredictPlot.flatten()
    traindf = pandas.DataFrame({'cost':trainPredictCost})
    traindf.index = ogdf.index

    testPredictCost = testPredictPlot.flatten()
    testdf = pandas.DataFrame({'cost':testPredictCost})
    testdf.index = ogdf.index

    print(testPredictPlot)
    ogdf['cost'].plot(markersize=6, marker='o', color='dimgray', linestyle=':',label='Original Spending ($)')
    traindf['cost'].plot(markersize=6, marker='o', color='darkorange',linestyle='-',label='Training Set Predicted Spending ($)')
    testdf['cost'].plot(markersize=6, marker='o', color='mediumseagreen',linestyle='-.',label="Testing Set Predicted Spending ($)")
    ax = plt.gcf().axes[0]
    fig = plt.gcf()
    fig.set_size_inches(16,7, forward=True)
    #axis tick formatting
    min_loc = dates.WeekdayLocator(byweekday=SU)
    ax.xaxis.set_minor_locator(min_loc)
    min_fmt = dates.DateFormatter('%d\n%a')
    ax.xaxis.set_minor_formatter(min_fmt)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b'))

    #grid formatting
    #sns.set_style("darkgrid")
    plt.style.use(u'seaborn-talk')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid()
    plt.xlabel('Date')
    plt.ylabel('Spending ($)')
    plt.title('Spending Prediction - LSTM')
    plt.legend()
    #https://stackoverflow.com/questions/23644020/matplotlib-string-to-dates
    plt.show()


else:
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = Yscaler.inverse_transform(trainPredict)
    trainY = Yscaler.inverse_transform(trainY)
    testPredict = Yscaler.inverse_transform(testPredict)
    testY = Yscaler.inverse_transform(testY)

    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[0:len(trainPredict), :] = trainPredict

    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict):len(dataset), :] = testPredict

    ogdf = dataframe
    #convert from 2d to 1d nparray
    trainPredictCost = trainPredictPlot.flatten()
    traindf = pandas.DataFrame({'cost':trainPredictCost})
    traindf.index = ogdf.index

    print(traindf)
    testPredictCost = testPredictPlot.flatten()
    testdf = pandas.DataFrame({'cost':testPredictCost})
    testdf.index = ogdf.index
    #print(testPredict)
    ogdf['cost'].plot(markersize=6, marker='o', color='dimgray', linestyle=':',label='Original Spending ($)')
    traindf['cost'].plot(markersize=6, marker='o', color='darkorange',linestyle='-',label='Training Set Predicted Spending ($)')
    testdf['cost'].plot(markersize=6, marker='o', color='mediumseagreen',linestyle='-.',label="Testing Set Predicted Spending ($)")
    ax = plt.gcf().axes[0]
    fig = plt.gcf()
    fig.set_size_inches(16,7, forward=True)
    #axis tick formatting
    min_loc = dates.WeekdayLocator(byweekday=SU)
    ax.xaxis.set_minor_locator(min_loc)
    min_fmt = dates.DateFormatter('%d\n%a')
    ax.xaxis.set_minor_formatter(min_fmt)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b'))

    #grid formatting
    #sns.set_style("darkgrid")
    plt.style.use(u'seaborn-talk')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid()
    plt.xlabel('Date')
    plt.ylabel('Spending ($)')
    plt.title('Spending Prediction - Non Linear Regression')
    plt.legend()
    #https://stackoverflow.com/questions/23644020/matplotlib-string-to-dates
    plt.show()