# import libraries
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# get covid data
data = pd.read_csv('archive/CONVENIENT_global_confirmed_cases.csv')
data = data.iloc[1:]
data['Total'] = data.sum(numeric_only=True, axis=1)
data = data[['Country/Region', 'Total']]
data.columns = ['date', 'total_cases']
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data_plot = data

# scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# format data
X = []
Y = []

for i in range(len(data)):
    if i+7 == len(data):
         break
    else:
        x = [data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6]]
        x = np.array(x)
        y = [data[i+7]]
        y = np.array(y)
    X.append(x)
    Y.append(y)

# set train test and validation data
X_train = X[:-30]
y_train = Y[:-30]

X_test = X[-30:]
y_test = Y[-30:]

# reshape data
X_train = np.array(X_train).reshape(-1,7,1)
y_train = np.array(y_train)

X_test = np.array(X_test).reshape(-1,7,1)

# create model
model = Sequential()

# input layer
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(7,1)))
model.add(Dropout(0.05))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(128, activation='relu'))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit model
model.fit(X_train, y_train, epochs=5, verbose=2)

# predict
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# revert data
predictions = []
train_prediction = scaler.inverse_transform(train_prediction)
test_prediction = scaler.inverse_transform(test_prediction)

# append train and test predictions
train_prediction = train_prediction.tolist()
train_prediction = [item for sublist in train_prediction for item in sublist]
train_prediction.extend([float('nan')]*(len(data_plot)-len(train_prediction)))
data_plot['predicted_cases'] = train_prediction

test_prediction = test_prediction.tolist()
test_prediction_data = [item for sublist in test_prediction for item in sublist]
test_prediction = [float('nan')] * (len(data_plot)-len(test_prediction_data))
test_prediction.extend(test_prediction_data)
data_plot['predicted_case'] = test_prediction

# predict future days
future_days = 30

for i in range(future_days):
    test = [data[-7], data[-6], data[-5], data[-4], data[-3], data[-2], data[-1]]
    test = np.array(test).reshape(-1,7,1)
    prediction = model.predict(test)
    data = np.concatenate((data, prediction), axis=0)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction.reshape(-1,1,1)
    prediction = prediction.tolist()
    prediction = [item for sublist in prediction for item in sublist]
    prediction = [item for sublist in prediction for item in sublist]
    for i in prediction:
        prediction = i
    data_plot = data_plot.append([{'total_cases':prediction}], ignore_index=True)

# plot
plt.plot(data_plot['total_cases'][:-30], color='lightpink', label='Actual Cases')
plt.plot(data_plot['predicted_cases'], color='m', label='Train Predictions')
plt.plot(data_plot['predicted_case'], color='c', label='Test Predictions')
plt.plot(data_plot['total_cases'][-30:], color='b', label='Future Predictions')
plt.title('Prediction of COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()
