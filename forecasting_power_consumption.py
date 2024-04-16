# Multivariate time series with LSTM

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('Tetuan_City_power_consumption.csv', index_col=0, infer_datetime_format=True)
df.head()

# Check for null values and data type
df.info()

# Plot last month power consumption in Zone 1
df['Zone 1 Power Consumption'].plot(figsize=(12,8))
plt.show()

# Plot last month power consumption in Zone 2
df['Zone 2  Power Consumption'].plot(figsize=(12,8))
plt.show()

# Plot last month power consumption in Zone 3
df['Zone 3  Power Consumption'].plot(figsize=(12,8))
plt.show()

# Train test split
print(len(df))
print(df.head(3))
print(df.tail(5))

# 72 hours in the future to be predicted, considering last month data
print(df.loc['12/1/2017 0:00':])
df = df.loc['12/1/2017 0:00':]

# Round off to two decimal
df = df.round(2)
print(df)
print(len(df))

# Number of rows per day
print(24*60/10)
test_days = 3
test_ind = test_days*144
print(test_ind)
train = df.iloc[:-test_ind]
test = df.iloc[-test_ind:]
print(train)
print(test)

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Time series generator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Generator defining
length = 144
batch_size = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
print(len(scaled_train))
print(len(generator))

# First batch prediction
X,y = generator[0]
print(f'Input given: \n{X.flatten()}')
print(f'Predicted y: \n {y}')

# Define and compile model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
print(scaled_train.shape)
model = Sequential()
model.add(LSTM(100, input_shape=(length,scaled_train.shape[1])))
model.add(Dense(scaled_train.shape[1]))
#model.compile(optimizer='adam', loss='mse')
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.summary()

# EarlyStopping is used to stop training when there is no improvement
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=1)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=batch_size)
model.fit_generator(generator, epochs=5, validation_data=validation_generator, callbacks=[early_stop])
model.history.history.keys()
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# Evaluate on test data
first_eval_batch = scaled_train[-length:]
print(first_eval_batch)
first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))
model.predict(first_eval_batch)
print(scaled_test[0])

# Predicting test set for 864 rows so that last 432 rows (144 rows per day. Since we need to predict for 3 days the length is given as 864) cover the future 3 days values
n_features = scaled_train.shape[1]
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

# print(len(test))

for i in range(864):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

print(test_predictions)
print(scaled_test)

# Inverse transformations and compare
true_predictions = scaler.inverse_transform(test_predictions)
print(true_predictions)
print(test)
true_predictions = pd.DataFrame(data=true_predictions, columns=test.columns)

# Forecasted values for future 3 days
print(true_predictions.tail(432))