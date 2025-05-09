#import
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime
#sometime tensorflow gives error warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#LOAD data
data = pd.read_csv("MicrosoftStock.csv")
print(data.head())
print(data.info())
print(data.describe())
#intaial data visualisation
#plot 1-open and close prices of time
plt.figure(figsize=(12,6))
plt.plot(data['date'],data['open'] ,label="open",color="blue")
plt.plot(data['date'],data['close'],label="close" ,color="red")
plt.title("open-close price over time")
plt.legend()
plt.show()

#plot 2-trading volume(check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'],data['volume'],label="volume",color="orange")
plt.title("stock volume overtime")
#plt.show()

#drop non numeric columns
numeric_data = data.select_dtypes(include=["int64","float"])  # Fixed: removed extra space at beginning of line

#plot 3 - checking correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation Heatmap")
#plt.show()

#convert data into date and time then create a date filter
data['date'] = pd.to_datetime(data['date'])

prediction = data.loc[
(data['date']>datetime(2013,1,1)) &
(data['date']<datetime(2018,1,1))
]

plt.figure(figsize=(12,6))
plt.plot(data['date'] ,data['close'],color="blue")
plt.xlabel("date")
plt.ylabel("close")
plt.title("price  over time")

#prepare for the LSTM Model (sequentially)
stock_close = data.filter(["close"])
dataset = stock_close.values #convert to numpy array

training_data_len = int(np.ceil(len(dataset) * 0.95))
#preprocessing stages
scalar = StandardScaler()
scaled_data = scalar.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #95% of all oout data
x_train = []
y_train = []


#creating a sliding window for our stock(60 days)
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i,0])
    y_train.append(training_data[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

#??
x_train =np.reshape(x_train ,(x_train.shape[0], x_train.shape[1],1))

#Build the model
model = keras.models.Sequential()

#first layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#second layer
model.add(keras.layers.LSTM(64,return_sequences=False))
#third layer(dense)
model.add(keras.layers.Dense(128,activation="relu"))
#fourth layer (dropout)
model.add(keras.layers.Dropout(0.5))
#final output layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam" ,
             loss= "mae",
             metrics=[keras.metrics.RootMeanSquaredError()])

#train the mpdel
training = model.fit(x_train , y_train , epochs=20 , batch_size= 32)

#prep the test data
test_data = scaled_data[training_data_len -60:]
x_test,y_test = [] , dataset[training_data_len:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#convert into numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test ,(x_test.shape[0],x_test.shape[1],1))

#make a prediction
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['predictions'] = predictions


plt.figure(figsize=(12,8))
plt.plot(train['date'],train['close'],label="train (Actual)",color='blue')
plt.plot(test['date'],test['close'],label="train (Actual)",color='orange')
plt.plot(test['date'],test['predictions'],label="predictions",color='red')
plt.title("our stock prediction")
plt.xlabel("Date")
plt.ylabel("close price")
plt.legend()
plt.show()
