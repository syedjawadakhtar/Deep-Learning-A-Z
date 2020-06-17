# Recurrent Neural Network
# To predict the Google stock prices

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2]   # dataframe with only one column = stock price

# Feature Scaling
# Standardisation and Normalization
# We will be using normalization function whenver there's a signmoid function as the actiavation function

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# 60 experimented value = 60 financial dates
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])  # 60 stock prices; a 2-D array with 1 row having 60 consecutive stock prices
    y_train.append(training_set_scaled[i, 0])   # Stock price at time t+1
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping 
# To add a new dimension in an array
# View Keras > layers > recurrent layers: (batch_size = No. of abservations 1197, timesteps = 60 columns, input_dim = indicators/predictors = 1)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))   



# Part 2 - Building the Stacked RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initalizing  the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))     # 50 neurons will capture the up/down trends; return_sequences as we are using stacked network
regressor.add(Dropout(0.2))

# 2nd LSTM layer with Dropout regularizatoin
regressor.add(LSTM(units = 50, return_sequences = True))    # no need to mention the shape
regressor.add(Dropout(0.2))

# 3rd LSTM layer with Dropout regularizatoin
regressor.add(LSTM(units = 50, return_sequences = True))    
regressor.add(Dropout(0.2))

# 4th LSTM layer with Dropout regularizatoin
regressor.add(LSTM(units = 50))    
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))  # One output value of stock price

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # Could use RMSprop but in this case we are Adam optimizer as it performs well on this data

# Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)




# Part 3 - Making the predictoins and visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# We need 60 produced stock prices before the date to be predicted
# Scaling the inputs but not changing the input values as we trained on a scaled version
# Concatenating
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  # vertical concatenation = 0; horizontal = 1; This will concatenation will contain a dataframe contaitnng Open google stock price of taining set and the test set of january 2017
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values     # Stock prices of January 
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80): # 20 values of January 2017
    x_test.append(inputs[i-60:i, 0])  
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))   # Reshaping
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Reversing the from scaling values

# Visualing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Improving RNN
# More data
# increase the previous step from 60 to 120
# Adding more indicators
# Adding moer LSTM layers
# More neurons to LSTM layers



























