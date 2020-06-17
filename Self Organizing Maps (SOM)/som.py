# Self Organizing Maps

# Finidng fradualant customers in credit card applications

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values  # Customer ID and all the column except the last one
y = dataset.iloc[:, -1].values   # Class coulumn showing whether the application was approved or not

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x) 

# Training the SOM
from minisom import MiniSom     # cloned from https://github.com/JustGlowing/minisom/
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5 )# Sigma= radius of the neighbourhood
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

# Visualizing the results
# MID - Mean Interneuron Distance
# We have to find the winning node with the highest MID
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']    # Two markers red circle = didn't get approval and green square = got approval
colors = ['r', 'g']
for i, X in enumerate(x): # i = indexes of customers and X = vectors of customers
    w = som.winner(X)
    plot(w[0] + 0.5,     # x coordinate  of the winning node 0.5 at the centre of the square
         w[1] + 0.5,     # y coordinate
         markers[y[i]],  # y[i] = value of the dependent variable of that customer; red or green if application is accepted
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show() 
# The outliers are the white color boxes. And of circle and box means that some got away with the fraud and some got rejected

# Finding the frauds
mappings = som.win_map(x)   # Dictionary gives winning nodes are no. of associated customers to it
frauds = np.concatenate((mappings[(9,2)], mappings[(2,2)]), axis = 0)    # View the coordinates of the outliers in the SOM and input it here
frauds = sc.inverse_transform(frauds)









