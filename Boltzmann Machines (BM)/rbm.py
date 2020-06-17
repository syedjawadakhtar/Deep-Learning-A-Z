# -*- coding: utf-8 -*-
"""
Spyder Editor

Boltzmann Machine Recmmender system
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel #parallel computation
import torch.optim as optim #optimizer
import torch.utils.data #utilities 
from torch.autograd import variable #stochastic gradient descent

#
# Preprocessing 
#

#importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') # age, code and zip code
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #same user, movies ids, rating, timestamps

# Preparing the training set and the test set
# .base and .test 1 pair of test and train. 5 test-train splits = 5 cross-validation
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') #80k values 80-20 split
training_set = np.array(training_set, dtype = 'int') #dataframe to array
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t') #20k values 80-20 split
test_set = np.array(test_set, dtype = 'int') #dataframe to array

# Getting the number of users and movies
# the maximum number can be in training set or test set, so take the maximum
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array for each users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]    #copying movies for each user
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)   #Putting zeroes for the ratings users have not given
        ratings[id_movies - 1] = id_ratings # -1 for starting from 0 index
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)    

# Converting the data into Torch tensors
# numpy arrays is can be used but we need multi dimensional arrays called tensors so torch tensors

training_set = torch.FloatTensor(training_set) #single type will be float
test_set = torch.FloatTensor(test_set)

#
# Boltzmann machines
#

#for the INPUT of RBM
# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked) 
training_set[training_set == 0] = -1 # erplacing all the 0 ratings to -1
training_set[training_set == 1] = 0 # not liked for rating 1
training_set[training_set == 2] = 0 # not liked for rating 2
training_set[training_set >= 3] = 1    # liked for rating 3 or above
test_set[test_set == 0] = -1 # erplacing all the 0 ratings to -1
test_set[test_set == 1] = 0 # not liked for rating 1
test_set[test_set == 2] = 0 # not liked for rating 2
test_set[test_set >= 3] = 1    # liked for rating 3 or above

# Creating the architecture of the neural network
class RBM():
    def __init__(self, nv, nh): # nv = no of visible node; nh = no. of hidden nodes
                                        # W: wt matrix for visible node given hidden nodes
        self.W = torch.randn(nh, nv) # Initialization of a tensor of size nh, nv with a normal distribution[randn](mean = 0; variance = 1)
                                    # setting bias for hidden nodes
        self.a = torch.randn(1, nh)  # In 2-D tensor: first dimension = batch and 2nd dimension = bias
                                        # setting bias for visible nodes
        self.b = torch.randn(1, nv)
        
    # Samples hidden nodes given visible nodes based on probability; for the movies the user hasn't rated
    def sample_h(self, x):
                                    # prob of given nodes 
        wx = torch.mm(x, self.W.t()) # mm: Multiplication of two tensors; with the transpose of weight matrix bcoz see line 76
                                    # act func for the hidden nodes according to the values of the visible nodes
        activation = wx + self.a.expand_as(wx) # bias is expanded as wx
        p_h_given_v = torch.sigmoid(activation) # Probability for hidden nodes given visible nodes
        return p_h_given_v, torch.bernoulli(p_h_given_v)     # The probability that the yes/no is present is taken using bernoulli sampling
    
    # return probability of visible nodes given the values of hidden nodes
    def sample_v(self, y):
        wy = torch.mm(y, self.W)    # no transpose
        activation = wy + self.b.expand_as(wy) 
        p_v_given_h = torch.sigmoid(activation) 
        return p_v_given_h, torch.bernoulli(p_v_given_h)     
    
    # Contrastive Divergence: Maximize the log-likelihood from the paper given in reference
    def train(self, v0, vk, ph0, phk): #v0, vk: visible nodes 0 to k | ph0, phk: probab of hidden nodes after 0 - k sampling gieven the visible nodes 0-k
        self.W += torch.mm(ph0, v0) - torch.mm(phk, vk)
        self.b += torch.sum((v0 - vk), 0) # 0 for only keeping the format of 2-D
        self.a +=  torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0]) # No of features in the training set is no. of visible nodes 
nh = 100    # Completely tuneable no of hidden nodes
batch_size = 100
rbm = RBM(nv, nh)   # Creating object

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size): # update wts after batch of users
        vk = training_set[id_user:id_user+batch_size] # vk: Input batch vk; All users from id user to id_user + 100 batch_size
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0) # ph0, _ : means recieving only the first return value from the sample_h function
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0]  = v0[v0<0]    # freeze ratings with -1
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1. # Counter
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))       #printng normalized values of loss

#Result 0.2583 means that 75% of successful prediction
  
# Testing the RNN
test_loss = 0
s = 0.
for id_user in range(nb_users): # all the users are our test set
    v = training_set[id_user:id_user+1] # i/p on which we will be making prediction; will activate the beurons of hidden to get the predicted rating of the test set 
    vt = test_set[id_user:id_user+1]   # target value
    # eg. of blind walk; making a single step and staying on a staright line; so our prediction will be single step of Gibbs algo
    if len(vt[vt>=0]) > 0:  #for valid 
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        vk[v0<0]  = v0[v0<0]    # freeze ratings with -1
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
    s += 1. # Counter
print('Test loss: ' + str(test_loss/s))       #printng normalized values of loss

#result 0.166 means that the 84% successfull prediction
  