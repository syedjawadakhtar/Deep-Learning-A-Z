# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:44:48 2020

@author: bon2die
"""

# Explaination more in Restricted Boltzmann Machine 
# Stacked Auto Encoder

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel #parallel computation
import torch.optim as optim #optimizer
import torch.utils.data #utilities 
from torch.autograd import Variable #stochastic gradient descent

#importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') # age, code and zip code
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #same user, movies ids, rating, timestamps

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') #80k values 80-20 split
training_set = np.array(training_set, dtype = 'int') #dataframe to array
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t') #20k values 80-20 split
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
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
training_set = torch.FloatTensor(training_set) 
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__() #All the inherited classess and methods of the parent class in a module
        # First full connection; 20 nodes in first layer; 20 & 10 experimental values
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)    # 20; layers b/w the first and second layer acts as dummy
        self.fc3 = nn.Linear(10, 20)    # Decoding; 
        self.fc4 = nn.Linear(20, nb_movies) #deconstructed to nb_movies
        self.activation = nn.Sigmoid()
    # Return the predicted ratings; imposng actication function in the layers
    def forward(self, x):
        x = self.activation(self.fc1(x))  #x is input vector
        x = self.activation(self.fc2(x))    # decoding to full connection
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss ()  # Object of the calss to measure the MSE
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # all the parameters of the class' 0.01 experimental value; 
                                                                    #decay is used to reduce the lr after few epochs i.e to regulate convergence
# Training the SAE                            
nb_epoch = 200
for epoch in range(1, nb_epoch + 1)                                                                    :
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):                             #0 - 942
        input = Variable(training_set[id_user]).unsqueeze(0)   # torch cant accept a single vector
        target = input.clone()                                 # modifiying input so cloning it
        if torch.sum(target.data > 0) > 0:                     #Observation contains at least one non - zero value
            output = sae(input)
            target.require_grad = False                        # Code Optimzer
            output[target == 0] = 0                            # These values wont count on computing error & have no impact wieght updation
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)    # 1-10 so that the denominator doesn't become zero; mean_collector average of the error for the movies that have got non zero rating
            loss.backward()                                   #  Backward method for the loss; in which dir should we move the wts: inr / derc
            train_loss += np.sqrt(loss.item() * mean_corrector)  # Changes according to the comment section
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
    
# Result : epoch: 200 loss: 0.9142395646464005
    
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):                             #0 - 942
    input = Variable(training_set[id_user]).unsqueeze(0)   # Part of the test set bcoz; we're predicting the movies that haven't watched by user that is in training set; SO we can compare the real rating to predicted rating
    target = Variable(test_set[id_user])                   # real rating of test_set
    if torch.sum(target.data > 0) > 0:                     # Observation contains at least one non - zero value
        output = sae(input)                                # Predicted rating that the user hasn't watched yet
        target.require_grad = False                        # Code Optimzer
        output[(target == 0).unsqueeze(0)] = 0                            # These values wont count on computing error & have no impact wieght updation
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)    # 1-10 so that the denominator doesn't become zero; mean_collector average of the error for the movies that have got non zero rating
        test_loss += np.sqrt(loss.item() * mean_corrector)  # Changes according to the comment section
        s += 1.
print('test_loss: ' + str(test_loss/s))

# Result: test_loss: 0.9543852216315367; this is less than 1 - pretty good
# Which means that after a watching a movie you give it a 4 star; this recommender system 
# will predict that the rating will be between 3 & 5; That it predicts how good you liked it