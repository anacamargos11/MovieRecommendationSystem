# ECE9063: Project
# Content Based Movie Recommendation Engine
# Part 2: Modeling
#
# Ana Carolina Camargos Couto
# Dept. of Applied Math. Western University

import numpy as np
import pandas as pd
import pickle
import ast
from scipy import linalg
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Loading the data files
Metadata = pd.read_pickle("./Metadata.pkl")
RatingsDataFrame =  pd.read_pickle("./RatingsDataFrame.pkl")
PCA_df = pd.read_pickle("./PCA_df.pkl")
pca_features = np.load('pca_features.npy')
genres = np.load('genressave.npy')
Nmovies = Metadata['id'].shape[0]
Nusers = RatingsDataFrame['userId'].nunique()

# this function extracts words out of dictionaries and formats them
def word_extractor(row_of_words):
    words_joined = []
    if (type(row_of_words)!=str or type(ast.literal_eval(row_of_words))!=list):
        words_joined = ['']
    else:
    # extract words from the dictionaries
        word_list = ast.literal_eval(row_of_words)
        for w in range(0,len(word_list)):
            word_list[w] = word_list[w]['name']
            word_list[w] = word_list[w].replace(" ","")
        words_joined.append(' '.join(word_list))    
    return words_joined

# this function creates a dataframe with ratings and movie features for an active user
def user_dataframe(active_user):
  user_df = RatingsDataFrame.groupby('userId').get_group(active_user)
  user_df = PCA_df.merge(user_df,left_on=0,right_on='movieId')
  return user_df

# this function splits a user dataframe into training and test data
def test_split(active_user):
  percentage = 0.85
  user_df = user_dataframe(active_user)
  X = user_df.iloc[:,1:3]
  y = user_df.iloc[:,5]
  x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15)
  return x_train, x_test, y_train, y_test                                                         
# this function returns the N largest elements of a list
def Nmaxelements(list1, N): 
    final_list = []  
    for i in range(0, N):  
        max1 = 0          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j];                  
        list1.remove(max1); 
        final_list.append(max1)          
    return final_list 

# this function finds the best hyperparameters for an svr user model
def svr_tuning(active_user):
  parameters = {'C':[0.1, 1, 10],'epsilon':[0.1,0.2,0.5],'gamma':['auto','scale']}
  x_train, x_test, y_train, y_test = test_split(active_user)
  svr = SVR(gamma='scale')
  svr = GridSearchCV(svr, parameters, random_state=0)
  search = svr.fit(x_train, y_train)
  return search.best_params_

def GBR_tuning(active_user):
  parameters = {'n_estimators':[100,300,500],'learning_rate':[0.001,0.01,0.1,1],\
    'loss':['ls','lad','huber','quantile']}
  x_train, x_test, y_train, y_test = test_split(active_user)
  GBR = ensemble.GradientBoostingRegressor()
  clf = GridSearchCV(GBR, parameters)
  search = clf.fit(x_train, y_train)
  return search.best_params_


# this function creates an svr model for an active user
def training(active_user):
  x_train, x_test, y_train, y_test = test_split(active_user)
  svr = SVR(gamma='auto', epsilon=0.2, C=0.1)
  LR = LinearRegression()
  GBR = ensemble.GradientBoostingRegressor(learning_rate=0.001,loss='ls',\
    n_estimators=100)
  svr.fit(x_train, y_train)
  train_score = svr.score(x_train, y_train)
  test_score = svr.score(x_test, y_test)
  predicted = svr.predict(x_test)
  rmse = np.sqrt(mean_squared_error(y_test,predicted))
  return svr, x_test, y_test


# this function creates an svr model for an active user
def recommendations(active_user,n_recom):
  svr, x_test, y_test = training(active_user)  
  recommend = svr.predict(pca_features)
  recommend_max = Nmaxelements(recommend.tolist(),n_recom)
  suggestions = []
  genres_array = np.zeros(20)
  for i in range(0,len(recommend)): 
     if recommend[i] in recommend_max:
       suggestions.append(Metadata['original_title'][i])
       suggestions.append(word_extractor(Metadata['genres'][i]))
       genres_array = genres_array + genres[i]
  return genres_array, suggestions

# this function computes a (N movies watched)-weighted accuracy avg for N users
def accuracy(N):
  counter = 0
  R2 = 0
  error_rsme = 0
  for i in range(1,N):
    active_user = random.randint(1,Nusers)
    user_df = user_dataframe(active_user)
    Nmovies_rated = user_df.shape[0]
    if (Nmovies_rated > 10):
      svr, x_test, y_test = training(active_user)
      test_score = svr.score(x_test, y_test)
      predicted = svr.predict(x_test)
      rmse = np.sqrt(mean_squared_error(y_test,predicted))
      error_rsme = error_rsme + Nmovies_rated*rmse
      R2 = R2 + Nmovies_rated*test_score
      counter = counter + Nmovies_rated
  error_rsme = error_rsme/counter
  R2 = R2/counter
  return error_rsme, R2

# this function computes a diversity index for an active user
def diversity(active_user,n_recom):
  genres_counter = 0
  Ngenres = 20
  user_df = user_dataframe(active_user)
  Nmovies_rated = user_df.shape[0]
  if (Nmovies_rated > 1):
    genres_array, suggestions = recommendations(active_user,n_recom)
    for i in range(0,genres_array.size):
      if (genres_array[i] > 0):
        genres_counter = genres_counter + 1
  genres_counter = genres_counter/Ngenres
  return genres_counter

# this function computes a diversity avg index for N users
def diversity_avg(N,n_recom):
  diversity_total = 0
  for user in range(1,N):
    diversity_total = diversity_total + diversity(user,n_recom)
  diversity_total = diversity_total/N
  return diversity_total

# this function measures the algorithmic runtime and accuracy for N user models
def model_performance(N):
  t1 = time.clock()
  error_rsme, R2 = weighted_accuracy(N)
  t2 = time.clock()
  t  = t2-t1
  return error_rsme, R2, t


