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
metadata = pd.read_pickle("./Metadata.pkl")
RatingsDataFrame =  pd.read_pickle("./RatingsDataFrame.pkl")
PCA_df = pd.read_pickle("./PCA_df.pkl")
pca_features = np.load('pca_features.npy')
genres = np.load('genressave.npy')
n_movies = metadata['id'].shape[0]
n_users = RatingsDataFrame['userId'].nunique()


def Nmaxelements(list1, N): 
    """Returns the N largest elements of a list."""
    final_list = []  
    for i in range(0, N):  
        max1 = 0          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j];                  
        list1.remove(max1); 
        final_list.append(max1)          
    return final_list 


def word_extractor(row_of_words):
    """Extracts words out of dictionaries and formats them."""
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


class Recommender:
    """Represents a recommendation system for a user."""

    RatingsDF = RatingsDataFrame
    PCAdf = PCA_df
    PCAarray = pca_features
    Metadata = metadata
    Genres = genres
    svr = SVR(gamma='auto', epsilon=0.2, C=0.1)
    GBR = ensemble.GradientBoostingRegressor() 

    def __init__(self, user):
        """Initializes the active user data."""
        self.user = user        
        self.UserDF = Recommender.RatingsDF.groupby('userId').get_group(self.user)
        self.UserDF = Recommender.PCAdf.merge(self.UserDF,left_on=0,right_on='movieId')

    def test_split(self):
        """Splits the active user dataframe into training and test data."""
        self.__X = self.UserDF.iloc[:,1:3]
        self.__y = self.UserDF['rating']
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.__X,self.__y,test_size=0.15)

    def SVRfit(self):
        self.test_split()
        svr_parameters = {'C':[0.1, 1, 10],'epsilon':[0.1,0.2,0.5],'gamma':['auto','scale']}
        self.svr = GridSearchCV(Recommender.svr, svr_parameters)
        self.svr = self.svr.fit(self.x_train, self.y_train)
        self.validation = self.svr.predict(self.x_test)
        self.svr_rmse = np.sqrt(mean_squared_error(self.y_test,self.validation))

    def GBRfit(self):
        self.test_split()
        gbr_parameters = {'n_estimators':[100,300,500],'learning_rate':[0.001,0.01,0.1,1],\
          'loss':['ls','lad','huber','quantile']}
        self.gbr = GridSearchCV(Recommender.GBR, gbr_parameters)
        self.gbr = self.gbr.fit(self.x_train, self.y_train)
        self.validation = self.model.predict(self.x_test)
        self.gbr_rmse = np.sqrt(mean_squared_error(self.y_test,self.validation))

    def recommend(self,Nrecom):
        """Creates an SVR model for an active user."""
        self.SVRfit()
        self.SVRgbr()

        if (svr_rmse <= gbr_rmse): 
            self.model = self.svr
        else:
            self.model = self.gbr

        self.recom = self.model.predict(Recommender.PCAarray)
        self.__largest_elements = Nmaxelements(self.recom.tolist(),Nrecom)
        self.suggestions = []
        self.genres_array = np.zeros(20)
        for i in range(0,len(self.recom)): 
           if self.recom[i] in self.__largest_elements:
             self.suggestions.append(Recommender.Metadata['original_title'][i])
             self.suggestions.append(word_extractor(Recommender.Metadata['genres'][i]))
             self.genres_array = self.genres_array + Recommender.Genres[i]
        return self.genres_array, self.suggestions

    def diversity(self,Nrecom):
        """Computes a diversity index for an active user."""
        self.diversity = 0
        Ngenres = 20
        self.__Nmovies_rated = self.UserDF.shape[0]
        if (self.__Nmovies_rated > 1):
          self.recommend(Nrecom)
          for i in range(0,self.genres_array.size):
            if (self.genres_array[i] > 0):
              self.diversity = self.diversity + 1
        self.diversity = self.diversity/Ngenres
        return self.diversity


def accuracy_avg(N):
  """Computes a (N movies watched)-weighted accuracy avg for N users."""
  counter = 0
  R2 = 0
  error_rsme = 0
  for i in range(1,N):
    active_user = random.randint(1,n_users)
    Recom = Recommender(active_user)
    Nmovies_rated = Recom.UserDF.shape[0]
    if (Nmovies_rated > 10):
      Recom.fit()
      test_score = Recom.svr.score(Recom.x_test, Recom.y_test)
      predicted = Recom.svr.predict(Recom.x_test)
      rmse = np.sqrt(mean_squared_error(Recom.y_test,predicted))
      error_rsme = error_rsme + Nmovies_rated*rmse
      R2 = R2 + Nmovies_rated*test_score
      counter = counter + Nmovies_rated
  error_rsme = error_rsme/counter
  R2 = R2/counter
  return error_rsme, R2


def diversity_avg(N,n_recom):
  """Computes a diversity avg index for N users."""
  diversity_avg = 0
  for i in range(1,N):
    active_user = random.randint(1,n_users)
    Recom = Recommender(active_user)
    diversity_avg = diversity_avg + Recom.diversity(n_recom)
  diversity_avg = diversity_avg/N
  return diversity_avg


def model_performance(N):
  """Measures the algorithmic runtime and accuracy for N user models."""
  t1 = time.clock()
  error_rsme, R2 = accuracy(N)
  t2 = time.clock()
  t  = t2-t1
  return error_rsme, R2, t
