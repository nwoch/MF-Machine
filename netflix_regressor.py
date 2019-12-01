# -*- coding: utf-8 -*-
"""
Regressor using a matrix factorization machine for recommendation systems/collaborative filtering 
on the original Netflix Prize data set

Created on Sat Nov 10 18:52:01 2018

@author: Nicole
"""
from matrix_factorization import MatrixFactorization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import copy
import os

class MatrixFactorizationRegressor:
    
    def __init__(self):
        self.movie_ratings = {}
        self.user_ratings = {}
        self.predictions = None
        self.n_ratings = 0 
        
    def parse_data(self):
        """Prepares Netflix data to be used by the learning machine, parsing it and storing it
        in the appropriate format in two dictionaries
        """
        directory = os.fsencode("training_set")
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if file_name.endswith(".txt"): 
                movie = file_name[3:-4] 
                path = "training_set/" + file_name
                df = pd.read_csv(path, header=0, names=['userID', 'rating', 'date'])
                ratings = dict([(user, rating) for user, rating in zip(df.userID, df.rating)])
                self.movie_ratings[movie] = ratings
                if len(self.user_ratings) <= 10000:
                    for user, rating in ratings.items():
                        self.n_ratings += 1
                        if user not in self.user_ratings:
                            self.user_ratings[user] = dict()
                        self.user_ratings[user][movie] = rating
                        
        # Updates movie ratings dictionary to only include 10000 users
        new_ratings = copy.deepcopy(self.movie_ratings)
        for movie in self.movie_ratings:
            for user, rating in self.movie_ratings[movie].items():
                if user not in self.user_ratings:
                    del new_ratings[movie][user]
        self.movie_ratings = new_ratings
    
    def store_machine(self, mf):
        with open("machine.pckl", "wb") as f:
            pickle.dump(mf, f)
    
    def retrieve_machine(self):
        with open("machine.pckl", "rb") as f:
            while True:
                try:
                    mf = pickle.load(f)
                except EOFError:
                    break
        return mf
    
    def plot_errors(self, mf):
        """Plots total error for each epoch after fitting"""
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        
        ax[0].plot(range(1, len(mf.errors) + 1), np.log10(mf.errors), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Mean-squared-error)')
        ax[0].set_title('Matrix Factorization - Learning rate 0.000001')
        ax[1].plot(range(1, len(mf.errors) + 1), mf.errors, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Mean-squared-error')
        ax[1].set_title('Matrix Factorization - Learning rate 0.000001')
        
    def train_and_predict(self):
        """Fits learning machine using training data and predicts regression values for test data
        Calculates final error rate for predictions
        """
        mf = MatrixFactorization()
        mf.fit(self.movie_ratings, self.user_ratings)
        self.store_machine(mf)
        self.plot_errors(mf)
        self.predictions = mf.predict(self.movie_ratings, self.user_ratings)
        mse = mf.errors[-1]
        root_mse = np.sqrt((mse/self.n_ratings))
        return root_mse
        
    
def main():
    mf_regressor = MatrixFactorizationRegressor()
    mf_regressor.parse_data()
    print("Root MSE:", mf_regressor.train_and_predict())

if __name__== "__main__":
  main()