# -*- coding: utf-8 -*-
"""
Matrix factorization machine for collaborative filtering

Created on Sun Nov 11 00:15:16 2018

@author: Nicole
"""
from scipy.spatial import distance
import numpy as np
import copy

class MatrixFactorization:
    
    def __init__(self, eta=0.000001, c1=1, c2=1, n_features=50, n_iter=10):
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.n_features = n_features
        self.n_iter = n_iter
        self.m = {}  # movie vectors
        self.u = {}  # user vectors
        self.errors = []
        
    def fit(self, movie_ratings, user_ratings):        
        """Fits training data"""
        for i in range(self.n_iter):
            self.update_movie_vector(movie_ratings)
            self.update_user_vector(user_ratings)
            self.errors.append(self.calc_error(movie_ratings))
            
    def update_movie_vector(self, movie_ratings):
        for movie in movie_ratings:
            gradient_sum = 0
            if movie not in self.m: 
                rgen = np.random.RandomState(1)
                self.m[movie] = rgen.normal(scale=1, size=self.n_features)
            for user, rating in movie_ratings[movie].items():
                if user not in self.u:
                    rgen = np.random.RandomState(2)
                    self.u[user] = rgen.normal(scale=1, size=self.n_features)
                gradient_obj_m = (2*(self.net_input(movie, user) - rating)) * self.u[user] 
                reg = self.c1 * self.m[movie] 
                gradient = np.add(gradient_obj_m, reg)
                gradient_sum = np.add(gradient_sum, gradient) 
            m_update = self.eta * gradient_sum 
            self.m[movie] = np.subtract(self.m[movie], m_update)
        
    def update_user_vector(self, user_ratings):
        for user in user_ratings:
            gradient_sum = 0
            for movie, rating in user_ratings[user].items():
                gradient_obj_u = (2*(self.net_input(movie, user) - rating)) * self.m[movie] 
                reg = self.c2 * self.u[user] 
                gradient = np.add(gradient_obj_u, reg)
                gradient_sum = np.add(gradient_sum, gradient)
            u_update = self.eta * gradient_sum
            self.u[user] = np.subtract(self.u[user], u_update)
    
    def calc_error(self, movie_ratings):
        """Calculates total error given the current movie and user vectors"""
        total_error = 0
        for movie in movie_ratings:
            for user, rating in movie_ratings[movie].items():
                obj = np.square(self.net_input(movie, user) - rating)
                total_error += obj
        return total_error
    
    def net_input(self, movie, user):
        return np.dot(self.m[movie], self.u[user])
    
    def find_nearest_neighbors(self, movie, k=10):
        """Finds the closest k vectors to a movie vectors"""

        # Finds distance between movie vector and every other movie vector
        movies = list(self.m.keys())
        movie_vectors = np.array(list(self.m.values()))
        euclidean_distances = distance.cdist(movie_vectors, self.m[movie].reshape(1, -1))
        euclidean_distances[np.where(euclidean_distances == 0)] = np.nan
        
        # Finds k nearest neighbors to movie vector
        nearest_neighbors = []
        for i in range(k):
            min_index = np.nanargmin(euclidean_distances, axis=0)
            euclidean_distances[min_index[0]] = np.nan
            nearest_neighbors.append(movies[min_index[0]])
        return nearest_neighbors
    
    def predict(self, movie_ratings, user_ratings):
        """Calculates and assigns regression values to test data"""
        predicted_movie_ratings = copy.deepcopy(movie_ratings)
        predicted_user_ratings = copy.deepcopy(user_ratings)
        for movie in movie_ratings:
            for user, rating in movie_ratings[movie].items():
                rating = self.net_input(movie, user)
                predicted_movie_ratings[movie][user] = rating
                predicted_user_ratings[user][movie] = rating
        return [predicted_movie_ratings, predicted_user_ratings]