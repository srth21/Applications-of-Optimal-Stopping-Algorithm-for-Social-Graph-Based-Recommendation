import pandas as pd
import numpy as np
import pickle

ratingsData = pd.read_csv("Datasets/ml-latest-small/ratings.csv")
print(ratingsData.head())

titlesAndGenresDataset = pd.read_csv("Datasets/ml-latest-small/movies.csv")
print(titlesAndGenresDataset.head())

data = pd.merge(ratingsData,titlesAndGenresDataset,on='movieId')
print(data.head())

moviesBasedOnUserId = {}

number = len(list(data['userId']))
print(number)

movies = []
for i in range(number):
	row = data.iloc[i]
	userId = row['userId']
	movieId = row['movieId']
	userRating = row['rating']
	movies.append(movieId)
	l=[movieId,userRating]
	if(userId not in moviesBasedOnUserId):
		moviesBasedOnUserId[userId]=[]
	moviesBasedOnUserId[userId].append(l)
path = "movieLensPickleFiles/"
pickle.dump(moviesBasedOnUserId, open(path+"userMovies.pickle", "wb"))

print(len(list(set(movies))))