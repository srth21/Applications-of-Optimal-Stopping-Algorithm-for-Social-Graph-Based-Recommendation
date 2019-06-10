import pandas as pd
import numpy as np
import pickle

ratingsData = pd.read_csv("Datasets/ml-latest-small/ratings.csv")
print(ratingsData.head())

titlesAndGenresDataset = pd.read_csv("Datasets/ml-latest-small/movies.csv")
print(titlesAndGenresDataset.head())

data = pd.merge(ratingsData,titlesAndGenresDataset,on='movieId')
print(data.head())

movies = {}

number = len(list(data['userId']))
print(number)

movieRating = {}
movieCount ={}
for i in range(number):
	row = data.iloc[i]
	movieId = row['movieId']
	rating = row['rating']

	if(movieId not in movieRating):
		movieRating[movieId] = 0
		movieCount[movieId] = 0

	movieRating[movieId]+=rating
	movieCount[movieId]+=1

for i in movieRating:
	movieRating[i]/=movieCount[i]

number = len(list(titlesAndGenresDataset['movieId']))

for i in range(number):
	row = titlesAndGenresDataset.iloc[i]
	movieId = row['movieId']
	title = row['title']
	genres = row['genres']
	averageRating = 0
	if(movieId not in movieRating):
		averageRating = 2.5
	else :
		averageRating = movieRating[movieId]

	movies[movieId]=[title,genres,averageRating]

path = "movieLensPickleFiles/"
pickle.dump(movies, open(path+"movies.pickle", "wb"))