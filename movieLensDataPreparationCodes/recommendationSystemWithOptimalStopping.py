import pickle
import networkx as nx
import numpy as np
import math

'''

Implementing Optimal Stopping

The optimal stopping solution is that we have to consider 37% percent of the total before we make a decision.
So to see how to consider these 37%.
We get scores for the weights of each edge between users.
This scores are obtained based on movie preference similarity.

'''

def loadGraph(graphFilename, mappingFilename):
	G,info = pickle.load(open(graphFilename,"rb"))
	userToGraphNodeMapping, graphNodeToUser = pickle.load(open(mappingFilename,"rb"))
	#print(nx.info(G))
	return G,userToGraphNodeMapping,graphNodeToUser

def loadMovieGraph(graphFilename):
	G = pickle.load(open(graphFilename,"rb"))
	return G

def findEgoNetworkOfANode(G,nodeName):
	egoGraph = nx.ego_graph(G,nodeName)
	#print(nx.info(egoGraph))
	#print(egoGraph.nodes())
	return egoGraph

def getUsersMovies(filename):
	usersToMoviesMapping = pickle.load(open(filename,"rb"))
	return usersToMoviesMapping

def getMoviesData(filename):
	movies = pickle.load(open(filename,"rb"))
	return movies

usersGraph,userToGraphNodeMapping,graphNodeToUserMapping = loadGraph("../movieLensPickleFiles/erdosRenyi.pickle","../movieLensPickleFiles/mappingErdosRenyi.pickle")
movieGraph = loadMovieGraph("../movieLensPickleFiles/moviesGraph.pickle")
usersToMoviesMapping = getUsersMovies("../movieLensPickleFiles/userMovies.pickle")
movies = getMoviesData("../movieLensPickleFiles/movies.pickle")

def softmax(l):
	lPowerX = np.exp(l - np.max(l))
	return lPowerX/lPowerX.sum()

def getGenreScores(moviesList):
	genreCount = {}

	for movie in moviesList:
		movieId = movie[0]
		rating = movie[1]

		genres = movies[movieId][1].split('|')

		for genre in genres:
			if(genre not in genreCount):
				genreCount[genre]=0
			genreCount[genre]+=rating


	'''
	getting a softmax over the scores of all the movie genres for each
	'''

	genreKeys = list(genreCount.keys())
	genreValues = list(genreCount.values())

	genreValuesSoftmax = softmax(genreValues)

	genreScoreSoftmax = {}

	for i in range(len(genreKeys)):
		genreScoreSoftmax[genreKeys[i]] = genreValuesSoftmax[i]

	return genreScoreSoftmax

def getRating(l):
	return l[1]


def getSimilarity(d1, d2):
	difference = 0

	for i in d1:
		if(i not in d2):
			difference+=d1[i]
		else:
			difference+=abs(d1[i]-d2[i])

	for i in d2:
		if(i not in d1):
			difference+=d2[i]

	return difference

def getWeights(graphNode, egoNetwork):
	
	neighbours = list(egoNetwork.nodes())

	graphNodeMovieGenreSimilarity = getGenreScores(usersToMoviesMapping[graphNodeToUserMapping[graphNode]])
	
	weights = {}

	for neighbour in neighbours:
		if(neighbour != graphNode):
			neighbourGenreScore = getGenreScores(usersToMoviesMapping[graphNodeToUserMapping[neighbour]])

			weights[neighbour] = getSimilarity(neighbourGenreScore,graphNodeMovieGenreSimilarity)

	return weights

def getRecommendationForAUser(userId,k):

	#PART 1 : SCORE OUT OF 5 BASED ON EGO NETWORK

	graphNode = userToGraphNodeMapping[userId]
	egoNetwork = findEgoNetworkOfANode(usersGraph,graphNode)

	edgeWeights = getWeights(graphNode,egoNetwork)

	optimalNodes = []

	for node in edgeWeights:
		optimalNodes.append([node,edgeWeights[node]])

	numberOfNeighbours = len(optimalNodes)

	optimalNodes.sort(key = getRating,reverse = True)

	optimalNodeCount = math.floor(numberOfNeighbours * 0.37)

	optimalNodes = optimalNodes[0:optimalNodeCount]

	'''
	Score Calculation for a movie inspired by Bayesian Probability

	We do not want the following case : 
		1. If one friend has rated a movie : 5 and no other has watched it
		2. If 200 friends have rated a movie 4
	Normal Scenrio movie 1 gets higher

	So we use a method inspired by Bayesian probability

	We have two factors : 
		1. R : The initial rating we assign to each movie. We take that as a 50% rating in this case
			so R = 2.5
		2. W : The weight we assign to the initial rating
		   This depends on the number of ratings for the movie

		   If a typical item obtains C ratings, then W should not exceed C, or 
		   else the final score will be more dependent on R than on the actual user ratings. 
		   Instead, W should be close to a fraction of C, perhaps between C/20 and C/5.

		   So we set W = C/5

	
		The formula to calculate the score for a movie is : 
		rating = R*W + sigma over all ratings ( number of ratings * rating score) / ( W + number of ratings)

	'''

	R = 2.5

	moviesScores = {}
	movieIds = []

	for node in optimalNodes:

		if(node!=graphNode):
			nodeMovies = usersToMoviesMapping[node[0]]
			for movie in nodeMovies:
				movieId = movie[0]
				movieRatingUser = movie[1]

				if(movieId not in moviesScores):
					movieIds.append(movieId)
					moviesScores[movieId] = {}
				if(movieRatingUser not in moviesScores[movieId]):
					moviesScores[movieId][movieRatingUser] = 0
				moviesScores[movieId][movieRatingUser]+=1

	movieBayesianRatings = {} 

	for movieId in moviesScores:
		ratingsDetails = moviesScores[movieId]
		totalNumberOfRatings = sum([ratingsDetails[i] for i in ratingsDetails])

		W = totalNumberOfRatings/5.0

		cumulativeRating = sum([i*ratingsDetails[i] for i in ratingsDetails])

		effectiveBayesianRating = (R*W + cumulativeRating)/(W + totalNumberOfRatings)

		movieBayesianRatings[movieId] = effectiveBayesianRating

	#PART 2 : For the same movies, score based on similarity with Movies User has watched in terms of genre

	#Will add this tomorrow.
	#continue with optimal stopping as that does not depend on the genre thing.

	genreSimilarityScores = {}

	currentUserMovies = usersToMoviesMapping[userId]

	genresScoreForUser = getGenreScores(currentUserMovies)

	for movie in movieIds:
		score = 0
		movieGenres = movies[movie][1].split('|')
		for movieGenre in movieGenres:
			if(movieGenre in genresScoreForUser):
				score += genresScoreForUser[movieGenre]

		score *= 5

		genreSimilarityScores[movie] = score


	totalScores = []
	currentUserMoviesIds = [i[0] for i in currentUserMovies]
	for movie in movieIds:
		if(movie not in currentUserMoviesIds):
			totalScores.append([movies[movie][0],movieBayesianRatings[movie] + genreSimilarityScores[movie]])

	totalScores.sort(key=getRating,reverse=True)

	if(k>len(totalScores)):
		return totalScores

	return totalScores[0:k] 
	
moviesRecommended = getRecommendationForAUser(2,10)
print(moviesRecommended)