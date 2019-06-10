import pandas as pd
import numpy as np
import pickle
import networkx as nx

ratingsData = pd.read_csv("Datasets/ml-latest-small/ratings.csv")
print(ratingsData.head())

userIds = list(set(ratingsData['userId']))
print(len(userIds))

print(userIds[0:10])


numberOfNodes = len(userIds)

G = nx.fast_gnp_random_graph(n=numberOfNodes,p=0.10,directed = False)
info = nx.info(G)
print(info)
graphToBeUsed = [G,info]
pickle.dump(graphToBeUsed,open("movieLensPickleFiles/erdosRenyi.pickle","wb"))
userToGraphNode={}
graphNodeToUser={}
j=0
for i in G.nodes():
	userToGraphNode[userIds[j]]=i
	graphNodeToUser[i]=userIds[j]
	j+=1
print(j)
pickle.dump([userToGraphNode,graphNodeToUser],open("movieLensPickleFiles/mappingErdosRenyi.pickle","wb"))

k=30
#given that we took an average degree of 60 in Erdos Renyi we take k as 60
G = nx.barabasi_albert_graph(numberOfNodes,k)
print(nx.info(G))
info = nx.info(G)
graphToBeUsed = [G,info]
pickle.dump(graphToBeUsed,open("movieLensPickleFiles/barabasiAlbert.pickle","wb"))
userToGraphNode={}
graphNodeToUser={}
j=0
for i in G.nodes():
	userToGraphNode[userIds[j]]=i
	graphNodeToUser[i]=userIds[j]
	j+=1
print(j)
pickle.dump([userToGraphNode,graphNodeToUser],open("movieLensPickleFiles/mappingBarabasiAlbert.pickle","wb"))

k=60
p=0.15
#given that we took an average degree of 250 in Erdos Renyi we take k as 250
G = nx.watts_strogatz_graph(numberOfNodes, k, p, seed=None)
print(nx.info(G))
info = nx.info(G)
graphToBeUsed = [G,info]
pickle.dump(graphToBeUsed,open("movieLensPickleFiles/graphWattsStrogatz.pickle","wb"))
userToGraphNode={}
graphNodeToUser={}
j=0
for i in G.nodes():
	userToGraphNode[userIds[j]]=i
	graphNodeToUser[i]=userIds[j]
	j+=1
print(j)
pickle.dump([userToGraphNode,graphNodeToUser],open("movieLensPickleFiles/mappingWattsStrogatz.pickle","wb"))