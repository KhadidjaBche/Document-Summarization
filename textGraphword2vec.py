import nltk
import nltk.data
import os
import gensim
from gensim.matutils import jaccard, cossim, dense2vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec
import networkx as nx
from networkx.algorithms.connectivity import minimum_st_edge_cut
from nltk.stem.porter import PorterStemmer
import community
import re, string

# get list of filenames in the directory
filenames = os.listdir(os.getcwd())
filenames = [filename for filename in filenames if filename.startswith("APW")]

pStemmer = PorterStemmer()

# flatten all documents into list of sentences
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
documents = [file(filename).read().replace("\n", '') for filename in filenames]
sentences = sum([sent_detector.tokenize(document) for document in documents], [])
regex = re.compile('[%s]' % re.escape(string.punctuation))
sentencesTokens = [regex.sub('', sen.lower()).split(' ') for sen in sentences]
for i in range(len(sentencesTokens)):
	tokens = []
	for s in sentencesTokens[i]:
		if s not in STOPWORDS:
			try:
				tokens.append(pStemmer.stem(s))
			except UnicodeDecodeError:
				pass
	sentencesTokens[i] = tokens
model = Word2Vec(sentencesTokens)

# compute similarities between sentences using stemming
similarities = []
threshold = 0.0
try:
	for i in range(len(sentences)):
		for j in range(i+1, len(sentences)):
			sen1 = sum([model[w] for w in sentencesTokens[i] if w in model])
			sen2 = sum([model[w] for w in sentencesTokens[j] if w in model])
			simScore = cossim(dense2vec(sen1), dense2vec(sen2))
			if simScore > threshold:
				similarities.append((simScore, sentences[i], sentences[j]))
except TypeError:
	pass
except UnicodeError:
	pass

sortSim = sorted(similarities, key = lambda x: -x[0])

sentenceGraph = nx.Graph()
sentenceGraph.add_edges_from([(sen1, sen2, {'weight': sim}) for (sim, sen1, sen2) in similarities])

def invertDictionary(clusterDict):
	"""Given a dictionary mapping sentences to cluster number, returns
	a dictionary mapping cluster number to a list of sentences in the cluster."""
	invertDict = {}
	for v in clusterDict.values():
		invertDict[v] = []
	for sen in clusterDict:
		invertDict[clusterDict[sen]].append(sen)
	return invertDict

def weightedDegree(graph, node):
	degree = 0
	for neighbor in graph.neighbors(node):
		degree += graph.get_edge_data(node, neighbor)['weight']
	return degree

# find clusters
partition = community.best_partition(sentenceGraph)
clusters = invertDictionary(partition)
subGraphs = [sentenceGraph.subgraph(clusters[cluster]) for cluster in clusters]

with open('summaryWord2Vec.txt', 'w') as summaryFile:
	for subG in subGraphs:
		summaryFile.write(max([(weightedDegree(subG, sen), sen) for sen in subG.nodes()])[1] + "\n")