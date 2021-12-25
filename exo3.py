"""
TP 4 - Réseaux de neurones récurrents
Exercice 3 : Extraction des embedding Glove des légendes
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from divers import *
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Téléchargement des données
download('http://cedric.cnam.fr/~thomen/cours/US330X/flickr_8k_train_dataset.txt', "flickr_8k_train_dataset.txt")

# Récupérer les mots
filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nb_samples = df.shape[0]
iter_word = df.iterrows()
allwords = []
for i in range(nb_samples):
    x = iter_word.__next__()
    cap_words = x[1][1].split()
    cap_wordsl = [w.lower() for w in cap_words]
    allwords.extend(cap_wordsl)

unique = list(set(allwords))
print(len(unique))

#  Téléchargement du fichier contenant les Embeddings vectoriels Glove
download('http://cedric.cnam.fr/~thomen/cours/US330X/glove.6B.100d.txt', "glove.6B.100d.txt")
GLOVE_MODEL = "glove.6B.100d.txt"
fglove = open(GLOVE_MODEL, "r", encoding='utf-8')

# Récupérer les mots
cpt = 0
listwords = []
listembeddings = []
for line in fglove:
    row = line.strip().split()
    word = row[0]
    if word in unique or word == 'unk':
        embedding = []
        listwords.append(word)
        embedding.append([float(row[i]) for i in range(1, 101)])
        embedding = np.array([embedding[0]])
        embedding = np.reshape(embedding, [100])
        listembeddings.append(embedding)

        cpt += 1
        print("word: " + word + " embedded " + str(cpt))

fglove.close()
nbwords = len(listembeddings)
tembedding = len(listembeddings[0])
print("Number of words=" + str(len(listembeddings)) + " Embedding size=" + str(tembedding))

embeddings = np.zeros((len(listembeddings) + 2, tembedding + 2))
for i in range(nbwords):
    embeddings[i, 0:tembedding] = listembeddings[i]

listwords.append('<start>')
embeddings[7001, 100] = 1

listwords.append('<end>')
embeddings[7002, 101] = 1

# Sauvegarde du fichier
outfile = 'Caption_Embeddings.p'
with open(outfile, "wb") as pickle_f:
    pickle.dump([listwords, embeddings], pickle_f)

# Récupérer le fichier
outfile = 'Caption_Embeddings.p'
[listwords, embeddings] = pickle.load(open(outfile, "rb"))
print("embeddings: " + str(embeddings.shape))

# Normalisation
for i in range(embeddings.shape[0]):
    embeddings[i, :] /= np.linalg.norm(embeddings[i, :])

# Clustering
kmeans = KMeans(10, init='random', max_iter=1000).fit(embeddings)
clustersID = kmeans.labels_
clusters = kmeans.cluster_centers_

# Visualisation
pointsclusters = np.zeros(([10, 2]))
indclusters = np.zeros(([10, len(listwords)]))

for i in range(10):
    norm = np.linalg.norm((clusters[i] - embeddings), axis=1)
    inorms = np.argsort(norm)
    indclusters[i, :] = inorms[:]
    indclusters = indclusters.astype(int)

    print("Cluster " + str(i) + " =" + listwords[indclusters[i][0]])
    for j in range(1, 21):
        print(" mot: " + listwords[indclusters[i][j]])

# T-SNE
tsne = TSNE(n_components=2, perplexity=30, verbose=2, init='pca', early_exaggeration=24)
points2D = tsne.fit_transform(embeddings)

# Affichage
for i in range(10):
    pointsclusters[i, :] = points2D[int(indclusters[i][0])]

cmap = cm.tab10
plt.figure(figsize=(3.841, 7.195), dpi=100)
plt.set_cmap(cmap)
plt.subplots_adjust(hspace=0.4)
plt.scatter(points2D[:, 0], points2D[:, 1], c=clustersID, s=3, edgecolors='none', cmap=cmap, alpha=1.0)
plt.scatter(pointsclusters[:, 0], pointsclusters[:, 1], c=range(10), marker='+', s=1000, edgecolors='none', cmap=cmap,
            alpha=1.0)

plt.colorbar(ticks=range(10))
plt.show()
