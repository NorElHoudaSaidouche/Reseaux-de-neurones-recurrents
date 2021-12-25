"""
TP 4 - Réseaux de neurones récurrents
Exercice 0 : Génération des données et étiquettes
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from divers import *
import numpy as np
import _pickle as pickle

# Téléchargement du fichier de données
download('http://cedric.cnam.fr/~thomen/cours/US330X/fleurs_mal.txt', "fleurs_mal.txt")

# Génération des données et étiquettes
bStart = False
fin = open("fleurs_mal.txt", 'r', encoding='utf8')
lines = fin.readlines()
lines2 = []
text = []

for line in lines:
    line = line.strip().lower()
    if "Charles Baudelaire avait un ami".lower() in line and not bStart:
        print("START")
        bStart = True
    if "End of the Project Gutenberg EBook of Les Fleurs du Mal, by Charles Baudelaire".lower() in line:
        print("END")
        break
    if not bStart or len(line) == 0:
        continue

    lines2.append(line)

fin.close()
text = " ".join(lines2)
chars = sorted(set([c for c in text]))
nb_chars = len(chars)

SEQLEN = 10
STEP = 1
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])
nbex = len(input_chars)

# Mapping char -> index
char2index = dict((c, i) for i, c in enumerate(chars))
# Mapping index -> char
index2char = dict((i, c) for i, c in enumerate(chars))

# Création des données d'entrainement
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)

for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        s = char2index.get(ch)
        X[i, j, s] = True
        s = char2index[label_chars[i]]
        y[i, s] = True

# Séparation des données apprentissage des données test
ratio_train = 0.8
nb_train = int(round(len(input_chars) * ratio_train))
print("nb tot=", len(input_chars), "nb_train=", nb_train)
X_train = X[0:nb_train, :, :]
y_train = y[0:nb_train, :]

X_test = X[nb_train:, :, :]
y_test = y[nb_train:, :]
print("X train.shape=", X_train.shape)
print("y train.shape=", y_train.shape)

print("X test.shape=", X_test.shape)
print("y test.shape=", y_test.shape)

outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"

with open(outfile, "wb") as pickle_f:
    pickle.dump([index2char, X_train, y_train, X_test, y_test], pickle_f)
