"""
TP 4 - Réseaux de neurones récurrents
Exercice 2 : Génération de texte avec le modèle appris
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import _pickle as pickle
from divers import *

# Chargement des données
SEQLEN = 10
outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
[index2char, X_train, y_train, X_test, y_test] = pickle.load(open(outfile, "rb"))

# Chargement du modèle
model = load_model("modele_exo1")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()
nb_chars = len(index2char)

# Sélection d'une chaîne de caractère initiale
seed = 15608
char_init = ""
for i in range(SEQLEN):
    char = index2char[np.argmax(X_train[seed, i, :])]
    char_init += char

print("CHAR INIT: " + char_init)

# Conversion de la séquence de départ au format one-hot
test = np.zeros((1, SEQLEN, nb_chars), dtype=np.bool)
test[0, :, :] = X_train[seed, :, :]

# Génération du texte
nbgen = 400
gen_char = char_init
temperature = 0.5

for i in range(nbgen):
    preds = model.predict(test)[0]
    next_ind = sampling(preds, temperature)
    next_char = index2char[next_ind]
    gen_char += next_char
    for j in range(SEQLEN - 1):
        test[0, j, :] = test[0, j + 1, :]
    test[0, SEQLEN - 1, :] = 0
    test[0, SEQLEN - 1, next_ind] = 1

print("Generated text: " + gen_char)
