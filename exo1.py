"""
TP 4 - Réseaux de neurones récurrents
Exercice 1 : Apprentissage d’un modèle autosupervisé pour la génération de texte
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import _pickle as pickle
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from divers import *

# Chargement des données
SEQLEN = 10
outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
[index2char, X_train, y_train, X_test, y_test] = pickle.load(open(outfile, "rb"))
nb_chars = 60

# Création du modèle
model = Sequential()
HSIZE = 128
model.add(SimpleRNN(HSIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars), unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

BATCH_SIZE = 128
NUM_EPOCHS = 50
learning_rate = 0.001
optim = RMSprop(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])
model.summary()

# Entrainement
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

# Evaluation
scores_train = model.evaluate(X_train, y_train, verbose=1)
scores_test = model.evaluate(X_test, y_test, verbose=1)
print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

# Sauvegarde du modèle
save_model(model, "modèle_exo1")
