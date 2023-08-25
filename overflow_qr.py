# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

import tensorflow as tf
print(tf.__version__)


# Télécharger les données de Stack Overflow
url = 'https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z'
df = pd.read_csv(url, compression='zip', usecols=['Title', 'Body'])

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# Concaténer le titre et le corps pour former la question
df['Question'] = df['Title'] + ' ' + df['Body']

# Sélectionner les colonnes pertinentes pour la construction du modèle
df = df[['Question', 'Answer']]

# Prétraiter les données
maxlen = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Question'].values)
X = tokenizer.texts_to_sequences(df['Question'].values)
X = pad_sequences(X, padding='post', maxlen=maxlen)
y = tokenizer.texts_to_sequences(df['Answer'].values)
y = pad_sequences(y, padding='post', maxlen=maxlen)

# Créer un modèle de question-réponse en utilisant un réseau de neurones LSTM
input_layer = Input(shape=(maxlen,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=maxlen)(input_layer)
lstm_layer = LSTM(100)(embedding_layer)
output_layer = Dense(len(tokenizer.word_index) + 1, activation='softmax')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
