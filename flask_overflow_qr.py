# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from flask import Flask, jsonify, request
from flask_caching import Cache

# Initialiser Flask et Flask-Caching
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Définir les paramètres de prétraitement des données
maxlen = 100
tokenizer = Tokenizer()

# Charger le modèle entraîné
model = load_model('model.h5')

# Définir une fonction pour prédire la réponse à partir d'une question
@cache.memoize(timeout=60)  # Définir une durée de cache de 60 secondes
def predict_answer(question):
    # Prétraiter la question
    X = tokenizer.texts_to_sequences([question])
    X = pad_sequences(X, padding='post', maxlen=maxlen)

    # Prédire la réponse
    y_pred = model.predict(X)[0]
    y_pred = np.argmax(y_pred)

    # Convertir la réponse en texte
    answer = tokenizer.index_word[y_pred]

    return answer

# Définir la route de l'API
@app.route('/api', methods=['POST'])
def predict():
    # Vérifier que la question a été envoyée
    if 'question' not in request.json:
        return jsonify({'error': 'La question est manquante.'})

    # Obtenir la question depuis la requête
    question = request.json['question']

    # Vérifier que la question n'est pas vide
    if not question:
        return jsonify({'error': 'La question est vide.'})

    # Prédire la réponse à partir de la question
    try:
        answer = predict_answer(question)
    except:
        return jsonify({'error': 'Une erreur s\'est produite lors de la prédiction de la réponse.'})

    # Retourner la réponse
    return jsonify({'answer': answer})

# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True)
