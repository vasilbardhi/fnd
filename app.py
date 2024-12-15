from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

app = Flask(__name__)

# Function to load tokenizer from JSON file
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = f.read()  # Read JSON file as string
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
    return tokenizer

# Paths to model and tokenizer files
model_path = 'fake_news_model.h5'
tokenizer_path = 'tokenizer.json'

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
tokenizer = load_tokenizer(tokenizer_path)

maxlen = 200  # Same as the maxlen used during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form.get('news')
        news_list = [news]

        sequences = tokenizer.texts_to_sequences(news_list)
        padded_sequences = pad_sequences(sequences, maxlen=maxlen)

        prediction = model.predict(padded_sequences)
        prediction = (prediction > 0.5).astype("int32")

        result = 'Real News' if prediction[0][0] == 1 else 'Fake News'

        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
