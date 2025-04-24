import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('simple_rnn_model.h5')

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_review(review):
    review = review.lower().split()
    encoded_review = [word_index.get(word, 0) + 3 for word in review]
    return sequence.pad_sequences([encoded_review], maxlen=500)


#prediction function
def predict_sentimant(review):
    preprocessed_input = preprocess_review(review)
    prediction  = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

st.title('Review Analysis using Simple RNN')
st.write('This is a simple sentiment analysis app using a Simple RNN model trained on the IMDB dataset.')
st.write('Enter a movie review below to get the sentiment prediction.')
st.write('The model predicts whether the review is positive or negative.')

user_input = st.text_area("Enter your review here:")

if st.button('Classify'):
        sentiment, score = predict_sentimant(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Score: {score:.2f}')
else:
    st.write('Please enter a review to classify.')
