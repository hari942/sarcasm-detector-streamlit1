import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer (save it during training using pickle)
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load trained model
model = tf.keras.models.load_model("sarcasm_model.h5")

# Parameters
max_length = 25
trunc_type = 'post'
padding_type = 'post'

# App Title
st.title("📰 Sarcasm Detection in News Headlines")
st.markdown("Enter a news headline and see if it's sarcastic!")

# Input text
headline = st.text_area("Enter a news headline:", "Government announces plan to make taxes fun again")

# Predict button
if st.button("Detect Sarcasm"):
    seq = tokenizer.texts_to_sequences([headline])
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred = model.predict(padded)[0][0]
    st.write(f"**Prediction Confidence:** {pred:.4f}")
    if pred > 0.5:
        st.markdown("🤣 **This headline is likely sarcastic!**")
    else:
        st.markdown("🙂 **This headline is likely NOT sarcastic.**")
