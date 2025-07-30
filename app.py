
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

#load the model
model=load_model('next_word_lstm.h5')

#load the tokenizer
with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)

#Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_length):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_length:
        token_list=token_list[-(max_sequence_length-1):]#Ensure the seq length matches the max sequence length
    token_list=pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

#Streamlit app
st.title("Next Word Prediction with LSTM")
input_text=st.text_input("Enter the sequence of words","To be or not to be")
if st.button("Predict the next word!"):
    max_sequence_length=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_length)
    st.write(f"Next word prediction: {next_word}")



