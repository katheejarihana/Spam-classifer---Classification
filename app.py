import streamlit as st
import pickle

model = pickle.load(open("model/spam_model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

st.title("Spam Email Classifier")

message = st.text_area("Enter Email")

if st.button("Predict"):

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)

    if prediction[0]==1:

        st.error("Spam Email")

    else:

        st.success("Not Spam")