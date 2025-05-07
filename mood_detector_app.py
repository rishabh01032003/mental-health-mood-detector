import streamlit as st
import pickle

# Load model and vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("mood_detector_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🧠 Mental Health Mood Detector")
st.write("Enter a text message and the model will predict the mood.")

# Input box
user_input = st.text_area("Enter your message here:")

# Prediction
if st.button("Detect Mood"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        st.success(f"Predicted Mood: **{prediction[0]}**")
    else:
        st.warning("Please enter a message to analyze.")
