import streamlit as st
import pickle
import re
import string
import os

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨")

# Safe model loading
def load_model():
    model_path = "model.pkl"
    tfidf_path = "tfidf.pkl"

    if not os.path.exists(model_path):
        st.error("❌ model.pkl NOT FOUND in repository!")
        return None, None
    if not os.path.exists(tfidf_path):
        st.error("❌ tfidf.pkl NOT FOUND in repository!")
        return None, None

    model = pickle.load(open(model_path, "rb"))
    tfidf = pickle.load(open(tfidf_path, "rb"))
    return model, tfidf

model, tfidf = load_model()

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🚨 Disaster Tweet Classifier")
st.write("Enter a tweet below to check whether it indicates a disaster or not.")

tweet = st.text_area("Type a tweet here...")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        if model is None or tfidf is None:
            st.error("Model not loaded. Upload model.pkl and tfidf.pkl to GitHub.")
        else:
            cleaned = clean_text(tweet)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("🚨 This is a **Disaster Tweet**.")
            else:
                st.success("✅ This is a **Non-Disaster Tweet**.")
