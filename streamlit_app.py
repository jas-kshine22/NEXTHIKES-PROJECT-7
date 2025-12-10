import streamlit as st
import pickle
import re
import string

# Load model and TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨")

st.title("🚨 Disaster Tweet Classifier")
st.write("Enter a tweet below to check whether it indicates a disaster or not.")

tweet = st.text_area("Type or paste a tweet here...")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 This is a **Disaster Tweet**.")
        else:
            st.success("✅ This is a **Non-Disaster Tweet**.")
