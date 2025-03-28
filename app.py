import streamlit as st
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment-to-GIF mapping
sentiment_gifs = {
   "POSITIVE": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2hsM21zbXhzejI1MXJ5NjN1Y3lnOWQ1eWRuM2V1YTZ2amRhMXNwdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT5LMHxhOfscxPfIfm/giphy.gif",  # Excited Happy 🎉
    "NEUTRAL": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnEyZTFnaGJmMm1odThseWdrbGc2YzVqcGNrcnlwaGN5eGx6Zm9tbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9xijGdDIMovchalhxN/giphy.gif",  # Meh 😐
}

# Streamlit UI
st.title("🔍 Sentiment Analyzer")
st.write("Enter text or upload a CSV file for sentiment analysis.")

# Text input
user_input = st.text_area("Enter text here:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        sentiment=result["label"].upper()
        st.subheader("Sentiment Analysis Result")
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence Score:** {result['score']:.2f}")
        #display GIF
        gif_url=sentiment_gifs.get(sentiment,sentiment_gifs["NEUTRAL"])
        st.image(gif_url,use_container_width=True)
    else:
        st.warning("Please enter some text to analyze.")

# CSV File Upload for Batch Analysis
st.subheader("📂 Upload CSV for Batch Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must have a 'text' column!")
    else:
        df["Sentiment"] = df["text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
        st.write(df[["text", "Sentiment"]])

        # Plot sentiment distribution
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="coolwarm")
        ax.set_title("Sentiment Distribution in CSV")
        st.pyplot(fig)
