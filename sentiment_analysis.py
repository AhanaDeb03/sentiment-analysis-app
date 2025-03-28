import nltk  
from textblob import TextBlob  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  

 
nltk.download('punkt')  


vader = SentimentIntensityAnalyzer()  

def analyze_sentiment(text):  
    """Analyze sentiment using VADER & TextBlob"""  
    if not text.strip():
        return {"error": "No text provided"}  

    # VADER Sentiment Score  
    vader_score = vader.polarity_scores(text)  

    # TextBlob Sentiment Score  
    blob_score = TextBlob(text).sentiment.polarity  

    # sentiment  
    sentiment = "Positive" if vader_score["compound"] > 0 else "Negative" if vader_score["compound"] < 0 else "Neutral"  

    return {
        "text": text,  
        "vader_score": vader_score,  
        "blob_score": blob_score,  
        "sentiment": sentiment  
    }  

# Testing the function  
if __name__ == "__main__":  
    sample_text = "I love this project! It's amazing."  
    print(analyze_sentiment(sample_text))  
