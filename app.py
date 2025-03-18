import os
import openai
import streamlit as st
import requests
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Load API keys securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
news_api_key = "c8feaea3b6b04bc48192f3dc9b698dc8"  # Hardcoded for fetching live data

# Debugging: Check if the keys are correctly set
if not openai_api_key.startswith("sk-"):
    st.error("⚠️ OpenAI API key not found or invalid. Ensure it's correctly set in the environment.")
    st.stop()

if not news_api_key:
    st.warning("⚠️ NEWS_API_KEY not set. Please make sure it's available.")

# Initialize OpenAI client
client = openai.Client(api_key=openai_api_key)

# ✅ Fetch news based on the user's risk category
def fetch_data(risk_category):
    """
    Fetch relevant news articles based on the user's input risk category.
    Fetches real-time data from the NewsAPI.

    Args:
        risk_category (str): The user-defined risk category.

    Returns:
        list: A list of news headlines relevant to the risk category.
    """
    url = f"https://newsapi.org/v2/everything?q={risk_category}&language=en&apiKey={news_api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract top 5 headlines related to the risk category
        headlines = [article["title"] for article in data.get("articles", [])[:5]]
        return headlines if headlines else [f"⚠️ No news found for '{risk_category}'."]
    
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return [f"⚠️ Could not fetch real-time data for '{risk_category}'."]

# ✅ Convert text data into embeddings
def get_embeddings(texts):
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        return [res.embedding for res in response.data]
    except openai.AuthenticationError:
        st.error("❌ OpenAI Authentication Failed: Check your API key!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error generating embeddings: {e}")
        st.stop()

# ✅ Cluster topics using BERTopic with better error handling and sparse matrix fix
def cluster_topics(texts):
    try:
        # Vectorizing the text data using CountVectorizer (you can also use TF-IDF)
        vectorizer = CountVectorizer(stop_words="english")
        embeddings = vectorizer.fit_transform(texts)
        
        # Convert sparse matrix to dense
        embeddings_dense = embeddings.toarray()  # Convert sparse matrix to dense

        # Dynamically adjust n_topics if needed to avoid the error (based on available data size)
        num_topics = min(10, len(texts) - 1)  # Ensure n_topics is less than the number of texts

        # Initialize BERTopic
        topic_model = BERTopic(n_topics=num_topics)  # Set the number of topics dynamically
        
        # Fit the topic model using the dense matrix
        topics, _ = topic_model.fit_transform(texts, embeddings_dense)
        
        return topic_model.get_topic_info()
    except Exception as e:
        st.error(f"⚠️ Error in topic modeling: {e}")
        return None

# ✅ Generate a risk hypothesis using GPT-4 Turbo with severity as a numeric score
def generate_risk_hypothesis(cluster_summary, risk_category):
    if not cluster_summary:
        return "⚠️ Not enough data to generate a risk hypothesis."

    prompt = f"""
    You are an AI risk analyst specializing in detecting novel risks in {risk_category}.

    Based on the following weak signals:
    {cluster_summary}

    Predict a significant emerging risk in {risk_category} that could arise by 2035.
    Provide:
    1. A structured risk hypothesis
    2. A probability score (0-1)
    3. An estimated severity score (0-100) based on:
       - Direct deaths (e.g., immediate loss of life due to the event)
       - Indirect deaths (e.g., deaths caused by secondary effects like infrastructure collapse, disease, or economic instability)
    4. A justification for the severity score:
       - Explain why the score falls in the given range.
       - Provide real-world analogies or past events that support the estimate.

    Format your response as:
    **Risk Hypothesis:** [Title]
    **Description:** [Detailed description]
    
    **1. Probability Score:** [Numeric 0-1]
    **2. Severity Score:** [Numeric 0-100]
    **3. Justification:** [Explanation including direct & indirect deaths]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content if response.choices else "No response from model."
    except openai.AuthenticationError:
        st.error("❌ OpenAI Authentication Failed: Check your API key!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error generating risk hypothesis: {e}")
        st.stop()

# ✅ Generate AI art using DALL·E with user-defined risk category
def generate_image(risk_text, risk_category):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"A conceptual artwork representing a major risk in {risk_category}: {risk_text}",
            n=1
        )
        return response.data[0].url if response.data else None
    except openai.AuthenticationError:
        st.error("❌ OpenAI Authentication Failed: Check your API key!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error generating image: {e}")
        st.stop()

# ✅ Streamlit UI Setup
st.title("Sentinel Risk Intelligence System")
st.write("Detecting novel risks for 2025-2035 using AI-powered weak signal analysis.")

# User input for risk category
query = st.text_input("Enter a risk category (e.g., 'climate change', 'cybersecurity', 'financial risks')")

# Button to generate risk
if st.button("Generate Risks"):
    if not query:
        st.error("⚠️ Please enter a risk category before generating risks!")
        st.stop()
    
    with st.spinner(f"Fetching data for '{query}'..."):
        data = fetch_data(query)
        embeddings = get_embeddings(data)
        clusters = cluster_topics(data)
        cluster_summary = "\n".join(data)

    with st.spinner(f"Generating risk hypothesis for '{query}'..."):
        risk_hypothesis = generate_risk_hypothesis(cluster_summary, query)

    with st.spinner(f"Generating AI image for '{query}'..."):
        image_url = generate_image(risk_hypothesis, query)

    # Display results
    st.subheader(f"🚨 Predicted Risk in {query}")
    st.write(risk_hypothesis)
    if image_url:
        st.image(image_url, caption=f"AI-Generated Visualization for '{query}'")
    else:
        st.write("⚠️ Image generation failed.")
