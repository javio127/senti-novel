import os
import openai
import streamlit as st
import requests
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load API keys securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
news_api_key = "c8feaea3b6b04bc48192f3dc9b698dc8"  # Hardcoded API key (consider storing in env variables)

# Debugging: Check if the keys are correctly set
if not openai_api_key.startswith("sk-"):
    st.error("⚠️ OpenAI API key not found or invalid. Ensure it's correctly set in the environment.")
    st.stop()

if not news_api_key:
    st.warning("⚠️ NEWS_API_KEY not set. Please make sure it's available.")

# Initialize OpenAI client
client = openai.Client(api_key=openai_api_key)

# ✅ Clean and expand user query
def clean_and_expand_query(user_input):
    """
    Cleans the user query and expands it using OpenAI.
    
    Args:
        user_input (str): The original risk category input.

    Returns:
        str: A comma-separated list of expanded search terms.
    """
    # Remove common stopwords
    words = user_input.lower().split()
    cleaned_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    cleaned_query = " ".join(cleaned_words)

    # Expand search terms using OpenAI
    prompt = f"""
    Expand this risk-related query into multiple relevant search keywords:
    
    Example:
    'cybersecurity threats' → 'cybersecurity risks, data breaches, hacking incidents, ransomware attacks'

    User input: {cleaned_query}
    Expanded query:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        expanded_terms = response.choices[0].message.content.strip()
        return expanded_terms.replace("\n", ", ")  # Convert to a comma-separated string
    except Exception as e:
        st.warning(f"⚠️ Error expanding query: {e}")
        return cleaned_query  # Fallback to original query if expansion fails

# ✅ Fetch detailed news articles
def fetch_data(risk_category, num_articles=50):
    """
    Fetch full news article texts based on the user's input risk category.
    
    Args:
        risk_category (str): The user-defined risk category.
        num_articles (int): Number of articles to fetch.

    Returns:
        list: A list of full article texts.
    """
    expanded_query = clean_and_expand_query(risk_category)
    url = f"https://newsapi.org/v2/everything?q={expanded_query}&language=en&pageSize={num_articles}&apiKey={news_api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract full content of the articles
        articles = [
            article["content"] or article["description"]  # Use description if content is missing
            for article in data.get("articles", []) if article.get("content")
        ]

        return articles if articles else [f"⚠️ No detailed news found for '{risk_category}'."]
    
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

# ✅ Cluster topics using BERTopic
def cluster_topics(texts):
    try:
        # Generate embeddings from OpenAI instead of CountVectorizer
        embeddings = get_embeddings(texts)

        # Dynamically adjust n_topics to avoid errors
        num_topics = min(10, len(texts) - 1) if len(texts) > 1 else 1

        # Initialize BERTopic
        topic_model = BERTopic(n_topics=num_topics)
        
        # Fit the topic model
        topics, _ = topic_model.fit_transform(texts, embeddings)
        
        return topic_model.get_topic_info()
    except Exception as e:
        st.error(f"⚠️ Error in topic modeling: {e}")
        return None

# ✅ Generate a risk hypothesis using GPT-4 Turbo
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
    3. An estimated severity score (0-100)
    4. A justification for the severity score, using real-world analogies.

    Format your response as:
    **Risk Hypothesis:** [Title]
    **Description:** [Detailed description]
    
    **1. Probability Score:** [0-1]
    **2. Severity Score:** [0-100]
    **3. Justification:** [Explanation]
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

# ✅ Generate AI art using DALL·E
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

