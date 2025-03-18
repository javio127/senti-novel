import os
import subprocess

# Ensure required packages are installed
missing_packages = ["openai", "bertopic", "pandas", "requests"]
for package in missing_packages:
    subprocess.run(["pip", "install", package])

import streamlit as st
import openai
import requests
from bertopic import BERTopic
import pandas as pd


# Set your OpenAI API Key
openai.api_key = "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"

# Simulated function to fetch real-time data (replace with actual API calls)
def fetch_data():
    return [
        "AI-driven misinformation is increasing political instability.",
        "New advancements in bioengineering raise ethical concerns.",
        "Financial markets are increasingly controlled by algorithmic trading."
    ]

# Convert fetched data into embeddings
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    return [res['embedding'] for res in response['data']]

# Cluster topics using BERTopic
def cluster_topics(texts):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(texts)
    return topic_model.get_topic_info()

# Generate a risk hypothesis using GPT-4 Turbo
def generate_risk_hypothesis(cluster_summary):
    prompt = f"""
    Analyze the following weak signals:
    {cluster_summary}

    Predict a novel risk that could emerge by 2035. Provide:
    1. A structured risk hypothesis
    2. A probability score (0-1)
    3. An estimated severity score (in direct or indirect deaths)
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Generate AI art using DALL·E
def generate_image(risk_text):
    response = openai.Image.create(
        prompt=f"A conceptual artwork representing: {risk_text}",
        model="dall-e-3"
    )
    return response['data'][0]['url']

# Streamlit UI Setup
st.title("Sentinel Risk Intelligence System")
st.write("Detecting novel risks for 2025-2035 using AI-powered weak signal analysis.")

# User input for query
query = st.text_input("Enter a risk category (or leave blank for automatic detection)")

# Button to generate risk
if st.button("Generate Risks"):
    with st.spinner("Fetching data..."):
        data = fetch_data()
        embeddings = get_embeddings(data)
        clusters = cluster_topics(data)
        cluster_summary = "\n".join(data)
    
    with st.spinner("Generating risk hypothesis..."):
        risk_hypothesis = generate_risk_hypothesis(cluster_summary)
    
    with st.spinner("Generating AI image..."):
        image_url = generate_image(risk_hypothesis)
    
    # Display results
    st.subheader("🚨 Predicted Risk")
    st.write(risk_hypothesis)
    st.image(image_url, caption="AI-Generated Risk Visualization")
