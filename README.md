
Novel Global Threat Predictor
Identify emerging risks using AI-driven analysis of real-time news data.

Overview
The Novel Global Threat Predictor is an AI-powered risk assessment tool that analyzes real-time news data to identify novel, underappreciated threats that could emerge between 2025-2035.

It processes unstructured news data, extracts key themes, and generates risk scenarios with probability and severity assessments using GPT-4 and BERTopic clustering.

How It Works
1. Data Collection
Sources: NewsData.io, GDELT Project
Scope: Scans real-time global news across various timeframes
Query Strategy: Uses AI to expand risk-related keywords and search perspectives
2. Data Processing
Embeddings: Converts text into numerical representations using OpenAI’s embedding model
Clustering: Groups related articles using BERTopic (based on UMAP + HDBSCAN)
Topic Extraction: Identifies key themes and weak signals in global news
3. Risk Analysis
Scenario Generation: GPT-4 synthesizes a risk description based on clustered news themes
Probability Assessment: Calculates likelihood based on:
Trend Momentum, Barriers to Realization, Historical Analogues, Expert Discourse, Emerging Warning Signs
Severity Assessment: Estimates potential impact (fatalities) based on:
Direct Impact, Cascade Effects, Geographic Scope, Vulnerable Populations, Temporal Persistence
4. Output and Insights
Risk Scenario: Detailed description of a novel threat
Quantitative Assessment: Probability score (0.0-1.0) and estimated fatalities
Early Warning Indicators: Signals that suggest the risk is materializing
Visual Representation: AI-generated conceptual artwork of the threat
Screenshots
Risk Scenario Output	Generated Risk Visualization
Installation and Setup
Prerequisites
Ensure you have the following installed:

Python 3.8+
pip package manager
OpenAI API key
NewsData.io API key
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/novel-threat-predictor.git
cd novel-threat-predictor
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Set Up Environment Variables
Create a .env file in the root directory and add your API keys:

ini
Copy code
OPENAI_API_KEY=your_openai_api_key
NEWSDATA_API_KEY=your_newsdata_api_key
Run the Streamlit App
bash
Copy code
streamlit run app.py
Usage
Open the Streamlit UI
Enter a risk category (e.g., "bioweapon threats", "AI cyberattacks")
The system collects, processes, and clusters news articles
GPT-4 generates a novel risk scenario
Review probability and severity assessments
Explore early warning signals and mitigation strategies
View a visual representation of the risk
Tech Stack
Python (Backend processing)
Streamlit (Web UI)
GPT-4 (Risk scenario generation)
BERTopic (Topic modeling)
UMAP & HDBSCAN (Dimensionality reduction & clustering)
NewsData.io & GDELT (Data sources)
OpenAI Embeddings API
Contributing
Fork the repository
Create a new feature branch (git checkout -b feature-name)
Commit your changes (git commit -m "Add new feature")
Push to your branch (git push origin feature-name)
Open a Pull Request
