import os
import openai
import streamlit as st
import requests
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import warnings

# Load environment variables from .env file
load_dotenv()

# Load API keys securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
news_api_key = os.getenv("NEWS_API_KEY", "c8feaea3b6b04bc48192f3dc9b698dc8").strip()  # Fallback to hardcoded key

# Debugging: Check if the keys are correctly set
if not (openai_api_key.startswith("sk-") or openai_api_key.startswith("sk-proj-")):
    st.error("⚠️ OpenAI API key not found or invalid. Ensure it's correctly set in the environment.")
    st.stop()

if not news_api_key:
    st.warning("⚠️ NEWS_API_KEY not set. Please make sure it's available.")

# Initialize OpenAI client without organization
client = OpenAI(api_key=openai_api_key)

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

# Add the generate_diverse_queries function here, before fetch_data
def generate_diverse_queries(risk_category):
    """Generate diverse queries to capture different perspectives on the risk category."""
    prompt = f"""
    Generate 5 diverse search queries to capture different perspectives on risks related to "{risk_category}".
    Include:
    1. A mainstream perspective
    2. An emerging or novel angle
    3. A historical precedent or analogy
    4. A contrarian or skeptical view
    5. A technical or domain-specific formulation
    
    Format: Return only the 5 queries as a comma-separated list with no quotation marks, no numbering, and no explanation.
    Each query should be 2-5 words maximum for best results.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        queries = response.choices[0].message.content.split(',')
        # Clean and return the queries, removing any quotes
        return [q.strip().replace('"', '').replace("'", "") for q in queries if q.strip()]
    except Exception as e:
        st.warning(f"⚠️ Error generating diverse queries: {e}")
        # Fallback to basic expansion
        return [
            risk_category,
            f"{risk_category} risks",
            f"{risk_category} future",
            f"{risk_category} analysis",
            f"{risk_category} concerns"
        ]

# Add this function to make API requests with retries
def make_api_request(url, max_retries=3, timeout=10):
    """Make an API request with retries"""
    import time
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"Retrying API request ({attempt+1}/{max_retries})...")
                time.sleep(1)  # Wait before retrying
            else:
                raise e

# Modify the fetch_data function to use more sources
def fetch_data(risk_category, num_articles=100, time_range="30d", min_sources=5):
    """Fetch diverse news article texts based on the user's input risk category."""
    # Expand query with multiple perspectives
    expanded_queries = generate_diverse_queries(risk_category)
    
    all_articles = []
    unique_sources = set()
    
    # Fetch from multiple time periods to avoid recency bias
    time_periods = [
        ("recent", f"from={datetime.now() - timedelta(days=7)}&to={datetime.now()}"),
        ("past_month", f"from={datetime.now() - timedelta(days=30)}&to={datetime.now() - timedelta(days=7)}"),
        ("past_quarter", f"from={datetime.now() - timedelta(days=90)}&to={datetime.now() - timedelta(days=30)}")
    ]
    
    st.info("Collecting data from multiple sources...")
    
    # Try News API with multiple time periods
    for query in expanded_queries:
        for period_name, period_param in time_periods:
            if len(all_articles) >= num_articles and len(unique_sources) >= min_sources:
                break
                
            url = f"https://newsapi.org/v2/everything?q={query}&{period_param}&language=en&pageSize=30&apiKey={news_api_key}"
            
            try:
                data = make_api_request(url)
                
                for article in data.get("articles", []):
                    if article.get("content") and article.get("source", {}).get("name"):
                        unique_sources.add(article["source"]["name"])
                        all_articles.append({
                            "content": article["content"] or article["description"],
                            "title": article["title"],
                            "source": article["source"]["name"],
                            "published": article["publishedAt"],
                            "period": period_name,
                            "query": query
                        })
            except Exception as e:
                st.warning(f"Error fetching for query '{query}' in period {period_name}: {e}")
    
    # Try GDELT as backup
    if len(all_articles) < num_articles or len(unique_sources) < min_sources:
        st.info("Supplementing with additional data from GDELT...")
        for query in expanded_queries:
            # Use a simpler GDELT URL format
            gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&format=json&maxrecords=30"
            
            try:
                data = make_api_request(gdelt_url)
                
                for article in data.get('articles', []):
                    if article.get('snippet') and article.get('domain'):
                        # Check if we already have this article (avoid duplicates)
                        if article.get("title") not in [a.get("title") for a in all_articles]:
                            unique_sources.add(article["domain"])
                            all_articles.append({
                                "content": article.get('snippet', ''),
                                "title": article.get('title', ''),
                                "source": article.get('domain', ''),
                                "published": article.get('seendate', ''),
                                "period": "gdelt",
                                "query": query,
                                "url": article.get('url', '')
                            })
            except Exception as e:
                st.warning(f"Error fetching from GDELT for query '{query}': {e}")
    
    # Extract just the content for processing
    article_texts = [article["content"] for article in all_articles]
    
    # Store the full metadata for later use
    st.session_state.article_metadata = all_articles
    
    # If we still don't have enough data, be transparent about it
    if len(article_texts) < 5:
        st.error("""
        ⚠️ **Insufficient Data**
        
        Unable to retrieve enough articles about this topic from web sources.
        Please try a different or broader risk category, or check your API keys.
        """)
        return []
    
    return article_texts

# ✅ Convert text data into embeddings
def get_embeddings(texts):
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        # Convert the embeddings to a numpy array
        embeddings = np.array([res.embedding for res in response.data])
        return embeddings
    except openai.AuthenticationError:
        st.error("❌ OpenAI Authentication Failed: Check your API key!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error generating embeddings: {e}")
        st.stop()

# ✅ Cluster topics using BERTopic
def cluster_topics(texts, metadata=None):
    """
    Cluster topics using a combination of embeddings and metadata.
    
    Args:
        texts (list): List of article texts.
        metadata (list): List of article metadata dictionaries.
        
    Returns:
        dict: Clustered topics with metadata.
    """
    if len(texts) < 10:
        st.warning("⚠️ Limited data available. Analysis may not be comprehensive.")
        
    # Generate embeddings
    embeddings = get_embeddings(texts)
    
    # Print debug info
    st.write(f"Embeddings shape: {embeddings.shape}")
    
    # Create a more sophisticated topic model with version-compatible parameters
    try:
        if len(texts) < 10:
            # Simpler model for small datasets
            topic_model = BERTopic(
                nr_topics="auto",
                min_topic_size=1,  # Allow smaller topics
                umap_model=UMAP(
                    n_neighbors=3,  # Smaller neighborhood
                    n_components=2,  # Fewer dimensions
                    min_dist=0.0,
                    metric='cosine'
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=2,
                    min_samples=1
                )
            )
        else:
            # Regular model for larger datasets
            topic_model = BERTopic(
                # Dynamic parameters based on data size
                nr_topics="auto",
                min_topic_size=max(2, len(texts) // 20),
                # UMAP parameters
                umap_model=UMAP(
                    n_neighbors=min(15, max(5, len(texts) // 10)),
                    n_components=min(10, max(5, len(texts) // 20)),
                    min_dist=0.1,
                    metric='cosine'
                ),
                # HDBSCAN parameters
                hdbscan_model=HDBSCAN(
                    min_cluster_size=max(2, len(texts) // 20),
                    min_samples=1,
                    prediction_data=True,
                    alpha=1.0
                ),
                # Use CountVectorizer for additional features
                vectorizer_model=CountVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=10000
                )
            )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(texts, embeddings)
        
        # Enhance topic representation with metadata if available
        if metadata:
            # Analyze source diversity per topic
            topic_sources = {}
            for i, topic_id in enumerate(topics):
                if topic_id not in topic_sources:
                    topic_sources[topic_id] = set()
                if i < len(metadata):
                    topic_sources[topic_id].add(metadata[i].get("source", "unknown"))
            
            # Add source diversity to topic info
            topic_info = topic_model.get_topic_info()
            topic_info["source_diversity"] = topic_info["Topic"].apply(
                lambda x: len(topic_sources.get(x, set())) if x in topic_sources else 0
            )
            
            return topic_info
        else:
            return topic_model.get_topic_info()
    except Exception as e:
        st.error(f"⚠️ Error clustering topics: {e}")
        return None

# ✅ Generate a focused, novel risk analysis
def generate_novel_risk(topic_model, texts, risk_category):
    """
    Generate a single, highly specific novel risk scenario based on clustered topics.
    """
    try:
        # Check if we have a proper topic model or a simplified one
        if hasattr(topic_model, 'get_topic_info'):
            # Extract the top topics and their representative documents
            topic_info = topic_model.get_topic_info()
            top_topics = topic_info[topic_info["Topic"] != -1].head(5)
            
            # For each top topic, get the most representative documents
            topic_summaries = []
            for topic_id in top_topics["Topic"]:
                # Get representative docs
                docs = topic_model.get_representative_docs(topic_id)
                # Get topic keywords
                keywords = [word for word, _ in topic_model.get_topic(topic_id)]
                
                topic_summaries.append({
                    "id": topic_id,
                    "keywords": ", ".join(keywords[:10]),
                    "sample_docs": docs[:3]  # Limit to 3 docs per topic
                })
        else:
            # Use a simplified approach with just the texts
            topic_summaries = [{"id": 0, "keywords": risk_category, "sample_docs": texts[:5]}]
        
        # Generate a single, highly specific risk scenario
        prompt = f"""
        You are an expert risk analyst specializing in identifying novel, emerging risks in {risk_category}.
        
        Based on the following information from news articles:
        
        {json.dumps(topic_summaries, indent=2)}
        
        Generate ONE highly specific, novel risk scenario for {risk_category} that could emerge by 2025-2035.
        
        Your analysis must:
        1. Focus on a single, coherent risk scenario (not multiple scenarios)
        2. Identify a risk that is novel and underappreciated - something we haven't seen before
        3. Be extremely specific about the mechanisms and pathways of the risk
        4. Explain why this risk hasn't been widely recognized yet
        5. Describe precisely how and why this risk could materialize
        
        Structure your response as follows:
        
        ## [Compelling Title for the Novel Risk]
        
        ### Executive Summary
        [2-3 sentence summary of the risk]
        
        ### Risk Specifics
        - **Probability**: [0.0-1.0] - Derive this score by analyzing: (1) technology readiness levels, (2) historical precedents, (3) expert consensus from the data, (4) trend analysis, and (5) barrier assessment. Explain your reasoning for each factor.
        
        - **Severity**: [Estimated number of direct or indirect deaths] - Calculate this by estimating: (1) direct fatalities, (2) cascade effects, (3) geographic scope, (4) temporal scale, and (5) vulnerable populations. Provide a specific number or range (e.g., "50,000-100,000 deaths").
        
        - **Time Horizon**: [When this risk might materialize between 2025-2035]
        
        - **Novelty Factor**: [Why this risk is novel/underappreciated]
        
        - **Data Confidence**: [High/Medium/Low] - Assess how confident we can be in this analysis based on the quality and comprehensiveness of the source data.
        
        ### Causal Pathway
        [Detailed explanation of exactly how this risk would unfold, with specific mechanisms]
        
        ### Early Warning Signals
        [5 specific, measurable indicators that would suggest this risk is materializing]
        
        ### Strategic Response Options
        [3-5 specific actions that could mitigate this risk]
        
        Be creative but grounded in reality. Focus on truly novel risks that could emerge from technological advances or complex system interactions. Your severity score MUST be expressed in terms of human lives lost (directly or indirectly).
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.8,  # Slightly higher temperature for creativity
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ Error generating risk analysis: {e}")
        return f"Error generating novel risk scenario: {str(e)}"

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
st.title("Novel Global Threat Predictor")
st.write("Identify novel or underappreciated global threats that could emerge between 2025-2035")

# Simple search interface
risk_category = st.text_input("What risk domain would you like to explore?", "emerging technology risks")

# Optional advanced settings in an expander
with st.expander("Advanced Settings", expanded=False):
    data_quality = st.select_slider(
        "Analysis Depth",
        options=["Quick", "Balanced", "Comprehensive"],
        value="Balanced",
        help="Controls the amount of data collected and analysis depth"
    )
    
    # Map the simple selector to actual parameters
    if data_quality == "Quick":
        num_articles, min_sources = 30, 3
    elif data_quality == "Balanced":
        num_articles, min_sources = 100, 5
    else:  # Comprehensive
        num_articles, min_sources = 200, 10

    st.markdown("### API Keys")
    st.info("API keys are stored securely in your .env file. Current status:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("OpenAI API:", "✅ Connected" if openai_api_key else "❌ Missing")
    with col2:
        st.write("News API:", "✅ Connected" if news_api_key else "❌ Missing")

# Run analysis button - prominent and clear
if st.button("Predict Novel Threats", type="primary", use_container_width=True):
    # Data collection with progress
    with st.status("Collecting diverse data...") as status:
        articles = fetch_data(risk_category, num_articles, min_sources=min_sources)
        
        # Check if we have enough data to proceed
        if not articles:
            st.error("Analysis cannot proceed without sufficient data. Please try again with a different risk category.")
            st.stop()
        
        # Create a more informative summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Articles Requested", num_articles)
            st.metric("Sources Requested", min_sources)
        with col2:
            st.metric("Articles Found", len(articles), 
                     delta=len(articles)-num_articles,
                     delta_color="inverse")
            
            source_count = len(set([a.get("source", "") for a in st.session_state.get('article_metadata', [])]))
            st.metric("Sources Found", source_count,
                     delta=source_count-min_sources,
                     delta_color="inverse")
        
        # Explain what happened if targets weren't met
        if len(articles) < num_articles or source_count < min_sources:
            st.info("""
            ℹ️ **Data Collection Note**: 
            The requested number of articles/sources couldn't be found. 
            This could be due to API limitations or the specificity of the topic.
            The analysis will proceed with the articles that were found.
            """)
        
        # Topic modeling
        topic_model = cluster_topics(articles, st.session_state.get('article_metadata'))
        
        # Check if topic modeling was successful
        if topic_model is None:
            st.error("❌ Topic modeling failed. Using simplified analysis.")
            # Create a simple fallback topic model
            from collections import namedtuple
            SimpleTopicInfo = namedtuple('SimpleTopicInfo', ['Topic', 'Count', 'Name'])
            topic_model = [SimpleTopicInfo(0, len(articles), "All Articles")]
            
        status.update(label="Generating risk analysis...", state="running")
        
        # Risk analysis
        risk_analysis = generate_novel_risk(topic_model, articles, risk_category)
        status.update(label="Analysis complete!", state="complete")
    
    # Display results
    st.markdown(risk_analysis)
    
    # Add explanation about the scoring system
    with st.expander("Understanding Risk Score Methodology", expanded=True):
        st.markdown("""
        ### How Risk Scores Are Calculated From Data
        
        Our system uses a multi-stage process to transform news data into quantitative risk assessments:
        
        #### 1. Data Collection & Processing
        - **News Articles**: We gather diverse perspectives from multiple sources
        - **Topic Modeling**: We identify key themes and patterns using NLP and clustering
        - **Weak Signal Detection**: We extract early indicators of emerging risks
        
        #### 2. Probability Score (0.0-1.0)
        The probability score is derived by analyzing:
        - **Technology Readiness Levels**: How mature are the enabling technologies?
        - **Historical Precedents**: Have similar risks emerged in the past?
        - **Expert Consensus**: What do domain experts say about this risk?
        - **Trend Analysis**: Are the necessary preconditions becoming more common?
        - **Barrier Assessment**: What obstacles would prevent this risk from materializing?
        
        #### 3. Severity Score (Human Lives)
        The severity is calculated by estimating:
        - **Direct Impact**: Immediate fatalities from the primary event
        - **Cascade Effects**: Deaths from system failures and secondary consequences
        - **Geographic Scope**: Local, regional, or global impact
        - **Temporal Scale**: Short-term vs. long-term mortality effects
        - **Vulnerability Factors**: Which populations would be most affected?
        
        #### 4. Confidence Assessment
        We also evaluate the confidence in our estimates based on:
        - **Data Quality**: How comprehensive and reliable is our source data?
        - **Model Uncertainty**: What are the limitations of our analytical approach?
        - **Expert Disagreement**: Is there consensus or controversy about this risk?
        
        This methodology combines quantitative data analysis with qualitative expert judgment to produce structured risk assessments that can inform decision-making.
        """)

    # Extract the title of the risk for the DALL-E prompt
    risk_title = risk_analysis.split('\n')[0].replace('#', '').strip() if '\n' in risk_analysis else risk_category

    # Generate and display an image
    with st.expander("Visualize this Risk", expanded=True):
        st.info("Generating visual representation...")
        image_url = generate_image(risk_title, risk_category)
        if image_url:
            st.image(image_url, caption=f"AI visualization of: {risk_title}")
        else:
            st.warning("Could not generate visualization")
    
    # Data transparency
    with st.expander("Data Sources"):
        if 'article_metadata' in st.session_state and len(st.session_state.article_metadata) > 0:
            try:
                source_df = pd.DataFrame(st.session_state.article_metadata)
                
                # Add data source explanation
                st.markdown("### Data Collection Summary")
                st.write(f"**Total Articles**: {len(source_df)}")
                
                # Check if 'source' column exists
                if 'source' in source_df.columns:
                    unique_sources = source_df['source'].nunique()
                    st.write(f"**Unique Sources**: {unique_sources}")
                    
                    # Show source breakdown
                    st.markdown("### Source Distribution")
                    st.bar_chart(source_df['source'].value_counts().head(10))
                    
                    # Show data origin
                    st.markdown("### Data Origin")
                    if 'period' in source_df.columns:
                        st.write(source_df['period'].value_counts())
                    
                    # Show full data
                    st.markdown("### Full Dataset")
                    st.dataframe(source_df[['title', 'source', 'published', 'period']])
                else:
                    st.write("Source information not available")
                    st.dataframe(source_df)
            except Exception as e:
                st.error(f"Error displaying source data: {e}")
        else:
            st.write("No article metadata available")

warnings.filterwarnings("ignore", category=RuntimeWarning)

