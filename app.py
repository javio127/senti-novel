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
import time
from random import choice
import concurrent.futures

# Update the environment variable loading to work with Streamlit Cloud
# Load environment variables from .env file if it exists (for local development)
if os.path.exists(".env"):
    load_dotenv()

# Load API keys securely from environment variables with fallbacks
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", "")).strip()

# Replace News API with NewsData.io API
newsdata_api_key = os.getenv("NEWS_DATA_API_KEY", 
                            st.secrets.get("NEWS_DATA_API_KEY", 
                            "pub_75348cb30f5a5794f5c8910692560fcf8d530")).strip()

if not newsdata_api_key:
    st.warning("⚠️ No NEWS_DATA_API_KEY found. Please make sure it's available.")

# Filter out empty keys
news_api_keys = [
    os.getenv("NEWS_API_KEY", "c8feaea3b6b04bc48192f3dc9b698dc8").strip(),  # Primary key
    os.getenv("NEWS_API_KEY_2", "").strip(),  # Secondary key
    # You can add more keys if needed
]

# Filter out empty keys
news_api_keys = [key for key in news_api_keys if key]

if not news_api_keys:
    st.warning("⚠️ No NEWS_API_KEY found. Please make sure at least one is available.")

# Debugging: Check if the keys are correctly set
if not (openai_api_key.startswith("sk-") or openai_api_key.startswith("sk-proj-")):
    st.error("⚠️ OpenAI API key not found or invalid. Ensure it's correctly set in the environment.")
    st.stop()

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

# Modify the make_api_request function to use different API keys
def make_api_request(url, max_retries=3, timeout=10):
    """Make an API request with retries and improved error handling, using multiple API keys if available"""
    # Check if this is a News API request
    is_news_api = "newsapi.org" in url
    
    # For News API requests, try each key in sequence
    if is_news_api and len(news_api_keys) > 1:
        # Try each key in sequence
        for key_index, api_key in enumerate(news_api_keys):
            # Replace the API key in the URL
            if "apiKey=" in url:
                url_parts = url.split("apiKey=")
                if len(url_parts) > 1:
                    # Replace everything after apiKey= up to the next & or end of string
                    key_part = url_parts[1]
                    if "&" in key_part:
                        key_part = key_part.split("&", 1)[1]
                        current_url = f"{url_parts[0]}apiKey={api_key}&{key_part}"
                    else:
                        current_url = f"{url_parts[0]}apiKey={api_key}"
                else:
                    current_url = url
            else:
                current_url = url
            
            # Try with this key
            try:
                response = requests.get(current_url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                
                # Check if the response contains an error message
                if "status" in data and data["status"] == "error":
                    if "rateLimited" in data.get("message", "").lower() or "429" in data.get("message", ""):
                        # This key is rate limited, try the next one
                        continue
                    return {"error": data.get("message", "Unknown API error"), "articles": []}
                    
                return data
            except requests.exceptions.RequestException as e:
                # If it's a 429 error, try the next key
                if "429" in str(e):
                    continue
                # For other errors, only try the next key if we've exhausted retries
                if key_index < len(news_api_keys) - 1:
                    continue
                return {"error": str(e), "articles": []}
            except ValueError as e:
                # JSON parsing error
                return {"error": f"Invalid JSON response: {str(e)}", "articles": []}
    
    # For non-News API requests or if we only have one key, use the original approach
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check if the response contains an error message
            if "status" in data and data["status"] == "error":
                return {"error": data.get("message", "Unknown API error"), "articles": []}
                
            return data
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Don't show warning for every retry to reduce clutter
                time.sleep(1)  # Longer wait between retries to avoid rate limits
            else:
                # Only return error on final attempt
                return {"error": str(e), "articles": []}
        except ValueError as e:
            # JSON parsing error
            return {"error": f"Invalid JSON response: {str(e)}", "articles": []}

# Update the fetch_data function to use NewsData.io instead of News API
def fetch_data(risk_category, num_articles=100, time_range="30d", min_sources=5):
    """Fetch diverse news article texts using NewsData.io API."""
    # Expand query with multiple perspectives - but limit to 3 queries to reduce API calls
    expanded_queries = generate_diverse_queries(risk_category)[:3]
    
    all_articles = []
    unique_sources = set()
    
    # Add a status message
    st.info("Collecting data from multiple sources...")
    
    # Process sequentially to avoid overwhelming the API
    for query in expanded_queries:
        # Check if we already have enough data
        if len(all_articles) >= num_articles and len(unique_sources) >= min_sources:
            break
            
        # NewsData.io API URL
        url = f"https://newsdata.io/api/1/news?apikey={newsdata_api_key}&q={query}&language=en&size=10"
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Check if we got any articles
            if "results" not in data or not data["results"]:
                st.warning(f"No results found for query: {query}")
                continue
                
            for article in data.get("results", []):
                if article.get("content") and article.get("source_id"):
                    # Only break if we have BOTH enough articles AND sources
                    if len(all_articles) >= num_articles and len(unique_sources) >= min_sources:
                        break
                        
                    unique_sources.add(article["source_id"])
                    all_articles.append({
                        "content": article.get("content", article.get("description", "")),
                        "title": article.get("title", ""),
                        "source": article.get("source_id", ""),
                        "published": article.get("pubDate", ""),
                        "period": "recent",
                        "query": query
                    })
            
            # If there's a nextPage token, get more articles
            if "nextPage" in data and data["nextPage"] and len(all_articles) < num_articles:
                next_page = data["nextPage"]
                next_url = f"https://newsdata.io/api/1/news?apikey={newsdata_api_key}&q={query}&language=en&size=10&page={next_page}"
                
                try:
                    response = requests.get(next_url, timeout=15)
                    response.raise_for_status()
                    next_data = response.json()
                    
                    if "results" in next_data and next_data["results"]:
                        for article in next_data.get("results", []):
                            if article.get("content") and article.get("source_id"):
                                # Only break if we have BOTH enough articles AND sources
                                if len(all_articles) >= num_articles and len(unique_sources) >= min_sources:
                                    break
                                    
                                unique_sources.add(article["source_id"])
                                all_articles.append({
                                    "content": article.get("content", article.get("description", "")),
                                    "title": article.get("title", ""),
                                    "source": article.get("source_id", ""),
                                    "published": article.get("pubDate", ""),
                                    "period": "recent",
                                    "query": query
                                })
                except Exception as e:
                    st.warning(f"Error fetching next page: {e}")
            
            # Add a small delay between requests to avoid rate limits
            time.sleep(1)
        except Exception as e:
            # Log errors but continue
            st.warning(f"Error processing NewsData.io results: {e}")
    
    # If we need more data, try GDELT too
    if len(all_articles) < num_articles or len(unique_sources) < min_sources:
        st.info(f"Found {len(all_articles)} articles from {len(unique_sources)} sources. Supplementing with GDELT data...")
        
        for query in expanded_queries:
            # Check if we already have enough data
            if len(all_articles) >= num_articles and len(unique_sources) >= min_sources:
                break
                
            gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&format=json&maxrecords=30"
            
            try:
                response = requests.get(gdelt_url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Check if we got any articles
                if "articles" not in data or not data["articles"]:
                    continue
                    
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
                
                # Add a small delay between requests
                time.sleep(1)
            except Exception as e:
                # Log errors but continue
                st.warning(f"Error processing GDELT results: {e}")
    
    # Extract just the content for processing
    article_texts = [article["content"] for article in all_articles]
    
    # Store the full metadata for later use
    st.session_state.article_metadata = all_articles
    
    # Show what we found
    st.success(f"Successfully collected {len(all_articles)} articles from {len(unique_sources)} unique sources.")
    
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
        # Use batching for large sets of texts
        if len(texts) > 20:
            # Process in batches of 20
            all_embeddings = []
            for i in range(0, len(texts), 20):
                batch = texts[i:i+20]
                response = client.embeddings.create(input=batch, model="text-embedding-ada-002")
                batch_embeddings = [res.embedding for res in response.data]
                all_embeddings.extend(batch_embeddings)
            embeddings = np.array(all_embeddings)
        else:
            # Process small batches directly
            response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
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

# ✅ Estimate probability using heuristic factors
def estimate_probability(risk_description, risk_category):
    """
    Estimate probability of a risk scenario using five heuristic factors.
    
    Args:
        risk_description (str): The full risk scenario description
        risk_category (str): The risk category being analyzed
        
    Returns:
        dict: Probability score and factor breakdown
    """
    prompt = f"""
    Analyze this risk scenario and estimate its probability of occurring between 2025-2035.
    
    RISK SCENARIO:
    {risk_description}
    
    Extract and score the following five factors that contribute to probability:
    
    1. Trend Momentum (0.0-1.0): How strongly are current trends moving toward this risk?
    2. Barriers to Realization (0.0-1.0): How easily can existing barriers be overcome? (Higher = fewer/weaker barriers)
    3. Historical Analogues (0.0-1.0): How closely does this risk match historical patterns that led to similar events?
    4. Expert Discourse (0.0-1.0): How much are experts already discussing this or related risks?
    5. Emerging Warning Signs (0.0-1.0): Are early indicators of this risk already visible?
    
    For each factor, provide:
    - A score between 0.0 and 1.0
    - A 1-2 sentence justification based on the risk description
    
    Then calculate an overall probability score as a weighted average:
    - Trend Momentum: 30%
    - Barriers to Realization: 25%
    - Historical Analogues: 15%
    - Expert Discourse: 15%
    - Emerging Warning Signs: 15%
    
    Return your analysis as a JSON object with this exact structure:
    {{
        "factors": {{
            "trend_momentum": {{
                "score": [0.0-1.0],
                "justification": "explanation"
            }},
            "barriers_to_realization": {{
                "score": [0.0-1.0],
                "justification": "explanation"
            }},
            "historical_analogues": {{
                "score": [0.0-1.0],
                "justification": "explanation"
            }},
            "expert_discourse": {{
                "score": [0.0-1.0],
                "justification": "explanation"
            }},
            "emerging_warning_signs": {{
                "score": [0.0-1.0],
                "justification": "explanation"
            }}
        }},
        "overall_probability": [0.0-1.0],
        "confidence": [0.0-1.0]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate the result structure
        if "overall_probability" not in result:
            # Calculate it from factors if possible
            factors = result.get("factors", {})
            if factors:
                weights = {
                    "trend_momentum": 0.30,
                    "barriers_to_realization": 0.25,
                    "historical_analogues": 0.15,
                    "expert_discourse": 0.15,
                    "emerging_warning_signs": 0.15
                }
                
                weighted_sum = 0
                weight_total = 0
                
                for factor, weight in weights.items():
                    if factor in factors and isinstance(factors[factor], dict) and "score" in factors[factor]:
                        weighted_sum += float(factors[factor]["score"]) * weight
                        weight_total += weight
                
                if weight_total > 0:
                    result["overall_probability"] = weighted_sum / weight_total
                else:
                    result["overall_probability"] = 0.5
            else:
                result["overall_probability"] = 0.5
        
        if "confidence" not in result:
            result["confidence"] = 0.7
            
        return result
    except Exception as e:
        st.error(f"Error estimating probability: {e}")
        # Return a default structure
        return {
            "factors": {
                "trend_momentum": {"score": 0.5, "justification": "Default value due to processing error"},
                "barriers_to_realization": {"score": 0.5, "justification": "Default value due to processing error"},
                "historical_analogues": {"score": 0.5, "justification": "Default value due to processing error"},
                "expert_discourse": {"score": 0.5, "justification": "Default value due to processing error"},
                "emerging_warning_signs": {"score": 0.5, "justification": "Default value due to processing error"}
            },
            "overall_probability": 0.5,
            "confidence": 0.3
        }

# ✅ Estimate severity using heuristic factors
def estimate_severity(risk_description, risk_category):
    """
    Estimate severity of a risk scenario in terms of potential fatalities.
    
    Args:
        risk_description (str): The full risk scenario description
        risk_category (str): The risk category being analyzed
        
    Returns:
        dict: Severity assessment with fatality estimates and factor breakdown
    """
    prompt = f"""
    Analyze this risk scenario and estimate its severity in terms of potential direct and indirect fatalities.
    
    RISK SCENARIO:
    {risk_description}
    
    Extract and score the following five factors that contribute to severity:
    
    1. Direct Impact: Estimate the base number of direct fatalities (provide a specific number)
    2. Cascade Effects (1.0-5.0): Multiplier for secondary disasters and system failures
    3. Geographic Scope (1.0-3.0): Multiplier for regional/global scale (1.0=local, 2.0=regional, 3.0=global)
    4. Vulnerable Populations (1.0-2.0): Multiplier based on proportion of vulnerable groups affected
    5. Temporal Persistence (1.0-2.0): Multiplier for duration of impact (1.0=short-term, 2.0=long-term)
    
    For each factor, provide:
    - A score within the specified range
    - A 1-2 sentence justification based on the risk description
    
    Then calculate:
    1. Total estimated fatalities = Direct Impact × Cascade Effects × Geographic Scope × Vulnerable Populations × Temporal Persistence
    2. Provide a range (low-high estimate) to account for uncertainty
    
    Return your analysis as a JSON object with this exact structure:
    {{
        "factors": {{
            "direct_impact": {{
                "estimate": [number],
                "justification": "explanation"
            }},
            "cascade_effects": {{
                "multiplier": [1.0-5.0],
                "justification": "explanation"
            }},
            "geographic_scope": {{
                "multiplier": [1.0-3.0],
                "justification": "explanation"
            }},
            "vulnerable_populations": {{
                "multiplier": [1.0-2.0],
                "justification": "explanation"
            }},
            "temporal_persistence": {{
                "multiplier": [1.0-2.0],
                "justification": "explanation"
            }}
        }},
        "fatality_estimate": {{
            "low": [number],
            "high": [number],
            "median": [number]
        }},
        "confidence": [0.0-1.0]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Error estimating severity: {e}")
        return {
            "factors": {},
            "fatality_estimate": {
                "low": 1000,
                "high": 10000,
                "median": 5000
            },
            "confidence": 0.3
        }

# ✅ Generate a focused, novel risk analysis with heuristic scoring
def generate_novel_risk(topic_model, texts, risk_category):
    """
    Generate a single, highly specific novel risk scenario based on clustered topics.
    Uses heuristic-based scoring for probability and severity.
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
        
        # [Compelling Title for the Novel Risk]
        
        ## Executive Summary
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
        
        risk_description = response.choices[0].message.content
        
        # Now estimate probability and severity using heuristic methods
        probability_assessment = estimate_probability(risk_description, risk_category)
        severity_assessment = estimate_severity(risk_description, risk_category)
        
        # Store the assessments in session state for UI display
        st.session_state.probability_assessment = probability_assessment
        st.session_state.severity_assessment = severity_assessment
        
        # Return the original risk description - we'll display the detailed assessments separately
        return risk_description
        
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
st.write("Identify novel or underappreciated global threats that could emerge between 2025-2035")

# Add this after the title and before the risk category input
with st.expander("How This System Works", expanded=False):
    st.markdown("""
    ## System Overview: From News Data to Novel Risk Prediction
    
    This system identifies novel global threats by analyzing real-time news data through a multi-stage pipeline:
    
    ### 1. Data Collection
    - **Sources**: News API and GDELT Project (Global Database of Events, Language, and Tone)
    - **Scope**: Diverse global news sources across multiple time periods
    - **Query Strategy**: Uses AI to generate multiple search perspectives on your risk category
    
    ### 2. Data Processing
    - **Embedding**: Converts article text into numerical vectors using OpenAI's embedding model
    - **Clustering**: Groups related articles using BERTopic algorithm to identify patterns
    - **Topic Extraction**: Identifies key themes and weak signals across sources
    
    ### 3. Risk Analysis
    - **Scenario Generation**: GPT-4 analyzes the clustered data to identify novel, underappreciated risks
    - **Probability Assessment**: Calculates likelihood using five heuristic factors:
      - Trend Momentum, Barriers to Realization, Historical Analogues, Expert Discourse, and Emerging Warning Signs
    - **Severity Assessment**: Estimates potential impact in human lives using:
      - Direct Impact, Cascade Effects, Geographic Scope, Vulnerable Populations, and Temporal Persistence
    
    ### 4. Output
    - **Risk Scenario**: Detailed description of a novel risk that could emerge by 2025-2035
    - **Quantitative Assessment**: Probability score (0.0-1.0) and severity estimate (fatalities)
    - **Strategic Insights**: Early warning signals and potential response options
    - **Visual Representation**: AI-generated visualization of the risk scenario
    
    All analysis is performed on real-time data collected from the web - no pre-written examples or fictional scenarios are used.
    """)

# Also add a brief explanation at the top of the page
st.markdown("""
# Novel Global Threat Predictor

This system analyzes real-time news data to identify novel, underappreciated global threats that could emerge between 2025-2035, providing probability and severity scores based on heuristic analysis.
""")

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
        st.write("News API:", "✅ Connected" if news_api_keys else "❌ Missing")

# 3. Optimize the show_processing_messages function to be faster
def show_processing_messages(container, process_name="analysis", duration=3.0):
    """Display rotating messages to keep users engaged during processing."""
    messages = {
        "data": [
            "Scanning global news sources...",
            "Analyzing emerging risk patterns...",
            "Identifying weak signals in the data...",
            "Extracting insights from multiple sources...",
            "Connecting related concepts across articles..."
        ],
        "analysis": [
            "Generating risk scenarios based on data patterns...",
            "Calculating probability factors...",
            "Estimating potential severity and impact...",
            "Identifying early warning signals...",
            "Formulating strategic response options..."
        ],
        "image": [
            "Creating visual representation...",
            "Generating conceptual artwork...",
            "Visualizing risk scenario..."
        ]
    }
    
    message_list = messages.get(process_name, messages["analysis"])
    placeholder = container.empty()
    
    # Add a progress bar
    progress_bar = container.progress(0)
    
    # Calculate time per message to fit within duration
    steps = min(len(message_list), 5)  # Show fewer messages
    sleep_time = duration / steps
    
    # Display rotating messages with progress updates
    for i in range(steps):
        # Update progress bar
        progress_bar.progress((i+1)/steps)
        
        # Display a message from the list
        idx = min(i, len(message_list)-1)
        placeholder.info(message_list[idx])
        
        # Sleep briefly
        time.sleep(sleep_time)
    
    # Keep the last message visible
    placeholder.empty()

# Run analysis button - prominent and clear
if st.button("Predict Novel Threats", type="primary", use_container_width=True):
    # Create a container for the entire analysis process
    analysis_container = st.container()
    
    with analysis_container:
        # Add a pulsing header to indicate activity
        st.markdown("""
        <style>
        @keyframes pulse {
          0% { opacity: 0.8; }
          50% { opacity: 1; }
          100% { opacity: 0.8; }
        }
        .pulse {
          animation: pulse 2s infinite;
          padding: 15px;
          border-radius: 8px;
          background-color: #2e6fdb;
          color: white;
          margin-bottom: 25px;
          font-weight: bold;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          z-index: 1000;
          position: relative;
        }
        </style>
        <div class="pulse">
        <h3>Analysis in progress... Please wait while we process your request.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a status container
        status_container = st.empty()
        
        # Data collection phase
        with status_container.status("Phase 1: Collecting diverse data...") as status:
            # Show engaging messages during data collection
            message_container = st.empty()
            show_processing_messages(message_container, "data")
            
            # Actual data collection
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
            
            status.update(label="Phase 1: Data collection complete ✓", state="complete")
        
        # Topic modeling phase
        with status_container.status("Phase 2: Identifying patterns in the data...") as status:
            # Show engaging messages during topic modeling
            message_container = st.empty()
            show_processing_messages(message_container, "data")
            
            # Actual topic modeling
            topic_model = cluster_topics(articles, st.session_state.get('article_metadata'))
            
            # Check if topic modeling was successful
            if topic_model is None:
                st.warning("⚠️ Topic modeling encountered challenges. Using simplified analysis approach.")
                # Create a simple fallback topic model
                from collections import namedtuple
                SimpleTopicInfo = namedtuple('SimpleTopicInfo', ['Topic', 'Count', 'Name'])
                topic_model = [SimpleTopicInfo(0, len(articles), "All Articles")]
            
            status.update(label="Phase 2: Pattern identification complete ✓", state="complete")
        
        # Risk analysis phase
        with status_container.status("Phase 3: Generating risk analysis...") as status:
            # Show engaging messages during risk analysis
            message_container = st.empty()
            show_processing_messages(message_container, "analysis")
            
            # Actual risk analysis
            risk_analysis = generate_novel_risk(topic_model, articles, risk_category)
            
            status.update(label="Phase 3: Risk analysis complete ✓", state="complete")
        
        # Clear the pulsing header
        st.markdown("")
        
        # Display a success message
        st.success("Analysis complete! Scroll down to explore the results.")
    
    # Display results in a new container for better visual separation
    results_container = st.container()
    
    with results_container:
        st.markdown("## Results")
        st.markdown(risk_analysis)
        
        # Display detailed probability and severity assessments
        if 'probability_assessment' in st.session_state and 'severity_assessment' in st.session_state:
            prob = st.session_state.probability_assessment
            sev = st.session_state.severity_assessment
            
            # Create a visual summary of the risk assessment
            st.markdown("## 📊 Risk Assessment Summary")
            
            # Create a more visually appealing summary with columns and metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability gauge - with error handling
                prob_value = prob.get('overall_probability', 0.5)  # Default to 0.5 if missing
                st.metric("🛠️ Probability Score", f"{prob_value:.2f}")
                
                # Add a color-coded probability bar
                prob_color = "🟢" if prob_value < 0.3 else "🟠" if prob_value < 0.7 else "🔴"
                
                # Create a colored progress bar based on probability
                st.markdown(f"""
                <style>
                .prob-container {{
                    width: 100%;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .prob-bar {{
                    width: {prob_value * 100}%;
                    height: 24px;
                    background-color: {prob_color.replace('🟢', 'green').replace('🟠', 'orange').replace('🔴', 'red')};
                    border-radius: 5px;
                    text-align: center;
                    line-height: 24px;
                    color: white;
                    font-weight: bold;
                }}
                </style>
                <div class="prob-container">
                    <div class="prob-bar">{prob_value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Likelihood:** {prob_color} {prob_value:.2f}")
                st.caption(f"Confidence: {prob.get('confidence', 0.5):.2f}")
            
            with col2:
                # Severity gauge - with error handling
                fatality_estimate = sev.get('fatality_estimate', {'median': 10000, 'low': 1000, 'high': 100000})
                median_fatalities = fatality_estimate.get('median', 10000)
                st.metric("☠️ Estimated Fatalities", f"{median_fatalities:,}")
                
                # Create a logarithmic scale for the severity
                # 0-1000: green, 1000-100k: orange, >100k: red
                severity_scale = min(1.0, np.log10(max(1, median_fatalities)) / 6)  # Log scale up to 1M deaths
                sev_color = "🟢" if median_fatalities < 1000 else "🟠" if median_fatalities < 100000 else "🔴"
                
                st.markdown(f"""
                <style>
                .sev-container {{
                    width: 100%;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .sev-bar {{
                    width: {severity_scale * 100}%;
                    height: 24px;
                    background-color: {sev_color.replace('🟢', 'green').replace('🟠', 'orange').replace('🔴', 'red')};
                    border-radius: 5px;
                    text-align: center;
                    line-height: 24px;
                    color: white;
                    font-weight: bold;
                }}
                </style>
                <div class="sev-container">
                    <div class="sev-bar">{median_fatalities:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Impact Level:** {sev_color} {median_fatalities:,}")
                
                low = fatality_estimate.get('low', median_fatalities // 2)
                high = fatality_estimate.get('high', median_fatalities * 2)
                st.caption(f"Range: {low:,} - {high:,} | Confidence: {sev.get('confidence', 0.5):.2f}")
            
            # Add explanation of methodology
            with st.expander("📚 How Risk Scores Are Calculated", expanded=False):
                st.markdown("""
                ### 🔢 How Probability & Severity Are Estimated
                
                **1️⃣ Probability Score (0.0 - 1.0)**
                - **Trend Momentum**: Is this risk increasing in frequency?
                - **Barriers to Realization**: How hard is it to prevent?
                - **Historical Analogues**: Have similar risks happened before?
                - **Expert Discourse**: Are professionals discussing this?
                - **Emerging Warning Signs**: Are there early indicators?

                **2️⃣ Severity Score (Measured in Estimated Fatalities)**
                - **Direct Impact**: Expected deaths from the event.
                - **Cascade Effects**: Multiplier for secondary disasters.
                - **Geographic Scope**: Local, regional, or global impact.
                - **Vulnerable Populations**: How many are affected?
                - **Temporal Persistence**: Will it last long-term?
                
                🏆 **Final Score = Weighted Sum of These Factors**
                """)
            
            # Create tabs for detailed factor breakdowns
            prob_tab, sev_tab = st.tabs(["📈 Probability Factors", "💀 Severity Factors"])
            
            with prob_tab:
                st.write("### 🔍 Breakdown of Probability Score")
                # Display factor breakdown with error handling
                factors = prob.get('factors', {})
                if not factors:
                    st.info("Detailed probability factors not available")
                else:
                    for factor, details in factors.items():
                        score = details.get('score', 0) if isinstance(details, dict) else 0
                        justification = details.get('justification', 'No justification provided') if isinstance(details, dict) else 'No details available'
                        with st.expander(f"🧠 {factor.replace('_', ' ').title()} - {score:.2f}"):
                            st.write(justification)
            
            with sev_tab:
                st.write("### 🔍 Breakdown of Severity Score")
                # Display factor breakdown with error handling
                factors = sev.get('factors', {})
                if not factors:
                    st.info("Detailed severity factors not available")
                else:
                    # Display direct impact
                    direct_impact = factors.get('direct_impact', {})
                    with st.expander("💥 Direct Impact"):
                        estimate = direct_impact.get('estimate', 0) if isinstance(direct_impact, dict) else 0
                        justification = direct_impact.get('justification', 'No justification provided') if isinstance(direct_impact, dict) else 'No details available'
                        st.write(f"**Estimate**: {estimate:,} fatalities")
                        st.write(justification)
                    
                    # Display multipliers
                    multiplier_factors = ['cascade_effects', 'geographic_scope', 'vulnerable_populations', 'temporal_persistence']
                    for factor in multiplier_factors:
                        if factor in factors:
                            factor_data = factors[factor]
                            multiplier = factor_data.get('multiplier', 1.0) if isinstance(factor_data, dict) else 1.0
                            justification = factor_data.get('justification', 'No justification provided') if isinstance(factor_data, dict) else 'No details available'
                            with st.expander(f"💥 {factor.replace('_', ' ').title()} - {multiplier}x Multiplier"):
                                st.write(justification)

        # Generate and display an image
        with st.expander("🖼️ Visual Representation of This Risk", expanded=True):
            image_container = st.empty()
            show_processing_messages(image_container, "image")
            
            risk_title = risk_analysis.split('\n')[0].replace('#', '').strip() if '\n' in risk_analysis else risk_category
            image_url = generate_image(risk_title, risk_category)
            
            if image_url:
                st.image(image_url, caption=f"🔮 AI Visualization: {risk_title}")
            else:
                st.warning("⚠️ Could not generate an image for this risk.")

warnings.filterwarnings("ignore", category=RuntimeWarning)



