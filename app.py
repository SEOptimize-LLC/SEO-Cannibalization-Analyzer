"""
SEO Cannibalization Analysis App
A comprehensive tool for detecting and analyzing keyword, content, and topic cannibalization
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, quote
import json
import asyncio
import aiohttp
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

# AI Provider Imports
import openai
import anthropic
import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .highlight-red {
        background-color: #ffcccc;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .highlight-yellow {
        background-color: #ffffcc;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .highlight-green {
        background-color: #ccffcc;
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

class AIProvider:
    """Handles AI model interactions for analysis and recommendations"""
    
    def __init__(self):
        self.provider = None
        self.model = None
        self.client = None
    
    def setup(self, provider: str, api_key: str, model: str):
        """Initialize AI provider with API key"""
        self.provider = provider
        self.model = model
        
        try:
            if provider == "OpenAI":
                openai.api_key = api_key
                self.client = openai
            elif provider == "Anthropic":
                self.client = anthropic.Anthropic(api_key=api_key)
            elif provider == "Google":
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
        except Exception as e:
            st.error(f"Failed to initialize {provider}: {str(e)}")
            return False
        return True
    
    def generate_analysis(self, data: Dict, analysis_type: str) -> str:
        """Generate AI-powered analysis and recommendations"""
        
        prompts = {
            "keyword": f"""Analyze this keyword cannibalization data and provide actionable recommendations:
                Data: {json.dumps(data, indent=2)}
                
                Please provide:
                1. Severity assessment (High/Medium/Low)
                2. Top 3 priority fixes
                3. Specific consolidation recommendations
                4. Expected impact of fixes""",
            
            "content": f"""Analyze this content/SERP cannibalization data:
                Data: {json.dumps(data, indent=2)}
                
                Please provide:
                1. SERP competition assessment
                2. Content differentiation opportunities
                3. Consolidation vs differentiation recommendations
                4. Priority keywords to focus on""",
            
            "topic": f"""Analyze this topic/semantic cannibalization data:
                Data: {json.dumps(data, indent=2)}
                
                Please provide:
                1. Topic cluster recommendations
                2. Content hierarchy suggestions
                3. Pages to merge or differentiate
                4. Internal linking strategy"""
        }
        
        prompt = prompts.get(analysis_type, prompts["keyword"])
        
        try:
            if self.provider == "OpenAI":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif self.provider == "Anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == "Google":
                response = self.client.generate_content(prompt)
                return response.text
                
        except Exception as e:
            return f"AI Analysis Error: {str(e)}"

class KeywordCannibalizationAnalyzer:
    """Analyzes keyword overlap between URLs"""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict:
        """Analyze keyword overlap between landing pages"""
        
        # Group by landing page
        page_keywords = df.groupby('Landing Page')['Query'].apply(list).to_dict()
        
        # Calculate overlap matrix
        pages = list(page_keywords.keys())
        overlap_matrix = pd.DataFrame(index=pages, columns=pages)
        overlap_details = {}
        
        for i, page1 in enumerate(pages):
            for j, page2 in enumerate(pages):
                if i < j:  # Only calculate upper triangle
                    keywords1 = set(page_keywords[page1])
                    keywords2 = set(page_keywords[page2])
                    
                    overlap = keywords1.intersection(keywords2)
                    overlap_pct = len(overlap) / max(len(keywords1.union(keywords2)), 1) * 100
                    
                    overlap_matrix.loc[page1, page2] = overlap_pct
                    overlap_matrix.loc[page2, page1] = overlap_pct
                    
                    if overlap_pct > 10:  # Threshold for significant overlap
                        overlap_details[f"{page1[:50]}...||{page2[:50]}..."] = {
                            "overlap_percentage": round(overlap_pct, 2),
                            "shared_keywords": list(overlap)[:20],  # Limit to top 20
                            "total_shared": len(overlap),
                            "page1_total": len(keywords1),
                            "page2_total": len(keywords2)
                        }
                elif i == j:
                    overlap_matrix.loc[page1, page2] = 100
        
        # Find top cannibalization issues
        top_issues = sorted(overlap_details.items(), 
                          key=lambda x: x[1]['overlap_percentage'], 
                          reverse=True)[:10]
        
        return {
            "overlap_matrix": overlap_matrix,
            "top_issues": dict(top_issues),
            "total_pages_analyzed": len(pages),
            "pages_with_overlap": len(overlap_details)
        }

class ContentCannibalizationAnalyzer:
    """Analyzes SERP overlap for queries using Serper API"""
    
    @staticmethod
    async def fetch_serp(session, query: str, api_key: str, num_results: int = 10) -> Dict:
        """Fetch top organic results for a query using Serper API"""
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results,
            'gl': 'us',  # Country code
            'hl': 'en'   # Language
        }
        
        try:
            async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract organic results
                    results = {
                        'domains': [],
                        'urls': [],
                        'titles': [],
                        'positions': []
                    }
                    
                    organic_results = data.get('organic', [])
                    for i, result in enumerate(organic_results[:num_results], 1):
                        domain = urlparse(result.get('link', '')).netloc
                        if domain:
                            results['domains'].append(domain)
                            results['urls'].append(result.get('link', ''))
                            results['titles'].append(result.get('title', ''))
                            results['positions'].append(i)
                    
                    return results
                else:
                    st.warning(f"Serper API error for '{query}': Status {response.status}")
                    return {'domains': [], 'urls': [], 'titles': [], 'positions': []}
                    
        except Exception as e:
            st.warning(f"Failed to fetch SERP for '{query}': {str(e)}")
            return {'domains': [], 'urls': [], 'titles': [], 'positions': []}
    
    @staticmethod
    async def analyze_serp_overlap(queries: List[str], api_key: str, sample_size: int = 50) -> Dict:
        """Analyze SERP overlap between queries using Serper API"""
        
        if not api_key:
            return {"error": "Serper API key is required for SERP analysis"}
        
        # Sample queries if too many
        if len(queries) > sample_size:
            queries = pd.Series(queries).sample(n=sample_size, random_state=42).tolist()
        
        serp_data = {}
        detailed_results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for query in queries:
                tasks.append(ContentCannibalizationAnalyzer.fetch_serp(session, query, api_key))
            
            # Add rate limiting to avoid hitting API limits
            batch_size = 10
            all_results = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = await asyncio.gather(*batch)
                all_results.extend(batch_results)
                
                # Add delay between batches to respect rate limits
                if i + batch_size < len(tasks):
                    await asyncio.sleep(1)
            
            for query, serp_result in zip(queries, all_results):
                if serp_result and serp_result['domains']:
                    serp_data[query] = serp_result['domains']
                    detailed_results[query] = serp_result
        
        # Calculate overlap with enhanced metrics
        overlap_matrix = {}
        query_metrics = {}
        
        for i, (query1, serp1) in enumerate(serp_data.items()):
            for j, (query2, serp2) in enumerate(serp_data.items()):
                if i < j:
                    overlap = set(serp1).intersection(set(serp2))
                    overlap_pct = len(overlap) / max(len(set(serp1).union(set(serp2))), 1) * 100
                    
                    if overlap_pct > 30:  # Significant SERP overlap threshold
                        key = f"{query1}||{query2}"
                        
                        # Get position-weighted overlap score
                        position_score = 0
                        if query1 in detailed_results and query2 in detailed_results:
                            for domain in overlap:
                                # Find positions of shared domain in both SERPs
                                pos1 = serp1.index(domain) + 1 if domain in serp1 else 10
                                pos2 = serp2.index(domain) + 1 if domain in serp2 else 10
                                # Higher score for domains appearing higher in both SERPs
                                position_score += (11 - pos1) * (11 - pos2) / 100
                        
                        overlap_matrix[key] = {
                            "overlap_percentage": round(overlap_pct, 2),
                            "shared_domains": list(overlap),
                            "query1_results": len(serp1),
                            "query2_results": len(serp2),
                            "position_weighted_score": round(position_score, 2),
                            "competition_level": "High" if overlap_pct > 60 else "Medium" if overlap_pct > 40 else "Low"
                        }
            
            # Store individual query metrics
            if query1 in detailed_results:
                query_metrics[query1] = {
                    "total_results": len(serp_data.get(query1, [])),
                    "top_domain": serp_data[query1][0] if serp_data.get(query1) else None,
                    "unique_domains": len(set(serp_data.get(query1, [])))
                }
        
        # Sort by overlap percentage
        top_overlaps = dict(sorted(overlap_matrix.items(), 
                                 key=lambda x: x[1]['overlap_percentage'], 
                                 reverse=True)[:20])
        
        # Calculate competition clusters (queries that frequently overlap)
        competition_clusters = []
        processed_queries = set()
        
        for key, data in overlap_matrix.items():
            q1, q2 = key.split('||')
            if q1 not in processed_queries:
                cluster = {q1, q2}
                # Find other queries that overlap with both
                for other_key, other_data in overlap_matrix.items():
                    other_q1, other_q2 = other_key.split('||')
                    if other_q1 in cluster or other_q2 in cluster:
                        cluster.add(other_q1)
                        cluster.add(other_q2)
                
                if len(cluster) > 2:
                    competition_clusters.append(list(cluster))
                    processed_queries.update(cluster)
        
        return {
            "total_queries_analyzed": len(serp_data),
            "queries_with_overlap": len(overlap_matrix),
            "top_overlaps": top_overlaps,
            "average_overlap": np.mean([v['overlap_percentage'] for v in overlap_matrix.values()]) if overlap_matrix else 0,
            "query_metrics": query_metrics,
            "competition_clusters": competition_clusters[:5],  # Top 5 clusters
            "api_credits_used": len(serp_data)  # Track API usage
        }

class TopicCannibalizationAnalyzer:
    """Analyzes semantic similarity between pages"""
    
    @staticmethod
    def parse_embeddings(embeddings_str: str) -> np.ndarray:
        """Parse embedding string to numpy array"""
        try:
            # Try to parse as JSON first
            embeddings = json.loads(embeddings_str)
        except:
            try:
                # Try to parse as Python literal
                embeddings = ast.literal_eval(embeddings_str)
            except:
                # If all else fails, try to extract numbers
                import re
                numbers = re.findall(r'-?\d+\.?\d*', embeddings_str)
                embeddings = [float(n) for n in numbers]
        
        return np.array(embeddings)
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict:
        """Analyze semantic similarity between pages"""
        
        # Parse embeddings
        embeddings_list = []
        valid_pages = []
        
        for idx, row in df.iterrows():
            try:
                embedding = TopicCannibalizationAnalyzer.parse_embeddings(
                    row['(ChatGPT) Extract embeddings from page content 1']
                )
                embeddings_list.append(embedding)
                valid_pages.append(row['Address'])
            except Exception as e:
                st.warning(f"Failed to parse embedding for {row['Address']}: {str(e)}")
        
        if len(embeddings_list) < 2:
            return {"error": "Not enough valid embeddings to analyze"}
        
        # Calculate similarity matrix
        embeddings_matrix = np.vstack(embeddings_list)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find high similarity pairs
        high_similarity_pairs = []
        
        for i in range(len(valid_pages)):
            for j in range(i + 1, len(valid_pages)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.8:  # High similarity threshold
                    high_similarity_pairs.append({
                        "page1": valid_pages[i],
                        "page2": valid_pages[j],
                        "similarity": round(float(similarity), 3)
                    })
        
        # Sort by similarity
        high_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Create similarity DataFrame for visualization
        sim_df = pd.DataFrame(similarity_matrix, 
                            index=valid_pages, 
                            columns=valid_pages)
        
        return {
            "similarity_matrix": sim_df,
            "high_similarity_pairs": high_similarity_pairs[:20],  # Top 20
            "total_pages": len(valid_pages),
            "pages_with_high_similarity": len(high_similarity_pairs),
            "average_similarity": float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        }

def generate_comprehensive_report(keyword_results: Dict, content_results: Dict, 
                                 topic_results: Dict, ai_provider: AIProvider) -> str:
    """Generate comprehensive analysis report"""
    
    report = f"""
# SEO Cannibalization Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### 1. Keyword Cannibalization
- **Pages Analyzed:** {keyword_results.get('total_pages_analyzed', 0)}
- **Pages with Overlap:** {keyword_results.get('pages_with_overlap', 0)}
- **Severity Level:** {'High' if keyword_results.get('pages_with_overlap', 0) > 10 else 'Medium' if keyword_results.get('pages_with_overlap', 0) > 5 else 'Low'}

### 2. Content/SERP Cannibalization
- **Queries Analyzed:** {content_results.get('total_queries_analyzed', 0)}
- **Queries with SERP Overlap:** {content_results.get('queries_with_overlap', 0)}
- **Average SERP Overlap:** {content_results.get('average_overlap', 0):.1f}%

### 3. Topic/Semantic Cannibalization
- **Pages Analyzed:** {topic_results.get('total_pages', 0)}
- **High Similarity Pairs:** {topic_results.get('pages_with_high_similarity', 0)}
- **Average Similarity Score:** {topic_results.get('average_similarity', 0):.3f}

## Detailed Findings

### Keyword Cannibalization Issues
"""
    
    # Add top keyword overlaps
    if keyword_results.get('top_issues'):
        for pages, data in list(keyword_results['top_issues'].items())[:5]:
            page1, page2 = pages.split('||')
            report += f"""
**Pages:** {page1} vs {page2}
- Overlap: {data['overlap_percentage']}%
- Shared Keywords: {data['total_shared']}
- Sample Keywords: {', '.join(data['shared_keywords'][:5])}
"""
    
    # Add AI recommendations if available
    if ai_provider.client:
        report += "\n## AI-Powered Recommendations\n"
        
        # Keyword recommendations
        if keyword_results.get('top_issues'):
            report += "\n### Keyword Cannibalization Fixes\n"
            report += ai_provider.generate_analysis(
                {"top_issues": list(keyword_results['top_issues'].items())[:3]}, 
                "keyword"
            )
        
        # Content recommendations
        if content_results.get('top_overlaps'):
            report += "\n### Content Cannibalization Fixes\n"
            report += ai_provider.generate_analysis(
                {"top_overlaps": list(content_results['top_overlaps'].items())[:3]}, 
                "content"
            )
        
        # Topic recommendations
        if topic_results.get('high_similarity_pairs'):
            report += "\n### Topic Cannibalization Fixes\n"
            report += ai_provider.generate_analysis(
                {"high_similarity_pairs": topic_results['high_similarity_pairs'][:3]}, 
                "topic"
            )
    
    report += """

## Implementation Roadmap

### Week 1-2: Quick Wins
1. Implement 301 redirects for pages with >70% keyword overlap
2. Update internal links to point to primary pages
3. Add canonical tags where content must remain

### Week 3-4: Content Optimization
1. Merge high-similarity content (>0.9 similarity score)
2. Differentiate content for different search intents
3. Create clear topic clusters

### Week 5-8: Monitoring & Refinement
1. Track ranking improvements
2. Monitor traffic changes
3. Adjust strategy based on results

## Expected Impact
- **Traffic Increase:** 25-110% based on severity
- **Ranking Improvement:** 2-5 positions for cannibalized keywords
- **CTR Improvement:** 10-30% for consolidated pages
"""
    
    return report

def main():
    st.title("🔍 SEO Cannibalization Analyzer")
    st.markdown("Comprehensive analysis of keyword, content, and topic cannibalization")
    
    # Initialize AI provider
    ai_provider = AIProvider()
    
    # Check for API keys in secrets
    try:
        serper_api_key_secret = st.secrets.get("api_keys", {}).get("serper_key", "")
    except:
        serper_api_key_secret = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys Section
        st.subheader("🔑 API Keys")
        
        # Serper API Key (Required for SERP Analysis)
        if serper_api_key_secret:
            st.success("✅ Serper API loaded from secrets")
            serper_api_key = serper_api_key_secret
            st.caption("Override with manual input below if needed")
            
            # Allow manual override
            manual_serper_key = st.text_input(
                "Serper API Key (Override)",
                type="password",
                help="Leave blank to use key from secrets"
            )
            if manual_serper_key:
                serper_api_key = manual_serper_key
                st.info("Using manually entered Serper key")
        else:
            serper_api_key = st.text_input(
                "Serper API Key (Required)",
                type="password",
                help="Get your API key from serper.dev - Required for SERP overlap analysis"
            )
            
            if serper_api_key:
                st.success("✅ Serper API configured")
            else:
                st.warning("⚠️ Serper API key required for content analysis")
        
        st.divider()
        
        # AI Provider Selection
        st.subheader("🤖 AI Provider Settings")
        
        # Check for AI keys in secrets
        try:
            openai_secret = st.secrets.get("api_keys", {}).get("openai_key", "")
            anthropic_secret = st.secrets.get("api_keys", {}).get("anthropic_key", "")
            google_secret = st.secrets.get("api_keys", {}).get("google_key", "")
            
            # Auto-detect available AI provider from secrets
            available_providers = ["None"]
            if openai_secret:
                available_providers.append("OpenAI")
            if anthropic_secret:
                available_providers.append("Anthropic")
            if google_secret:
                available_providers.append("Google")
            
            if len(available_providers) > 1:
                st.info(f"Found {len(available_providers)-1} AI provider(s) in secrets")
        except:
            openai_secret = anthropic_secret = google_secret = ""
            available_providers = ["None", "OpenAI", "Anthropic", "Google"]
        
        ai_choice = st.selectbox(
            "Select AI Provider",
            available_providers if len(available_providers) > 1 else ["None", "OpenAI", "Anthropic", "Google"]
        )
        
        if ai_choice != "None":
            # Check if API key exists in secrets
            api_key_from_secret = ""
            if ai_choice == "OpenAI":
                api_key_from_secret = openai_secret
                model_options = ["gpt-4", "gpt-3.5-turbo"]
            elif ai_choice == "Anthropic":
                api_key_from_secret = anthropic_secret
                model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
            elif ai_choice == "Google":
                api_key_from_secret = google_secret
                model_options = ["gemini-pro"]
            
            if api_key_from_secret:
                st.success(f"✅ {ai_choice} API loaded from secrets")
                api_key = api_key_from_secret
                st.caption("Override with manual input below if needed")
                
                # Allow manual override
                manual_api_key = st.text_input(
                    f"{ai_choice} API Key (Override)",
                    type="password",
                    help="Leave blank to use key from secrets"
                )
                if manual_api_key:
                    api_key = manual_api_key
                    st.info("Using manually entered API key")
            else:
                api_key = st.text_input(
                    f"{ai_choice} API Key",
                    type="password",
                    help="Enter your API key for AI-powered recommendations"
                )
            
            model = st.selectbox("Model", model_options)
            
            if api_key and st.button("Initialize AI"):
                if ai_provider.setup(ai_choice, api_key, model):
                    st.success(f"✅ {ai_choice} initialized successfully!")
                else:
                    st.error("Failed to initialize AI provider")
        
        st.divider()
        
        # Analysis Settings
        st.subheader("Analysis Settings")
        
        keyword_threshold = st.slider(
            "Keyword Overlap Threshold (%)",
            min_value=5, max_value=50, value=10,
            help="Minimum overlap percentage to flag as cannibalization"
        )
        
        serp_sample_size = st.number_input(
            "SERP Analysis Sample Size",
            min_value=10, max_value=100, value=30,
            help="Number of queries to analyze for SERP overlap"
        )
        
        similarity_threshold = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05,
            help="Minimum similarity score to flag as topic cannibalization"
        )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload", "🔤 Keyword Analysis", 
                                       "📑 Content Analysis", "🧠 Topic Analysis"])
    
    with tab1:
        st.header("Upload Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("GSC Performance Report")
            gsc_file = st.file_uploader(
                "Upload GSC Report CSV",
                type=['csv'],
                help="Export from Google Search Console: Performance > Export"
            )
            
            if gsc_file:
                try:
                    gsc_df = pd.read_csv(gsc_file)
                    st.success(f"✅ Loaded {len(gsc_df)} rows")
                    st.dataframe(gsc_df.head(), use_container_width=True)
                    st.session_state['gsc_df'] = gsc_df
                except Exception as e:
                    st.error(f"Error loading GSC file: {str(e)}")
        
        with col2:
            st.subheader("Embeddings File")
            embeddings_file = st.file_uploader(
                "Upload Embeddings CSV",
                type=['csv'],
                help="CSV with URL and embedding columns"
            )
            
            if embeddings_file:
                try:
                    embeddings_df = pd.read_csv(embeddings_file)
                    st.success(f"✅ Loaded {len(embeddings_df)} pages")
                    st.dataframe(embeddings_df.head(), use_container_width=True)
                    st.session_state['embeddings_df'] = embeddings_df
                except Exception as e:
                    st.error(f"Error loading embeddings file: {str(e)}")
    
    with tab2:
        st.header("🔤 Keyword Cannibalization Analysis")
        
        if 'gsc_df' in st.session_state:
            if st.button("Analyze Keyword Overlap", type="primary", key="keyword_btn"):
                with st.spinner("Analyzing keyword overlap..."):
                    results = KeywordCannibalizationAnalyzer.analyze(st.session_state['gsc_df'])
                    st.session_state['keyword_results'] = results
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Pages", results['total_pages_analyzed'])
                    with col2:
                        st.metric("Pages with Overlap", results['pages_with_overlap'])
                    with col3:
                        severity = "High" if results['pages_with_overlap'] > 10 else "Medium" if results['pages_with_overlap'] > 5 else "Low"
                        st.metric("Severity", severity)
                    
                    # Display top issues
                    st.subheader("Top Cannibalization Issues")
                    if results['top_issues']:
                        for pages, data in list(results['top_issues'].items())[:5]:
                            with st.expander(f"📍 {pages.replace('||', ' vs ')[:100]}..."):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Overlap %", f"{data['overlap_percentage']}%")
                                    st.metric("Shared Keywords", data['total_shared'])
                                with col2:
                                    st.write("**Sample Shared Keywords:**")
                                    st.write(", ".join(data['shared_keywords'][:10]))
                    
                    # Heatmap visualization
                    if len(results['overlap_matrix']) < 20:  # Only show if not too many pages
                        st.subheader("Overlap Heatmap")
                        fig = px.imshow(
                            results['overlap_matrix'].fillna(0).values,
                            labels=dict(x="Pages", y="Pages", color="Overlap %"),
                            x=results['overlap_matrix'].columns[:20],
                            y=results['overlap_matrix'].index[:20],
                            color_continuous_scale="Reds",
                            aspect="auto"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📤 Please upload a GSC report in the Data Upload tab")
    
    with tab3:
        st.header("📑 Content/SERP Cannibalization Analysis")
        
        if 'gsc_df' in st.session_state:
            if not serper_api_key:
                st.error("🔑 Please enter your Serper API key in the sidebar to run SERP analysis")
                st.info("Get your API key at [serper.dev](https://serper.dev) - Free tier includes 2,500 queries/month")
                st.code("""
# Or add to Streamlit secrets:
[api_keys]
serper_key = "your-serper-api-key"
                """)
            elif st.button("Analyze SERP Overlap", type="primary", key="serp_btn"):
                with st.spinner(f"Fetching SERPs for top {serp_sample_size} queries using Serper API..."):
                    # Get unique queries
                    queries = st.session_state['gsc_df']['Query'].unique()[:serp_sample_size]
                    
                    # Run async analysis with Serper API
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        ContentCannibalizationAnalyzer.analyze_serp_overlap(
                            queries.tolist(), 
                            serper_api_key,
                            serp_sample_size
                        )
                    )
                    st.session_state['content_results'] = results
                    
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Queries Analyzed", results['total_queries_analyzed'])
                        with col2:
                            st.metric("Overlapping Queries", results['queries_with_overlap'])
                        with col3:
                            st.metric("Avg SERP Overlap", f"{results['average_overlap']:.1f}%")
                        with col4:
                            st.metric("API Credits Used", results.get('api_credits_used', 0))
                        
                        # Display competition clusters if found
                        if results.get('competition_clusters'):
                            st.subheader("🎯 Competition Clusters")
                            st.info("Groups of queries competing for similar SERP positions")
                            for i, cluster in enumerate(results['competition_clusters'], 1):
                                with st.expander(f"Cluster {i} ({len(cluster)} queries)"):
                                    st.write("**Competing Queries:**")
                                    for query in cluster[:10]:  # Show max 10 queries per cluster
                                        st.write(f"• {query}")
                        
                        # Display top overlaps
                        st.subheader("Top SERP Overlaps")
                        if results['top_overlaps']:
                            for query_pair, data in list(results['top_overlaps'].items())[:10]:
                                q1, q2 = query_pair.split('||')
                                with st.expander(f"🔍 '{q1}' vs '{q2}'"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("SERP Overlap", f"{data['overlap_percentage']}%")
                                    with col2:
                                        st.metric("Competition", data.get('competition_level', 'Medium'))
                                    with col3:
                                        st.metric("Position Score", data.get('position_weighted_score', 0))
                                    
                                    st.write("**Shared Domains:**")
                                    for domain in data['shared_domains'][:5]:
                                        st.write(f"• {domain}")
                        
                        # Query metrics summary
                        if results.get('query_metrics'):
                            st.subheader("📊 Query Performance Insights")
                            metrics_df = pd.DataFrame(results['query_metrics']).T
                            st.dataframe(metrics_df.head(20), use_container_width=True)
        else:
            st.info("📤 Please upload a GSC report in the Data Upload tab")
    
    with tab4:
        st.header("🧠 Topic/Semantic Cannibalization Analysis")
        
        if 'embeddings_df' in st.session_state:
            if st.button("Analyze Semantic Similarity", type="primary", key="topic_btn"):
                with st.spinner("Calculating semantic similarities..."):
                    results = TopicCannibalizationAnalyzer.analyze(st.session_state['embeddings_df'])
                    st.session_state['topic_results'] = results
                    
                    if 'error' not in results:
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pages", results['total_pages'])
                        with col2:
                            st.metric("High Similarity Pairs", results['pages_with_high_similarity'])
                        with col3:
                            st.metric("Avg Similarity", f"{results['average_similarity']:.3f}")
                        
                        # Display top similar pairs
                        st.subheader("Highly Similar Page Pairs")
                        if results['high_similarity_pairs']:
                            for pair in results['high_similarity_pairs'][:10]:
                                with st.expander(f"📄 Similarity: {pair['similarity']:.3f}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Page 1:**")
                                        st.code(pair['page1'])
                                    with col2:
                                        st.write("**Page 2:**")
                                        st.code(pair['page2'])
                        
                        # Similarity distribution
                        st.subheader("Similarity Distribution")
                        sim_values = results['similarity_matrix'].values[np.triu_indices_from(results['similarity_matrix'].values, k=1)]
                        fig = px.histogram(
                            x=sim_values,
                            nbins=30,
                            title="Distribution of Semantic Similarities",
                            labels={'x': 'Similarity Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(results['error'])
        else:
            st.info("📤 Please upload an embeddings file in the Data Upload tab")
    
    # Generate Report Section
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📋 Generate Comprehensive Report", type="primary", use_container_width=True):
            if any(key in st.session_state for key in ['keyword_results', 'content_results', 'topic_results']):
                with st.spinner("Generating comprehensive report..."):
                    report = generate_comprehensive_report(
                        st.session_state.get('keyword_results', {}),
                        st.session_state.get('content_results', {}),
                        st.session_state.get('topic_results', {}),
                        ai_provider
                    )
                    
                    st.markdown("### 📊 Analysis Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"seo_cannibalization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Please run at least one analysis before generating the report")

if __name__ == "__main__":
    main()
