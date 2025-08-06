"""
SEO Cannibalization Analysis App
Improved version with URL normalization, click-based prioritization, and enhanced SERP analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import ast
from typing import Dict, List, Tuple, Optional
import time
import io
import re

# AI Provider Imports (optional)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
    .severity-critical {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-high {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-medium {
        background-color: #ffbb33;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .severity-low {
        background-color: #00C851;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# URL NORMALIZATION AND PAGE TYPE DETECTION FUNCTIONS
# ============================================================================

def detect_page_type(url: str) -> str:
    """
    Detect the type of page based on URL patterns and keywords
    Returns: 'blog', 'service', 'product', 'legal', 'about', 'home', or 'other'
    """
    url_lower = url.lower()
    
    # Blog patterns
    blog_patterns = ['/blog/', '/article/', '/post/', '/news/', '/insights/', '/resources/']
    if any(pattern in url_lower for pattern in blog_patterns):
        return 'blog'
    
    # Service/solution patterns
    service_patterns = ['/services/', '/solutions/', '/what-we-do/', '/offerings/']
    if any(pattern in url_lower for pattern in service_patterns):
        return 'service'
    
    # Product patterns
    product_patterns = ['/product/', '/products/', '/shop/', '/store/', '/item/']
    if any(pattern in url_lower for pattern in product_patterns):
        return 'product'
    
    # Legal/policy patterns
    legal_patterns = ['/privacy', '/terms', '/legal/', '/policy', '/disclaimer', '/cookie']
    if any(pattern in url_lower for pattern in legal_patterns):
        return 'legal'
    
    # About/company patterns
    about_patterns = ['/about', '/team', '/company', '/who-we-are', '/our-story', '/mission', '/values']
    if any(pattern in url_lower for pattern in about_patterns):
        return 'about'
    
    # Home page
    if url_lower.endswith('/') and url_lower.count('/') <= 3:
        return 'home'
    
    return 'other'

def normalize_url(url: str) -> str:
    """
    Normalize URL by removing fragments, sorting query parameters, and handling trailing slashes
    This ensures URLs with different fragments or parameter orders are treated as the same page
    """
    try:
        # Parse the URL
        parsed = urlparse(url.lower().strip())
        
        # Remove fragment (everything after #)
        parsed = parsed._replace(fragment='')
        
        # Parse and sort query parameters
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        
        # Remove common tracking parameters that don't affect content
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 
                           'utm_content', 'fbclid', 'gclid', 'msclkid', '_ga']
        for param in tracking_params:
            query_params.pop(param, None)
        
        # Sort parameters for consistent ordering
        sorted_query = urlencode(sorted(query_params.items()), doseq=True)
        parsed = parsed._replace(query=sorted_query)
        
        # Remove trailing slash from path (except for root)
        path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
        parsed = parsed._replace(path=path)
        
        # Reconstruct URL
        normalized = urlunparse(parsed)
        
        # Remove default ports
        normalized = normalized.replace(':80/', '/').replace(':443/', '/')
        
        return normalized
    except:
        # If normalization fails, return original URL
        return url.lower().strip()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_gsc_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate GSC report format"""
    required_columns = ['Query', 'Landing Page', 'Clicks', 'Impressions']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    try:
        df['Clicks'] = pd.to_numeric(df['Clicks'])
        df['Impressions'] = pd.to_numeric(df['Impressions'])
    except:
        return False, "Clicks and Impressions must be numeric"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # Remove any rows with null URLs
    null_urls = df['Landing Page'].isna().sum()
    if null_urls > 0:
        df = df.dropna(subset=['Landing Page'])
    
    # Normalize URLs
    df['Normalized_URL'] = df['Landing Page'].apply(normalize_url)
    
    return True, "Data validation successful"

def validate_embeddings_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate embeddings file format"""
    if 'Address' not in df.columns:
        return False, "Missing 'Address' column"
    
    embeddings_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if not embeddings_cols:
        return False, "No embeddings column found"
    
    # Normalize URLs in embeddings data
    df['Normalized_URL'] = df['Address'].apply(normalize_url)
    
    return True, "Embeddings data validation successful"

def calculate_severity(pages_affected: int, overlap_percentage: float, issue_type: str, total_clicks: int = 0) -> Dict:
    """Calculate severity and impact for different cannibalization types"""
    
    # Adjust severity based on click impact
    click_multiplier = 1.0
    if total_clicks > 1000:
        click_multiplier = 1.5
    elif total_clicks > 100:
        click_multiplier = 1.2
    
    if issue_type == "keyword":
        if (pages_affected > 20 or overlap_percentage > 50) and click_multiplier > 1:
            severity = "CRITICAL"
            impact = "50-110% potential traffic loss"
            priority = "Immediate action required"
            color = "🔴"
        elif pages_affected > 10 or overlap_percentage > 30:
            severity = "HIGH"
            impact = "25-50% potential traffic loss"
            priority = "Address within 1 week"
            color = "🟠"
        elif pages_affected > 5 or overlap_percentage > 15:
            severity = "MEDIUM"
            impact = "10-25% potential traffic loss"
            priority = "Address within 2-4 weeks"
            color = "🟡"
        else:
            severity = "LOW"
            impact = "5-10% potential traffic loss"
            priority = "Monitor and address if resources available"
            color = "🟢"
    
    elif issue_type == "content":
        if overlap_percentage > 65: # Updated threshold per user request
            severity = "CRITICAL"
            impact = "Severe SERP competition"
            priority = "Immediate content differentiation needed"
            color = "🔴"
        elif overlap_percentage > 40:
            severity = "HIGH"
            impact = "Significant SERP competition"
            priority = "Content optimization within 1 week"
            color = "🟠"
        elif overlap_percentage > 25:
            severity = "MEDIUM"
            impact = "Moderate SERP competition"
            priority = "Review content strategy"
            color = "🟡"
        else:
            severity = "LOW"
            impact = "Minor SERP competition"
            priority = "Monitor performance"
            color = "🟢"
    
    else: # topic
        if pages_affected > 10 or overlap_percentage > 0.9:
            severity = "CRITICAL"
            impact = "Severe topic dilution"
            priority = "Immediate consolidation needed"
            color = "🔴"
        elif pages_affected > 5 or overlap_percentage > 0.8:
            severity = "HIGH"
            impact = "Significant topic overlap"
            priority = "Create topic clusters within 1 week"
            color = "🟠"
        elif pages_affected > 2 or overlap_percentage > 0.7:
            severity = "MEDIUM"
            impact = "Moderate topic overlap"
            priority = "Review content architecture"
            color = "🟡"
        else:
            severity = "LOW"
            impact = "Minor topic overlap"
            priority = "Consider future optimization"
            color = "🟢"
    
    return {
        "severity": severity,
        "impact": impact,
        "priority": priority,
        "color": color
    }

# ============================================================================
# AI PROVIDER CLASS
# ============================================================================

class AIProvider:
    """Handles AI model interactions for analysis and recommendations"""
    
    def __init__(self):
        self.provider = None
        self.model = None
        self.client = None
    
    def setup(self, provider: str, api_key: str, model: str) -> bool:
        """Initialize AI provider with API key"""
        self.provider = provider
        self.model = model
        
        try:
            if provider == "OpenAI" and openai:
                openai.api_key = api_key
                self.client = openai
            elif provider == "Anthropic" and anthropic:
                self.client = anthropic.Anthropic(api_key=api_key)
            elif provider == "Google" and genai:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
            else:
                return False
        except Exception as e:
            st.error(f"Failed to initialize {provider}: {str(e)}")
            return False
        return True
    
    def generate_detailed_analysis(self, keyword_data: Dict, content_data: Dict, topic_data: Dict) -> str:
        """Generate comprehensive AI analysis with specific recommendations"""
        
        prompt = f"""You are an expert SEO consultant analyzing cannibalization issues. 
        
Based on this data, provide SPECIFIC, ACTIONABLE recommendations:

KEYWORD CANNIBALIZATION DATA:
- Pages with issues: {keyword_data.get('pages_with_cannibalization', 0)}
- Total overlap pairs: {keyword_data.get('total_overlap_pairs', 0)}
- Average overlap: {keyword_data.get('average_overlap', 0):.1f}%
- Total clicks affected: {keyword_data.get('total_clicks_affected', 0)}
- Top issues: {json.dumps(list(keyword_data.get('top_issues', {}).items())[:3], indent=2)}

CONTENT/SERP CANNIBALIZATION DATA:
- Queries analyzed: {content_data.get('total_queries_analyzed', 0)}
- Queries with overlap: {content_data.get('queries_with_overlap', 0)}
- Average SERP overlap: {content_data.get('average_overlap', 0):.1f}%
- Same URL pairs correctly skipped: {content_data.get('same_url_pairs_skipped', 0)}

TOPIC CANNIBALIZATION DATA:
- Pages analyzed: {topic_data.get('total_pages', 0)}
- High similarity pairs: {topic_data.get('pages_with_high_similarity', 0)}
- Average similarity: {topic_data.get('average_similarity', 0):.3f}

Provide:
1. TOP 3 CRITICAL ISSUES to fix immediately with specific page examples
2. CONSOLIDATION STRATEGY: Which specific pages to merge/redirect
3. CONTENT DIFFERENTIATION: How to make competing pages unique
4. ESTIMATED IMPACT: Traffic increase potential if fixed
5. 30-DAY ACTION PLAN: Week-by-week implementation roadmap
6. QUICK WINS: 3 things that can be done today

Be specific with page URLs and keywords when making recommendations."""

        try:
            if self.provider == "OpenAI" and openai:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif self.provider == "Anthropic" and anthropic:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == "Google" and genai:
                response = self.client.generate_content(prompt)
                return response.text
                
        except Exception as e:
            return f"Could not generate AI analysis: {str(e)}"
        
        return ""

# ============================================================================
# IMPROVED ANALYZER CLASSES
# ============================================================================

class KeywordCannibalizationAnalyzer:
    """Analyzes keyword overlap between URLs with click prioritization"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, threshold: float = 10, min_clicks: int = 1, branded_terms: List[str] = None) -> Dict:
        """Analyze keyword overlap between landing pages with click-based prioritization"""
        
        # Use normalized URLs for comparison
        df_clean = df.drop_duplicates(subset=['Normalized_URL', 'Query'])
        
        # Filter out branded terms if provided
        if branded_terms:
            # Create a mask for non-branded queries
            branded_mask = df_clean['Query'].str.lower().apply(
                lambda x: not any(brand in x for brand in branded_terms)
            )
            df_clean = df_clean[branded_mask]
            branded_queries_removed = (~branded_mask).sum()
        else:
            branded_queries_removed = 0
        
        # Aggregate clicks by normalized URL and query
        df_agg = df_clean.groupby(['Normalized_URL', 'Query']).agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Landing Page': 'first' # Keep original URL for display
        }).reset_index()
        
        # Filter by minimum clicks if specified
        if min_clicks > 0:
            df_filtered = df_agg[df_agg['Clicks'] >= min_clicks]
            if df_filtered.empty:
                df_filtered = df_agg # Fall back to all data if no keywords meet criteria
        else:
            df_filtered = df_agg
        
        # Calculate total clicks per keyword for prioritization
        keyword_clicks = df_filtered.groupby('Query')['Clicks'].sum().to_dict()
        
        # Group by normalized URL
        page_keywords = df_filtered.groupby('Normalized_URL').apply(
            lambda x: list(zip(x['Query'].tolist(), x['Clicks'].tolist()))
        ).to_dict()
        
        # Get unique pages
        pages = list(page_keywords.keys())
        overlap_details = {}
        pages_with_issues = set()
        total_clicks_affected = 0
        
        # Calculate overlap - comparing normalized URLs
        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):
                page1 = pages[i]
                page2 = pages[j]
                
                # Get keywords and clicks
                keywords1_with_clicks = page_keywords[page1]
                keywords2_with_clicks = page_keywords[page2]
                
                keywords1 = set([k for k, c in keywords1_with_clicks])
                keywords2 = set([k for k, c in keywords2_with_clicks])
                
                overlap = keywords1.intersection(keywords2)
                
                if len(overlap) > 0:
                    union = keywords1.union(keywords2)
                    overlap_pct = (len(overlap) / len(union)) * 100 if union else 0
                    
                    # Calculate total clicks for overlapping keywords
                    overlap_clicks = sum(keyword_clicks.get(kw, 0) for kw in overlap)
                    
                    if overlap_pct > threshold and overlap_clicks > 0:
                        pages_with_issues.add(page1)
                        pages_with_issues.add(page2)
                        total_clicks_affected += overlap_clicks
                        
                        # Get original URLs for display
                        original_url1 = df_filtered[df_filtered['Normalized_URL'] == page1]['Landing Page'].iloc[0]
                        original_url2 = df_filtered[df_filtered['Normalized_URL'] == page2]['Landing Page'].iloc[0]
                        
                        # Sort overlapping keywords by clicks
                        overlap_with_clicks = [(kw, keyword_clicks.get(kw, 0)) for kw in overlap]
                        overlap_with_clicks.sort(key=lambda x: x[1], reverse=True)
                        
                        key = f"{original_url1[:50]}...||{original_url2[:50]}..."
                        
                        overlap_details[key] = {
                            "page1_full": original_url1,
                            "page2_full": original_url2,
                            "normalized_page1": page1,
                            "normalized_page2": page2,
                            "overlap_percentage": round(overlap_pct, 2),
                            "shared_keywords": [kw for kw, _ in overlap_with_clicks[:20]],
                            "shared_keywords_with_clicks": overlap_with_clicks[:20],
                            "total_shared": len(overlap),
                            "total_clicks_affected": overlap_clicks,
                            "page1_total": len(keywords1),
                            "page2_total": len(keywords2),
                            "page1_unique": len(keywords1 - keywords2),
                            "page2_unique": len(keywords2 - keywords1)
                        }
        
        # Sort by total clicks affected (prioritize high-traffic overlaps)
        top_issues = sorted(overlap_details.items(), 
                            key=lambda x: x[1]['total_clicks_affected'], 
                            reverse=True)[:15]
        
        # Calculate severity
        avg_overlap = np.mean([d['overlap_percentage'] for d in overlap_details.values()]) if overlap_details else 0
        severity_info = calculate_severity(
            len(pages_with_issues), 
            avg_overlap, 
            "keyword",
            total_clicks_affected
        )
        
        return {
            "overlap_details": overlap_details,
            "top_issues": dict(top_issues),
            "total_pages_analyzed": len(pages),
            "pages_with_cannibalization": len(pages_with_issues),
            "total_overlap_pairs": len(overlap_details),
            "average_overlap": avg_overlap,
            "total_clicks_affected": total_clicks_affected,
            "severity_info": severity_info,
            "filtered_by_clicks": min_clicks > 0,
            "branded_queries_removed": branded_queries_removed
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
            'gl': 'us',
            'hl': 'en'
        }
        
        try:
            async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
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
                    return {'domains': [], 'urls': [], 'titles': [], 'positions': [], 'error': f'Status {response.status}'}
                    
        except Exception as e:
            return {'domains': [], 'urls': [], 'titles': [], 'positions': [], 'error': str(e)}
    
    @staticmethod
    async def analyze_serp_overlap(queries_df: pd.DataFrame, api_key: str, sample_size: int = 50, 
                                   progress_callback=None, min_clicks: int = 1, branded_terms: List[str] = None) -> Dict:
        """Analyze SERP overlap between queries using Serper API"""
        
        if not api_key:
            return {"error": "Serper API key is required for SERP analysis"}
        
        # Get the client's domain from the data
        client_domain = None
        if not queries_df.empty and 'Landing Page' in queries_df.columns:
            sample_url = queries_df['Landing Page'].iloc[0]
            client_domain = urlparse(sample_url).netloc.lower()
        
        # Filter out branded queries if provided
        if branded_terms:
            # Create a mask for non-branded queries
            branded_mask = queries_df['Query'].str.lower().apply(
                lambda x: not any(brand in x for brand in branded_terms)
            )
            queries_df = queries_df[branded_mask]
            branded_queries_removed = (~branded_mask).sum()
        else:
            branded_queries_removed = 0
        
        # Group by query to check which queries come from multiple URLs
        query_url_counts = queries_df.groupby('Query')['Landing Page'].nunique()
        
        # Sort queries by total clicks and filter
        query_clicks = queries_df.groupby('Query')['Clicks'].sum().sort_values(ascending=False)
        
        if min_clicks > 0:
            query_clicks = query_clicks[query_clicks >= min_clicks]
        
        # Get queries based on sample_size (0 means all)
        if sample_size == 0:
            queries = list(query_clicks.index)
        else:
            queries = list(query_clicks.index[:sample_size])
        
        if len(queries) < 2:
            return {"error": "Need at least 2 unique queries to analyze overlap"}
        
        serp_data = {}
        detailed_results = {}
        failed_queries = []
        
        async with aiohttp.ClientSession() as session:
            batch_size = 5
            
            for batch_start in range(0, len(queries), batch_size):
                batch_end = min(batch_start + batch_size, len(queries))
                batch = queries[batch_start:batch_end]
                
                tasks = [ContentCannibalizationAnalyzer.fetch_serp(session, q, api_key) for q in batch]
                batch_results = await asyncio.gather(*tasks)
                
                for query, result in zip(batch, batch_results):
                    if result and result.get('domains'):
                        serp_data[query] = result['domains']
                        detailed_results[query] = result
                    elif result.get('error'):
                        failed_queries.append(query)
                
                # Update progress
                if progress_callback:
                    progress = (batch_end / len(queries)) * 100
                    progress_callback(progress)
                
                # Rate limiting
                if batch_end < len(queries):
                    await asyncio.sleep(1)
        
        if not serp_data:
            return {"error": "No SERP data retrieved. Check your API key and queries."}
        
        # Calculate overlap
        overlap_matrix = {}
        query_metrics = {}
        queries_with_issues = set()
        same_url_skipped = 0
        
        serp_items = list(serp_data.items())
        for i in range(len(serp_items)):
            for j in range(i + 1, len(serp_items)):
                query1, serp1 = serp_items[i]
                query2, serp2 = serp_items[j]
                
                # Check if both queries come from the same URL
                query1_urls = set(queries_df[queries_df['Query'] == query1]['Landing Page'].unique())
                query2_urls = set(queries_df[queries_df['Query'] == query2]['Landing Page'].unique())
                
                # Skip if both queries are from the same URL(s)
                if query1_urls == query2_urls and len(query1_urls) == 1:
                    same_url_skipped += 1
                    continue
                
                overlap = set(serp1).intersection(set(serp2))
                union = set(serp1).union(set(serp2))
                overlap_pct = (len(overlap) / len(union)) * 100 if union else 0
                
                if overlap_pct > 65: # Updated threshold per user request
                    queries_with_issues.add(query1)
                    queries_with_issues.add(query2)
                    
                    key = f"{query1}||{query2}"
                    
                    # Get click data for both queries
                    q1_clicks = query_clicks.get(query1, 0)
                    q2_clicks = query_clicks.get(query2, 0)
                    
                    # Calculate position-weighted score
                    position_score = 0
                    if query1 in detailed_results and query2 in detailed_results:
                        for domain in overlap:
                            if domain in serp1 and domain in serp2:
                                pos1 = serp1.index(domain) + 1
                                pos2 = serp2.index(domain) + 1
                                position_score += (11 - pos1) * (11 - pos2) / 100
                    
                    # Build full SERP data for display
                    serp1_full = []
                    serp2_full = []
                    
                    if query1 in detailed_results:
                        for idx, (url, title) in enumerate(zip(detailed_results[query1]['urls'], 
                                                              detailed_results[query1]['titles'])):
                            domain = urlparse(url).netloc
                            is_client = domain.lower() == client_domain if client_domain else False
                            serp1_full.append({
                                'position': idx + 1,
                                'url': url,
                                'title': title,
                                'domain': domain,
                                'is_client': is_client
                            })
                    
                    if query2 in detailed_results:
                        for idx, (url, title) in enumerate(zip(detailed_results[query2]['urls'], 
                                                              detailed_results[query2]['titles'])):
                            domain = urlparse(url).netloc
                            is_client = domain.lower() == client_domain if client_domain else False
                            serp2_full.append({
                                'position': idx + 1,
                                'url': url,
                                'title': title,
                                'domain': domain,
                                'is_client': is_client
                            })
                    
                    overlap_matrix[key] = {
                        "query1": query1,
                        "query2": query2,
                        "query1_clicks": q1_clicks,
                        "query2_clicks": q2_clicks,
                        "total_clicks": q1_clicks + q2_clicks,
                        "overlap_percentage": round(overlap_pct, 2),
                        "total_shared": len(overlap),
                        "position_weighted_score": round(position_score, 2),
                        "competition_level": "High" if overlap_pct > 65 else "Medium" if overlap_pct > 40 else "Low",
                        "serp1_results": serp1_full,
                        "serp2_results": serp2_full,
                        "query1_client_urls": list(query1_urls),
                        "query2_client_urls": list(query2_urls)
                    }
        
        # Individual query metrics
        for query, serp in serp_data.items():
            query_metrics[query] = {
                "total_results": len(serp),
                "top_domain": serp[0] if serp else None,
                "unique_domains": len(set(serp)),
                "clicks": query_clicks.get(query, 0)
            }
        
        # Sort results by overlap percentage descending
        top_overlaps = dict(sorted(overlap_matrix.items(), 
                                   key=lambda x: x[1]['overlap_percentage'], 
                                   reverse=True)[:20])
        
        # Calculate severity
        avg_overlap = np.mean([v['overlap_percentage'] for v in overlap_matrix.values()]) if overlap_matrix else 0
        severity_info = calculate_severity(len(queries_with_issues), avg_overlap, "content")
        
        return {
            "total_queries_analyzed": len(serp_data),
            "total_queries_available": len(query_clicks),
            "queries_with_overlap": len(queries_with_issues),
            "total_overlap_pairs": len(overlap_matrix),
            "top_overlaps": top_overlaps,
            "average_overlap": avg_overlap,
            "query_metrics": query_metrics,
            "api_credits_used": len(serp_data),
            "failed_queries": failed_queries,
            "severity_info": severity_info,
            "selection_criteria": f"{'ALL' if sample_size == 0 else f'{sample_size}'} queries analyzed",
            "branded_queries_removed": branded_queries_removed,
            "client_domain": client_domain,
            "same_url_pairs_skipped": same_url_skipped
        }

class TopicCannibalizationAnalyzer:
    """Analyzes semantic similarity between pages with proper dimension handling"""
    
    @staticmethod
    def parse_embeddings(embeddings_str: str) -> np.ndarray:
        """Parse embedding string to numpy array"""
        try:
            # Try JSON first
            embeddings = json.loads(embeddings_str)
        except:
            try:
                # Try Python literal
                embeddings = ast.literal_eval(embeddings_str)
            except:
                # Extract numbers as last resort
                import re
                numbers = re.findall(r'-?\d+\.?\d*', embeddings_str)
                embeddings = [float(n) for n in numbers]
        
        return np.array(embeddings)
    
    @staticmethod
    def analyze(df: pd.DataFrame, threshold: float = 0.8) -> Dict:
        """Analyze semantic similarity between pages"""
        
        embeddings_list = []
        valid_pages = []
        normalized_urls = []
        
        # Find embeddings column
        embeddings_col = None
        for col in df.columns:
            if 'embedding' in col.lower():
                embeddings_col = col
                break
        
        if not embeddings_col:
            return {"error": "No embeddings column found in the uploaded file"}
        
        # Parse embeddings and check dimensions
        embedding_dims = []
        for idx, row in df.iterrows():
            try:
                embedding = TopicCannibalizationAnalyzer.parse_embeddings(row[embeddings_col])
                if len(embedding) > 0:
                    embedding_dims.append(len(embedding))
                    embeddings_list.append(embedding)
                    valid_pages.append(row['Address'])
                    normalized_urls.append(row['Normalized_URL'])
            except Exception as e:
                continue
        
        if len(embeddings_list) < 2:
            return {"error": f"Not enough valid embeddings. Found {len(embeddings_list)} valid embeddings, need at least 2."}
        
        # Check dimension consistency
        unique_dims = set(embedding_dims)
        if len(unique_dims) > 1:
            most_common_dim = max(set(embedding_dims), key=embedding_dims.count)
            
            # Filter to consistent dimensions
            filtered_embeddings = []
            filtered_pages = []
            filtered_normalized = []
            for emb, page, norm, dim in zip(embeddings_list, valid_pages, normalized_urls, embedding_dims):
                if dim == most_common_dim:
                    filtered_embeddings.append(emb)
                    filtered_pages.append(page)
                    filtered_normalized.append(norm)
            
            embeddings_list = filtered_embeddings
            valid_pages = filtered_pages
            normalized_urls = filtered_normalized
            
            if len(embeddings_list) < 2:
                return {"error": f"Inconsistent embedding dimensions. Found: {unique_dims}. Need at least 2 embeddings with same dimension."}
        
        try:
            # Calculate similarity matrix
            embeddings_matrix = np.vstack(embeddings_list)
            similarity_matrix = cosine_similarity(embeddings_matrix)
        except Exception as e:
            return {"error": f"Error calculating similarities: {str(e)}"}
        
        # Find high similarity pairs (excluding self-comparison)
        high_similarity_pairs = []
        pages_with_issues = set()
        
        for i in range(len(valid_pages)):
            for j in range(i + 1, len(valid_pages)):
                # Skip if same normalized URL
                if normalized_urls[i] == normalized_urls[j]:
                    continue
                    
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    pages_with_issues.add(valid_pages[i])
                    pages_with_issues.add(valid_pages[j])
                    
                    high_similarity_pairs.append({
                        "page1": valid_pages[i],
                        "page2": valid_pages[j],
                        "normalized_page1": normalized_urls[i],
                        "normalized_page2": normalized_urls[j],
                        "similarity": round(float(similarity), 3)
                    })
        
        # Sort by similarity
        high_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Create similarity DataFrame
        sim_df = pd.DataFrame(similarity_matrix, 
                              index=valid_pages, 
                              columns=valid_pages)
        
        # Calculate average (excluding diagonal)
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, 0)
        avg_similarity = similarity_matrix[mask].mean() if mask.any() else 0
        
        # Calculate severity
        severity_info = calculate_severity(len(pages_with_issues), avg_similarity, "topic")
        
        return {
            "similarity_matrix": sim_df,
            "high_similarity_pairs": high_similarity_pairs[:20],
            "total_pages": len(valid_pages),
            "pages_with_high_similarity": len(pages_with_issues),
            "total_similarity_pairs": len(high_similarity_pairs),
            "average_similarity": float(avg_similarity),
            "embedding_dimension": embeddings_matrix.shape[1] if len(embeddings_list) > 0 else 0,
            "severity_info": severity_info
        }

# ============================================================================
# ENHANCED REPORT GENERATION
# ============================================================================

def generate_comprehensive_report(keyword_results: Dict, content_results: Dict, 
                                  topic_results: Dict, ai_provider: AIProvider,
                                  ai_recommendations: Dict = None) -> str:
    """Generate comprehensive analysis report with real insights"""
    
    # Get severity info
    keyword_severity = keyword_results.get('severity_info', {})
    content_severity = content_results.get('severity_info', {})
    topic_severity = topic_results.get('severity_info', {})
    
    report = f"""# 🔍 SEO Cannibalization Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

### Overall Health Score
"""
    
    # Calculate overall score
    issues_found = (
        keyword_results.get('pages_with_cannibalization', 0) +
        content_results.get('queries_with_overlap', 0) +
        topic_results.get('pages_with_high_similarity', 0)
    )
    
    total_clicks_affected = keyword_results.get('total_clicks_affected', 0)
    
    if issues_found > 50 or total_clicks_affected > 1000:
        report += "### 🔴 CRITICAL - Immediate Action Required\n"
    elif issues_found > 20 or total_clicks_affected > 500:
        report += "### 🟠 HIGH - Significant Issues Found\n"
    elif issues_found > 10 or total_clicks_affected > 100:
        report += "### 🟡 MEDIUM - Moderate Issues Found\n"
    else:
        report += "### 🟢 LOW - Minor Issues Found\n"
    
    report += f"""
**Total Issues Identified:** {issues_found}
**Total Clicks Affected:** {total_clicks_affected:,}
**Estimated Traffic Impact:** {keyword_severity.get('impact', 'N/A')}

---

## 1️⃣ Keyword Cannibalization Analysis

{keyword_severity.get('color', '')} **Severity: {keyword_severity.get('severity', 'N/A')}**

### Key Metrics:
- **Pages Analyzed:** {keyword_results.get('total_pages_analyzed', 0)}
- **Pages with Cannibalization:** {keyword_results.get('pages_with_cannibalization', 0)}
- **Overlap Pairs Found:** {keyword_results.get('total_overlap_pairs', 0)}
- **Average Overlap:** {keyword_results.get('average_overlap', 0):.1f}%
- **Total Clicks Affected:** {keyword_results.get('total_clicks_affected', 0):,}

### Impact Assessment:
- {keyword_severity.get('impact', 'N/A')}
- {keyword_severity.get('priority', 'N/A')}
"""
    
    # Add top keyword issues with specific recommendations
    if keyword_results.get('top_issues'):
        report += "\n### Top 5 Critical Keyword Overlaps (Sorted by Click Impact):\n"
        for i, (pages, data) in enumerate(list(keyword_results['top_issues'].items())[:5], 1):
            report += f"""
**Issue #{i}: {data['overlap_percentage']}% Overlap - {data['total_clicks_affected']:,} Clicks Affected**
- **Pages Competing:** - Page 1: `{data.get('page1_full', pages.split('||')[0])[:100]}`
  - Page 2: `{data.get('page2_full', pages.split('||')[1])[:100]}`
- **Shared Keywords:** {data['total_shared']} keywords
- **Top Competing Keywords (by clicks):** """
            for kw, clicks in data.get('shared_keywords_with_clicks', [])[:5]:
                report += f"  - {kw} ({clicks:,} clicks)\n"
            
            report += f"- **Recommended Action:** {"Merge pages (>50% overlap)" if data['overlap_percentage'] > 50 else "Differentiate content (30-50% overlap)" if data['overlap_percentage'] > 30 else "Monitor performance"}\n"
    
    report += f"""
---

## 2️⃣ Content/SERP Cannibalization Analysis

{content_severity.get('color', '')} **Severity: {content_severity.get('severity', 'N/A')}**

### Key Metrics:
- **Queries Analyzed:** {content_results.get('total_queries_analyzed', 0)} out of {content_results.get('total_queries_available', 0)} total
- **Selection Criteria:** {content_results.get('selection_criteria', 'N/A')}
- **Queries with SERP Overlap (>65%):** {content_results.get('queries_with_overlap', 0)}
- **Average SERP Overlap:** {content_results.get('average_overlap', 0):.1f}%
- **API Credits Used:** {content_results.get('api_credits_used', 0)}

### Impact Assessment:
- {content_severity.get('impact', 'N/A')}
- {content_severity.get('priority', 'N/A')}
"""
    
    # Add top SERP overlaps
    if content_results.get('top_overlaps'):
        report += "\n### Top SERP Competitions (Sorted by Click Impact):\n"
        for query_pair, data in list(content_results.get('top_overlaps', {}).items())[:5]:
            report += f"""
**{data['competition_level']} Competition: {data['overlap_percentage']}% SERP Overlap - {data['total_clicks']:,} Total Clicks**
- Query 1: "{data.get('query1', query_pair.split('||')[0])}" ({data.get('query1_clicks', 0):,} clicks)
- Query 2: "{data.get('query2', query_pair.split('||')[1])}" ({data.get('query2_clicks', 0):,} clicks)
- Shared Domains: {data.get('total_shared', len(data.get('shared_domains', [])))}
- Action: {"Target different search intents" if data['overlap_percentage'] > 65 else "Optimize for featured snippets"}
"""
    
    report += f"""
---

## 3️⃣ Topic/Semantic Cannibalization Analysis

{topic_severity.get('color', '')} **Severity: {topic_severity.get('severity', 'N/A')}**

### Key Metrics:
- **Pages Analyzed:** {topic_results.get('total_pages', 0)}
- **Pages with High Similarity:** {topic_results.get('pages_with_high_similarity', 0)}
- **Similarity Pairs Found:** {topic_results.get('total_similarity_pairs', 0)}
- **Average Similarity Score:** {topic_results.get('average_similarity', 0):.3f}
- **Embedding Dimension:** {topic_results.get('embedding_dimension', 0)}

### Impact Assessment:
- {topic_severity.get('impact', 'N/A')}
- {topic_severity.get('priority', 'N/A')}
"""
    
    # Add top similarity pairs
    if topic_results.get('high_similarity_pairs'):
        report += "\n### Highly Similar Pages (Merge Candidates):\n"
        for i, pair in enumerate(topic_results.get('high_similarity_pairs', [])[:5], 1):
            similarity_pct = pair['similarity'] * 100
            report += f"""
**Pair #{i}: {similarity_pct:.1f}% Similar**
- Page 1: `{pair['page1'][:100]}`
- Page 2: `{pair['page2'][:100]}`
- Action: {"Merge immediately (>95% similar)" if pair['similarity'] > 0.95 else "Consolidate content (>90% similar)" if pair['similarity'] > 0.9 else "Differentiate topics"}
"""
    
    # Add AI analysis if available
    if ai_provider and ai_provider.client:
        report += "\n---\n\n## 🤖 AI-Powered Strategic Recommendations\n\n"
        ai_analysis = ai_provider.generate_detailed_analysis(
            keyword_results, content_results, topic_results
        )
        if ai_analysis and not ai_analysis.startswith("Could not"):
            report += ai_analysis
        else:
            report += "*(AI analysis not available - configure AI provider for detailed recommendations)*"
    
    # Add specific actionable recommendations if available
    if ai_recommendations and ai_recommendations.get("immediate_actions"):
        report += "\n\n### 🎯 Prioritized Action Items\n\n"
        
        if ai_recommendations.get("immediate_actions"):
            report += "**Immediate Actions (This Week):**\n"
            for i, action in enumerate(ai_recommendations["immediate_actions"][:5], 1):
                report += f"\n{i}. **{action['action']}**\n"
                report += f"   - Priority: {action['priority']}\n"
                report += f"   - Reason: {action['reason']}\n"
                if action.get('expected_impact'):
                    report += f"   - Expected Impact: {action['expected_impact']}\n"
        
        if ai_recommendations.get("quick_wins"):
            report += "\n**Quick Wins (Can be done today):**\n"
            for i, win in enumerate(ai_recommendations["quick_wins"][:3], 1):
                report += f"\n{i}. {win['action']} ({win['time']})\n"
    
    # Add implementation roadmap
    report += """

---

## 📋 Implementation Roadmap

### Week 1: Quick Wins (Immediate Impact)
1. **301 Redirects:** Implement for pages with >70% keyword overlap and high click volumes
2. **Canonical Tags:** Add to duplicate content that must remain
3. **Internal Linking:** Update to point to primary pages for high-traffic keywords
4. **Meta Tags:** Differentiate title tags and meta descriptions

### Week 2-3: Content Optimization
1. **Content Merging:** Combine pages with >90% semantic similarity
2. **Content Differentiation:** Rewrite pages targeting different search intents
3. **Topic Clusters:** Create pillar pages for high-traffic topics
4. **URL Structure:** Implement clear hierarchical URL structure

### Week 4: Monitoring & Refinement
1. **Track Rankings:** Monitor keyword position changes for affected queries
2. **Measure Traffic:** Compare before/after organic traffic
3. **User Metrics:** Check bounce rate and time on page
4. **Iterate:** Adjust strategy based on results

---

## 📈 Expected Results

Based on the severity of issues found and clicks affected:

- **Traffic Recovery Timeline:** 4-8 weeks
- **Expected Traffic Increase:** {keyword_severity.get('impact', '10-25%')}
- **Ranking Improvements:** 2-5 positions for cannibalized keywords
- **CTR Improvement:** 15-30% for consolidated pages

---

## ⚠️ Risk Mitigation

1. **Before Making Changes:**
   - Back up all content
   - Document current rankings
   - Set up proper tracking
   - Note all URLs with parameters/fragments

2. **During Implementation:**
   - Make changes gradually
   - Monitor for 404 errors
   - Check Google Search Console daily
   - Ensure proper redirects for normalized URLs

3. **After Changes:**
   - Submit updated sitemap
   - Monitor Core Web Vitals
   - Track user engagement metrics

---

*Report generated by SEO Cannibalization Analyzer v1.0*
"""
    
    return report

# ============================================================================
# AI RECOMMENDATIONS FUNCTIONS
# ============================================================================

def generate_ai_recommendations(keyword_data: Dict, content_data: Dict, 
                                topic_data: Dict, ai_provider: AIProvider) -> Dict:
    """Generate comprehensive AI-powered recommendations based on analysis results"""
    
    recommendations = {
        "immediate_actions": [],
        "consolidation_candidates": [],
        "content_optimization": [],
        "technical_fixes": [],
        "long_term_strategy": [],
        "quick_wins": [],
        "priority_order": []
    }
    
    # Analyze keyword cannibalization issues
    if keyword_data and keyword_data.get('top_issues'):
        for pages, data in list(keyword_data.get('top_issues', {}).items())[:5]:
            overlap_pct = data['overlap_percentage']
            clicks_affected = data.get('total_clicks_affected', 0)
            
            # Detect page types
            page1_type = detect_page_type(data.get('page1_full', ''))
            page2_type = detect_page_type(data.get('page2_full', ''))
            
            # Check if pages are compatible for consolidation
            incompatible_types = [
                ('blog', 'service'), ('blog', 'product'), ('service', 'product'),
                ('legal', 'blog'), ('legal', 'service'), ('legal', 'product'),
                ('about', 'blog'), ('about', 'service'), ('about', 'product')
            ]
            
            pages_incompatible = (
                (page1_type, page2_type) in incompatible_types or 
                (page2_type, page1_type) in incompatible_types
            )
            
            if overlap_pct > 70 and clicks_affected > 100 and not pages_incompatible:
                # Critical: Immediate consolidation needed
                recommendations["immediate_actions"].append({
                    "type": "consolidation",
                    "priority": "CRITICAL",
                    "page1": data.get('page1_full', ''),
                    "page2": data.get('page2_full', ''),
                    "action": "Merge pages with 301 redirect",
                    "reason": f"{overlap_pct}% keyword overlap affecting {clicks_affected:,} clicks",
                    "expected_impact": "50-70% traffic recovery within 4-6 weeks",
                    "implementation": [
                        "1. Combine best content from both pages",
                        "2. Create comprehensive page targeting main keyword",
                        "3. Implement 301 redirect from weaker page",
                        "4. Update all internal links",
                        "5. Submit updated sitemap"
                    ]
                })
            elif overlap_pct > 50 or pages_incompatible:
                # High: Content differentiation needed (or incompatible page types)
                recommendations["content_optimization"].append({
                    "type": "differentiation",
                    "priority": "HIGH" if not pages_incompatible else "CRITICAL",
                    "page1": data.get('page1_full', ''),
                    "page2": data.get('page2_full', ''),
                    "page1_type": page1_type,
                    "page2_type": page2_type,
                    "action": "Differentiate content focus" if not pages_incompatible else "CANNOT merge - incompatible page types",
                    "reason": f"{overlap_pct}% overlap on {data['total_shared']} keywords" + 
                              (f" - {page1_type} vs {page2_type} pages" if pages_incompatible else ""),
                    "strategy": "Re-optimize for different search intents" if not pages_incompatible else 
                                "Keep pages separate but differentiate keywords",
                    "suggestions": [
                        f"Page 1 ({page1_type}): Target transactional intent for top keywords" if not pages_incompatible else
                        f"Page 1 ({page1_type}): Focus on {page1_type}-specific keywords",
                        f"Page 2 ({page2_type}): Target informational/educational intent" if not pages_incompatible else
                        f"Page 2 ({page2_type}): Focus on {page2_type}-specific keywords",
                        "Use different long-tail variations",
                        "Update meta titles and descriptions",
                        "Ensure internal linking respects page hierarchy" if pages_incompatible else ""
                    ]
                })
            elif overlap_pct > 30:
                # Medium: Technical optimization
                recommendations["technical_fixes"].append({
                    "type": "technical",
                    "priority": "MEDIUM",
                    "pages": [data.get('page1_full', ''), data.get('page2_full', '')],
                    "action": "Implement canonical tags or adjust internal linking",
                    "reason": f"Moderate overlap ({overlap_pct}%) causing ranking confusion"
                })
    
    # Analyze content/SERP cannibalization
    if content_data and content_data.get('top_overlaps'):
        for query_pair, data in list(content_data.get('top_overlaps', {}).items())[:5]:
            if data['overlap_percentage'] > 65:
                # Check if queries come from different URLs
                q1_urls = data.get('query1_client_urls', [])
                q2_urls = data.get('query2_client_urls', [])
                
                # Only recommend action if queries come from different URLs
                if q1_urls != q2_urls or len(q1_urls) > 1 or len(q2_urls) > 1:
                    recommendations["immediate_actions"].append({
                        "type": "serp_consolidation",
                        "priority": "HIGH",
                        "query1": data.get('query1', ''),
                        "query2": data.get('query2', ''),
                        "clicks_impact": data.get('total_clicks', 0),
                        "action": "Consolidate search intent",
                        "reason": f"{data['overlap_percentage']}% SERP overlap indicates same search intent from different pages",
                        "strategy": "Create single authoritative page for both queries",
                        "implementation": [
                            "1. Identify pages ranking for these queries",
                            "2. Merge content into comprehensive resource",
                            "3. Target primary query with secondary variations",
                            "4. Implement schema markup for better SERP visibility"
                        ]
                    })
    
    # Analyze topic/semantic similarity
    if topic_data and topic_data.get('high_similarity_pairs'):
        for pair in topic_data.get('high_similarity_pairs', [])[:5]:
            similarity = pair['similarity']
            
            # Detect page types
            page1_type = detect_page_type(pair['page1'])
            page2_type = detect_page_type(pair['page2'])
            
            # Check compatibility
            incompatible_types = [
                ('blog', 'service'), ('blog', 'product'), ('service', 'product'),
                ('legal', 'blog'), ('legal', 'service'), ('legal', 'product'),
                ('about', 'blog'), ('about', 'service'), ('about', 'product')
            ]
            
            pages_incompatible = (
                (page1_type, page2_type) in incompatible_types or 
                (page2_type, page1_type) in incompatible_types
            )
            
            if similarity > 0.95 and not pages_incompatible:
                recommendations["consolidation_candidates"].append({
                    "type": "semantic_merge",
                    "priority": "CRITICAL",
                    "page1": pair['page1'],
                    "page2": pair['page2'],
                    "page1_type": page1_type,
                    "page2_type": page2_type,
                    "similarity": f"{similarity * 100:.1f}%",
                    "action": "Immediate consolidation required",
                    "reason": "Near-duplicate content detected",
                    "steps": [
                        "1. Choose stronger page (check backlinks, traffic)",
                        "2. Merge unique content elements",
                        "3. 301 redirect duplicate page",
                        "4. Update XML sitemap"
                    ]
                })
            elif similarity > 0.85 or (similarity > 0.75 and pages_incompatible):
                recommendations["content_optimization"].append({
                    "type": "topic_differentiation",
                    "priority": "HIGH" if not pages_incompatible else "CRITICAL",
                    "pages": [pair['page1'], pair['page2']],
                    "page_types": [page1_type, page2_type],
                    "similarity": f"{similarity * 100:.1f}%",
                    "action": "Differentiate topic focus" if not pages_incompatible else 
                              f"CANNOT merge - incompatible page types ({page1_type} vs {page2_type})",
                    "suggestions": [
                        "Target different audience segments" if not pages_incompatible else
                        f"Ensure {page1_type} page focuses on {page1_type}-specific content",
                        "Focus on different aspects of the topic" if not pages_incompatible else
                        f"Ensure {page2_type} page focuses on {page2_type}-specific content",
                        "Use distinct keyword variations for each page type" if pages_incompatible else
                        "Use different long-tail keyword variations",
                        "Create clear content hierarchy"
                    ]
                })
    
    # Generate quick wins
    if keyword_data.get('top_issues'):
        for pages, data in list(keyword_data.get('top_issues', {}).items())[:3]:
            if data.get('total_clicks_affected', 0) > 50:
                recommendations["quick_wins"].append({
                    "action": f"Update internal links for '{data.get('shared_keywords', [''])[0]}'",
                    "impact": "Immediate ranking signal consolidation",
                    "time": "1-2 hours",
                    "how": "Point all internal links to the stronger performing page"
                })
    
    # Long-term strategy recommendations
    total_issues = (
        keyword_data.get('pages_with_cannibalization', 0) +
        content_data.get('queries_with_overlap', 0) +
        topic_data.get('pages_with_high_similarity', 0)
    )
    
    if total_issues > 20:
        recommendations["long_term_strategy"] = [
            {
                "strategy": "Implement Content Governance",
                "priority": "CRITICAL",
                "actions": [
                    "Create keyword mapping document",
                    "Establish content approval process",
                    "Define clear page type hierarchies",
                    "Train content team on cannibalization prevention"
                ]
            },
            {
                "strategy": "Establish Clear Content Guidelines",
                "priority": "HIGH",
                "actions": [
                    "Define unique purpose for each page type",
                    "Create keyword assignment rules",
                    "Document internal linking best practices",
                    "Set up monitoring for new cannibalization"
                ]
            }
        ]
    
    # Generate priority order
    all_actions = []
    for action in recommendations["immediate_actions"]:
        all_actions.append({
            "priority_score": 100 if action["priority"] == "CRITICAL" else 80,
            "action": action["action"],
            "type": action["type"],
            "details": action
        })
    
    for action in recommendations["consolidation_candidates"]:
        all_actions.append({
            "priority_score": 90,
            "action": action["action"],
            "type": action["type"],
            "details": action
        })
    
    # Sort by priority
    all_actions.sort(key=lambda x: x["priority_score"], reverse=True)
    recommendations["priority_order"] = all_actions[:10]
    
    # Use AI for enhanced recommendations if available
    if ai_provider and ai_provider.client:
        ai_insights = generate_ai_enhanced_recommendations(
            keyword_data, content_data, topic_data, recommendations, ai_provider
        )
        recommendations["ai_insights"] = ai_insights
    
    return recommendations

def generate_ai_enhanced_recommendations(keyword_data: Dict, content_data: Dict, 
                                           topic_data: Dict, base_recommendations: Dict,
                                           ai_provider: AIProvider) -> str:
    """Generate AI-enhanced recommendations with specific examples"""
    
    # Prepare context for AI
    top_keyword_issues = []
    if keyword_data.get('top_issues'):
        for pages, data in list(keyword_data.get('top_issues', {}).items())[:3]:
            top_keyword_issues.append({
                "pages": [data.get('page1_full', ''), data.get('page2_full', '')],
                "overlap": data['overlap_percentage'],
                "keywords": data.get('shared_keywords', [])[:5],
                "clicks": data.get('total_clicks_affected', 0)
            })
    
    prompt = f"""As an SEO expert, analyze this cannibalization data and provide SPECIFIC, ACTIONABLE recommendations:

CRITICAL FINDINGS:
- Keyword Cannibalization: {keyword_data.get('pages_with_cannibalization', 0)} pages affected
- Content/SERP Overlap: {content_data.get('queries_with_overlap', 0)} queries competing
- Topic Similarity: {topic_data.get('pages_with_high_similarity', 0)} near-duplicate pages

TOP ISSUES REQUIRING IMMEDIATE ACTION:
{json.dumps(top_keyword_issues, indent=2)}

Based on SEO best practices for fixing cannibalization, provide:

1. **IMMEDIATE ACTIONS** (Do this week):
   - Which specific pages to merge (with URLs)
   - Exactly how to implement 301 redirects
   - Quick internal linking fixes

2. **CONTENT CONSOLIDATION STRATEGY**:
   - Step-by-step merging process
   - How to preserve link equity
   - Content migration checklist

3. **DIFFERENTIATION TACTICS**:
   - How to rewrite titles/meta descriptions
   - Long-tail keyword opportunities
   - Different search intent angles

4. **EXPECTED RESULTS**:
   - Traffic recovery timeline
   - Ranking improvement expectations
   - Risk mitigation steps

5. **30-DAY IMPLEMENTATION PLAN**:
   Week 1: [Specific tasks]
   Week 2: [Specific tasks]
   Week 3: [Specific tasks]
   Week 4: [Monitoring & adjustment]

Be specific with URLs and keywords. Focus on high-impact, practical solutions."""

    try:
        if ai_provider.provider == "OpenAI" and openai:
            response = openai.ChatCompletion.create(
                model=ai_provider.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif ai_provider.provider == "Anthropic" and anthropic:
            response = ai_provider.client.messages.create(
                model=ai_provider.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        elif ai_provider.provider == "Google" and genai:
            response = ai_provider.client.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"Could not generate AI insights: {str(e)}"
    
    return ""

def display_ai_recommendations(recommendations: Dict):
    """Display AI recommendations in a structured, actionable format"""
    
    # Immediate Actions
    if recommendations.get("immediate_actions"):
        st.subheader("🚨 Immediate Actions Required")
        st.markdown("*Complete these within the next 7 days for maximum impact*")
        
        for i, action in enumerate(recommendations["immediate_actions"], 1):
            with st.expander(f"Action #{i}: {action['action']} ({action['priority']} Priority)", expanded=True):
                st.error(f"**Reason:** {action['reason']}")
                
                if action.get('page1'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Page 1:**")
                        st.code(action['page1'][:100] + "..." if len(action['page1']) > 100 else action['page1'])
                        if action.get('page1_type'):
                            st.caption(f"Type: {action['page1_type']}")
                    with col2:
                        st.write("**Page 2:**")
                        st.code(action.get('page2', '')[:100] + "..." if action.get('page2', '') and len(action.get('page2', '')) > 100 else action.get('page2', ''))
                        if action.get('page2_type'):
                            st.caption(f"Type: {action.get('page2_type', 'unknown')}")
                
                if action.get('implementation'):
                    st.write("**Implementation Steps:**")
                    for step in action['implementation']:
                        st.write(step)
                
                if action.get('expected_impact'):
                    st.success(f"**Expected Impact:** {action['expected_impact']}")
    
    # Quick Wins
    if recommendations.get("quick_wins"):
        st.subheader("⚡ Quick Wins")
        st.markdown("*Low-effort, high-impact changes you can make today*")
        
        cols = st.columns(len(recommendations["quick_wins"][:3]))
        for i, (col, win) in enumerate(zip(cols, recommendations["quick_wins"][:3])):
            with col:
                st.info(f"**Quick Win #{i+1}**")
                st.write(f"**Action:** {win['action']}")
                st.write(f"**Time:** {win['time']}")
                st.write(f"**Impact:** {win['impact']}")
                if win.get('how'):
                    st.write(f"**How:** {win['how']}")
    
    # Consolidation Candidates
    if recommendations.get("consolidation_candidates"):
        st.subheader("🔄 Pages to Consolidate")
        st.markdown("*These pages are so similar they should be merged*")
        
        for candidate in recommendations["consolidation_candidates"]:
            with st.expander(f"{candidate['action']} - {candidate['similarity']} similarity"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Page 1:**")
                    st.code(candidate['page1'][:80] + "...")
                    if candidate.get('page1_type'):
                        st.caption(f"Type: {candidate['page1_type']}")
                with col2:
                    st.write("**Page 2:**")
                    st.code(candidate['page2'][:80] + "...")
                    if candidate.get('page2_type'):
                        st.caption(f"Type: {candidate['page2_type']}")
                
                st.warning(f"**Reason:** {candidate['reason']}")
                
                if candidate.get('steps'):
                    st.write("**Consolidation Steps:**")
                    for step in candidate['steps']:
                        st.write(step)
    
    # Content Optimization
    if recommendations.get("content_optimization"):
        st.subheader("✏️ Content Optimization Needed")
        st.markdown("*Differentiate these pages to avoid competition*")
        
        for opt in recommendations["content_optimization"]:
            with st.expander(f"{opt['action']} ({opt['priority']} Priority)"):
                if opt.get('pages'):
                    st.write("**Affected Pages:**")
                    for i, page in enumerate(opt['pages']):
                        page_type = opt.get('page_types', ['', ''])[i] if opt.get('page_types') else ''
                        st.write(f"• {page[:100]}... ({page_type})" if page_type else f"• {page[:100]}...")
                
                if opt.get('suggestions'):
                    st.write("**Optimization Suggestions:**")
                    for suggestion in opt['suggestions']:
                        st.write(f"• {suggestion}")
                
                if opt.get('reason'):
                    st.info(f"**Why:** {opt['reason']}")
    
    # Technical Fixes
    if recommendations.get("technical_fixes"):
        st.subheader("🔧 Technical Fixes")
        st.markdown("*Backend optimizations to clarify page hierarchy*")
        
        for fix in recommendations["technical_fixes"]:
            with st.expander(f"{fix['action']} ({fix['priority']} Priority)"):
                st.write(f"**Issue:** {fix['reason']}")
                st.write("**Affected Pages:**")
                for page in fix['pages']:
                    st.write(f"• {page[:100]}...")
    
    # Long-term Strategy
    if recommendations.get("long_term_strategy"):
        st.subheader("📅 Long-term Strategy")
        st.markdown("*Prevent future cannibalization with these strategic changes*")
        
        for strategy in recommendations["long_term_strategy"]:
            with st.expander(f"{strategy['strategy']} ({strategy['priority']} Priority)"):
                st.write("**Action Items:**")
                for action in strategy['actions']:
                    st.write(f"• {action}")
    
    # AI Insights
    if recommendations.get("ai_insights"):
        st.subheader("🤖 AI-Powered Strategic Insights")
        with st.expander("Detailed AI Analysis & Recommendations", expanded=True):
            st.markdown(recommendations["ai_insights"])
    
    # Priority Timeline
    st.subheader("📋 Implementation Timeline")
    st.markdown("*Your prioritized action plan*")
    
    timeline_data = {
        "Week 1": [],
        "Week 2": [],
        "Week 3": [],
        "Week 4": []
    }
    
    for i, action in enumerate(recommendations.get("priority_order", [])[:10]):
        week = f"Week {min((i // 3) + 1, 4)}"
        timeline_data[week].append(f"• {action['action']} ({action['type']})")
    
    cols = st.columns(4)
    for i, (week, tasks) in enumerate(timeline_data.items()):
        with cols[i]:
            st.write(f"**{week}**")
            for task in tasks:
                st.write(task)
            if not tasks:
                st.write("• Monitor & adjust")
    
    # Download recommendations
    if st.button("📥 Download Recommendations", use_container_width=True):
        report_text = generate_recommendations_report(recommendations)
        st.download_button(
            label="Download Recommendations Report",
            data=report_text,
            file_name=f"cannibalization_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def generate_recommendations_report(recommendations: Dict) -> str:
    """Generate a downloadable recommendations report"""
    
    report = f"""# SEO Cannibalization Fix Recommendations
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides actionable recommendations to fix SEO cannibalization issues identified on your website.

"""
    
    # Add immediate actions
    if recommendations.get("immediate_actions"):
        report += "## Immediate Actions (Do This Week)\n\n"
        for i, action in enumerate(recommendations["immediate_actions"], 1):
            report += f"### Action {i}: {action['action']}\n"
            report += f"**Priority:** {action['priority']}\n"
            report += f"**Reason:** {action['reason']}\n\n"
            if action.get('implementation'):
                report += "**Steps:**\n"
                for step in action['implementation']:
                    report += f"{step}\n"
            report += "\n---\n\n"
    
    # Add other sections
    if recommendations.get("quick_wins"):
        report += "## Quick Wins\n\n"
        for win in recommendations["quick_wins"]:
            report += f"- **{win['action']}** ({win['time']})\n"
            report += f"  - Impact: {win['impact']}\n\n"
    
    # Add AI insights if available
    if recommendations.get("ai_insights"):
        report += "\n## AI Strategic Analysis\n\n"
        report += recommendations["ai_insights"]
    
    return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🔍 SEO Cannibalization Analyzer")
    st.markdown("Comprehensive analysis of keyword, content, and topic cannibalization with AI-powered recommendations")
    
    # Initialize session state for progress tracking
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = {}
    
    # Initialize AI provider
    ai_provider = AIProvider()
    
    # Check for API keys in secrets
    try:
        serper_api_key_secret = st.secrets.get("api_keys", {}).get("serper_key", "")
        openai_secret = st.secrets.get("api_keys", {}).get("openai_key", "")
        anthropic_secret = st.secrets.get("api_keys", {}).get("anthropic_key", "")
        google_secret = st.secrets.get("api_keys", {}).get("google_key", "")
    except:
        serper_api_key_secret = openai_secret = anthropic_secret = google_secret = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys Section
        st.subheader("🔑 API Keys")
        
        # Serper API Key
        if serper_api_key_secret:
            st.success("✅ Serper API loaded from secrets")
            serper_api_key = serper_api_key_secret
            manual_serper_key = st.text_input(
                "Serper API Key (Override)",
                type="password",
                help="Leave blank to use key from secrets"
            )
            if manual_serper_key:
                serper_api_key = manual_serper_key
        else:
            serper_api_key = st.text_input(
                "Serper API Key (Required for SERP)",
                type="password",
                help="Get your API key from serper.dev"
            )
            if serper_api_key:
                st.success("✅ Serper API configured")
        
        st.divider()
        
        # AI Provider Selection
        st.subheader("🤖 AI Provider (Optional)")
        
        available_providers = ["None"]
        if openai and openai_secret:
            available_providers.append("OpenAI")
        if anthropic and anthropic_secret:
            available_providers.append("Anthropic")
        if genai and google_secret:
            available_providers.append("Google")
        
        ai_choice = st.selectbox("Select AI Provider", available_providers)
        
        if ai_choice != "None":
            api_key_from_secret = ""
            model_options = []
            
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
                api_key = api_key_from_secret
                st.success(f"✅ {ai_choice} API loaded from secrets")
            else:
                api_key = st.text_input(
                    f"{ai_choice} API Key",
                    type="password",
                    help="Enter your API key for AI-powered recommendations"
                )
            
            if model_options:
                model = st.selectbox("Model", model_options)
                
                if api_key and st.button("Initialize AI"):
                    if ai_provider.setup(ai_choice, api_key, model):
                        st.success(f"✅ {ai_choice} initialized!")
        
        st.divider()
        
        # Analysis Settings
        st.subheader("Analysis Settings")
        
        # Branded terms exclusion
        branded_terms = st.text_area(
            "Branded Terms to Exclude",
            placeholder="Enter branded terms (one per line) to exclude from cannibalization analysis",
            help="Add your brand names, company names, and variations to avoid false positives"
        )
        branded_terms_list = [term.strip().lower() for term in branded_terms.split('\n') if term.strip()]
        
        keyword_threshold = st.slider(
            "Keyword Overlap Threshold (%)",
            min_value=5, max_value=50, value=10,
            help="Minimum overlap percentage to flag"
        )
        
        min_clicks_keyword = st.number_input(
            "Min Clicks for Keywords",
            min_value=0, max_value=100, value=1,
            help="Only analyze keywords with at least this many clicks"
        )
        
        serp_sample_size = st.number_input(
            "SERP Analysis Sample Size",
            min_value=0, max_value=50000, value=0,
            help="Number of queries to analyze. 0 = ALL queries. Set a limit only if you want to reduce API usage."
        )
        
        min_clicks_serp = st.number_input(
            "Min Clicks for SERP Analysis",
            min_value=0, max_value=100, value=1,
            help="Only analyze queries with at least this many clicks"
        )
        
        similarity_threshold = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05,
            help="Minimum similarity score to flag"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🔤 Keyword Analysis", 
                                            "📑 Content Analysis", "🧠 Topic Analysis", 
                                            "🤖 AI Insights & Recommendations"])
    
    with tab1:
        st.header("Upload Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("GSC Performance Report")
            gsc_file = st.file_uploader(
                "Upload GSC Report CSV",
                type=['csv'],
                help="Export from Google Search Console"
            )
            
            if gsc_file:
                try:
                    gsc_df = pd.read_csv(gsc_file)
                    is_valid, message = validate_gsc_data(gsc_df)
                    
                    if is_valid:
                        st.success(f"✅ Loaded {len(gsc_df)} rows")
                        
                        # Show URL normalization info
                        unique_before = gsc_df['Landing Page'].nunique()
                        unique_after = gsc_df['Normalized_URL'].nunique()
                        if unique_before != unique_after:
                            st.info(f"📊 URL Normalization: {unique_before} unique URLs → {unique_after} normalized URLs")
                        
                        # Show data preview
                        with st.expander("Preview Data"):
                            preview_df = gsc_df[['Query', 'Landing Page', 'Normalized_URL', 'Clicks', 'Impressions']].head(10)
                            st.dataframe(preview_df, use_container_width=True)
                        
                        # Show statistics
                        col1_1, col1_2, col1_3 = st.columns(3)
                        with col1_1:
                            st.metric("Unique Pages", gsc_df['Normalized_URL'].nunique())
                        with col1_2:
                            st.metric("Unique Queries", gsc_df['Query'].nunique())
                        with col1_3:
                            st.metric("Total Clicks", f"{gsc_df['Clicks'].sum():,}")
                        
                        st.session_state['gsc_df'] = gsc_df
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
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
                    is_valid, message = validate_embeddings_data(embeddings_df)
                    
                    if is_valid:
                        st.success(f"✅ Loaded {len(embeddings_df)} pages")
                        
                        # Show URL normalization info
                        unique_before = embeddings_df['Address'].nunique()
                        unique_after = embeddings_df['Normalized_URL'].nunique()
                        if unique_before != unique_after:
                            st.info(f"📊 URL Normalization: {unique_before} unique URLs → {unique_after} normalized URLs")
                        
                        with st.expander("Preview Data"):
                            st.dataframe(embeddings_df[['Address', 'Normalized_URL']].head(10), use_container_width=True)
                        
                        st.metric("Total Pages", len(embeddings_df))
                        
                        st.session_state['embeddings_df'] = embeddings_df
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.header("🔤 Keyword Cannibalization Analysis")
        
        if 'gsc_df' in st.session_state:
            # Show preview stats
            df = st.session_state['gsc_df']
            st.info(f"Ready to analyze {df['Normalized_URL'].nunique()} unique pages and {df['Query'].nunique()} unique queries")
            
            if min_clicks_keyword > 0:
                filtered_queries = df[df['Clicks'] >= min_clicks_keyword]['Query'].nunique()
                st.warning(f"⚠️ Filtering to {filtered_queries} queries with at least {min_clicks_keyword} clicks")
            
            if st.button("Analyze Keyword Overlap", type="primary", key="keyword_btn"):
                with st.spinner("Analyzing keyword overlap (this may take a moment)..."):
                    results = KeywordCannibalizationAnalyzer.analyze(
                        st.session_state['gsc_df'], 
                        keyword_threshold,
                        min_clicks_keyword,
                        branded_terms_list
                    )
                    st.session_state['keyword_results'] = results
                    
                    # Display severity badge
                    severity = results.get('severity_info', {})
                    st.markdown(f"<div class='severity-{severity.get('severity', '').lower()}'>{severity.get('color', '')} {severity.get('severity', 'Unknown')} SEVERITY - {severity.get('impact', '')}</div>", unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Pages", results['total_pages_analyzed'])
                    with col2:
                        st.metric("Pages with Issues", results['pages_with_cannibalization'])
                    with col3:
                        st.metric("Overlap Pairs", results['total_overlap_pairs'])
                    with col4:
                        st.metric("Clicks Affected", f"{results['total_clicks_affected']:,}")
                    
                    # Show branded terms filtered if any
                    if results.get('branded_queries_removed', 0) > 0:
                        st.info(f"ℹ️ Filtered out {results['branded_queries_removed']} queries containing branded terms")
                    
                    # Display top issues
                    if results['top_issues']:
                        st.subheader("🔥 Top Cannibalization Issues (Sorted by Click Impact)")
                        for i, (pages, data) in enumerate(list(results['top_issues'].items())[:10], 1):
                            with st.expander(f"Issue #{i}: {data['total_clicks_affected']:,} clicks affected - {data['overlap_percentage']}% overlap - {data['total_shared']} shared keywords"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Page 1:**")
                                    st.code(data.get('page1_full', pages.split('||')[0]))
                                    if data.get('normalized_page1'):
                                        st.caption(f"Normalized: {data['normalized_page1'][:80]}...")
                                    st.metric("Unique Keywords", data.get('page1_unique', 0))
                                with col2:
                                    st.write("**Page 2:**")
                                    st.code(data.get('page2_full', pages.split('||')[1]))
                                    if data.get('normalized_page2'):
                                        st.caption(f"Normalized: {data['normalized_page2'][:80]}...")
                                    st.metric("Unique Keywords", data.get('page2_unique', 0))
                                
                                st.write("**Top Shared Keywords (by clicks):**")
                                if data.get('shared_keywords_with_clicks'):
                                    for kw, clicks in data['shared_keywords_with_clicks'][:10]:
                                        st.write(f"• {kw} ({clicks:,} clicks)")
                                else:
                                    st.write(", ".join(data['shared_keywords'][:20]))
                                
                                if data['overlap_percentage'] > 70:
                                    st.error("⚠️ Critical: Consider merging these pages")
                                elif data['overlap_percentage'] > 40:
                                    st.warning("⚠️ High: Differentiate content significantly")
                                else:
                                    st.info("ℹ️ Moderate: Monitor and optimize")
        else:
            st.info("📤 Please upload a GSC report in the Data Upload tab")
    
    with tab3:
        st.header("📑 Content/SERP Cannibalization Analysis")
        
        if 'gsc_df' in st.session_state:
            if not serper_api_key:
                st.error("🔑 Please enter your Serper API key in the sidebar")
                st.info("[Get your API key at serper.dev](https://serper.dev) - 2,500 free queries/month")
            else:
                # Show preview
                total_queries = st.session_state['gsc_df']['Query'].nunique()
                
                # Calculate how many queries meet the click threshold
                queries_meeting_threshold = len(st.session_state['gsc_df'][
                    st.session_state['gsc_df']['Clicks'] >= min_clicks_serp
                ]['Query'].unique())
                
                # Consider branded terms filtering
                if branded_terms_list:
                    queries_meeting_threshold_after_filter = len(st.session_state['gsc_df'][
                        (st.session_state['gsc_df']['Clicks'] >= min_clicks_serp) & 
                        (~st.session_state['gsc_df']['Query'].str.lower().apply(
                            lambda x: any(brand in x for brand in branded_terms_list)
                        ))
                    ]['Query'].unique())
                    queries_meeting_threshold = queries_meeting_threshold_after_filter
                
                if serp_sample_size == 0:
                    st.warning(f"⚠️ Analyzing ALL {queries_meeting_threshold} queries will use {queries_meeting_threshold} API credits!")
                    st.info(f"Ready to analyze ALL {queries_meeting_threshold} queries with ≥{min_clicks_serp} clicks")
                else:
                    st.info(f"Ready to analyze {min(serp_sample_size, queries_meeting_threshold)} queries (limited from {queries_meeting_threshold} total)")
                
                if st.button("Analyze SERP Overlap", type="primary", key="serp_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    actual_queries_to_analyze = queries_meeting_threshold if serp_sample_size == 0 else min(serp_sample_size, queries_meeting_threshold)
                    progress_message = f"Analyzing SERPs for {actual_queries_to_analyze} queries..."
                    
                    with st.spinner(progress_message):
                        # Progress callback
                        def update_progress(value):
                            progress_bar.progress(int(value))
                            status_text.text(f"Progress: {int(value)}%")
                        
                        # Run analysis
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            results = loop.run_until_complete(
                                ContentCannibalizationAnalyzer.analyze_serp_overlap(
                                    st.session_state['gsc_df'], 
                                    serper_api_key,
                                    serp_sample_size,
                                    update_progress,
                                    min_clicks_serp,
                                    branded_terms_list
                                )
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("Analysis complete!")
                            
                            if 'error' not in results:
                                st.session_state['content_results'] = results
                                
                                # Display severity
                                severity = results.get('severity_info', {})
                                st.markdown(f"<div class='severity-{severity.get('severity', '').lower()}'>{severity.get('color', '')} {severity.get('severity', 'Unknown')} SEVERITY - {severity.get('impact', '')}</div>", unsafe_allow_html=True)
                                
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Queries Analyzed", results['total_queries_analyzed'])
                                with col2:
                                    st.metric("Queries with Overlap (>65%)", results['queries_with_overlap'])
                                with col3:
                                    st.metric("Avg SERP Overlap", f"{results['average_overlap']:.1f}%")
                                with col4:
                                    st.metric("API Credits Used", results.get('api_credits_used', 0))
                                
                                # Show detected client domain
                                if results.get('client_domain'):
                                    st.info(f"🌐 Detected client domain: **{results['client_domain']}**")
                                
                                # Failed queries warning
                                if results.get('failed_queries'):
                                    st.warning(f"⚠️ Failed to fetch SERPs for {len(results['failed_queries'])} queries")
                                
                                # Branded terms filtered info
                                if results.get('branded_queries_removed', 0) > 0:
                                    st.info(f"ℹ️ Filtered out {results['branded_queries_removed']} queries containing branded terms")
                                
                                # Same URL pairs skipped info
                                if results.get('same_url_pairs_skipped', 0) > 0:
                                    st.success(f"✅ Correctly skipped {results['same_url_pairs_skipped']} query pairs from the same URL (not cannibalization)")
                                
                                # Top overlaps
                                if results.get('top_overlaps'):
                                    st.subheader("🔥 Top SERP Overlaps (Sorted by Overlap %)")
                                    for query_pair, data in list(results['top_overlaps'].items())[:10]:
                                        with st.expander(f"{data['overlap_percentage']}% SERP Overlap - {data['competition_level']} Competition - {data['total_clicks']:,} total clicks"):
                                            # Query info
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write(f"**Query 1:** {data.get('query1', 'N/A')}")
                                                st.metric("Clicks", f"{data.get('query1_clicks', 0):,}")
                                                if data.get('query1_client_urls'):
                                                    st.caption(f"Your URL: {data['query1_client_urls'][0][:60]}...")
                                            with col2:
                                                st.write(f"**Query 2:** {data.get('query2', 'N/A')}")
                                                st.metric("Clicks", f"{data.get('query2_clicks', 0):,}")
                                                if data.get('query2_client_urls'):
                                                    st.caption(f"Your URL: {data['query2_client_urls'][0][:60]}...")
                                            
                                            st.write(f"**SERP Overlap: {data['overlap_percentage']}% ({data['total_shared']} shared results)**")
                                            
                                            # Show SERP results side by side
                                            serp_col1, serp_col2 = st.columns(2)
                                            
                                            with serp_col1:
                                                st.write(f"**SERP for: {data.get('query1', '')}**")
                                                for result in data.get('serp1_results', [])[:10]:
                                                    if result['is_client']:
                                                        st.success(f"{result['position']}. **[YOUR SITE]** {result['title'][:50]}...")
                                                        st.caption(f"   {result['url'][:60]}...")
                                                    else:
                                                        st.write(f"{result['position']}. {result['title'][:50]}...")
                                                        st.caption(f"   {result['domain']}")
                                            
                                            with serp_col2:
                                                st.write(f"**SERP for: {data.get('query2', '')}**")
                                                for result in data.get('serp2_results', [])[:10]:
                                                    if result['is_client']:
                                                        st.success(f"{result['position']}. **[YOUR SITE]** {result['title'][:50]}...")
                                                        st.caption(f"   {result['url'][:60]}...")
                                                    else:
                                                        st.write(f"{result['position']}. {result['title'][:50]}...")
                                                        st.caption(f"   {result['domain']}")
                                            
                                            if data['overlap_percentage'] > 65:
                                                st.error("⚠️ Critical: Same search intent - consider consolidating content")
                                            else:
                                                st.info("ℹ️ Moderate overlap - differentiate content angles")
                            else:
                                st.error(f"❌ {results.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.info("Please check your Serper API key and try again")
        else:
            st.info("📤 Please upload a GSC report in the Data Upload tab")
    
    with tab4:
        st.header("🧠 Topic/Semantic Cannibalization Analysis")
        
        if 'embeddings_df' in st.session_state:
            df = st.session_state['embeddings_df']
            st.info(f"Ready to analyze {len(df)} pages for semantic similarity (normalized to {df['Normalized_URL'].nunique()} unique URLs)")
            
            if st.button("Analyze Semantic Similarity", type="primary", key="topic_btn"):
                with st.spinner("Calculating semantic similarities..."):
                    results = TopicCannibalizationAnalyzer.analyze(
                        st.session_state['embeddings_df'],
                        similarity_threshold
                    )
                    st.session_state['topic_results'] = results
                    
                    if 'error' not in results:
                        # Display severity
                        severity = results.get('severity_info', {})
                        st.markdown(f"<div class='severity-{severity.get('severity', '').lower()}'>{severity.get('color', '')} {severity.get('severity', 'Unknown')} SEVERITY - {severity.get('impact', '')}</div>", unsafe_allow_html=True)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Pages", results['total_pages'])
                        with col2:
                            st.metric("Pages with Issues", results['pages_with_high_similarity'])
                        with col3:
                            st.metric("Similarity Pairs", results['total_similarity_pairs'])
                        with col4:
                            st.metric("Avg Similarity", f"{results['average_similarity']:.3f}")
                        
                        st.info(f"Embedding dimension: {results.get('embedding_dimension', 'N/A')}")
                        
                        # Top similar pairs
                        if results.get('high_similarity_pairs'):
                            st.subheader("🔥 Highly Similar Pages (Excluding Same Normalized URLs)")
                            for i, pair in enumerate(results['high_similarity_pairs'][:10], 1):
                                similarity_pct = pair['similarity'] * 100
                                with st.expander(f"Pair #{i}: {similarity_pct:.1f}% Similar"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Page 1:**")
                                        st.code(pair['page1'])
                                        if pair.get('normalized_page1'):
                                            st.caption(f"Normalized: {pair['normalized_page1'][:80]}...")
                                    with col2:
                                        st.write("**Page 2:**")
                                        st.code(pair['page2'])
                                        if pair.get('normalized_page2'):
                                            st.caption(f"Normalized: {pair['normalized_page2'][:80]}...")
                                    
                                    if pair['similarity'] > 0.95:
                                        st.error("⚠️ Critical: These pages are nearly identical - merge immediately")
                                    elif pair['similarity'] > 0.9:
                                        st.warning("⚠️ High: Strong candidate for consolidation")
                                    else:
                                        st.info("ℹ️ Moderate: Consider content differentiation")
                        
                        # Similarity distribution
                        if len(results.get('similarity_matrix', pd.DataFrame()).values) > 0:
                            st.subheader("📊 Similarity Distribution")
                            sim_values = results['similarity_matrix'].values[
                                np.triu_indices_from(results['similarity_matrix'].values, k=1)
                            ]
                            fig = px.histogram(
                                x=sim_values,
                                nbins=30,
                                title="Distribution of Semantic Similarities Between Pages",
                                labels={'x': 'Similarity Score', 'y': 'Number of Page Pairs'}
                            )
                            fig.add_vline(x=similarity_threshold, line_dash="dash", line_color="red",
                                          annotation_text=f"Threshold: {similarity_threshold}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"❌ {results.get('error', 'Unknown error')}")
        else:
            st.info("📤 Please upload an embeddings file in the Data Upload tab")
    
    with tab5:
        st.header("🤖 AI-Powered Insights & Actionable Recommendations")
        
        # Check if any analysis has been run
        has_keyword_results = 'keyword_results' in st.session_state
        has_content_results = 'content_results' in st.session_state
        has_topic_results = 'topic_results' in st.session_state
        
        if not any([has_keyword_results, has_content_results, has_topic_results]):
            st.info("📊 Please run at least one analysis in the previous tabs to generate AI insights")
        else:
            # Display analysis summary
            st.subheader("📈 Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_keyword_results:
                    kr = st.session_state['keyword_results']
                    severity = kr.get('severity_info', {})
                    st.metric(
                        "Keyword Cannibalization",
                        f"{severity.get('severity', 'N/A')}",
                        f"{kr.get('pages_with_cannibalization', 0)} pages affected"
                    )
                else:
                    st.metric("Keyword Cannibalization", "Not analyzed", "Run analysis")
            
            with col2:
                if has_content_results:
                    cr = st.session_state['content_results']
                    severity = cr.get('severity_info', {})
                    st.metric(
                        "Content/SERP Overlap",
                        f"{severity.get('severity', 'N/A')}",
                        f"{cr.get('queries_with_overlap', 0)} queries affected"
                    )
                else:
                    st.metric("Content/SERP Overlap", "Not analyzed", "Run analysis")
            
            with col3:
                if has_topic_results:
                    tr = st.session_state['topic_results']
                    severity = tr.get('severity_info', {})
                    st.metric(
                        "Topic Similarity",
                        f"{severity.get('severity', 'N/A')}",
                        f"{tr.get('pages_with_high_similarity', 0)} pages affected"
                    )
                else:
                    st.metric("Topic Similarity", "Not analyzed", "Run analysis")
            
            st.divider()
            
            # AI Provider Check
            if not ai_provider or not ai_provider.client:
                st.warning("⚠️ Please configure an AI provider in the sidebar to get detailed recommendations")
                st.info("Without AI, you'll still see rule-based recommendations based on SEO best practices")
            
            # Generate insights button
            if st.button("🎯 Generate Actionable Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing your data and generating personalized recommendations..."):
                    
                    # Prepare data for analysis
                    keyword_data = st.session_state.get('keyword_results', {})
                    content_data = st.session_state.get('content_results', {})
                    topic_data = st.session_state.get('topic_results', {})
                    
                    # Generate recommendations based on severity and data
                    recommendations = generate_ai_recommendations(
                        keyword_data, content_data, topic_data, ai_provider
                    )
                    
                    # Store recommendations in session state
                    st.session_state['ai_recommendations'] = recommendations
                    
                    # Display success message
                    st.success("✅ Recommendations generated successfully!")
                    
                    # Display recommendations
                    display_ai_recommendations(recommendations)
            
            # If recommendations already exist, display them
            elif 'ai_recommendations' in st.session_state:
                display_ai_recommendations(st.session_state['ai_recommendations'])
    
    # Report Generation Section (only at the bottom)
    st.divider()
    
    # Check if any analysis has been run
    has_results = any([
        'keyword_results' in st.session_state,
        'content_results' in st.session_state,
        'topic_results' in st.session_state
    ])
    
    if has_results:
        # AI Insights reminder
        if 'ai_recommendations' not in st.session_state:
            st.info("💡 Don't forget to check the **AI Insights & Recommendations** tab for actionable fixes!")
    else:
        st.info("📊 Run at least one analysis to generate insights")

if __name__ == "__main__":
    main()
