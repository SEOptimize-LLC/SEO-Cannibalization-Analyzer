"""
SEO Cannibalization Analysis App
Complete FIXED version for Streamlit Cloud deployment
All bugs corrected - no self-comparison, proper counting, real AI analysis
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
from urllib.parse import urlparse
import ast
from typing import Dict, List, Tuple, Optional
import time
import io

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
    
    return True, "Data validation successful"

def validate_embeddings_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate embeddings file format"""
    if 'Address' not in df.columns:
        return False, "Missing 'Address' column"
    
    embeddings_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if not embeddings_cols:
        return False, "No embeddings column found"
    
    return True, "Embeddings data validation successful"

def calculate_severity(pages_affected: int, overlap_percentage: float, issue_type: str) -> Dict:
    """Calculate severity and impact for different cannibalization types"""
    
    if issue_type == "keyword":
        if pages_affected > 20 or overlap_percentage > 50:
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
        if overlap_percentage > 60:
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
    
    else:  # topic
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
- Top issues: {json.dumps(list(keyword_data.get('top_issues', {}).items())[:3], indent=2)}

CONTENT/SERP CANNIBALIZATION DATA:
- Queries analyzed: {content_data.get('total_queries_analyzed', 0)}
- Queries with overlap: {content_data.get('queries_with_overlap', 0)}
- Average SERP overlap: {content_data.get('average_overlap', 0):.1f}%
- Competition clusters: {len(content_data.get('competition_clusters', []))}

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
# FIXED ANALYZER CLASSES
# ============================================================================

class KeywordCannibalizationAnalyzer:
    """FIXED: Analyzes keyword overlap between URLs - no self-comparison"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, threshold: float = 10) -> Dict:
        """Analyze keyword overlap between landing pages"""
        
        # Clean data - remove any duplicate URL-keyword pairs
        df_clean = df.drop_duplicates(subset=['Landing Page', 'Query'])
        
        # Group by landing page
        page_keywords = df_clean.groupby('Landing Page')['Query'].apply(list).to_dict()
        
        # Get unique pages
        pages = list(page_keywords.keys())
        overlap_details = {}
        pages_with_issues = set()
        
        # Calculate overlap - SKIP SELF-COMPARISON
        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):  # Start from i+1 to avoid self-comparison
                page1 = pages[i]
                page2 = pages[j]
                
                keywords1 = set(page_keywords[page1])
                keywords2 = set(page_keywords[page2])
                
                overlap = keywords1.intersection(keywords2)
                
                if len(overlap) > 0:  # Only if there's actual overlap
                    union = keywords1.union(keywords2)
                    overlap_pct = (len(overlap) / len(union)) * 100 if union else 0
                    
                    if overlap_pct > threshold:
                        # Track unique pages with issues
                        pages_with_issues.add(page1)
                        pages_with_issues.add(page2)
                        
                        # Create unique key for this pair
                        key = f"{page1[:50]}...||{page2[:50]}..."
                        
                        overlap_details[key] = {
                            "page1_full": page1,
                            "page2_full": page2,
                            "overlap_percentage": round(overlap_pct, 2),
                            "shared_keywords": sorted(list(overlap))[:20],
                            "total_shared": len(overlap),
                            "page1_total": len(keywords1),
                            "page2_total": len(keywords2),
                            "page1_unique": len(keywords1 - keywords2),
                            "page2_unique": len(keywords2 - keywords1)
                        }
        
        # Sort by overlap percentage
        top_issues = sorted(overlap_details.items(), 
                          key=lambda x: x[1]['overlap_percentage'], 
                          reverse=True)[:15]
        
        # Calculate severity
        avg_overlap = np.mean([d['overlap_percentage'] for d in overlap_details.values()]) if overlap_details else 0
        severity_info = calculate_severity(len(pages_with_issues), avg_overlap, "keyword")
        
        return {
            "overlap_details": overlap_details,
            "top_issues": dict(top_issues),
            "total_pages_analyzed": len(pages),
            "pages_with_cannibalization": len(pages_with_issues),
            "total_overlap_pairs": len(overlap_details),
            "average_overlap": avg_overlap,
            "severity_info": severity_info
        }

class ContentCannibalizationAnalyzer:
    """FIXED: Analyzes SERP overlap for queries using Serper API"""
    
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
    async def analyze_serp_overlap(queries: List[str], api_key: str, sample_size: int = 50, progress_callback=None) -> Dict:
        """Analyze SERP overlap between queries using Serper API"""
        
        if not api_key:
            return {"error": "Serper API key is required for SERP analysis"}
        
        # Remove duplicates and limit sample size
        queries = list(set(queries))[:sample_size]
        
        if len(queries) < 2:
            return {"error": "Need at least 2 unique queries to analyze overlap"}
        
        serp_data = {}
        detailed_results = {}
        failed_queries = []
        
        async with aiohttp.ClientSession() as session:
            batch_size = 5  # Smaller batches for rate limiting
            
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
        
        serp_items = list(serp_data.items())
        for i in range(len(serp_items)):
            for j in range(i + 1, len(serp_items)):
                query1, serp1 = serp_items[i]
                query2, serp2 = serp_items[j]
                
                overlap = set(serp1).intersection(set(serp2))
                union = set(serp1).union(set(serp2))
                overlap_pct = (len(overlap) / len(union)) * 100 if union else 0
                
                if overlap_pct > 30:  # Significant overlap threshold
                    queries_with_issues.add(query1)
                    queries_with_issues.add(query2)
                    
                    key = f"{query1}||{query2}"
                    
                    # Calculate position-weighted score
                    position_score = 0
                    if query1 in detailed_results and query2 in detailed_results:
                        for domain in overlap:
                            if domain in serp1 and domain in serp2:
                                pos1 = serp1.index(domain) + 1
                                pos2 = serp2.index(domain) + 1
                                position_score += (11 - pos1) * (11 - pos2) / 100
                    
                    overlap_matrix[key] = {
                        "query1": query1,
                        "query2": query2,
                        "overlap_percentage": round(overlap_pct, 2),
                        "shared_domains": list(overlap)[:10],
                        "total_shared": len(overlap),
                        "position_weighted_score": round(position_score, 2),
                        "competition_level": "High" if overlap_pct > 60 else "Medium" if overlap_pct > 40 else "Low"
                    }
        
        # Individual query metrics
        for query, serp in serp_data.items():
            query_metrics[query] = {
                "total_results": len(serp),
                "top_domain": serp[0] if serp else None,
                "unique_domains": len(set(serp))
            }
        
        # Find competition clusters
        competition_clusters = []
        processed = set()
        
        for key in overlap_matrix:
            q1, q2 = key.split('||')
            if q1 not in processed:
                cluster = {q1, q2}
                
                # Find related queries
                for other_key in overlap_matrix:
                    other_q1, other_q2 = other_key.split('||')
                    if other_q1 in cluster or other_q2 in cluster:
                        cluster.add(other_q1)
                        cluster.add(other_q2)
                
                if len(cluster) > 2:
                    competition_clusters.append(sorted(list(cluster)))
                    processed.update(cluster)
        
        # Sort results
        top_overlaps = dict(sorted(overlap_matrix.items(), 
                                 key=lambda x: x[1]['overlap_percentage'], 
                                 reverse=True)[:20])
        
        # Calculate severity
        avg_overlap = np.mean([v['overlap_percentage'] for v in overlap_matrix.values()]) if overlap_matrix else 0
        severity_info = calculate_severity(len(queries_with_issues), avg_overlap, "content")
        
        return {
            "total_queries_analyzed": len(serp_data),
            "queries_with_overlap": len(queries_with_issues),
            "total_overlap_pairs": len(overlap_matrix),
            "top_overlaps": top_overlaps,
            "average_overlap": avg_overlap,
            "query_metrics": query_metrics,
            "competition_clusters": competition_clusters[:5],
            "api_credits_used": len(serp_data),
            "failed_queries": failed_queries,
            "severity_info": severity_info
        }

class TopicCannibalizationAnalyzer:
    """FIXED: Analyzes semantic similarity between pages with proper dimension handling"""
    
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
            for emb, page, dim in zip(embeddings_list, valid_pages, embedding_dims):
                if dim == most_common_dim:
                    filtered_embeddings.append(emb)
                    filtered_pages.append(page)
            
            embeddings_list = filtered_embeddings
            valid_pages = filtered_pages
            
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
            for j in range(i + 1, len(valid_pages)):  # Start from i+1 to avoid self-comparison
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    pages_with_issues.add(valid_pages[i])
                    pages_with_issues.add(valid_pages[j])
                    
                    high_similarity_pairs.append({
                        "page1": valid_pages[i],
                        "page2": valid_pages[j],
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
                                 topic_results: Dict, ai_provider: AIProvider) -> str:
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
    
    if issues_found > 50:
        report += "### 🔴 CRITICAL - Immediate Action Required\n"
    elif issues_found > 20:
        report += "### 🟠 HIGH - Significant Issues Found\n"
    elif issues_found > 10:
        report += "### 🟡 MEDIUM - Moderate Issues Found\n"
    else:
        report += "### 🟢 LOW - Minor Issues Found\n"
    
    report += f"""
**Total Issues Identified:** {issues_found}
**Estimated Traffic Impact:** {keyword_severity.get('impact', 'N/A')}

---

## 1️⃣ Keyword Cannibalization Analysis

{keyword_severity.get('color', '')} **Severity: {keyword_severity.get('severity', 'N/A')}**

### Key Metrics:
- **Pages Analyzed:** {keyword_results.get('total_pages_analyzed', 0)}
- **Pages with Cannibalization:** {keyword_results.get('pages_with_cannibalization', 0)}
- **Overlap Pairs Found:** {keyword_results.get('total_overlap_pairs', 0)}
- **Average Overlap:** {keyword_results.get('average_overlap', 0):.1f}%

### Impact Assessment:
- {keyword_severity.get('impact', 'N/A')}
- {keyword_severity.get('priority', 'N/A')}
"""
    
    # Add top keyword issues with specific recommendations
    if keyword_results.get('top_issues'):
        report += "\n### Top 5 Critical Keyword Overlaps:\n"
        for i, (pages, data) in enumerate(list(keyword_results['top_issues'].items())[:5], 1):
            report += f"""
**Issue #{i}: {data['overlap_percentage']}% Overlap**
- **Pages Competing:** 
  - Page 1: `{data.get('page1_full', pages.split('||')[0])[:100]}`
  - Page 2: `{data.get('page2_full', pages.split('||')[1])[:100]}`
- **Shared Keywords:** {data['total_shared']} keywords
- **Top Competing Keywords:** {', '.join(data['shared_keywords'][:5])}
- **Recommended Action:** {"Merge pages (>50% overlap)" if data['overlap_percentage'] > 50 else "Differentiate content (30-50% overlap)" if data['overlap_percentage'] > 30 else "Monitor performance"}
"""
    
    report += f"""
---

## 2️⃣ Content/SERP Cannibalization Analysis

{content_severity.get('color', '')} **Severity: {content_severity.get('severity', 'N/A')}**

### Key Metrics:
- **Queries Analyzed:** {content_results.get('total_queries_analyzed', 0)}
- **Queries with SERP Overlap:** {content_results.get('queries_with_overlap', 0)}
- **Average SERP Overlap:** {content_results.get('average_overlap', 0):.1f}%
- **Competition Clusters:** {len(content_results.get('competition_clusters', []))}
- **API Credits Used:** {content_results.get('api_credits_used', 0)}

### Impact Assessment:
- {content_severity.get('impact', 'N/A')}
- {content_severity.get('priority', 'N/A')}
"""
    
    # Add competition clusters
    if content_results.get('competition_clusters'):
        report += "\n### Query Competition Clusters:\n"
        for i, cluster in enumerate(content_results.get('competition_clusters', [])[:3], 1):
            report += f"\n**Cluster {i}:** {len(cluster)} competing queries\n"
            report += f"- Queries: {', '.join(cluster[:5])}\n"
            report += f"- Action: Create distinct content angles for each query\n"
    
    # Add top SERP overlaps
    if content_results.get('top_overlaps'):
        report += "\n### Top SERP Competitions:\n"
        for query_pair, data in list(content_results.get('top_overlaps', {}).items())[:3]:
            report += f"""
**{data['competition_level']} Competition: {data['overlap_percentage']}% SERP Overlap**
- Query 1: "{data.get('query1', query_pair.split('||')[0])}"
- Query 2: "{data.get('query2', query_pair.split('||')[1])}"
- Shared Domains: {data.get('total_shared', len(data.get('shared_domains', [])))}
- Action: {"Target different search intents" if data['overlap_percentage'] > 50 else "Optimize for featured snippets"}
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
    
    # Add implementation roadmap
    report += """

---

## 📋 Implementation Roadmap

### Week 1: Quick Wins (Immediate Impact)
1. **301 Redirects:** Implement for pages with >70% keyword overlap
2. **Canonical Tags:** Add to duplicate content that must remain
3. **Internal Linking:** Update to point to primary pages
4. **Meta Tags:** Differentiate title tags and meta descriptions

### Week 2-3: Content Optimization
1. **Content Merging:** Combine pages with >90% semantic similarity
2. **Content Differentiation:** Rewrite pages targeting different intents
3. **Topic Clusters:** Create pillar pages and supporting content
4. **URL Structure:** Implement clear hierarchical URL structure

### Week 4: Monitoring & Refinement
1. **Track Rankings:** Monitor keyword position changes
2. **Measure Traffic:** Compare before/after organic traffic
3. **User Metrics:** Check bounce rate and time on page
4. **Iterate:** Adjust strategy based on results

---

## 📈 Expected Results

Based on the severity of issues found:

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

2. **During Implementation:**
   - Make changes gradually
   - Monitor for 404 errors
   - Check Google Search Console daily

3. **After Changes:**
   - Submit updated sitemap
   - Monitor Core Web Vitals
   - Track user engagement metrics

---

*Report generated by SEO Cannibalization Analyzer v1.0*
"""
    
    return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🔍 SEO Cannibalization Analyzer")
    st.markdown("Comprehensive analysis of keyword, content, and topic cannibalization")
    
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
        
        keyword_threshold = st.slider(
            "Keyword Overlap Threshold (%)",
            min_value=5, max_value=50, value=10,
            help="Minimum overlap percentage to flag"
        )
        
        serp_sample_size = st.number_input(
            "SERP Analysis Sample Size",
            min_value=10, max_value=100, value=30,
            help="Number of queries to analyze"
        )
        
        similarity_threshold = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05,
            help="Minimum similarity score to flag"
        )
    
    # Main content tabs
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
                help="Export from Google Search Console"
            )
            
            if gsc_file:
                try:
                    gsc_df = pd.read_csv(gsc_file)
                    is_valid, message = validate_gsc_data(gsc_df)
                    
                    if is_valid:
                        st.success(f"✅ Loaded {len(gsc_df)} rows")
                        # Show data preview
                        with st.expander("Preview Data"):
                            st.dataframe(gsc_df.head(10), use_container_width=True)
                        
                        # Show statistics
                        st.metric("Unique Pages", gsc_df['Landing Page'].nunique())
                        st.metric("Unique Queries", gsc_df['Query'].nunique())
                        st.metric("Total Clicks", gsc_df['Clicks'].sum())
                        
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
                        with st.expander("Preview Data"):
                            st.dataframe(embeddings_df.head(10), use_container_width=True)
                        
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
            st.info(f"Ready to analyze {df['Landing Page'].nunique()} unique pages and {df['Query'].nunique()} unique queries")
            
            if st.button("Analyze Keyword Overlap", type="primary", key="keyword_btn"):
                with st.spinner("Analyzing keyword overlap (this may take a moment)..."):
                    results = KeywordCannibalizationAnalyzer.analyze(
                        st.session_state['gsc_df'], 
                        keyword_threshold
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
                        st.metric("Avg Overlap", f"{results['average_overlap']:.1f}%")
                    
                    # Display top issues
                    if results['top_issues']:
                        st.subheader("🔥 Top Cannibalization Issues")
                        for i, (pages, data) in enumerate(list(results['top_issues'].items())[:10], 1):
                            with st.expander(f"Issue #{i}: {data['overlap_percentage']}% Overlap - {data['total_shared']} shared keywords"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Page 1:**")
                                    st.code(data.get('page1_full', pages.split('||')[0]))
                                    st.metric("Unique Keywords", data.get('page1_unique', 0))
                                with col2:
                                    st.write("**Page 2:**")
                                    st.code(data.get('page2_full', pages.split('||')[1]))
                                    st.metric("Unique Keywords", data.get('page2_unique', 0))
                                
                                st.write("**Shared Keywords:**")
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
                st.info(f"Ready to analyze top {min(serp_sample_size, total_queries)} queries out of {total_queries} total unique queries")
                
                if st.button("Analyze SERP Overlap", type="primary", key="serp_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner(f"Analyzing SERPs for {min(serp_sample_size, total_queries)} queries..."):
                        queries = st.session_state['gsc_df']['Query'].unique()[:serp_sample_size]
                        
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
                                    queries.tolist(), 
                                    serper_api_key,
                                    serp_sample_size,
                                    update_progress
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
                                    st.metric("Queries with Overlap", results['queries_with_overlap'])
                                with col3:
                                    st.metric("Avg SERP Overlap", f"{results['average_overlap']:.1f}%")
                                with col4:
                                    st.metric("API Credits Used", results.get('api_credits_used', 0))
                                
                                # Failed queries warning
                                if results.get('failed_queries'):
                                    st.warning(f"⚠️ Failed to fetch SERPs for {len(results['failed_queries'])} queries")
                                
                                # Competition clusters
                                if results.get('competition_clusters'):
                                    st.subheader("🎯 Competition Clusters")
                                    for i, cluster in enumerate(results['competition_clusters'], 1):
                                        with st.expander(f"Cluster {i}: {len(cluster)} competing queries"):
                                            st.write("These queries are competing for similar SERP positions:")
                                            for query in cluster[:10]:
                                                st.write(f"• {query}")
                                            st.info("💡 Consider creating distinct content angles for each query")
                                
                                # Top overlaps
                                if results.get('top_overlaps'):
                                    st.subheader("🔥 Top SERP Overlaps")
                                    for query_pair, data in list(results['top_overlaps'].items())[:10]:
                                        with st.expander(f"{data['competition_level']} Competition: {data['overlap_percentage']}% overlap"):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write(f"**Query 1:** {data.get('query1', 'N/A')}")
                                            with col2:
                                                st.write(f"**Query 2:** {data.get('query2', 'N/A')}")
                                            
                                            st.write(f"**Shared Domains ({data.get('total_shared', 0)}):**")
                                            for domain in data.get('shared_domains', [])[:5]:
                                                st.write(f"• {domain}")
                                            
                                            st.write(f"**Position Score:** {data.get('position_weighted_score', 0)}")
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
            st.info(f"Ready to analyze {len(df)} pages for semantic similarity")
            
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
                            st.subheader("🔥 Highly Similar Pages")
                            for i, pair in enumerate(results['high_similarity_pairs'][:10], 1):
                                similarity_pct = pair['similarity'] * 100
                                with st.expander(f"Pair #{i}: {similarity_pct:.1f}% Similar"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Page 1:**")
                                        st.code(pair['page1'])
                                    with col2:
                                        st.write("**Page 2:**")
                                        st.code(pair['page2'])
                                    
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
    
    # Report Generation Section
    st.divider()
    
    # Check if any analysis has been run
    has_results = any([
        'keyword_results' in st.session_state,
        'content_results' in st.session_state,
        'topic_results' in st.session_state
    ])
    
    if has_results:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 Generate Comprehensive Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive report with insights..."):
                    report = generate_comprehensive_report(
                        st.session_state.get('keyword_results', {}),
                        st.session_state.get('content_results', {}),
                        st.session_state.get('topic_results', {}),
                        ai_provider
                    )
                    
                    st.markdown("### 📊 Complete Analysis Report")
                    
                    # Display report in expandable section
                    with st.expander("View Full Report", expanded=True):
                        st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=report,
                        file_name=f"seo_cannibalization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
    else:
        st.info("📊 Run at least one analysis to generate a report")

if __name__ == "__main__":
    main()
