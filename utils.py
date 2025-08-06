"""
Utility functions for SEO Cannibalization Analyzer
Helper functions for data processing, visualization, and analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import re
from urllib.parse import urlparse
import hashlib
import json

class DataValidator:
    """Validates input data formats and integrity"""
    
    @staticmethod
    def validate_gsc_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate GSC report format"""
        required_columns = ['Query', 'Landing Page', 'Clicks', 'Impressions']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check data types
        try:
            df['Clicks'] = pd.to_numeric(df['Clicks'])
            df['Impressions'] = pd.to_numeric(df['Impressions'])
        except:
            return False, "Clicks and Impressions must be numeric"
        
        # Check for empty data
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check for valid URLs
        invalid_urls = df[~df['Landing Page'].str.startswith(('http://', 'https://'))]
        if not invalid_urls.empty:
            return False, f"Found {len(invalid_urls)} invalid URLs"
        
        return True, "Data validation successful"
    
    @staticmethod
    def validate_embeddings_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate embeddings file format"""
        
        # Check for required columns
        if 'Address' not in df.columns:
            return False, "Missing 'Address' column"
        
        # Check for embeddings column (flexible naming)
        embeddings_cols = [col for col in df.columns if 'embedding' in col.lower()]
        if not embeddings_cols:
            return False, "No embeddings column found"
        
        # Validate URLs
        invalid_urls = df[~df['Address'].str.startswith(('http://', 'https://'))]
        if not invalid_urls.empty:
            return False, f"Found {len(invalid_urls)} invalid URLs"
        
        return True, "Embeddings data validation successful"

class URLProcessor:
    """Utilities for URL processing and normalization"""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for comparison"""
        # Remove trailing slash
        url = url.rstrip('/')
        
        # Remove www
        url = re.sub(r'://www\.', '://', url)
        
        # Remove fragment
        url = url.split('#')[0]
        
        # Sort query parameters for consistency
        if '?' in url:
            base, params = url.split('?', 1)
            params = '&'.join(sorted(params.split('&')))
            url = f"{base}?{params}"
        
        return url.lower()
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    @staticmethod
    def get_url_slug(url: str) -> str:
        """Extract slug/path from URL"""
        try:
            parsed = urlparse(url)
            return parsed.path
        except:
            return ""
    
    @staticmethod
    def group_urls_by_pattern(urls: List[str]) -> Dict[str, List[str]]:
        """Group URLs by common patterns"""
        patterns = {}
        
        for url in urls:
            # Extract path pattern (replace numbers with placeholders)
            path = URLProcessor.get_url_slug(url)
            pattern = re.sub(r'\d+', '{id}', path)
            pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '{uuid}', pattern)
            
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(url)
        
        return patterns

class VisualizationHelper:
    """Helper functions for creating visualizations"""
    
    @staticmethod
    def create_cannibalization_heatmap(overlap_matrix: pd.DataFrame, 
                                      title: str = "Cannibalization Heatmap") -> go.Figure:
        """Create an interactive heatmap for overlap visualization"""
        
        # Limit to top 30 pages for readability
        if len(overlap_matrix) > 30:
            # Get pages with highest total overlap
            total_overlap = overlap_matrix.sum(axis=1)
            top_pages = total_overlap.nlargest(30).index
            overlap_matrix = overlap_matrix.loc[top_pages, top_pages]
        
        # Create shortened labels for better display
        labels = [URLProcessor.get_url_slug(url)[-30:] for url in overlap_matrix.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=overlap_matrix.values,
            x=labels,
            y=labels,
            colorscale='Reds',
            hovertemplate='Page 1: %{y}<br>Page 2: %{x}<br>Overlap: %{z:.1f}%<extra></extra>',
            colorbar=dict(title="Overlap %")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Pages",
            yaxis_title="Pages",
            height=600,
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    @staticmethod
    def create_similarity_network(similarity_pairs: List[Dict], 
                                 threshold: float = 0.8) -> go.Figure:
        """Create network graph of similar pages"""
        
        # Extract nodes and edges
        nodes = set()
        edges = []
        
        for pair in similarity_pairs:
            if pair['similarity'] >= threshold:
                nodes.add(pair['page1'])
                nodes.add(pair['page2'])
                edges.append((pair['page1'], pair['page2'], pair['similarity']))
        
        # Create node positions using simple circular layout
        node_list = list(nodes)
        n = len(node_list)
        
        if n == 0:
            return go.Figure()
        
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        pos = {node: (np.cos(theta[i]), np.sin(theta[i])) for i, node in enumerate(node_list)}
        
        # Create edge traces
        edge_trace = []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge[2]*5, color='rgba(125,125,125,0.5)'),
                hoverinfo='none'
            ))
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in node_list],
            y=[pos[node][1] for node in node_list],
            mode='markers+text',
            text=[URLProcessor.get_url_slug(node)[-20:] for node in node_list],
            textposition="top center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hovertext=node_list,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Page Similarity Network",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_cannibalization_summary_chart(keyword_overlap: int, 
                                            serp_overlap: int, 
                                            topic_similarity: float) -> go.Figure:
        """Create summary gauge charts for cannibalization metrics"""
        
        fig = go.Figure()
        
        # Keyword Cannibalization Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=keyword_overlap,
            title={'text': "Keyword Overlap"},
            domain={'x': [0, 0.3], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if keyword_overlap > 50 else "orange" if keyword_overlap > 20 else "green"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 50], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}
        ))
        
        # SERP Overlap Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=serp_overlap,
            title={'text': "SERP Overlap"},
            domain={'x': [0.35, 0.65], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if serp_overlap > 60 else "orange" if serp_overlap > 30 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60}}
        ))
        
        # Topic Similarity Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=topic_similarity * 100,
            title={'text': "Topic Similarity"},
            domain={'x': [0.7, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if topic_similarity > 0.9 else "orange" if topic_similarity > 0.7 else "green"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        
        fig.update_layout(
            title="Cannibalization Severity Dashboard",
            height=300
        )
        
        return fig

class ReportGenerator:
    """Generate detailed reports and recommendations"""
    
    @staticmethod
    def generate_fix_priority_matrix(keyword_data: Dict, 
                                    content_data: Dict, 
                                    topic_data: Dict) -> pd.DataFrame:
        """Generate priority matrix for fixes"""
        
        fixes = []
        
        # Process keyword cannibalization
        if 'top_issues' in keyword_data:
            for pages, data in list(keyword_data['top_issues'].items())[:10]:
                page1, page2 = pages.split('||')
                fixes.append({
                    'Type': 'Keyword',
                    'Page 1': page1[:50],
                    'Page 2': page2[:50],
                    'Severity': data['overlap_percentage'],
                    'Impact': 'High' if data['overlap_percentage'] > 50 else 'Medium',
                    'Action': 'Consolidate' if data['overlap_percentage'] > 70 else 'Differentiate',
                    'Priority': 1 if data['overlap_percentage'] > 70 else 2
                })
        
        # Process content cannibalization
        if 'top_overlaps' in content_data:
            for query_pair, data in list(content_data['top_overlaps'].items())[:10]:
                q1, q2 = query_pair.split('||')
                fixes.append({
                    'Type': 'Content',
                    'Page 1': q1,
                    'Page 2': q2,
                    'Severity': data['overlap_percentage'],
                    'Impact': 'High' if data['overlap_percentage'] > 60 else 'Medium',
                    'Action': 'Target different intents',
                    'Priority': 1 if data['overlap_percentage'] > 60 else 3
                })
        
        # Process topic cannibalization
        if 'high_similarity_pairs' in topic_data:
            for pair in topic_data['high_similarity_pairs'][:10]:
                fixes.append({
                    'Type': 'Topic',
                    'Page 1': pair['page1'][:50],
                    'Page 2': pair['page2'][:50],
                    'Severity': pair['similarity'] * 100,
                    'Impact': 'High' if pair['similarity'] > 0.9 else 'Medium',
                    'Action': 'Merge content' if pair['similarity'] > 0.95 else 'Create topic cluster',
                    'Priority': 1 if pair['similarity'] > 0.95 else 2
                })
        
        df = pd.DataFrame(fixes)
        if not df.empty:
            df = df.sort_values('Priority')
        
        return df
    
    @staticmethod
    def generate_redirect_map(overlap_data: Dict, threshold: float = 70) -> pd.DataFrame:
        """Generate 301 redirect recommendations"""
        
        redirects = []
        
        for pages, data in overlap_data.items():
            if data['overlap_percentage'] > threshold:
                page1, page2 = pages.split('||')
                
                # Determine which page to keep (simplified logic)
                # In production, would use traffic, backlinks, etc.
                redirects.append({
                    'From URL': page2,
                    'To URL': page1,
                    'Overlap %': data['overlap_percentage'],
                    'Shared Keywords': data['total_shared'],
                    'Status': 'Recommended'
                })
        
        return pd.DataFrame(redirects)
    
    @staticmethod
    def calculate_potential_impact(cannibalization_data: Dict) -> Dict:
        """Calculate potential traffic impact of fixing cannibalization"""
        
        # Research shows 25-110% traffic increase potential
        keyword_issues = cannibalization_data.get('keyword_results', {}).get('pages_with_overlap', 0)
        content_issues = cannibalization_data.get('content_results', {}).get('queries_with_overlap', 0)
        topic_issues = cannibalization_data.get('topic_results', {}).get('pages_with_high_similarity', 0)
        
        # Calculate severity score (0-100)
        severity = min(100, (keyword_issues * 2) + (content_issues * 1.5) + (topic_issues * 3))
        
        # Estimate traffic increase potential
        if severity > 70:
            traffic_increase = "50-110%"
            ranking_improvement = "3-5 positions"
        elif severity > 40:
            traffic_increase = "25-50%"
            ranking_improvement = "2-3 positions"
        else:
            traffic_increase = "10-25%"
            ranking_improvement = "1-2 positions"
        
        return {
            'severity_score': severity,
            'estimated_traffic_increase': traffic_increase,
            'expected_ranking_improvement': ranking_improvement,
            'priority_level': 'Critical' if severity > 70 else 'High' if severity > 40 else 'Medium',
            'estimated_time_to_fix': f"{severity // 10} weeks",
            'roi_potential': 'Very High' if severity > 70 else 'High' if severity > 40 else 'Medium'
        }

class ExportHelper:
    """Helper functions for exporting data and reports"""
    
    @staticmethod
    def export_to_excel(data_dict: Dict, filename: str = "cannibalization_analysis.xlsx") -> bytes:
        """Export analysis results to Excel file"""
        import io
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export each analysis type to separate sheet
            for sheet_name, data in data_dict.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def generate_csv_redirect_file(redirects: pd.DataFrame) -> str:
        """Generate CSV file content for redirects"""
        return redirects.to_csv(index=False)
    
    @staticmethod
    def generate_json_report(analysis_results: Dict) -> str:
        """Generate JSON report of all findings"""
        
        # Clean data for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        cleaned_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                cleaned_results[key] = {k: clean_for_json(v) for k, v in value.items()}
            else:
                cleaned_results[key] = clean_for_json(value)
        
        return json.dumps(cleaned_results, indent=2, default=str)
