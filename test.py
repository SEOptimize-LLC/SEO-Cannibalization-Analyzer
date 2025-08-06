"""
Test script for SEO Cannibalization Analyzer
Generates sample data and tests functionality locally
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

class TestDataGenerator:
    """Generate sample data for testing the app"""
    
    @staticmethod
    def test_serper_api(api_key: str = None):
        """Test Serper API functionality"""
        import asyncio
        import aiohttp
        from app import ContentCannibalizationAnalyzer
        
        print("\n🌐 Testing Serper API Integration...")
        
        if not api_key:
            print("⚠️ No Serper API key provided. Skipping test.")
            print("To test: python test.py serper YOUR_API_KEY")
            return None
        
        async def test_single_query():
            async with aiohttp.ClientSession() as session:
                result = await ContentCannibalizationAnalyzer.fetch_serp(
                    session, "SEO tools", api_key
                )
                return result
        
        # Test single query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_single_query())
        
        if result and result['domains']:
            print(f"✅ Successfully fetched {len(result['domains'])} results")
            print(f"Top 3 domains: {', '.join(result['domains'][:3])}")
            return True
        else:
            print("❌ Failed to fetch SERP results")
            return False
    
    @staticmethod
    def generate_gsc_report(num_queries: int = 500, num_pages: int = 50) -> pd.DataFrame:
        """Generate sample GSC report data"""
        
        # Sample keywords for realistic data
        keyword_templates = [
            "best {product} {year}",
            "how to {action} {object}",
            "{product} reviews",
            "{product} vs {competitor}",
            "buy {product} online",
            "{action} guide",
            "{product} {location}",
            "cheap {product}",
            "{product} near me",
            "what is {concept}"
        ]
        
        products = ["shoes", "laptop", "phone", "camera", "headphones", "watch", "tablet", "monitor", "keyboard", "mouse"]
        actions = ["choose", "buy", "find", "compare", "review", "test", "use", "install", "configure", "optimize"]
        objects = ["software", "hardware", "website", "app", "tool", "device", "system", "platform", "service", "product"]
        concepts = ["seo", "marketing", "analytics", "conversion", "optimization", "strategy", "automation", "integration"]
        locations = ["usa", "uk", "canada", "australia", "europe", "asia", "online", "store", "shop", "market"]
        
        # Generate queries
        queries = []
        for _ in range(num_queries):
            template = random.choice(keyword_templates)
            query = template.format(
                product=random.choice(products),
                action=random.choice(actions),
                object=random.choice(objects),
                concept=random.choice(concepts),
                location=random.choice(locations),
                competitor=random.choice(products),
                year=random.choice(["2024", "2025", ""])
            ).strip()
            queries.append(query)
        
        # Generate pages with intentional overlap
        base_urls = [
            "https://example.com/products/",
            "https://example.com/blog/",
            "https://example.com/guides/",
            "https://example.com/reviews/",
            "https://example.com/compare/",
            "https://example.com/category/"
        ]
        
        pages = []
        for i in range(num_pages):
            base = random.choice(base_urls)
            slug = f"{random.choice(products)}-{random.choice(['guide', 'review', 'comparison', 'best', 'top'])}-{i}"
            pages.append(f"{base}{slug}")
        
        # Create DataFrame with intentional cannibalization
        data = []
        for query in queries:
            # 30% chance of multiple pages ranking for same query (cannibalization)
            if random.random() < 0.3:
                num_pages_for_query = random.randint(2, 4)
            else:
                num_pages_for_query = 1
            
            selected_pages = random.sample(pages, min(num_pages_for_query, len(pages)))
            
            for page in selected_pages:
                data.append({
                    'Query': query,
                    'Landing Page': page,
                    'Clicks': random.randint(0, 100),
                    'Impressions': random.randint(10, 1000),
                    'Avg. Pos': round(random.uniform(1, 50), 1)
                })
        
        df = pd.DataFrame(data)
        
        # Sort by impressions descending
        df = df.sort_values('Impressions', ascending=False)
        
        return df
    
    @staticmethod
    def generate_embeddings_file(num_pages: int = 50, embedding_dim: int = 768) -> pd.DataFrame:
        """Generate sample embeddings file"""
        
        pages = []
        embeddings = []
        
        # Generate URLs
        for i in range(num_pages):
            pages.append(f"https://example.com/page-{i}")
        
        # Generate embeddings with intentional similarity clusters
        num_clusters = 5
        cluster_centers = [np.random.randn(embedding_dim) for _ in range(num_clusters)]
        
        for i in range(num_pages):
            # Assign page to a cluster
            cluster_id = i % num_clusters
            
            # Generate embedding close to cluster center
            noise_level = 0.1 if i < 10 else 0.3  # First 10 pages are very similar
            embedding = cluster_centers[cluster_id] + np.random.randn(embedding_dim) * noise_level
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Convert to list and then to JSON string
            embeddings.append(json.dumps(embedding.tolist()))
        
        df = pd.DataFrame({
            'Address': pages,
            '(ChatGPT) Extract embeddings from page content 1': embeddings
        })
        
        return df
    
    @staticmethod
    def save_test_data(output_dir: str = "./test_data/"):
        """Save test data files"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save GSC report
        gsc_df = TestDataGenerator.generate_gsc_report()
        gsc_df.to_csv(f"{output_dir}test_gsc_report.csv", index=False)
        print(f"✅ Saved test GSC report: {len(gsc_df)} rows")
        
        # Generate and save embeddings
        embeddings_df = TestDataGenerator.generate_embeddings_file()
        embeddings_df.to_csv(f"{output_dir}test_embeddings.csv", index=False)
        print(f"✅ Saved test embeddings: {len(embeddings_df)} pages")
        
        return gsc_df, embeddings_df

class LocalTester:
    """Test app functionality locally"""
    
    @staticmethod
    def test_keyword_analysis(df: pd.DataFrame):
        """Test keyword cannibalization analysis"""
        from app import KeywordCannibalizationAnalyzer
        
        print("\n🔤 Testing Keyword Cannibalization Analysis...")
        
        results = KeywordCannibalizationAnalyzer.analyze(df)
        
        print(f"Total pages analyzed: {results['total_pages_analyzed']}")
        print(f"Pages with overlap: {results['pages_with_overlap']}")
        
        if results['top_issues']:
            print("\nTop 3 cannibalization issues:")
            for pages, data in list(results['top_issues'].items())[:3]:
                print(f"  - {pages[:100]}: {data['overlap_percentage']}% overlap")
        
        return results
    
    @staticmethod
    def test_topic_analysis(df: pd.DataFrame):
        """Test topic cannibalization analysis"""
        from app import TopicCannibalizationAnalyzer
        
        print("\n🧠 Testing Topic Cannibalization Analysis...")
        
        results = TopicCannibalizationAnalyzer.analyze(df)
        
        if 'error' not in results:
            print(f"Total pages analyzed: {results['total_pages']}")
            print(f"High similarity pairs: {results['pages_with_high_similarity']}")
            print(f"Average similarity: {results['average_similarity']:.3f}")
            
            if results['high_similarity_pairs']:
                print("\nTop 3 similar page pairs:")
                for pair in results['high_similarity_pairs'][:3]:
                    print(f"  - Similarity {pair['similarity']:.3f}: {pair['page1'][:40]} <-> {pair['page2'][:40]}")
        else:
            print(f"Error: {results['error']}")
        
        return results
    
    @staticmethod
    def test_utils():
        """Test utility functions"""
        from utils import DataValidator, URLProcessor, ReportGenerator
        
        print("\n🔧 Testing Utility Functions...")
        
        # Test URL normalization
        test_urls = [
            "https://www.example.com/page/",
            "https://example.com/page",
            "HTTPS://EXAMPLE.COM/PAGE?param=1&another=2",
            "https://example.com/page#section"
        ]
        
        print("\nURL Normalization:")
        for url in test_urls:
            normalized = URLProcessor.normalize_url(url)
            print(f"  {url[:40]} -> {normalized}")
        
        # Test impact calculation
        test_data = {
            'keyword_results': {'pages_with_overlap': 15},
            'content_results': {'queries_with_overlap': 20},
            'topic_results': {'pages_with_high_similarity': 10}
        }
        
        impact = ReportGenerator.calculate_potential_impact(test_data)
        print(f"\nImpact Assessment:")
        print(f"  Severity Score: {impact['severity_score']}")
        print(f"  Traffic Increase: {impact['estimated_traffic_increase']}")
        print(f"  Priority Level: {impact['priority_level']}")
        
        return True
    
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        print("=" * 60)
        print("SEO Cannibalization Analyzer - Test Suite")
        print("=" * 60)
        
        # Generate test data
        print("\n📁 Generating test data...")
        gsc_df, embeddings_df = TestDataGenerator.save_test_data()
        
        # Run tests
        try:
            keyword_results = LocalTester.test_keyword_analysis(gsc_df)
            topic_results = LocalTester.test_topic_analysis(embeddings_df)
            utils_results = LocalTester.test_utils()
            
            print("\n" + "=" * 60)
            print("✅ All tests completed successfully!")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def generate_sample_config():
    """Generate sample configuration files"""
    
    # Sample secrets.toml
    secrets_content = """
# Streamlit Secrets Configuration
# Add this to .streamlit/secrets.toml or configure in Streamlit Cloud

[api_keys]
serper_key = "your-serper-api-key-here"  # REQUIRED for SERP analysis
openai_key = "sk-your-openai-key-here"  # Optional for AI recommendations
anthropic_key = "sk-ant-your-anthropic-key-here"  # Optional for AI recommendations
google_key = "your-google-ai-key-here"  # Optional for AI recommendations

[app_config]
max_serp_queries = 50
default_similarity_threshold = 0.8
default_overlap_threshold = 10

[rate_limits]
serper_requests_per_second = 10
serper_batch_size = 10
delay_between_batches = 1
"""
    
    # Sample config.toml
    config_content = """
# Streamlit Configuration
# Add this to .streamlit/config.toml

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
"""
    
    # Sample .env file for local development
    env_content = """
# Environment Variables for Local Development
# Create a .env file with these variables

# Required
SERPER_API_KEY=your-serper-api-key-here

# Optional AI Providers
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-ai-key-here

# App Configuration
MAX_SERP_QUERIES=50
SERP_BATCH_SIZE=10
"""
    
    # Save files
    import os
    
    os.makedirs(".streamlit", exist_ok=True)
    
    with open(".streamlit/secrets.toml.example", "w") as f:
        f.write(secrets_content)
    
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content)
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("✅ Generated sample configuration files:")
    print("  - .streamlit/secrets.toml.example")
    print("  - .streamlit/config.toml")
    print("  - .env.example")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run tests
            LocalTester.run_all_tests()
        elif sys.argv[1] == "generate":
            # Generate test data only
            TestDataGenerator.save_test_data()
            print("\n✅ Test data generated successfully!")
        elif sys.argv[1] == "config":
            # Generate configuration files
            generate_sample_config()
        elif sys.argv[1] == "serper" and len(sys.argv) > 2:
            # Test Serper API
            LocalTester.test_serper_api(sys.argv[2])
    else:
        print("""
SEO Cannibalization Analyzer - Test Script

Usage:
    python test.py test                  # Run all tests
    python test.py generate              # Generate test data files
    python test.py config                # Generate configuration files
    python test.py serper YOUR_API_KEY  # Test Serper API integration

To run the Streamlit app:
    streamlit run app.py
    
To get a Serper API key:
    Visit https://serper.dev and sign up for free (2,500 queries/month)
        """)
