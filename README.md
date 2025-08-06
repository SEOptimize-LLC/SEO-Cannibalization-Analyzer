# 🔍 SEO Cannibalization Analyzer

A comprehensive Streamlit application for detecting and analyzing keyword, content, and topic cannibalization issues that can impact your website's SEO performance.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.29.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

### Three Types of Cannibalization Analysis

1. **Keyword Cannibalization**
   - Detects URL overlap for keywords
   - Calculates overlap percentages
   - Identifies top cannibalization issues
   - Interactive heatmap visualization

2. **Content/SERP Cannibalization**
   - Professional SERP data via Serper API
   - Position-weighted overlap scoring
   - Competition cluster identification
   - Query performance insights

3. **Topic Cannibalization**
   - Semantic similarity analysis using embeddings
   - Cosine similarity calculations
   - High-similarity pair detection
   - Topic cluster recommendations

### AI-Powered Recommendations
- Integration with OpenAI, Anthropic, and Google AI
- Context-aware analysis and fixes
- Priority-based action plans
- Expected impact predictions

### Comprehensive Reporting
- Detailed analysis reports
- Export to Markdown, Excel, CSV, JSON
- 301 redirect recommendations
- Implementation roadmaps

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Serper API key (required) - [Get free key](https://serper.dev)
- AI API keys (optional) - OpenAI, Anthropic, or Google

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/seo-cannibalization-analyzer.git
cd seo-cannibalization-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**

Create `.streamlit/secrets.toml`:
```toml
[api_keys]
serper_key = "your-serper-api-key"  # Required
openai_key = "sk-your-openai-key"   # Optional
anthropic_key = "sk-ant-your-key"   # Optional
google_key = "your-google-key"      # Optional
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📊 Data Requirements

### GSC Report (CSV)
Export from Google Search Console with columns:
- `Query` - Search query
- `Landing Page` - URL
- `Clicks` - Click count
- `Impressions` - Impression count
- `Avg. Pos` - Average position

### Embeddings File (CSV)
- `Address` - Page URL
- `(ChatGPT) Extract embeddings from page content 1` - JSON array of embeddings

## 🔧 Configuration

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add secrets in dashboard:
```toml
[api_keys]
serper_key = "your-key"
# Add other keys as needed
```
5. Deploy

### Environment Variables (Local)

Create `.env` file:
```bash
SERPER_API_KEY=your-serper-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## 🧪 Testing

### Generate Test Data
```bash
python test.py generate
```

### Run Tests
```bash
python test.py test
```

### Test Serper API
```bash
python test.py serper YOUR_API_KEY
```

### Generate Config Files
```bash
python test.py config
```

## 📈 Usage Guide

### Step 1: Upload Data
1. Navigate to "Data Upload" tab
2. Upload GSC Performance Report
3. Upload Embeddings file (for topic analysis)

### Step 2: Configure APIs
1. Enter Serper API key (required for SERP analysis)
2. Optionally configure AI provider for recommendations

### Step 3: Run Analyses
1. **Keyword Analysis**: Detect URL overlap
2. **Content Analysis**: Analyze SERP competition
3. **Topic Analysis**: Calculate semantic similarities

### Step 4: Generate Report
- Click "Generate Comprehensive Report"
- Review AI recommendations
- Download report in preferred format

## 🏗️ Architecture

```
├── app.py                  # Main Streamlit application
├── utils.py               # Helper functions and utilities
├── test.py                # Testing and data generation
├── requirements.txt       # Python dependencies
├── .streamlit/
│   ├── config.toml       # UI configuration
│   └── secrets.toml      # API keys (not in repo)
└── test_data/            # Sample data for testing
```

## 📊 Metrics & Impact

Based on SEO research, fixing cannibalization can result in:
- **25-110%** increase in organic traffic
- **2-5 positions** ranking improvement
- **10-30%** CTR improvement

## 🔑 API Pricing

### Serper API
- **Free Tier**: 2,500 queries/month
- **Paid Plans**: Starting at $50/month

### AI Providers (Optional)
- **OpenAI**: Pay-per-token
- **Anthropic**: Pay-per-token
- **Google AI**: Free tier available

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the amazing framework
- [Serper](https://serper.dev) for reliable SERP data
- SEO community for research and insights

## 📧 Support

For issues or questions:
- Create an issue in GitHub
- Check [documentation](docs/)
- Review [FAQ](FAQ.md)

## 🚀 Roadmap

### Version 1.1 (Q1 2025)
- [ ] Direct GSC API integration
- [ ] Bulk site processing
- [ ] Historical tracking

### Version 1.2 (Q2 2025)
- [ ] Automated fix implementation
- [ ] Custom embedding generation
- [ ] Advanced filtering

### Version 2.0 (Q3 2025)
- [ ] Multi-site management
- [ ] Team collaboration
- [ ] REST API endpoint

---

**Created with ❤️ for the SEO community**

*Last Updated: 2024*
