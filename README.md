# SMOL Agents Market Research

A comprehensive AI-powered market research system using Small Language Model (SMOL) agents for specialized analytical tasks.

## Overview

This system leverages multiple specialized AI agents, each focused on a specific aspect of market research:

1. **Data Collection Agent** - Gathers data from various sources including web search, news APIs, social media, and more
2. **Analysis Agent** - Processes market data to identify trends, competitors, market size, and growth projections
3. **Sentiment Agent** - Analyzes market sentiment and emotions in the collected data
4. **Report Agent** - Compiles all findings into comprehensive, human-readable reports

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market_research_smol.git
cd market_research_smol
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create a `.env` file in the project root):
```
OPENAI_API_KEY=your_openai_api_key
SERP_API_KEY=your_serp_api_key
NEWS_API_KEY=your_news_api_key
```

## Usage

### Command Line Interface

Run market research from the command line:

```bash
# Full research pipeline (collect data, analyze, and generate report)
python main.py --market-domain "artificial intelligence" --mode full

# Data collection only
python main.py --market-domain "blockchain" --mode collect

# Analysis only (using existing data files)
python main.py --market-domain "electric vehicles" --mode analyze --data-files data/raw/web_search_ev_20230815_120000.json data/raw/news_api_ev_20230815_120000.json

# Report generation only (using existing analysis files)
python main.py --market-domain "renewable energy" --mode report --data-files data/processed/trend_analysis_renewable_energy_20230815_120000.json data/processed/sentiment_renewable_energy_20230815_120000.json --output-format html
```

### API

Start the API server:

```bash
python -m api.server
```

The API will be available at `http://localhost:8000`. You can use the following endpoints:

- `POST /research` - Start a new market research task
- `GET /research/{task_id}/status` - Get the status of a research task
- `GET /research/{task_id}/report` - Get the report from a completed research task
- `POST /analyze` - Run analysis on existing data files
- `POST /report` - Generate a report from existing analysis files
- `GET /files/data` - List available data files
- `GET /files/reports` - List available report files

Example API request to start a new research task:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "market_domain": "quantum computing",
    "search_terms": ["market size", "applications", "companies", "trends", "investments"],
    "include_visualizations": true,
    "output_format": "markdown"
  }'
```

## Project Structure

```
market_research_smol/
│
├── config/                # Configuration settings
│   ├── settings.py        # Global settings
│   └── agent_configs.py   # Agent-specific configurations
│
├── agents/                # SMOL agents
│   ├── base_agent.py      # Base agent class
│   ├── data_collector.py  # Web scraping and data collection agent
│   ├── analyzer.py        # Data analysis agent
│   ├── sentiment_agent.py # Sentiment analysis agent
│   └── report_agent.py    # Report generation agent
│
├── models/                # AI models
│   └── embedding_model.py # Text embedding functionality
│
├── data/                  # Data storage
│   ├── raw/               # Raw collected data
│   ├── processed/         # Processed analysis data
│   └── reports/           # Generated reports
│
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data handling utilities
│   ├── web_utils.py       # Web scraping utilities
│   └── nlp_utils.py       # Text processing utilities
│
├── pipelines/             # Workflow pipelines
│   ├── collection_pipeline.py # Data collection workflow
│   └── analysis_pipeline.py  # Analysis workflow
│
├── api/                   # API endpoints
│   ├── routes.py          # API routes
│   └── server.py          # API server
│
├── requirements.txt       # Project dependencies
├── main.py                # Main entry point
└── README.md              # Project documentation
```

## Extending the System

### Adding New Data Sources

To add a new data source to the data collector agent:

1. Open `agents/data_collector.py`
2. Add a new collection method following the pattern of existing ones:
```python
def collect_from_new_source(self, domain: str, search_terms: List[str], max_results: int) -> List[Dict]:
    """Collect data from new source"""
    # Implementation here
    return results
```
3. Add the new source to the list of sources in `config/agent_configs.py`

### Adding New Analysis Methods

To add a new analysis method to the analyzer agent:

1. Open `agents/analyzer.py`
2. Add a new analysis method following the pattern of existing ones:
```python
def analyze_new_aspect(self, data: List[Dict], domain: str) -> Dict[str, Any]:
    """Analyze a new aspect of the market"""
    # Implementation here
    return results
```
3. Add the new analysis type to the list in `config/agent_configs.py`

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses several open-source libraries and APIs:
- FastAPI for the API server
- Transformers for NLP models
- Requests for web data collection
- Pandas for data manipulation
- Matplotlib for visualizations