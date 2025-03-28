"""
Configuration settings for different SMOL agents
"""

# General agent settings
DEFAULT_AGENT_CONFIG = {
    "memory_size": 10,  # Number of previous interactions to remember
    "max_iterations": 5,  # Maximum iterations for a single task
    "thinking_style": "step_by_step",  # How the agent approaches problems
    "verbose": True,  # Whether to print detailed logs
}

# Data collection agent
DATA_COLLECTOR_CONFIG = {
    **DEFAULT_AGENT_CONFIG,
    "sources": [
        "web_search",
        "news_api",
        "sec_filings",
        "social_media",
    ],
    "max_results_per_source": 20,
    "date_range_days": 30,  # How far back to look for data
    "check_duplicates": True,
    "extract_metadata": True,
}

# Analyzer agent
ANALYZER_CONFIG = {
    **DEFAULT_AGENT_CONFIG,
    "analysis_types": [
        "trend_analysis",
        "competitor_comparison",
        "market_size_estimation",
        "growth_projections",
    ],
    "time_periods": ["quarterly", "annual"],
    "statistical_tests": ["t_test", "correlation", "regression"],
    "outlier_detection": True,
    "significance_threshold": 0.05,
}

# Sentiment agent
SENTIMENT_AGENT_CONFIG = {
    **DEFAULT_AGENT_CONFIG,
    "sentiment_categories": ["positive", "negative", "neutral"],
    "analyze_emotions": True,
    "emotion_categories": ["joy", "anger", "fear", "surprise", "sadness"],
    "topic_extraction": True,
    "max_topics": 5,
    "sentiment_aggregation": "weighted_average",
}

# Report agent
REPORT_AGENT_CONFIG = {
    **DEFAULT_AGENT_CONFIG,
    "report_formats": ["markdown", "html", "pdf"],
    "include_executive_summary": True,
    "include_methodology": True,
    "include_visualizations": True,
    "visualization_types": ["bar", "line", "pie", "heatmap", "wordcloud"],
    "citation_style": "APA",
}

# Agent communication settings
AGENT_COMMUNICATION = {
    "message_format": "json",
    "broadcast_results": True,
    "require_confirmation": False,
    "error_handling": "retry",  # Options: retry, skip, abort
    "max_message_size": 10240,  # bytes
}

# Agent tools and capabilities
AGENT_TOOLS = {
    "web_search": True,
    "file_reading": True,
    "data_visualization": True,
    "text_analysis": True,
    "numeric_computation": True,
    "database_access": False,  # Requires additional configuration
}