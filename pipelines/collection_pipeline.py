"""
Pipeline for data collection from various sources
"""

import logging
from typing import Dict, List, Any
import time

from ..agents.data_collector import DataCollectorAgent

logger = logging.getLogger("pipelines.collection")

def run_collection_pipeline(market_domain: str, search_terms: List[str]) -> Dict[str, Any]:
    """
    Run the data collection pipeline
    
    Args:
        market_domain: Market domain to research
        search_terms: List of search terms to collect data for
        
    Returns:
        Dictionary with collection results
    """
    logger.info(f"Starting data collection pipeline for {market_domain}")
    start_time = time.time()
    
    # Initialize data collector agent
    collector = DataCollectorAgent()
    
    # Define collection task
    task = {
        "market_domain": market_domain,
        "search_terms": search_terms,
        "sources": ["web_search", "news_api", "social_media", "sec_filings"],
        "max_results": 20
    }
    
    # Run collection
    collection_results = collector.act(task)
    
    # Log completion
    duration = time.time() - start_time
    logger.info(f"Data collection completed in {duration:.2f} seconds")
    
    if collection_results.get("status") != "success":
        logger.error("Data collection failed")
        logger.error(collection_results.get("message", "Unknown error"))
    else:
        sources = collection_results.get("sources", {})
        total = sum(s.get("count", 0) for s in sources.values())
        logger.info(f"Collected {total} items from {len(sources)} sources")
    
    return collection_results