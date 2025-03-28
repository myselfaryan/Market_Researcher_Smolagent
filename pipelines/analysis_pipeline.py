"""
Pipeline for data analysis using multiple analysis agents
"""

import logging
from typing import Dict, List, Any
import time

from ..agents.analyzer import AnalyzerAgent
from ..agents.sentiment_agent import SentimentAgent

logger = logging.getLogger("pipelines.analysis")

def run_analysis_pipeline(market_domain: str, data_files: List[str]) -> Dict[str, Any]:
    """
    Run the data analysis pipeline
    
    Args:
        market_domain: Market domain being researched
        data_files: List of files containing data to analyze
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Starting analysis pipeline for {market_domain}")
    start_time = time.time()
    
    analysis_results = {
        "market_domain": market_domain,
        "analyses": {},
        "analysis_files": []
    }
    
    # Run market analysis
    analyzer = AnalyzerAgent()
    market_analysis_task = {
        "market_domain": market_domain,
        "data_files": data_files,
        "analysis_types": [
            "trend_analysis",
            "competitor_comparison",
            "market_size_estimation",
            "growth_projections"
        ]
    }
    
    logger.info("Running market analysis")
    market_analysis = analyzer.act(market_analysis_task)
    
    if market_analysis.get("status") == "success":
        analysis_results["analyses"]["market_analysis"] = market_analysis
        
        # Extract output files for each analysis type
        for analysis_type, results in market_analysis.get("analysis", {}).items():
            if "output_file" in results:
                analysis_results["analysis_files"].append(results["output_file"])
                
        logger.info("Market analysis completed successfully")
    else:
        logger.error("Market analysis failed")
        logger.error(market_analysis.get("message", "Unknown error"))
    
    # Run sentiment analysis
    sentiment_agent = SentimentAgent()
    sentiment_task = {
        "market_domain": market_domain,
        "data_files": data_files,
        "analyze_emotions": True,
        "extract_topics": True
    }
    
    logger.info("Running sentiment analysis")
    sentiment_analysis = sentiment_agent.act(sentiment_task)
    
    if sentiment_analysis.get("status") == "success":
        analysis_results["analyses"]["sentiment_analysis"] = sentiment_analysis
        
        # Extract sentiment output file
        if "output_file" in sentiment_analysis.get("sentiment_analysis", {}):
            analysis_results["analysis_files"].append(
                sentiment_analysis["sentiment_analysis"]["output_file"]
            )
            
        logger.info("Sentiment analysis completed successfully")
    else:
        logger.error("Sentiment analysis failed")
        logger.error(sentiment_analysis.get("message", "Unknown error"))
    
    # Log completion
    duration = time.time() - start_time
    logger.info(f"Analysis pipeline completed in {duration:.2f} seconds")
    
    return analysis_results