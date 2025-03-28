"""
Main entry point for the SMOL Agents Market Research system
"""

import os
import logging
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any

from config.settings import DATA_DIR, LOG_LEVEL
from agents.data_collector import DataCollectorAgent
from agents.analyzer import AnalyzerAgent
from agents.sentiment_agent import SentimentAgent
from agents.report_agent import ReportAgent
from pipelines.collection_pipeline import run_collection_pipeline
from pipelines.analysis_pipeline import run_analysis_pipeline

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_research.log')
    ]
)

logger = logging.getLogger("main")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SMOL Agents Market Research')
    
    parser.add_argument('--market-domain', type=str, required=True,
                        help='Market domain to research (e.g., "artificial intelligence")')
    
    parser.add_argument('--search-terms', type=str, nargs='+',
                        help='Specific search terms to collect data for')
    
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'report', 'full'],
                        default='full', help='Mode of operation')
    
    parser.add_argument('--data-files', type=str, nargs='*',
                        help='Data files to use for analysis or reporting (if not running collection)')
    
    parser.add_argument('--output-format', type=str, choices=['markdown', 'html'],
                        default='markdown', help='Output format for the report')
    
    parser.add_argument('--include-visualizations', action='store_true',
                        help='Include visualizations in the report')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    logger.info("Starting SMOL Agents Market Research")
    
    # Parse arguments
    args = parse_arguments()
    market_domain = args.market_domain
    
    # Default search terms if not provided
    if not args.search_terms:
        search_terms = [
            "market size", "trends", "growth", "competitors", "forecast",
            "opportunities", "challenges", "technologies", "innovations", "demand"
        ]
    else:
        search_terms = args.search_terms
    
    logger.info(f"Market domain: {market_domain}")
    logger.info(f"Search terms: {search_terms}")
    logger.info(f"Mode: {args.mode}")
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{market_domain.replace(' ', '_')}_{timestamp}"
    
    # Create results dictionary to store outputs
    results = {
        "run_id": run_id,
        "market_domain": market_domain,
        "search_terms": search_terms,
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": args.mode,
        "output_files": {}
    }
    
    # Data collection
    if args.mode in ['collect', 'full']:
        logger.info("Running data collection pipeline")
        
        collection_results = run_collection_pipeline(
            market_domain=market_domain,
            search_terms=search_terms
        )
        
        results["data_collection"] = collection_results
        data_files = [file for source in collection_results.get("sources", {}).values() 
                    for file in [source.get("data_file")] if file]
    else:
        data_files = args.data_files or []
    
    # Data analysis
    if args.mode in ['analyze', 'full']:
        logger.info("Running analysis pipeline")
        
        if not data_files:
            logger.warning("No data files available for analysis")
            return
        
        analysis_results = run_analysis_pipeline(
            market_domain=market_domain,
            data_files=data_files
        )
        
        results["data_analysis"] = analysis_results
        analysis_files = [
            file for analysis_type in analysis_results.get("analyses", {}).values() 
            for file in [analysis_type.get("output_file")] if file
        ]
    else:
        analysis_files = args.data_files or []
    
    # Report generation
    if args.mode in ['report', 'full']:
        logger.info("Generating report")
        
        if not analysis_files:
            logger.warning("No analysis files available for report generation")
            return
        
        # Initialize report agent
        report_agent = ReportAgent()
        
        # Generate report
        report_results = report_agent.act({
            "analysis_files": analysis_files,
            "market_domain": market_domain,
            "format": args.output_format,
            "include_sections": {
                "visualizations": args.include_visualizations
            }
        })
        
        results["report"] = report_results
        
        if report_results.get("status") == "success":
            report_file = report_results.get("report", {}).get("file")
            if report_file:
                logger.info(f"Report generated: {report_file}")
                results["output_files"]["report"] = report_file
    
    # Save results summary
    results_file = DATA_DIR / f"results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info("SMOL Agents Market Research completed")
    
    return results

if __name__ == "__main__":
    main()