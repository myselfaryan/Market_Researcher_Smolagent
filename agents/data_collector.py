"""
Data collection agent responsible for gathering market research data
from various sources like web pages, news articles, and databases.
"""

import os
import json
import datetime
import hashlib
from typing import Dict, List, Any, Optional, Union
import logging
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

from .base_agent import BaseAgent
from ..config.settings import (
    REQUEST_HEADERS, REQUEST_TIMEOUT, MAX_RETRIES, 
    RETRY_DELAY, RAW_DATA_DIR, SERP_API_KEY, NEWS_API_KEY
)
from ..utils.web_utils import make_request, extract_text_from_html
from ..utils.data_utils import save_json, save_dataframe

class DataCollectorAgent(BaseAgent):
    """Agent for collecting data from various sources"""
    
    def __init__(self, name: str = "data_collector", config: Dict[str, Any] = None):
        """
        Initialize the data collector agent
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        from ..config.agent_configs import DATA_COLLECTOR_CONFIG
        super().__init__(name, config or DATA_COLLECTOR_CONFIG)
        
        self.sources = self.config.get("sources", [])
        self.max_results_per_source = self.config.get("max_results_per_source", 20)
        self.date_range_days = self.config.get("date_range_days", 30)
        self.check_duplicates = self.config.get("check_duplicates", True)
        
        # Create a directory for this collector's data
        self.data_dir = RAW_DATA_DIR / name
        self.data_dir.mkdir(exist_ok=True)
        
        # Store collected data hashes to avoid duplicates
        self.collected_hashes = set()
        
        self.logger.info(f"Data collector initialized with sources: {', '.join(self.sources)}")
    
    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect data based on the specified task
        
        Args:
            task: Dictionary with task details including:
                - market_domain: Domain/industry to research
                - search_terms: List of search terms
                - sources: Optional override of default sources
                - max_results: Optional max results to collect
                
        Returns:
            Dictionary with results of data collection
        """
        self.logger.info(f"Starting data collection for: {task.get('market_domain', 'Unknown domain')}")
        
        start_time = time.time()
        market_domain = task.get("market_domain", "")
        search_terms = task.get("search_terms", [])
        sources = task.get("sources", self.sources)
        max_results = task.get("max_results", self.max_results_per_source)
        
        if not market_domain or not search_terms:
            return {
                "status": "error",
                "message": "Missing required parameters: market_domain and search_terms",
                "agent": self.name
            }
        
        # Initialize results container
        results = {
            "status": "success",
            "agent": self.name,
            "market_domain": market_domain,
            "search_terms": search_terms,
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": {},
            "summary": {
                "total_collected": 0,
                "sources_used": [],
                "collection_time_seconds": 0
            }
        }
        
        # Collect data from each source
        for source in sources:
            source_method = getattr(self, f"collect_from_{source}", None)
            
            if source_method:
                try:
                    self.logger.info(f"Collecting from {source}...")
                    source_data = source_method(market_domain, search_terms, max_results)
                    
                    if source_data and len(source_data) > 0:
                        results["sources"][source] = {
                            "count": len(source_data),
                            "data_file": self._save_source_data(source, market_domain, source_data)
                        }
                        results["summary"]["total_collected"] += len(source_data)
                        results["summary"]["sources_used"].append(source)
                        self.logger.info(f"Collected {len(source_data)} items from {source}")
                    else:
                        self.logger.warning(f"No data collected from {source}")
                        results["sources"][source] = {"count": 0, "error": "No data found"}
                        
                except Exception as e:
                    self.logger.error(f"Error collecting from {source}: {str(e)}")
                    results["sources"][source] = {"count": 0, "error": str(e)}
            else:
                self.logger.warning(f"Collection method for {source} not implemented")
                results["sources"][source] = {"count": 0, "error": "Source not implemented"}
        
        # Complete the summary
        collection_time = time.time() - start_time
        results["summary"]["collection_time_seconds"] = round(collection_time, 2)
        
        # Record the results in memory
        self._update_memory({"type": "collection_results", "content": results["summary"]})
        
        self.logger.info(f"Data collection completed in {collection_time:.2f} seconds")
        return results
    
    def _save_source_data(self, source: str, domain: str, data: List[Dict]) -> str:
        """Save collected data to a file and return the filename"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{domain.replace(' ', '_')}_{timestamp}.json"
        filepath = self.data_dir / filename
        
        save_json(data, filepath)
        
        return str(filepath)
    
    def _is_duplicate(self, data: Dict[str, Any]) -> bool:
        """Check if this data has been collected before"""
        if not self.check_duplicates:
            return False
            
        # Create a hash of the data content
        content = data.get("content", "") or data.get("text", "") or str(data)
        url = data.get("url", "")
        hash_content = hashlib.md5(f"{url}:{content[:500]}".encode()).hexdigest()
        
        if hash_content in self.collected_hashes:
            return True
            
        self.collected_hashes.add(hash_content)
        return False
    
    def collect_from_web_search(self, domain: str, search_terms: List[str], max_results: int) -> List[Dict]:
        """Collect data from web search results"""
        if not SERP_API_KEY:
            self.logger.warning("SERP API key not set, skipping web search")
            return []
            
        results = []
        
        for term in search_terms:
            search_query = f"{domain} {term}"
            self.logger.info(f"Searching web for: {search_query}")
            
            # Use a search API (e.g., SerpAPI) to get search results
            # This is a simplified implementation
            search_url = f"https://serpapi.com/search.json?q={search_query}&api_key={SERP_API_KEY}"
            
            try:
                response = make_request(search_url)
                search_results = response.json().get("organic_results", [])
                
                for result in search_results[:max_results]:
                    url = result.get("link")
                    if not url:
                        continue
                        
                    # Get the content of the page
                    try:
                        page_response = make_request(url)
                        if page_response.status_code != 200:
                            continue
                            
                        soup = BeautifulSoup(page_response.text, "html.parser")
                        text_content = extract_text_from_html(soup)
                        
                        data_item = {
                            "url": url,
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "content": text_content,
                            "source": "web_search",
                            "domain": domain,
                            "search_term": term,
                            "collected_at": datetime.datetime.now().isoformat()
                        }
                        
                        if not self._is_duplicate(data_item):
                            results.append(data_item)
                            
                    except Exception as e:
                        self.logger.error(f"Error fetching content from {url}: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error performing web search for '{search_query}': {str(e)}")
        
        return results
    
    def collect_from_news_api(self, domain: str, search_terms: List[str], max_results: int) -> List[Dict]:
        """Collect data from News API"""
        if not NEWS_API_KEY:
            self.logger.warning("News API key not set, skipping news collection")
            return []
            
        results = []
        
        # Calculate date range for news articles
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.date_range_days)
        
        for term in search_terms:
            search_query = f"{domain} {term}"
            self.logger.info(f"Searching news for: {search_query}")
            
            # Use News API to get articles
            news_url = (
                f"https://newsapi.org/v2/everything"
                f"?q={search_query}"
                f"&from={start_date.strftime('%Y-%m-%d')}"
                f"&to={end_date.strftime('%Y-%m-%d')}"
                f"&sortBy=relevancy"
                f"&apiKey={NEWS_API_KEY}"
            )
            
            try:
                response = make_request(news_url)
                news_data = response.json()
                
                if news_data.get("status") != "ok":
                    self.logger.error(f"News API error: {news_data.get('message', 'Unknown error')}")
                    continue
                    
                articles = news_data.get("articles", [])
                
                for article in articles[:max_results]:
                    data_item = {
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "author": article.get("author", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source_name": article.get("source", {}).get("name", ""),
                        "source": "news_api",
                        "domain": domain,
                        "search_term": term,
                        "collected_at": datetime.datetime.now().isoformat()
                    }
                    
                    if not self._is_duplicate(data_item):
                        results.append(data_item)
                        
            except Exception as e:
                self.logger.error(f"Error fetching news for '{search_query}': {str(e)}")
        
        return results
    
    def collect_from_social_media(self, domain: str, search_terms: List[str], max_results: int) -> List[Dict]:
        """Collect data from social media platforms"""
        # This would typically use platform-specific APIs
        # Implementing a simplified mock version
        self.logger.info("Social media collection not fully implemented - returning mock data")
        
        results = []
        platforms = ["twitter", "linkedin", "reddit"]
        
        for platform in platforms:
            for term in search_terms[:2]:  # Limit terms for mock data
                for i in range(min(3, max_results)):  # Generate a few mock entries
                    data_item = {
                        "platform": platform,
                        "content": f"Mock {platform} post about {domain} and {term}",
                        "author": f"user_{i}_{platform}",
                        "posted_at": (datetime.datetime.now() - datetime.timedelta(days=i)).isoformat(),
                        "engagement": {"likes": i*10, "shares": i*2, "comments": i*5},
                        "source": "social_media",
                        "domain": domain,
                        "search_term": term,
                        "collected_at": datetime.datetime.now().isoformat()
                    }
                    results.append(data_item)
        
        return results
    
    def collect_from_sec_filings(self, domain: str, search_terms: List[str], max_results: int) -> List[Dict]:
        """Collect data from SEC filings (for company research)"""
        # This would typically use SEC EDGAR API
        # Implementing a simplified mock version
        self.logger.info("SEC filings collection not fully implemented - returning mock data")
        
        results = []
        filing_types = ["10-K", "10-Q", "8-K"]
        companies = [f"{domain} Inc", f"{domain} Corp", f"Leading {domain} Company"]
        
        for company in companies:
            for filing_type in filing_types:
                data_item = {
                    "company": company,
                    "filing_type": filing_type,
                    "filed_date": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
                    "highlights": f"Mock financial data for {company} in the {domain} sector",
                    "source": "sec_filings",
                    "domain": domain,
                    "collected_at": datetime.datetime.now().isoformat()
                }
                results.append(data_item)
                
                # Add only a few mock entries
                if len(results) >= max_results:
                    break
        
        return results