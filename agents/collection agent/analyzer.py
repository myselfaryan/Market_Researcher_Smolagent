"""
Analyzer agent that processes collected data to extract insights and trends
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter

from .base_agent import BaseAgent
from ..config.settings import PROCESSED_DATA_DIR
from ..utils.data_utils import load_json, save_json, save_dataframe
from ..models.embedding_model import EmbeddingModel

class AnalyzerAgent(BaseAgent):
    """Agent for analyzing market research data"""
    
    def __init__(self, name: str = "analyzer", config: Dict[str, Any] = None):
        """
        Initialize the analyzer agent
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        from ..config.agent_configs import ANALYZER_CONFIG
        super().__init__(name, config or ANALYZER_CONFIG)
        
        self.analysis_types = self.config.get("analysis_types", [])
        self.time_periods = self.config.get("time_periods", ["quarterly", "annual"])
        self.statistical_tests = self.config.get("statistical_tests", [])
        self.outlier_detection = self.config.get("outlier_detection", True)
        self.significance_threshold = self.config.get("significance_threshold", 0.05)
        
        # Create a directory for processed data
        self.data_dir = PROCESSED_DATA_DIR / name
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model for semantic analysis
        self.embedding_model = None
        
        self.logger.info(f"Analyzer initialized with analysis types: {', '.join(self.analysis_types)}")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model if not already done"""
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
            
    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data based on the specified task
        
        Args:
            task: Dictionary with task details including:
                - data_files: List of files containing data to analyze
                - analysis_types: Types of analysis to perform
                - market_domain: Domain being researched
                
        Returns:
            Dictionary with analysis results
        """
        self.logger.info(f"Starting analysis for: {task.get('market_domain', 'Unknown domain')}")
        
        start_time = datetime.datetime.now()
        data_files = task.get("data_files", [])
        analysis_types = task.get("analysis_types", self.analysis_types)
        market_domain = task.get("market_domain", "")
        
        if not data_files:
            return {
                "status": "error",
                "message": "No data files provided for analysis",
                "agent": self.name
            }
        
        # Initialize results
        results = {
            "status": "success",
            "agent": self.name,
            "market_domain": market_domain,
            "timestamp": start_time.isoformat(),
            "analysis": {},
            "summary": {
                "total_documents": 0,
                "analyses_performed": [],
                "analysis_time_seconds": 0
            }
        }
        
        # Load and consolidate data
        try:
            data = self._load_data(data_files)
            results["summary"]["total_documents"] = len(data)
            self.logger.info(f"Loaded {len(data)} documents for analysis")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading data: {str(e)}",
                "agent": self.name
            }
        
        # Perform each requested analysis
        for analysis_type in analysis_types:
            analysis_method = getattr(self, f"analyze_{analysis_type}", None)
            
            if analysis_method:
                try:
                    self.logger.info(f"Performing {analysis_type} analysis...")
                    analysis_result = analysis_method(data, market_domain)
                    
                    if analysis_result:
                        results["analysis"][analysis_type] = {
                            "results": analysis_result,
                            "output_file": self._save_analysis_result(analysis_type, market_domain, analysis_result)
                        }
                        results["summary"]["analyses_performed"].append(analysis_type)
                        self.logger.info(f"Completed {analysis_type} analysis")
                    else:
                        self.logger.warning(f"No results from {analysis_type} analysis")
                        results["analysis"][analysis_type] = {"error": "No results generated"}
                        
                except Exception as e:
                    self.logger.error(f"Error in {analysis_type} analysis: {str(e)}")
                    results["analysis"][analysis_type] = {"error": str(e)}
            else:
                self.logger.warning(f"Analysis method for {analysis_type} not implemented")
                results["analysis"][analysis_type] = {"error": "Analysis type not implemented"}
        
        # Complete the summary
        end_time = datetime.datetime.now()
        analysis_time = (end_time - start_time).total_seconds()
        results["summary"]["analysis_time_seconds"] = round(analysis_time, 2)
        
        # Record the results in memory
        self._update_memory({"type": "analysis_results", "content": results["summary"]})
        
        self.logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        return results
    
    def _load_data(self, data_files: List[str]) -> List[Dict]:
        """Load and consolidate data from multiple files"""
        all_data = []
        
        for file_path in data_files:
            try:
                data = load_json(file_path)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {str(e)}")
        
        return all_data
    
    def _save_analysis_result(self, analysis_type: str, domain: str, result: Any) -> str:
        """Save analysis result to a file and return the filename"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_{domain.replace(' ', '_')}_{timestamp}.json"
        filepath = self.data_dir / filename
        
        save_json(result, filepath)
        
        return str(filepath)
    
    def analyze_trend_analysis(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """
        Identify trends in the data over time
        
        Args:
            data: List of data items to analyze
            domain: Market domain being analyzed
            
        Returns:
            Dictionary with trend analysis results
        """
        self.logger.info("Performing trend analysis")
        
        # Extract dates and organize data by time
        time_series_data = {}
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        
        for item in data:
            # Extract date from various possible fields
            date_str = None
            for date_field in ["published_at", "posted_at", "collected_at", "filed_date"]:
                if date_field in item and item[date_field]:
                    date_match = date_pattern.search(item[date_field])
                    if date_match:
                        date_str = date_match.group()
                        break
            
            if not date_str:
                continue
                
            # Use just the month for grouping
            month_str = date_str[:7]  # YYYY-MM format
            
            if month_str not in time_series_data:
                time_series_data[month_str] = []
                
            time_series_data[month_str].append(item)
        
        # Count documents per month
        monthly_counts = {month: len(items) for month, items in time_series_data.items()}
        
        # Extract key terms by month
        monthly_terms = {}
        for month, items in time_series_data.items():
            # Combine all text content for this month
            combined_text = " ".join([
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "") 
                for item in items
            ])
            
            # Extract important terms (simple approach)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
            word_counts = Counter(words)
            
            # Remove common stopwords (simplified)
            stopwords = {"the", "and", "is", "of", "to", "in", "that", "for", "on", "with"}
            monthly_terms[month] = {word: count for word, count in word_counts.most_common(10) 
                                  if word not in stopwords}
        
        # Sort months chronologically
        sorted_months = sorted(time_series_data.keys())
        
        # Identify change in volume over time
        if len(sorted_months) > 1:
            first_month = sorted_months[0]
            last_month = sorted_months[-1]
            
            first_count = monthly_counts.get(first_month, 0)
            last_count = monthly_counts.get(last_month, 0)
            
            if first_count > 0:
                volume_change_pct = ((last_count - first_count) / first_count) * 100
            else:
                volume_change_pct = 0
        else:
            volume_change_pct = 0
        
        # Prepare results
        results = {
            "monthly_document_counts": monthly_counts,
            "monthly_terms": monthly_terms,
            "volume_change_percentage": round(volume_change_pct, 2),
            "time_period": {
                "start": sorted_months[0] if sorted_months else None,
                "end": sorted_months[-1] if sorted_months else None,
                "total_months": len(sorted_months)
            },
            "trend_summary": self._generate_trend_summary(monthly_counts, monthly_terms, volume_change_pct)
        }
        
        return results
    
    def _generate_trend_summary(self, monthly_counts, monthly_terms, volume_change_pct):
        """Generate a human-readable summary of the trends"""
        sorted_months = sorted(monthly_counts.keys())
        
        if not sorted_months:
            return "Insufficient data for trend analysis"
            
        summary = []
        
        # Volume trend
        if volume_change_pct > 20:
            summary.append(f"Significant increase in volume ({volume_change_pct:.1f}%) from {sorted_months[0]} to {sorted_months[-1]}")
        elif volume_change_pct < -20:
            summary.append(f"Significant decrease in volume ({volume_change_pct:.1f}%) from {sorted_months[0]} to {sorted_months[-1]}")
        else:
            summary.append(f"Relatively stable volume ({volume_change_pct:.1f}% change) from {sorted_months[0]} to {sorted_months[-1]}")
        
        # Term evolution
        if len(sorted_months) >= 2:
            first_month = sorted_months[0]
            last_month = sorted_months[-1]
            
            if first_month in monthly_terms and last_month in monthly_terms:
                first_terms = set(monthly_terms[first_month].keys())
                last_terms = set(monthly_terms[last_month].keys())
                
                new_terms = last_terms - first_terms
                disappeared_terms = first_terms - last_terms
                
                if new_terms:
                    summary.append(f"New prominent terms in recent data: {', '.join(list(new_terms)[:5])}")
                
                if disappeared_terms:
                    summary.append(f"Terms no longer prominent: {', '.join(list(disappeared_terms)[:5])}")
        
        return ". ".join(summary)
    
    def analyze_competitor_comparison(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """
        Compare competitors mentioned in the data
        
        Args:
            data: List of data items to analyze
            domain: Market domain being analyzed
            
        Returns:
            Dictionary with competitor comparison results
        """
        self.logger.info("Performing competitor comparison analysis")
        
        # Extract competitor mentions from data
        competitor_mentions = {}
        competitor_contexts = {}
        
        # First, determine potential competitors by looking for company names
        # This is a simplified approach - a real implementation would use NER
        all_text = " ".join([
            item.get("content", "") or 
            item.get("description", "") or 
            item.get("text", "") or 
            item.get("title", "") 
            for item in data
        ])
        
        # Look for company patterns like "X Inc", "X Corp", "X Company"
        company_pattern = re.compile(r'([A-Z][a-zA-Z\d\s]+) (Inc|Corp|Company|Ltd|LLC)')
        potential_companies = company_pattern.findall(all_text)
        
        # Count company mentions
        companies = [name.strip() for name, _ in potential_companies]
        company_counts = Counter(companies)
        
        # Keep only companies with multiple mentions
        competitors = {company: count for company, count in company_counts.most_common(10) 
                      if count > 1}
        
        # Now analyze each competitor's context
        for item in data:
            content = (
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "")
            )
            
            for competitor in competitors:
                if competitor in content:
                    # Update mention count
                    competitor_mentions[competitor] = competitor_mentions.get(competitor, 0) + 1
                    
                    # Extract context (sentences containing the competitor)
                    sentences = re.split(r'[.!?]', content)
                    relevant_sentences = [s.strip() for s in sentences if competitor in s]
                    
                    if competitor not in competitor_contexts:
                        competitor_contexts[competitor] = []
                        
                    competitor_contexts[competitor].extend(relevant_sentences[:2])  # Limit to avoid excessive data
        
        # Calculate market presence percentage
        total_mentions = sum(competitor_mentions.values())
        market_presence = {}
        
        if total_mentions > 0:
            for competitor, mentions in competitor_mentions.items():
                market_presence[competitor] = round((mentions / total_mentions) * 100, 2)
        
        # Extract common associations with each competitor
        competitor_associations = {}
        
        for competitor, contexts in competitor_contexts.items():
            combined_context = " ".join(contexts)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_context.lower())
            word_counts = Counter(words)
            
            # Remove the competitor name and common stopwords
            competitor_lower = competitor.lower()
            stopwords = {"the", "and", "is", "of", "to", "in", "that", "for", "on", "with"}
            
            competitor_associations[competitor] = {
                word: count for word, count in word_counts.most_common(5) 
                if word not in stopwords and word not in competitor_lower
            }
        
        # Prepare results
        results = {
            "competitors": list(competitors.keys()),
            "mention_counts": competitor_mentions,
            "market_presence_percentage": market_presence,
            "top_associations": competitor_associations,
            "sample_contexts": {comp: contexts[:3] for comp, contexts in competitor_contexts.items()},
            "comparison_summary": self._generate_competitor_summary(
                competitor_mentions, market_presence, competitor_associations
            )
        }
        
        return results
    
    def _generate_competitor_summary(self, mentions, market_presence, associations):
        """Generate a human-readable summary of the competitor comparison"""
        if not mentions:
            return "No significant competitors identified"
            
        top_competitors = sorted(mentions.keys(), key=lambda x: mentions[x], reverse=True)[:3]
        
        summary = [f"Top competitors identified: {', '.join(top_competitors)}"]
        
        for competitor in top_competitors:
            presence = market_presence.get(competitor, 0)
            mention = mentions.get(competitor, 0)
            
            competitor_summary = f"{competitor} has {presence}% market presence with {mention} mentions"
            
            # Add key associations if available
            if competitor in associations and associations[competitor]:
                top_assoc = list(associations[competitor].keys())[:3]
                if top_assoc:
                    competitor_summary += f", associated with {', '.join(top_assoc)}"
                    
            summary.append(competitor_summary)
            
        return ". ".join(summary)
    
    def analyze_market_size_estimation(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """
        Estimate market size based on available data
        
        Args:
            data: List of data items to analyze
            domain: Market domain being analyzed
            
        Returns:
            Dictionary with market size estimation results
        """
        self.logger.info("Performing market size estimation")
        
        # This would typically use more sophisticated methods with real data
        # Implementing a simplified mock approach based on data mentions
        
        # Extract revenue and market size mentions
        revenue_pattern = re.compile(r'(\$\d+(?:\.\d+)?)\s*(million|billion|trillion|M|B|T)')
        market_pattern = re.compile(r'market\s+size.*?(\$\d+(?:\.\d+)?)\s*(million|billion|trillion|M|B|T)', re.IGNORECASE)
        growth_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:annual|yearly|CAGR)', re.IGNORECASE)
        
        revenue_mentions = []
        market_size_mentions = []
        growth_rate_mentions = []
        
        for item in data:
            content = (
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "")
            )
            
            # Extract revenue mentions
            for amount, unit in revenue_pattern.findall(content):
                if "million" in unit.lower() or unit.upper() == "M":
                    multiplier = 1_000_000
                elif "billion" in unit.lower() or unit.upper() == "B":
                    multiplier = 1_000_000_000
                elif "trillion" in unit.lower() or unit.upper() == "T":
                    multiplier = 1_000_000_000_000
                else:
                    multiplier = 1
                
                value = float(amount.replace("$", "")) * multiplier
                revenue_mentions.append(value)
            
            # Extract market size mentions
            for amount, unit in market_pattern.findall(content):
                if "million" in unit.lower() or unit.upper() == "M":
                    multiplier = 1_000_000
                elif "billion" in unit.lower() or unit.upper() == "B":
                    multiplier = 1_000_000_000
                elif "trillion" in unit.lower() or unit.upper() == "T":
                    multiplier = 1_000_000_000_000
                else:
                    multiplier = 1
                
                value = float(amount.replace("$", "")) * multiplier
                market_size_mentions.append(value)
            
            # Extract growth rate mentions
            for rate in growth_pattern.findall(content):
                growth_rate_mentions.append(float(rate))
        
        # Calculate estimated market size
        if market_size_mentions:
            est_market_size = sum(market_size_mentions) / len(market_size_mentions)
        elif revenue_mentions:
            # Rough estimation based on revenue mentions
            est_market_size = sum(revenue_mentions) / len(revenue_mentions) * 10
        else:
            est_market_size = None
        
        # Calculate estimated growth rate
        if growth_rate_mentions:
            est_growth_rate = sum(growth_rate_mentions) / len(growth_rate_mentions)
        else:
            est_growth_rate = None
        
        # Format for human readability
        formatted_market_size = None
        if est_market_size is not None:
            if est_market_size >= 1_000_000_000_000:
                formatted_market_size = f"${est_market_size / 1_000_000_000_000:.2f} trillion"
            elif est_market_size >= 1_000_000_000:
                formatted_market_size = f"${est_market_size / 1_000_000_000:.2f} billion"
            elif est_market_size >= 1_000_000:
                formatted_market_size = f"${est_market_size / 1_000_000:.2f} million"
            else:
                formatted_market_size = f"${est_market_size:.2f}"
        
        # Prepare results
        results = {
            "estimated_market_size": est_market_size,
            "formatted_market_size": formatted_market_size,
            "estimated_growth_rate": est_growth_rate,
            "formatted_growth_rate": f"{est_growth_rate:.2f}%" if est_growth_rate is not None else None,
            "confidence_level": "low" if len(market_size_mentions) < 3 else "medium",
            "data_points": {
                "market_size_mentions": len(market_size_mentions),
                "revenue_mentions": len(revenue_mentions),
                "growth_rate_mentions": len(growth_rate_mentions)
            },
            "estimation_summary": self._generate_market_size_summary(
                formatted_market_size, est_growth_rate, len(market_size_mentions)
            )
        }
        
        return results
    
    def _generate_market_size_summary(self, market_size, growth_rate, num_mentions):
        """Generate a human-readable summary of the market size estimation"""
        if not market_size:
            return "Insufficient data for market size estimation"
            
        summary = [f"Estimated market size: {market_size}"]
        
        if growth_rate is not None:
            summary.append(f"with projected annual growth rate of {growth_rate:.1f}%")
            
        confidence = "low" if num_mentions < 3 else "medium" if num_mentions < 10 else "high"
        summary.append(f"(Confidence level: {confidence}, based on {num_mentions} data points)")
        
        return " ".join(summary)
    
    def analyze_growth_projections(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """
        Project market growth based on available data
        
        Args:
            data: List of data items to analyze
            domain: Market domain being analyzed
            
        Returns:
            Dictionary with growth projection results
        """
        self.logger.info("Performing growth projections analysis")
        
        # Extract growth rate mentions
        growth_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:annual|yearly|CAGR|growth)', re.IGNORECASE)
        forecast_pattern = re.compile(r'(?:forecast|project|predict|expect).*?(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)
        year_pattern = re.compile(r'by\s+(\d{4})', re.IGNORECASE)
        
        growth_rate_mentions = []
        forecast_mentions = []
        target_years = []
        
        for item in data:
            content = (
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "")
            )
            
            # Extract general growth rate mentions
            for rate in growth_pattern.findall(content):
                rate_val = float(rate)
                if 0 <= rate_val <= 100:  # Filter out unrealistic values
                    growth_rate_mentions.append(rate_val)
            
            # Extract forecast-specific mentions
            for rate in forecast_pattern.findall(content):
                rate_val = float(rate)
                if 0 <= rate_val <= 100:  # Filter out unrealistic values
                    forecast_mentions.append(rate_val)
            
            # Extract target years
            for year in year_pattern.findall(content):
                year_val = int(year)
                if 2020 <= year_val <= 2050:  # Filter to reasonable range
                    target_years.append(year_val)
        
        # Calculate average growth rate
        if forecast_mentions:
            # Prioritize explicit forecasts
            avg_growth_rate = sum(forecast_mentions) / len(forecast_mentions)
        elif growth_rate_mentions:
            avg_growth_rate = sum(growth_rate_mentions) / len(growth_rate_mentions)
        else:
            avg_growth_rate = None
        
        # Determine forecast period
        current_year = datetime.datetime.now().year
        if target_years:
            avg_target_year = sum(target_years) / len(target_years)
            forecast_years = round(avg_target_year - current_year)
        else:
            forecast_years = 5  # Default to 5-year forecast
        
        # Project growth
        projections = {}
        
        if avg_growth_rate is not None:
            # Get estimated current market size from market_size_estimation if available
            market_size_analysis = self._get_previous_analysis("market_size_estimation")
            if market_size_analysis and "estimated_market_size" in market_size_analysis:
                current_size = market_size_analysis["estimated_market_size"]
            else:
                # Default value if no real data
                current_size = 1_000_000_000  # $1B placeholder
            
            # Calculate compound growth
            for year in range(current_year, current_year + forecast_years + 1):
                year_offset = year - current_year
                growth_factor = (1 + (avg_growth_rate / 100)) ** year_offset
                projected_size = current_size * growth_factor
                
                # Format for display
                if projected_size >= 1_000_000_000_000:
                    formatted_size = f"${projected_size / 1_000_000_000_000:.2f}T"
                elif projected_size >= 1_000_000_000:
                    formatted_size = f"${projected_size / 1_000_000_000:.2f}B"
                else:
                    formatted_size = f"${projected_size / 1_000_000:.2f}M"
                
                projections[str(year)] = {
                    "value": projected_size,
                    "formatted": formatted_size
                }
        
        # Prepare results
        results = {
            "average_growth_rate": avg_growth_rate,
            "formatted_growth_rate": f"{avg_growth_rate:.2f}%" if avg_growth_rate is not None else None,
            "forecast_period_years": forecast_years,
            "target_year": current_year + forecast_years,
            "projections": projections,
            "confidence_level": "low" if len(forecast_mentions) < 3 else "medium",
            "data_points": {
                "growth_mentions": len(growth_rate_mentions),
                "forecast_mentions": len(forecast_mentions),
                "year_mentions": len(target_years)
            },
            "projection_summary": self._generate_growth_summary(
                avg_growth_rate, forecast_years, projections, current_year
            )
        }
        
        return results
    
    def _generate_growth_summary(self, growth_rate, forecast_years, projections, current_year):
        """Generate a human-readable summary of the growth projections"""
        if growth_rate is None:
            return "Insufficient data for growth projections"
            
        end_year = current_year + forecast_years
        
        if str(end_year) in projections:
            end_size = projections[str(end_year)]["formatted"]
            summary = [
                f"Market projected to grow at {growth_rate:.1f}% annually",
                f"reaching {end_size} by {end_year}",
                f"(a {forecast_years}-year forecast period)"
            ]
        else:
            summary = [
                f"Market projected to grow at {growth_rate:.1f}% annually",
                f"over the next {forecast_years} years"
            ]
            
        return " ".join(summary)
    
    def _get_previous_analysis(self, analysis_type):
        """
        Retrieve results from a previous analysis if available in memory
        
        This allows different analysis methods to build on each other's results
        """
        for entry in reversed(self.memory):
            if entry.get("type") == "analysis_results":
                content = entry.get("content", {})
                if "analyses_performed" in content and analysis_type in content["analyses_performed"]:
                    # Result exists somewhere, check saved file
                    for entry2 in reversed(self.memory):
                        if entry2.get("type") == "analysis_result" and entry2.get("analysis_type") == analysis_type:
                            return entry2.get("result", {})
                            
        return None