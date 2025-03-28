"""
Report generation agent that compiles analysis results into a coherent report
"""

import os
import json
import datetime
import markdown
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

from .base_agent import BaseAgent
from ..config.settings import REPORT_DIR, CHART_DPI, DEFAULT_CHART_STYLE
from ..utils.data_utils import load_json, save_text

class ReportAgent(BaseAgent):
    """Agent for generating market research reports"""
    
    def __init__(self, name: str = "report_agent", config: Dict[str, Any] = None):
        """
        Initialize the report agent
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        from ..config.agent_configs import REPORT_AGENT_CONFIG
        super().__init__(name, config or REPORT_AGENT_CONFIG)
        
        self.report_formats = self.config.get("report_formats", ["markdown"])
        self.include_executive_summary = self.config.get("include_executive_summary", True)
        self.include_methodology = self.config.get("include_methodology", True)
        self.include_visualizations = self.config.get("include_visualizations", True)
        self.visualization_types = self.config.get("visualization_types", [])
        
        # Create a directory for reports
        self.report_dir = REPORT_DIR
        self.report_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use(DEFAULT_CHART_STYLE)
        
        self.logger.info(f"Report agent initialized with formats: {', '.join(self.report_formats)}")
    
    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report based on analysis results
        
        Args:
            task: Dictionary with task details including:
                - analysis_files: Files containing analysis results
                - market_domain: Domain being researched
                - format: Report format (e.g. markdown, html)
                - include_sections: Which sections to include
                
        Returns:
            Dictionary with report generation results
        """
        self.logger.info(f"Starting report generation for: {task.get('market_domain', 'Unknown domain')}")
        
        start_time = datetime.datetime.now()
        analysis_files = task.get("analysis_files", [])
        market_domain = task.get("market_domain", "")
        report_format = task.get("format", self.report_formats[0])
        include_sections = task.get("include_sections", {
            "executive_summary": self.include_executive_summary,
            "methodology": self.include_methodology,
            "visualizations": self.include_visualizations
        })
        
        if not analysis_files:
            return {
                "status": "error",
                "message": "No analysis files provided for report generation",
                "agent": self.name
            }
        
        # Initialize results
        results = {
            "status": "success",
            "agent": self.name,
            "market_domain": market_domain,
            "timestamp": start_time.isoformat(),
            "report": {
                "format": report_format,
                "sections": []
            },
            "summary": {
                "generation_time_seconds": 0
            }
        }
        
        # Load analysis data
        try:
            analysis_data = self._load_analysis_data(analysis_files)
            self.logger.info(f"Loaded {len(analysis_data)} analysis datasets")
        except Exception as e:
            self.logger.error(f"Error loading analysis data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading analysis data: {str(e)}",
                "agent": self.name
            }
        
        # Generate report content
        try:
            report_content, sections = self._generate_report_content(
                analysis_data, market_domain, include_sections
            )
            
            results["report"]["content"] = report_content
            results["report"]["sections"] = sections
            self.logger.info(f"Generated report with {len(sections)} sections")
            
            # Save the report
            report_file = self._save_report(report_content, market_domain, report_format)
            results["report"]["file"] = report_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating report: {str(e)}",
                "agent": self.name
            }
        
        # Complete the summary
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        results["summary"]["generation_time_seconds"] = round(generation_time, 2)
        
        # Record the results in memory
        self._update_memory({"type": "report_results", "content": results["summary"]})
        
        self.logger.info(f"Report generation completed in {generation_time:.2f} seconds")
        return results
    
    def _load_analysis_data(self, analysis_files: List[str]) -> Dict[str, Any]:
        """
        Load analysis data from files
        
        Args:
            analysis_files: List of paths to analysis files
            
        Returns:
            Dictionary mapping analysis types to their data
        """
        analysis_data = {}
        
        for file_path in analysis_files:
            try:
                data = load_json(file_path)
                
                # Determine the analysis type from the file name
                file_name = Path(file_path).name
                if "trend" in file_name:
                    analysis_type = "trend_analysis"
                elif "competitor" in file_name:
                    analysis_type = "competitor_comparison"
                elif "market_size" in file_name:
                    analysis_type = "market_size_estimation"
                elif "growth" in file_name:
                    analysis_type = "growth_projections"
                elif "sentiment" in file_name:
                    analysis_type = "sentiment_analysis"
                else:
                    # Extract from content if possible
                    if isinstance(data, dict):
                        for key in data.keys():
                            if key in [
                                "trend_analysis", "competitor_comparison", 
                                "market_size_estimation", "growth_projections",
                                "sentiment_analysis", "sentiment"
                            ]:
                                analysis_type = key
                                break
                        else:
                            analysis_type = f"unknown_{Path(file_path).stem}"
                    else:
                        analysis_type = f"unknown_{Path(file_path).stem}"
                
                analysis_data[analysis_type] = data
                self.logger.info(f"Loaded {analysis_type} data from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {str(e)}")
        
        return analysis_data
    
    def _generate_report_content(self, analysis_data: Dict[str, Any], 
                                market_domain: str,
                                include_sections: Dict[str, bool]) -> tuple:
        """
        Generate the content of the report
        
        Args:
            analysis_data: Dictionary mapping analysis types to their data
            market_domain: Domain being researched
            include_sections: Dictionary specifying which sections to include
            
        Returns:
            Tuple of (report_content, sections_list)
        """
        # Format title case for market domain
        formatted_domain = ' '.join(word.capitalize() for word in market_domain.split())
        
        # Start with the title and date
        report_parts = [
            f"# Market Research Report: {formatted_domain}",
            f"*Generated on {datetime.datetime.now().strftime('%B %d, %Y')}*",
            ""
        ]
        
        sections = ["title"]
        
        # Add table of contents
        toc_parts = ["## Table of Contents", ""]
        toc_items = []
        
        # Add executive summary if requested
        if include_sections.get("executive_summary", True):
            toc_items.append("1. [Executive Summary](#executive-summary)")
            sections.append("executive_summary")
        
        # Add methodology if requested
        if include_sections.get("methodology", True):
            toc_items.append("2. [Methodology](#methodology)")
            sections.append("methodology")
        
        # Add analysis sections
        section_count = 3
        
        # Market size section
        if "market_size_estimation" in analysis_data:
            toc_items.append(f"{section_count}. [Market Size Analysis](#market-size-analysis)")
            sections.append("market_size")
            section_count += 1
        
        # Growth projections section
        if "growth_projections" in analysis_data:
            toc_items.append(f"{section_count}. [Growth Projections](#growth-projections)")
            sections.append("growth")
            section_count += 1
        
        # Competitor analysis section
        if "competitor_comparison" in analysis_data:
            toc_items.append(f"{section_count}. [Competitor Analysis](#competitor-analysis)")
            sections.append("competitors")
            section_count += 1
        
        # Trend analysis section
        if "trend_analysis" in analysis_data:
            toc_items.append(f"{section_count}. [Market Trends](#market-trends)")
            sections.append("trends")
            section_count += 1
        
        # Sentiment analysis section
        if "sentiment_analysis" in analysis_data or "sentiment" in analysis_data:
            toc_items.append(f"{section_count}. [Market Sentiment](#market-sentiment)")
            sections.append("sentiment")
            section_count += 1
        
        # Conclusion section
        toc_items.append(f"{section_count}. [Conclusion](#conclusion)")
        sections.append("conclusion")
        
        # Add ToC items to the report
        toc_parts.extend(toc_items)
        toc_parts.append("")
        report_parts.extend(toc_parts)
        
        # Generate each section
        
        # Executive Summary
        if include_sections.get("executive_summary", True):
            exec_summary = self._generate_executive_summary(analysis_data, market_domain)
            report_parts.extend(exec_summary)
        
        # Methodology
        if include_sections.get("methodology", True):
            methodology = self._generate_methodology(analysis_data)
            report_parts.extend(methodology)
        
        # Market Size
        if "market_size_estimation" in analysis_data:
            market_size = self._generate_market_size_section(analysis_data["market_size_estimation"])
            report_parts.extend(market_size)
        
        # Growth Projections
        if "growth_projections" in analysis_data:
            growth = self._generate_growth_section(analysis_data["growth_projections"])
            report_parts.extend(growth)
        
        # Competitor Analysis
        if "competitor_comparison" in analysis_data:
            competitors = self._generate_competitor_section(analysis_data["competitor_comparison"])
            report_parts.extend(competitors)
        
        # Market Trends
        if "trend_analysis" in analysis_data:
            trends = self._generate_trends_section(analysis_data["trend_analysis"])
            report_parts.extend(trends)
        
        # Market Sentiment
        if "sentiment_analysis" in analysis_data:
            sentiment = self._generate_sentiment_section(analysis_data["sentiment_analysis"])
            report_parts.extend(sentiment)
        elif "sentiment" in analysis_data:
            sentiment = self._generate_sentiment_section(analysis_data["sentiment"])
            report_parts.extend(sentiment)
        
        # Conclusion
        conclusion = self._generate_conclusion(analysis_data, market_domain)
        report_parts.extend(conclusion)
        
        # Join all parts with line breaks
        report_content = "\n".join(report_parts)
        
        return report_content, sections
    
    def _save_report(self, content: str, domain: str, format: str) -> str:
        """
        Save the report to a file
        
        Args:
            content: Report content
            domain: Market domain
            format: Report format
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_research_{domain.replace(' ', '_')}_{timestamp}"
        
        if format == "html":
            # Convert markdown to HTML
            html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Market Research: {domain}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                    h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 12px; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    code {{ background: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
                    pre {{ background: #f8f8f8; padding: 15px; border-radius: 3px; overflow-x: auto; }}
                    blockquote {{ background: #f9f9f9; border-left: 4px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            filepath = self.report_dir / f"{filename}.html"
            save_text(html_template, filepath)
            
        else:  # Default to markdown
            filepath = self.report_dir / f"{filename}.md"
            save_text(content, filepath)
        
        return str(filepath)
    
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], domain: str) -> List[str]:
        """Generate the executive summary section"""
        summary_parts = [
            "## Executive Summary",
            ""
        ]
        
        # Introduction
        intro = f"This report presents a comprehensive analysis of the {domain} market, "
        intro += "synthesizing data from multiple sources and applying various analytical techniques. "
        intro += "The following summarizes the key findings from our research:"
        summary_parts.append(intro)
        summary_parts.append("")
        
        # Key points
        key_points = []
        
        # Market size
        if "market_size_estimation" in analysis_data:
            ms_data = analysis_data["market_size_estimation"]
            if "formatted_market_size" in ms_data:
                size = ms_data["formatted_market_size"]
                key_points.append(f"**Market Size**: The {domain} market is estimated at {size}.")
            
            if "formatted_growth_rate" in ms_data:
                growth = ms_data["formatted_growth_rate"]
                key_points.append(f"**Growth Rate**: The market is projected to grow at {growth} annually.")
        
        # Growth projections
        elif "growth_projections" in analysis_data:
            gp_data = analysis_data["growth_projections"]
            if "formatted_growth_rate" in gp_data:
                growth = gp_data["formatted_growth_rate"]
                key_points.append(f"**Growth Projection**: The market is projected to grow at {growth} annually.")
            
            if "projections" in gp_data:
                projections = gp_data["projections"]
                if projections and str(gp_data.get("target_year", "")) in projections:
                    target = str(gp_data.get("target_year", ""))
                    proj_size = projections[target]["formatted"]
                    key_points.append(f"**Future Value**: By {target}, the market is expected to reach {proj_size}.")
        
        # Competitors
        if "competitor_comparison" in analysis_data:
            comp_data = analysis_data["competitor_comparison"]
            if "competitors" in comp_data and comp_data["competitors"]:
                top_competitors = comp_data["competitors"][:3]
                comp_str = ", ".join(top_competitors)
                key_points.append(f"**Leading Competitors**: {comp_str} dominate the market.")
        
        # Sentiment
        sentiment_key = "sentiment_analysis" if "sentiment_analysis" in analysis_data else "sentiment"
        if sentiment_key in analysis_data:
            sent_data = analysis_data[sentiment_key]
            if "summary" in sent_data and "text" in sent_data["summary"]:
                sentiment_summary = sent_data["summary"]["text"]
                key_points.append(f"**Market Sentiment**: {sentiment_summary}")
        
        # Trends
        if "trend_analysis" in analysis_data:
            trend_data = analysis_data["trend_analysis"]
            if "trend_summary" in trend_data:
                trend_summary = trend_data["trend_summary"]
                key_points.append(f"**Key Trends**: {trend_summary}")
        
        # Add the key points to the summary
        if key_points:
            for point in key_points:
                summary_parts.append(point)
                summary_parts.append("")
        else:
            summary_parts.append("*Detailed analysis results are presented in the following sections.*")
            summary_parts.append("")
        
        return summary_parts
    
    def _generate_methodology(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate the methodology section"""
        method_parts = [
            "## Methodology",
            ""
        ]
        
        # Overview
        method_parts.append("This market research was conducted using a multi-phase approach:")
        method_parts.append("")
        
        # Data collection
        method_parts.append("### Data Collection")
        method_parts.append("")
        method_parts.append("Data was gathered from multiple sources including:")
        method_parts.append("")
        method_parts.append("- Web search results")
        method_parts.append("- Industry news articles")
        method_parts.append("- Social media mentions")
        method_parts.append("- Financial filings and reports")
        method_parts.append("")
        
        # Analysis techniques
        method_parts.append("### Analysis Techniques")
        method_parts.append("")
        
        techniques = []
        
        if "market_size_estimation" in analysis_data:
            techniques.append("**Market Size Estimation**: Analysis of market size mentions and revenue data to determine the current market valuation.")
        
        if "growth_projections" in analysis_data:
            techniques.append("**Growth Projection Analysis**: Synthesis of growth forecasts and trend indicators to project future market development.")
        
        if "competitor_comparison" in analysis_data:
            techniques.append("**Competitor Analysis**: Identification of key market players and analysis of their market presence and positioning.")
        
        if "trend_analysis" in analysis_data:
            techniques.append("**Trend Analysis**: Temporal analysis of market discussions to identify emerging and declining trends.")
        
        if "sentiment_analysis" in analysis_data or "sentiment" in analysis_data:
            techniques.append("**Sentiment Analysis**: Natural language processing to gauge market sentiment and emotional context.")
        
        if techniques:
            for technique in techniques:
                method_parts.append(technique)
                method_parts.append("")
        else:
            method_parts.append("Various analytical techniques were applied to extract insights from the collected data.")
            method_parts.append("")
        
        # AI-powered analysis
        method_parts.append("### AI Analysis Framework")
        method_parts.append("")
        method_parts.append("This research utilized a system of specialized AI agents, each focused on a specific aspect of market analysis:")
        method_parts.append("")
        method_parts.append("1. **Data Collection Agent**: Gathered and pre-processed raw market data")
        method_parts.append("2. **Analysis Agent**: Performed core market analysis including trend identification and competitor analysis")
        method_parts.append("3. **Sentiment Agent**: Analyzed emotional and subjective aspects of market discussions")
        method_parts.append("4. **Report Agent**: Synthesized findings into this coherent report")
        method_parts.append("")
        
        return method_parts
    
    def _generate_market_size_section(self, market_size_data: Dict[str, Any]) -> List[str]:
        """Generate the market size analysis section"""
        section_parts = [
            "## Market Size Analysis",
            ""
        ]
        
        # Key findings
        if "formatted_market_size" in market_size_data:
            size = market_size_data["formatted_market_size"]
            section_parts.append(f"The current market size is estimated at **{size}**.")
            section_parts.append("")
        
        if "estimation_summary" in market_size_data:
            summary = market_size_data["estimation_summary"]
            section_parts.append(summary)
            section_parts.append("")
        
        # Confidence level
        if "confidence_level" in market_size_data:
            confidence = market_size_data["confidence_level"]
            if confidence == "low":
                conf_text = "This estimation has a **low confidence level** due to limited data points. Additional research is recommended for more precise valuation."
            elif confidence == "medium":
                conf_text = "This estimation has a **medium confidence level** based on the available data points. While reasonably reliable, additional data would strengthen this estimation."
            else:
                conf_text = "This estimation has a **high confidence level** based on numerous consistent data points across multiple sources."
            
            section_parts.append(conf_text)
            section_parts.append("")
        
        # Data points
        if "data_points" in market_size_data:
            data_points = market_size_data["data_points"]
            section_parts.append("### Data Sources")
            section_parts.append("")
            section_parts.append("This estimation is based on:")
            section_parts.append("")
            section_parts.append(f"- **{data_points.get('market_size_mentions', 0)}** direct market size references")
            section_parts.append(f"- **{data_points.get('revenue_mentions', 0)}** revenue mentions")
            section_parts.append(f"- **{data_points.get('growth_rate_mentions', 0)}** growth rate indicators")
            section_parts.append("")
        
        return section_parts
    
    def _generate_growth_section(self, growth_data: Dict[str, Any]) -> List[str]:
        """Generate the growth projections section"""
        section_parts = [
            "## Growth Projections",
            ""
        ]
        
        # Key findings
        if "projection_summary" in growth_data:
            summary = growth_data["projection_summary"]
            section_parts.append(summary)
            section_parts.append("")
        
        # Growth rate details
        if "formatted_growth_rate" in growth_data:
            growth = growth_data["formatted_growth_rate"]
            section_parts.append(f"**Annual Growth Rate**: {growth}")
            section_parts.append("")
        
        # Projection table
        if "projections" in growth_data and growth_data["projections"]:
            section_parts.append("### Projected Market Size by Year")
            section_parts.append("")
            section_parts.append("| Year | Projected Market Size |")
            section_parts.append("|------|----------------------|")
            
            projections = growth_data["projections"]
            for year in sorted(projections.keys()):
                size = projections[year]["formatted"]
                section_parts.append(f"| {year} | {size} |")
            
            section_parts.append("")
        
        # Confidence level
        if "confidence_level" in growth_data:
            confidence = growth_data["confidence_level"]
            if confidence == "low":
                conf_text = "These projections have a **low confidence level** due to limited growth indicators. They should be considered as general directional guidance rather than precise forecasts."
            elif confidence == "medium":
                conf_text = "These projections have a **medium confidence level** based on the available growth indicators. While reasonably reliable, market conditions may cause actual results to vary."
            else:
                conf_text = "These projections have a **high confidence level** based on numerous consistent growth indicators across multiple sources."
            
            section_parts.append(conf_text)
            section_parts.append("")
        
        return section_parts
    
    def _generate_competitor_section(self, competitor_data: Dict[str, Any]) -> List[str]:
        """Generate the competitor analysis section"""
        section_parts = [
            "## Competitor Analysis",
            ""
        ]
        
        # Overview
        if "comparison_summary" in competitor_data:
            summary = competitor_data["comparison_summary"]
            section_parts.append(summary)
            section_parts.append("")
        
        # Market presence table
        if "competitors" in competitor_data and "market_presence_percentage" in competitor_data:
            competitors = competitor_data["competitors"]
            market_presence = competitor_data["market_presence_percentage"]
            
            if competitors and market_presence:
                section_parts.append("### Market Share Distribution")
                section_parts.append("")
                section_parts.append("| Competitor | Market Presence |")
                section_parts.append("|------------|-----------------|")
                
                for competitor in competitors:
                    if competitor in market_presence:
                        presence = f"{market_presence[competitor]}%"
                        section_parts.append(f"| {competitor} | {presence} |")
                
                section_parts.append("")
        
        # Top associations
        if "top_associations" in competitor_data:
            associations = competitor_data["top_associations"]
            if associations:
                section_parts.append("### Competitor Positioning")
                section_parts.append("")
                section_parts.append("Key terms associated with each competitor:")
                section_parts.append("")
                
                for competitor, terms in associations.items():
                    if terms:
                        term_str = ", ".join(list(terms.keys())[:5])
                        section_parts.append(f"- **{competitor}**: {term_str}")
                
                section_parts.append("")
        
        # Sample contexts
        if "sample_contexts" in competitor_data:
            contexts = competitor_data["sample_contexts"]
            if contexts:
                section_parts.append("### Notable Mentions")
                section_parts.append("")
                
                for competitor, mentions in list(contexts.items())[:3]:  # Limit to top 3 competitors
                    if mentions:
                        section_parts.append(f"**{competitor}**:")
                        section_parts.append("")
                        for mention in mentions[:2]:  # Limit to 2 mentions per competitor
                            section_parts.append(f"> {mention}")
                            section_parts.append("")
        
        return section_parts
    
    def _generate_trends_section(self, trend_data: Dict[str, Any]) -> List[str]:
        """Generate the market trends section"""
        section_parts = [
            "## Market Trends",
            ""
        ]
        
        # Overview
        if "trend_summary" in trend_data:
            summary = trend_data["trend_summary"]
            section_parts.append(summary)
            section_parts.append("")
        
        # Temporal distribution
        if "monthly_document_counts" in trend_data:
            monthly_counts = trend_data["monthly_document_counts"]
            if monthly_counts:
                section_parts.append("### Discussion Volume Trends")
                section_parts.append("")
                section_parts.append("| Month | Discussion Volume |")
                section_parts.append("|-------|-------------------|")
                
                for month in sorted(monthly_counts.keys()):
                    count = monthly_counts[month]
                    section_parts.append(f"| {month} | {count} |")
                
                section_parts.append("")
        
        # Key terms by month
        if "monthly_terms" in trend_data:
            monthly_terms = trend_data["monthly_terms"]
            if monthly_terms:
                section_parts.append("### Evolving Terminology")
                section_parts.append("")
                section_parts.append("Key terms by month:")
                section_parts.append("")
                
                for month in sorted(monthly_terms.keys())[-3:]:  # Show only the most recent 3 months
                    terms = monthly_terms[month]
                    if terms:
                        term_str = ", ".join(list(terms.keys())[:5])
                        section_parts.append(f"- **{month}**: {term_str}")
                
                section_parts.append("")
        
        # Volume change
        if "volume_change_percentage" in trend_data:
            change = trend_data["volume_change_percentage"]
            direction = "increase" if change > 0 else "decrease" if change < 0 else "no change"
            section_parts.append(f"### Market Interest Trend")
            section_parts.append("")
            section_parts.append(f"There has been a {abs(change)}% {direction} in market discussions over the analyzed period.")
            section_parts.append("")
        
        return section_parts
    
    def _generate_sentiment_section(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate the market sentiment section"""
        section_parts = [
            "## Market Sentiment",
            ""
        ]
        
        # Check if we're dealing with nested sentiment data
        if "sentiment" in sentiment_data:
            sentiment = sentiment_data["sentiment"]
        else:
            sentiment = sentiment_data
        
        # Overview
        if "summary" in sentiment_data and "text" in sentiment_data["summary"]:
            summary = sentiment_data["summary"]["text"]
            section_parts.append(summary)
            section_parts.append("")
        
        # Sentiment distribution
        if "distribution" in sentiment:
            distribution = sentiment["distribution"]
            if distribution:
                section_parts.append("### Sentiment Distribution")
                section_parts.append("")
                section_parts.append("| Sentiment | Percentage | Count |")
                section_parts.append("|-----------|------------|-------|")
                
                for sentiment_type in ["positive", "neutral", "negative"]:
                    if sentiment_type in distribution:
                        info = distribution[sentiment_type]
                        section_parts.append(f"| {sentiment_type.capitalize()} | {info['percentage']}% | {info['count']} |")
                
                section_parts.append("")
        
        # Emotion analysis
        if "emotions" in sentiment_data and "distribution" in sentiment_data["emotions"]:
            emotions = sentiment_data["emotions"]["distribution"]
            if emotions:
                section_parts.append("### Emotional Context")
                section_parts.append("")
                section_parts.append("| Emotion | Percentage |")
                section_parts.append("|---------|------------|")
                
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1]["percentage"], reverse=True)
                for emotion, info in sorted_emotions[:5]:  # Show top 5 emotions
                    section_parts.append(f"| {emotion.capitalize()} | {info['percentage']}% |")
                
                section_parts.append("")
        
        # Topics by sentiment
        if "topics" in sentiment_data:
            topics = sentiment_data["topics"]
            if topics:
                section_parts.append("### Topics by Sentiment")
                section_parts.append("")
                
                for sentiment_type, terms in topics.items():
                    if terms:
                        term_str = ", ".join(terms[:5])
                        section_parts.append(f"- **{sentiment_type.capitalize()}**: {term_str}")
                
                section_parts.append("")
        
        return section_parts
    
    def _generate_conclusion(self, analysis_data: Dict[str, Any], domain: str) -> List[str]:
        """Generate the conclusion section"""
        section_parts = [
            "## Conclusion",
            ""
        ]
        
        # Format domain for readability
        formatted_domain = ' '.join(word.capitalize() for word in domain.split())
        
        # Build conclusion based on available analyses
        conclusion_points = []
        
        # Market status
        market_status = f"The {formatted_domain} market "
        
        # Size and growth
        if "market_size_estimation" in analysis_data:
            market_size = analysis_data["market_size_estimation"]
            size = market_size.get("formatted_market_size", "of significant value")
            growth = market_size.get("formatted_growth_rate", "")
            
            if growth:
                market_status += f"is currently valued at approximately {size} and growing at {growth} annually."
            else:
                market_status += f"is currently valued at approximately {size}."
        elif "growth_projections" in analysis_data:
            growth = analysis_data["growth_projections"].get("formatted_growth_rate", "")
            if growth:
                market_status += f"is growing at an estimated rate of {growth} annually."
            else:
                market_status += "shows promising growth potential."
        else:
            market_status += "presents significant opportunities based on current analysis."
        
        conclusion_points.append(market_status)
        
        # Competitive landscape
        if "competitor_comparison" in analysis_data:
            competitor_data = analysis_data["competitor_comparison"]
            competitors = competitor_data.get("competitors", [])
            
            if competitors:
                top_competitors = competitors[:3]
                comp_str = ", ".join(top_competitors)
                competitive_landscape = f"The competitive landscape is dominated by {comp_str}, "
                
                # Add market positioning if available
                if "top_associations" in competitor_data and top_competitors[0] in competitor_data["top_associations"]:
                    top_comp = top_competitors[0]
                    associations = competitor_data["top_associations"][top_comp]
                    if associations:
                        top_assoc = list(associations.keys())[:2]
                        assoc_str = " and ".join(top_assoc)
                        competitive_landscape += f"with {top_comp} particularly known for {assoc_str}."
                    else:
                        competitive_landscape += "each with their unique market positioning."
                else:
                    competitive_landscape += "each competing for market share."
                    
                conclusion_points.append(competitive_landscape)
        
        # Market sentiment
        sentiment_key = "sentiment_analysis" if "sentiment_analysis" in analysis_data else "sentiment"
        if sentiment_key in analysis_data:
            sentiment_data = analysis_data[sentiment_key]
            
            if "sentiment" in sentiment_data:
                sentiment = sentiment_data["sentiment"]
            else:
                sentiment = sentiment_data
                
            if "overall_sentiment" in sentiment:
                overall = sentiment["overall_sentiment"]
                sentiment_text = f"Overall market sentiment is primarily {overall}, "
                
                # Add emotion if available
                if "emotions" in sentiment_data and "dominant_emotion" in sentiment_data["emotions"]:
                    emotion = sentiment_data["emotions"]["dominant_emotion"]
                    sentiment_text += f"with communications often conveying a sense of {emotion}."
                else:
                    sentiment_text += "which indicates the market's general perception."
                    
                conclusion_points.append(sentiment_text)
        
        # Future outlook
        future_outlook = "Looking ahead, "
        
        if "growth_projections" in analysis_data:
            growth_data = analysis_data["growth_projections"]
            
            if "projections" in growth_data and growth_data["projections"]:
                target_year = str(growth_data.get("target_year", ""))
                if target_year in growth_data["projections"]:
                    projected_size = growth_data["projections"][target_year]["formatted"]
                    future_outlook += f"the market is projected to reach {projected_size} by {target_year}, "
            
            if "formatted_growth_rate" in growth_data:
                growth = growth_data["formatted_growth_rate"]
                future_outlook += f"growing at {growth} annually."
            else:
                future_outlook += "showing continued growth potential."
        elif "trend_analysis" in analysis_data:
            trend_data = analysis_data["trend_analysis"]
            
            if "volume_change_percentage" in trend_data:
                change = trend_data["volume_change_percentage"]
                if change > 20:
                    future_outlook += "market interest is increasing significantly, suggesting strong future potential."
                elif change > 0:
                    future_outlook += "market interest is gradually increasing, indicating steady future development."
                elif change < -20:
                    future_outlook += "market interest is declining, which may signal challenges ahead."
                else:
                    future_outlook += "market interest remains relatively stable."
            else:
                future_outlook += "the market is expected to evolve with industry trends."
        else:
            future_outlook += "this market presents both opportunities and challenges that warrant continued monitoring."
            
        conclusion_points.append(future_outlook)
        
        # Recommendations
        recommendations = "Based on this analysis, we recommend "
        
        if "sentiment" in analysis_data or "sentiment_analysis" in analysis_data:
            sentiment_key = "sentiment_analysis" if "sentiment_analysis" in analysis_data else "sentiment"
            sentiment_data = analysis_data[sentiment_key]
            
            if "sentiment" in sentiment_data:
                sentiment = sentiment_data["sentiment"]["overall_sentiment"]
            else:
                sentiment = sentiment_data.get("overall_sentiment", "")
                
            if sentiment == "positive":
                recommendations += "considering an increased investment in this market, "
            elif sentiment == "negative":
                recommendations += "approaching this market with caution, "
            else:
                recommendations += "developing a balanced strategy for this market, "
        else:
            recommendations += "developing a strategic approach to this market, "
            
        recommendations += "focusing on key trends identified in this report, and continuing to monitor market developments as the landscape evolves."
        
        conclusion_points.append(recommendations)
        
        # Add all conclusion points
        for point in conclusion_points:
            section_parts.append(point)
            section_parts.append("")
        
        # Final note
        section_parts.append("*This report was generated using AI-powered market analysis. For specific investment decisions, consult with appropriate financial and industry experts.*")
        section_parts.append("")
        
        return section_parts