"""
Sentiment analysis agent that processes text data to extract sentiment and emotions
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
from transformers import pipeline

from .base_agent import BaseAgent
from ..config.settings import PROCESSED_DATA_DIR, SENTIMENT_MODEL
from ..utils.data_utils import load_json, save_json, save_dataframe
from ..utils.nlp_utils import extract_keywords, preprocess_text

class SentimentAgent(BaseAgent):
    """Agent for sentiment analysis of market research data"""
    
    def __init__(self, name: str = "sentiment", config: Dict[str, Any] = None):
        """
        Initialize the sentiment agent
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        from ..config.agent_configs import SENTIMENT_AGENT_CONFIG
        super().__init__(name, config or SENTIMENT_AGENT_CONFIG)
        
        self.sentiment_categories = self.config.get("sentiment_categories", ["positive", "negative", "neutral"])
        self.analyze_emotions = self.config.get("analyze_emotions", True)
        self.emotion_categories = self.config.get("emotion_categories", [])
        self.topic_extraction = self.config.get("topic_extraction", True)
        self.max_topics = self.config.get("max_topics", 5)
        
        # Create a directory for processed sentiment data
        self.data_dir = PROCESSED_DATA_DIR / name
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize models (lazy loading)
        self.sentiment_model = None
        self.emotion_model = None
        
        self.logger.info(f"Sentiment agent initialized with categories: {', '.join(self.sentiment_categories)}")
    
    def _load_sentiment_model(self):
        """Load the sentiment analysis model"""
        if self.sentiment_model is None:
            self.logger.info(f"Loading sentiment model {SENTIMENT_MODEL}")
            try:
                self.sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
                self.logger.info("Sentiment model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading sentiment model: {str(e)}")
                raise
    
    def _load_emotion_model(self):
        """Load the emotion analysis model"""
        if self.emotion_model is None and self.analyze_emotions:
            self.logger.info("Loading emotion model")
            try:
                # Using a pre-trained emotion detection model
                self.emotion_model = pipeline("text-classification", 
                                            model="j-hartmann/emotion-english-distilroberta-base")
                self.logger.info("Emotion model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading emotion model: {str(e)}")
                self.analyze_emotions = False  # Disable if can't load
    
    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment in data based on the specified task
        
        Args:
            task: Dictionary with task details including:
                - data_files: List of files containing data to analyze
                - market_domain: Domain being researched
                - analyze_emotions: Whether to analyze emotions (optional)
                - extract_topics: Whether to extract topics (optional)
                
        Returns:
            Dictionary with sentiment analysis results
        """
        self.logger.info(f"Starting sentiment analysis for: {task.get('market_domain', 'Unknown domain')}")
        
        start_time = datetime.datetime.now()
        data_files = task.get("data_files", [])
        market_domain = task.get("market_domain", "")
        analyze_emotions = task.get("analyze_emotions", self.analyze_emotions)
        extract_topics = task.get("extract_topics", self.topic_extraction)
        
        if not data_files:
            return {
                "status": "error",
                "message": "No data files provided for sentiment analysis",
                "agent": self.name
            }
        
        # Initialize results
        results = {
            "status": "success",
            "agent": self.name,
            "market_domain": market_domain,
            "timestamp": start_time.isoformat(),
            "sentiment_analysis": {},
            "summary": {
                "total_documents": 0,
                "analysis_time_seconds": 0
            }
        }
        
        # Load models
        self._load_sentiment_model()
        if analyze_emotions:
            self._load_emotion_model()
        
        # Load and process data
        try:
            data = self._load_data(data_files)
            results["summary"]["total_documents"] = len(data)
            self.logger.info(f"Loaded {len(data)} documents for sentiment analysis")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading data: {str(e)}",
                "agent": self.name
            }
        
        # Perform sentiment analysis
        try:
            sentiment_results = self._analyze_sentiment(data)
            results["sentiment_analysis"]["sentiment"] = sentiment_results
            self.logger.info("Completed sentiment analysis")
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            results["sentiment_analysis"]["sentiment"] = {"error": str(e)}
        
        # Perform emotion analysis if requested
        if analyze_emotions and self.emotion_model:
            try:
                emotion_results = self._analyze_emotions(data)
                results["sentiment_analysis"]["emotions"] = emotion_results
                self.logger.info("Completed emotion analysis")
            except Exception as e:
                self.logger.error(f"Error in emotion analysis: {str(e)}")
                results["sentiment_analysis"]["emotions"] = {"error": str(e)}
        
        # Extract topics related to sentiments if requested
        if extract_topics:
            try:
                topic_results = self._extract_sentiment_topics(data, sentiment_results)
                results["sentiment_analysis"]["topics"] = topic_results
                self.logger.info("Completed topic extraction")
            except Exception as e:
                self.logger.error(f"Error in topic extraction: {str(e)}")
                results["sentiment_analysis"]["topics"] = {"error": str(e)}
        
        # Generate summary
        try:
            sentiment_summary = self._generate_sentiment_summary(
                sentiment_results, 
                results["sentiment_analysis"].get("emotions", {}),
                results["sentiment_analysis"].get("topics", {})
            )
            results["sentiment_analysis"]["summary"] = sentiment_summary
            self.logger.info("Generated sentiment summary")
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            results["sentiment_analysis"]["summary"] = {"error": str(e)}
        
        # Save the full results
        try:
            output_file = self._save_analysis_result(market_domain, results["sentiment_analysis"])
            results["sentiment_analysis"]["output_file"] = output_file
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
        
        # Complete the summary
        end_time = datetime.datetime.now()
        analysis_time = (end_time - start_time).total_seconds()
        results["summary"]["analysis_time_seconds"] = round(analysis_time, 2)
        
        # Record the results in memory
        self._update_memory({"type": "sentiment_results", "content": results["summary"]})
        
        self.logger.info(f"Sentiment analysis completed in {analysis_time:.2f} seconds")
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
    
    def _save_analysis_result(self, domain: str, result: Any) -> str:
        """Save analysis result to a file and return the filename"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_{domain.replace(' ', '_')}_{timestamp}.json"
        filepath = self.data_dir / filename
        
        save_json(result, filepath)
        
        return str(filepath)
    
    def _analyze_sentiment(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment in the data
        
        Args:
            data: List of data items to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        self.logger.info(f"Analyzing sentiment for {len(data)} documents")
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for item in data:
            # Extract text content
            text = (
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "")
            )
            
            # Skip empty content
            if not text.strip():
                continue
            
            # Preprocess text - truncate long texts to avoid overloading model
            text = text[:1000]  # Limit to 1000 chars for performance
            
            # Process with sentiment model
            try:
                sentiment_result = self.sentiment_model(text)[0]
                label = sentiment_result["label"]
                score = sentiment_result["score"]
                
                # Map to standard categories
                if "positive" in label.lower():
                    sentiment = "positive"
                    positive_count += 1
                elif "negative" in label.lower():
                    sentiment = "negative"
                    negative_count += 1
                else:
                    sentiment = "neutral"
                    neutral_count += 1
                
                # Store result
                sentiments.append({
                    "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": sentiment,
                    "score": score,
                    "source": item.get("source", "unknown"),
                    "url": item.get("url", None)
                })
                
            except Exception as e:
                self.logger.error(f"Error processing text: {str(e)}")
        
        # Calculate overall sentiment distribution
        total = len(sentiments)
        
        if total > 0:
            sentiment_distribution = {
                "positive": {
                    "count": positive_count,
                    "percentage": round((positive_count / total) * 100, 2)
                },
                "negative": {
                    "count": negative_count,
                    "percentage": round((negative_count / total) * 100, 2)
                },
                "neutral": {
                    "count": neutral_count,
                    "percentage": round((neutral_count / total) * 100, 2)
                }
            }
            
            # Determine overall sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                overall_sentiment = "positive"
            elif negative_count > positive_count and negative_count > neutral_count:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
                
            sentiment_score = (positive_count - negative_count) / total
        else:
            sentiment_distribution = {
                "positive": {"count": 0, "percentage": 0},
                "negative": {"count": 0, "percentage": 0},
                "neutral": {"count": 0, "percentage": 0}
            }
            overall_sentiment = "neutral"
            sentiment_score = 0
        
        return {
            "items": sentiments,
            "distribution": sentiment_distribution,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "total_analyzed": total
        }
    
    def _analyze_emotions(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze emotions in the data
        
        Args:
            data: List of data items to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        self.logger.info(f"Analyzing emotions for {len(data)} documents")
        
        emotions = []
        emotion_counts = Counter()
        
        for item in data:
            # Extract text content
            text = (
                item.get("content", "") or 
                item.get("description", "") or 
                item.get("text", "") or 
                item.get("title", "")
            )
            
            # Skip empty content
            if not text.strip():
                continue
            
            # Preprocess text
            text = text[:1000]  # Limit to 1000 chars for performance
            
            # Process with emotion model
            try:
                emotion_result = self.emotion_model(text)[0]
                emotion = emotion_result["label"]
                score = emotion_result["score"]
                
                # Store result
                emotions.append({
                    "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                    "emotion": emotion,
                    "score": score,
                    "source": item.get("source", "unknown"),
                    "url": item.get("url", None)
                })
                
                # Update emotion count
                emotion_counts[emotion] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing text for emotions: {str(e)}")
        
        # Calculate emotion distribution
        total = len(emotions)
        
        if total > 0:
            emotion_distribution = {}
            for emotion, count in emotion_counts.items():
                emotion_distribution[emotion] = {
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                }
            
            # Determine dominant emotion
            if emotion_counts:
                dominant_emotion = emotion_counts.most_common(1)[0][0]
            else:
                dominant_emotion = "neutral"
        else:
            emotion_distribution = {}
            dominant_emotion = "neutral"
        
        return {
            "items": emotions,
            "distribution": emotion_distribution,
            "dominant_emotion": dominant_emotion,
            "total_analyzed": total
        }
    
    def _extract_sentiment_topics(self, data: List[Dict], sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract topics related to different sentiments
        
        Args:
            data: List of data items to analyze
            sentiment_results: Results from sentiment analysis
            
        Returns:
            Dictionary with topic extraction results
        """
        self.logger.info("Extracting sentiment-related topics")
        
        # Group by sentiment
        positive_texts = []
        negative_texts = []
        neutral_texts = []
        
        for item in sentiment_results.get("items", []):
            sentiment = item.get("sentiment")
            text = item.get("text_snippet", "")
            
            if sentiment == "positive":
                positive_texts.append(text)
            elif sentiment == "negative":
                negative_texts.append(text)
            else:
                neutral_texts.append(text)
        
        # Extract keywords for each sentiment group
        topics = {}
        
        if positive_texts:
            topics["positive"] = self._extract_topics_from_texts(positive_texts)
        
        if negative_texts:
            topics["negative"] = self._extract_topics_from_texts(negative_texts)
        
        if neutral_texts:
            topics["neutral"] = self._extract_topics_from_texts(neutral_texts)
        
        return topics
    
    def _extract_topics_from_texts(self, texts: List[str]) -> List[str]:
        """Extract important topics/keywords from a list of texts"""
        combined_text = " ".join(texts)
        
        # Simple keyword extraction (in a real implementation, use more sophisticated NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
        word_counts = Counter(words)
        
        # Remove common stopwords
        stopwords = {"the", "and", "is", "of", "to", "in", "that", "for", "on", "with", 
                   "as", "this", "by", "be", "are", "from", "have", "has", "was", "were"}
        
        topics = [word for word, count in word_counts.most_common(self.max_topics * 2) 
                if word not in stopwords]
        
        return topics[:self.max_topics]
    
    def _generate_sentiment_summary(self, sentiment_results: Dict[str, Any], 
                                   emotion_results: Dict[str, Any], 
                                   topic_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a human-readable summary of sentiment analysis results
        
        Args:
            sentiment_results: Results from sentiment analysis
            emotion_results: Results from emotion analysis
            topic_results: Results from topic extraction
            
        Returns:
            Dictionary with summary text and key points
        """
        summary_points = []
        
        # Sentiment distribution
        if "distribution" in sentiment_results:
            dist = sentiment_results["distribution"]
            overall = sentiment_results.get("overall_sentiment", "neutral")
            
            sentiment_text = f"Market sentiment is predominantly {overall}"
            
            if overall == "positive":
                sentiment_text += f" ({dist['positive']['percentage']}% positive vs {dist['negative']['percentage']}% negative)"
            elif overall == "negative":
                sentiment_text += f" ({dist['negative']['percentage']}% negative vs {dist['positive']['percentage']}% positive)"
            else:
                sentiment_text += f" ({dist['neutral']['percentage']}% neutral, {dist['positive']['percentage']}% positive, {dist['negative']['percentage']}% negative)"
                
            summary_points.append(sentiment_text)
        
        # Emotion insights
        if emotion_results and "distribution" in emotion_results and emotion_results["distribution"]:
            dominant = emotion_results.get("dominant_emotion", "")
            
            if dominant:
                emotion_text = f"The dominant emotion expressed is {dominant}"
                
                # Add supporting emotions
                other_emotions = [e for e, data in emotion_results["distribution"].items() 
                               if e != dominant and data["percentage"] >= 10]
                
                if other_emotions:
                    emotion_text += f", with significant presence of {', '.join(other_emotions[:2])}"
                    
                summary_points.append(emotion_text)
        
        # Topic insights
        if topic_results:
            # Positive topics
            if "positive" in topic_results and topic_results["positive"]:
                positive_topics = topic_results["positive"][:3]
                if positive_topics:
                    summary_points.append(f"Positive sentiment is associated with: {', '.join(positive_topics)}")
            
            # Negative topics
            if "negative" in topic_results and topic_results["negative"]:
                negative_topics = topic_results["negative"][:3]
                if negative_topics:
                    summary_points.append(f"Negative sentiment is associated with: {', '.join(negative_topics)}")
        
        # Create the final summary
        summary_text = ". ".join(summary_points)
        
        return {
            "text": summary_text,
            "key_points": summary_points
        }