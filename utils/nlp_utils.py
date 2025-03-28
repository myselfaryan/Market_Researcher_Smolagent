"""
Utility functions for text processing and NLP
"""

import re
import string
import logging
from typing import Dict, List, Any, Set, Tuple
from collections import Counter

logger = logging.getLogger("utils.nlp")

# Common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
}

def preprocess_text(text: str, remove_stopwords: bool = True, 
                   remove_punctuation: bool = True, lowercase: bool = True, 
                   remove_numbers: bool = False) -> str:
    """
    Preprocess text for NLP tasks
    
    Args:
        text: Text to preprocess
        remove_stopwords: Whether to remove stopwords
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert to lowercase
        remove_numbers: Whether to remove numbers
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    if remove_stopwords:
        words = text.split()
        words = [word for word in words if word.lower() not in STOPWORDS]
        text = ' '.join(words)
    
    return text

def extract_keywords(text: str, top_n: int = 10, min_word_length: int = 3, 
                     custom_stopwords: Set[str] = None) -> List[str]:
    """
    Extract keywords from text using simple frequency-based approach
    
    Args:
        text: Text to extract keywords from
        top_n: Number of keywords to extract
        min_word_length: Minimum word length to consider
        custom_stopwords: Additional stopwords to exclude
        
    Returns:
        List of keywords
    """
    logger.debug(f"Extracting top {top_n} keywords")
    
    # Preprocess text
    processed_text = preprocess_text(text, remove_stopwords=True, 
                                     remove_punctuation=True, lowercase=True)
    
    # Tokenize
    words = processed_text.split()
    
    # Filter by length and remove custom stopwords
    stopwords = STOPWORDS.copy()
    if custom_stopwords:
        stopwords.update(custom_stopwords)
        
    words = [word for word in words 
             if len(word) >= min_word_length and word not in stopwords]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get top N keywords
    keywords = [word for word, _ in word_counts.most_common(top_n)]
    
    return keywords

def extract_ngrams(text: str, n: int = 2, top_n: int = 10) -> List[str]:
    """
    Extract n-grams from text
    
    Args:
        text: Text to extract n-grams from
        n: Size of n-grams
        top_n: Number of n-grams to extract
        
    Returns:
        List of n-grams
    """
    logger.debug(f"Extracting top {top_n} {n}-grams")
    
    # Preprocess text
    processed_text = preprocess_text(text, remove_stopwords=False, 
                                    remove_punctuation=True, lowercase=True)
    
    # Tokenize
    words = processed_text.split()
    
    # Generate n-grams
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Count n-gram frequencies
    ngram_counts = Counter(ngrams)
    
    # Get top N n-grams
    top_ngrams = [ngram for ngram, _ in ngram_counts.most_common(top_n)]
    
    return top_ngrams

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using regex patterns
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Dictionary mapping entity types to lists of entities
    """
    logger.debug("Extracting named entities")
    
    entities = {
        "organizations": [],
        "people": [],
        "locations": [],
        "dates": [],
        "money": []
    }
    
    # Simple patterns for entity extraction
    # These are basic patterns - a production system would use a proper NER model
    
    # Organization pattern (e.g., "Google Inc.", "Apple Corporation")
    org_pattern = r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*(?:\s(?:Inc|Corp|LLC|Ltd|Company|Group|Foundation))?)'
    
    # People pattern (e.g., "John Smith", "Jane Doe")
    people_pattern = r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,2})'
    
    # Location pattern (basic countries, cities)
    loc_pattern = r'(?:in|at|from|to)\s([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)'
    
    # Date pattern (e.g., "January 1, 2023", "2023-01-01")
    date_pattern = r'(?:\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})'
    
    # Money pattern (e.g., "$100", "100 dollars")
    money_pattern = r'(?:\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s(?:dollars|USD|euros|EUR|pounds|GBP))'
    
    # Extract entities
    entities["organizations"] = list(set(re.findall(org_pattern, text)))
    entities["people"] = list(set(re.findall(people_pattern, text)))
    entities["locations"] = list(set([loc for _, loc in re.findall(loc_pattern, text)]))
    entities["dates"] = list(set(re.findall(date_pattern, text)))
    entities["money"] = list(set(re.findall(money_pattern, text)))
    
    return entities

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculate statistics about a text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with text statistics
    """
    logger.debug("Calculating text statistics")
    
    # Preprocess text to remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Calculate basic statistics
    char_count = len(text)
    word_count = len(text.split())
    
    # Calculate sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = sum(1 for s in sentences if s.strip())
    
    # Calculate average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in text.split()) / word_count
    else:
        avg_word_length = 0
    
    # Calculate average sentence length
    if sentence_count > 0:
        words_per_sentence = word_count / sentence_count
    else:
        words_per_sentence = 0
    
    # Calculate readability (Flesch Reading Ease)
    if sentence_count > 0 and word_count > 0:
        flesch_score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (char_count / word_count))
    else:
        flesch_score = 0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "words_per_sentence": round(words_per_sentence, 2),
        "readability_score": round(flesch_score, 2)
    }

def extract_topics(texts: List[str], num_topics: int = 5, 
                  words_per_topic: int = 5) -> List[List[str]]:
    """
    Extract topics from a collection of texts using simple keyword clustering
    
    Args:
        texts: List of texts to analyze
        num_topics: Number of topics to extract
        words_per_topic: Number of words per topic
        
    Returns:
        List of topics, each represented as a list of keywords
    """
    logger.debug(f"Extracting {num_topics} topics from {len(texts)} texts")
    
    # Try to import sklearn for topic modeling
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize texts using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            min_df=2, 
            max_df=0.85, 
            stop_words='english'
        )
        tfidf = vectorizer.fit_transform(processed_texts)
        
        # Extract topics using Non-negative Matrix Factorization
        nmf = NMF(n_components=num_topics, random_state=42)
        nmf.fit(tfidf)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get topics
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-words_per_topic-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            
        return topics
        
    except ImportError:
        # Fallback to simple keyword extraction if sklearn is not available
        logger.warning("sklearn not available, falling back to simple keyword extraction")
        
        # Combine all texts
        combined_text = " ".join(texts)
        
        # Extract keywords
        all_keywords = extract_keywords(combined_text, top_n=num_topics * words_per_topic)
        
        # Split into topics (simple approach)
        topics = [all_keywords[i:i+words_per_topic] for i in range(0, len(all_keywords), words_per_topic)]
        
        return topics[:num_topics]

def extract_sentiment_terms(text: str, sentiment_lexicon: Dict[str, float] = None) -> Tuple[List[str], List[str]]:
    """
    Extract positive and negative terms from text
    
    Args:
        text: Text to analyze
        sentiment_lexicon: Dictionary mapping words to sentiment scores
        
    Returns:
        Tuple of (positive_terms, negative_terms)
    """
    logger.debug("Extracting sentiment terms")
    
    # Simple sentiment lexicon if none provided
    if sentiment_lexicon is None:
        sentiment_lexicon = {
            # Positive terms
            'good': 0.8, 'great': 0.9, 'excellent': 1.0, 'best': 1.0,
            'amazing': 0.9, 'awesome': 0.9, 'fantastic': 0.9, 'wonderful': 0.9,
            'positive': 0.7, 'benefit': 0.7, 'advantage': 0.7, 'success': 0.8,
            'improve': 0.6, 'innovative': 0.7, 'growth': 0.7, 'profit': 0.8,
            'efficient': 0.7, 'quality': 0.7, 'superior': 0.8, 'valuable': 0.7,
            
            # Negative terms
            'bad': -0.8, 'poor': -0.7, 'terrible': -0.9, 'worst': -1.0,
            'awful': -0.9, 'horrible': -0.9, 'negative': -0.7, 'fail': -0.8,
            'problem': -0.6, 'issue': -0.5, 'drawback': -0.6, 'concern': -0.5,
            'risk': -0.6, 'loss': -0.7, 'decrease': -0.6, 'decline': -0.7,
            'costly': -0.6, 'expensive': -0.6, 'difficult': -0.5, 'challenge': -0.5
        }
    
    # Preprocess text
    processed_text = preprocess_text(text, remove_stopwords=True, lowercase=True)
    
    # Tokenize
    words = processed_text.split()
    
    # Extract sentiment terms
    positive_terms = []
    negative_terms = []
    
    for word in words:
        if word in sentiment_lexicon:
            score = sentiment_lexicon[word]
            if score > 0:
                positive_terms.append(word)
            elif score < 0:
                negative_terms.append(word)
    
    return positive_terms, negative_terms