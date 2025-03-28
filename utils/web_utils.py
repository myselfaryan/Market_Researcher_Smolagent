"""
Utility functions for web scraping and data collection
"""

import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import re
from urllib.parse import urlparse

from ..config.settings import (
    REQUEST_HEADERS, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY
)

logger = logging.getLogger("utils.web")

def make_request(url: str, headers: Dict[str, str] = None, 
                params: Dict[str, Any] = None, method: str = "get", 
                retry: int = 0) -> requests.Response:
    """
    Make an HTTP request with retry logic
    
    Args:
        url: URL to request
        headers: HTTP headers
        params: Query parameters
        method: HTTP method (get, post)
        retry: Current retry count
        
    Returns:
        Response object
    """
    headers = headers or REQUEST_HEADERS
    
    try:
        logger.debug(f"Making {method.upper()} request to {url}")
        
        if method.lower() == "post":
            response = requests.post(url, headers=headers, json=params, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        
        # Check if request was successful
        response.raise_for_status()
        
        return response
    
    except requests.exceptions.RequestException as e:
        if retry < MAX_RETRIES:
            retry_wait = RETRY_DELAY * (2 ** retry)  # Exponential backoff
            logger.warning(f"Error requesting {url}: {str(e)}. Retrying in {retry_wait}s (attempt {retry + 1}/{MAX_RETRIES})")
            time.sleep(retry_wait)
            return make_request(url, headers, params, method, retry + 1)
        else:
            logger.error(f"Failed to request {url} after {MAX_RETRIES} retries: {str(e)}")
            raise

def extract_text_from_html(soup: BeautifulSoup) -> str:
    """
    Extract clean text from HTML content
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Extracted text
    """
    # Remove script and style elements
    for script in soup(["script", "style", "meta", "noscript", "iframe"]):
        script.extract()
    
    # Get text
    text = soup.get_text(separator=" ", strip=True)
    
    # Clean text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """
    Extract links from HTML content
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for resolving relative links
        
    Returns:
        List of dictionaries with link information
    """
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        
        # Skip empty or anchor links
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue
        
        # Resolve relative URLs
        if not href.startswith(('http://', 'https://')):
            parsed_base = urlparse(base_url)
            if href.startswith('/'):
                href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            else:
                href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
        
        links.append({
            'url': href,
            'text': text[:100] if text else '',
            'title': link.get('title', '')
        })
    
    return links

def extract_metadata(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """
    Extract metadata from HTML content
    
    Args:
        soup: BeautifulSoup object
        url: URL of the page
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        'url': url,
        'title': '',
        'description': '',
        'keywords': '',
        'author': '',
        'published_date': ''
    }
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    
    # Extract meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name', '').lower()
        property = meta.get('property', '').lower()
        content = meta.get('content', '')
        
        if name == 'description' or property == 'og:description':
            metadata['description'] = content
        elif name == 'keywords':
            metadata['keywords'] = content
        elif name == 'author':
            metadata['author'] = content
        elif name in ['date', 'pubdate'] or property in ['article:published_time', 'og:published_time']:
            metadata['published_date'] = content
    
    return metadata

def fetch_and_parse(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and parse its content
    
    Args:
        url: URL to fetch
        
    Returns:
        Dictionary with parsed content
    """
    logger.info(f"Fetching and parsing {url}")
    
    result = {
        'url': url,
        'status': 'error',
        'content': {},
        'error': None
    }
    
    try:
        # Make the request
        response = make_request(url)
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract content
        result['content'] = {
            'metadata': extract_metadata(soup, url),
            'text': extract_text_from_html(soup),
            'links': extract_links(soup, url),
            'html': response.text[:10000]  # Store first 10K of HTML for reference
        }
        
        result['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        result['error'] = str(e)
    
    return result

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid
    
    Args:
        url: URL to check
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def clean_url(url: str) -> str:
    """
    Clean and normalize a URL
    
    Args:
        url: URL to clean
        
    Returns:
        Cleaned URL
    """
    # Add https if no scheme
    if not urlparse(url).scheme:
        url = 'https://' + url
    
    # Remove tracking parameters
    try:
        parsed = urlparse(url)
        params = parsed.query.split('&')
        filtered_params = []
        
        # List of common tracking parameters to remove
        tracking_params = [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'ref', 'source', 'mc_cid', 'mc_eid'
        ]
        
        for param in params:
            if '=' in param:
                name = param.split('=')[0]
                if name.lower() not in tracking_params:
                    filtered_params.append(param)
        
        # Reconstruct URL
        clean = parsed._replace(query='&'.join(filtered_params)).geturl()
        return clean
    except:
        return url

def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL
    
    Args:
        url: URL to extract from
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    except:
        return ""