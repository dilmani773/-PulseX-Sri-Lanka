"""
Advanced Multi-Source News Scraper for Sri Lankan Media
Implements adaptive scraping with rate limiting and error handling
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import hashlib
import re
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Data model for news articles"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    language: str
    category: str
    scraped_at: datetime
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['published_date'] = self.published_date.isoformat()
        data['scraped_at'] = self.scraped_at.isoformat()
        return data


class NewsScraperEngine:
    """
    Adaptive news scraping engine with multiple strategies
    """
    
    def __init__(self, sources: List[Dict], max_concurrent: int = 5):
        self.sources = sources
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_articles: List[NewsArticle] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'PulseX Research Bot 1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_article_id(self, title: str, url: str) -> str:
        """Generate unique article ID using hash"""
        unique_string = f"{title}{url}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with error handling"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def extract_ada_derana(self, html: str, source: Dict) -> List[NewsArticle]:
        """Extract articles from Ada Derana"""
        articles = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find article containers (adapt selectors as needed)
        article_blocks = soup.find_all('div', class_=['news-story', 'article-item'])
        
        for block in article_blocks[:20]:  # Limit to 20 articles
            try:
                title_elem = block.find(['h2', 'h3', 'a'])
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href') or block.find('a')['href']
                
                if not link.startswith('http'):
                    link = source['url'] + link
                
                # Extract snippet/content
                content_elem = block.find(['p', 'div'], class_=['summary', 'description'])
                content = content_elem.get_text(strip=True) if content_elem else title
                
                article = NewsArticle(
                    id=self.generate_article_id(title, link),
                    title=title,
                    content=content,
                    source=source['name'],
                    url=link,
                    published_date=datetime.now(),  # Would extract from article
                    language=source['language'],
                    category=source['category'],
                    scraped_at=datetime.now()
                )
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing article block: {str(e)}")
                continue
        
        return articles
    
    def extract_daily_mirror(self, html: str, source: Dict) -> List[NewsArticle]:
        """Extract articles from Daily Mirror"""
        articles = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Adapt selectors based on actual website structure
        article_blocks = soup.find_all(['article', 'div'], class_=['post', 'article'])
        
        for block in article_blocks[:20]:
            try:
                title_elem = block.find(['h2', 'h3', 'h4'])
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link_elem = title_elem.find('a') or block.find('a')
                
                if not link_elem:
                    continue
                
                link = link_elem.get('href')
                if not link.startswith('http'):
                    link = source['url'] + link
                
                content_elem = block.find('p')
                content = content_elem.get_text(strip=True) if content_elem else title
                
                article = NewsArticle(
                    id=self.generate_article_id(title, link),
                    title=title,
                    content=content,
                    source=source['name'],
                    url=link,
                    published_date=datetime.now(),
                    language=source['language'],
                    category=source['category'],
                    scraped_at=datetime.now()
                )
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing article: {str(e)}")
                continue
        
        return articles
    
    def generic_extractor(self, html: str, source: Dict) -> List[NewsArticle]:
        """Generic extractor for any news website"""
        articles = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all links that look like articles
        potential_links = soup.find_all('a', href=True)
        
        seen_titles = set()
        
        for link in potential_links:
            try:
                title = link.get_text(strip=True)
                
                # Filter out navigation links, etc.
                if len(title) < 20 or len(title) > 200:
                    continue
                
                if title in seen_titles:
                    continue
                
                seen_titles.add(title)
                
                url = link['href']
                if not url.startswith('http'):
                    url = source['url'] + url
                
                # Skip non-article URLs
                if any(skip in url for skip in ['#', 'javascript:', 'mailto:', '/category/', '/tag/']):
                    continue
                
                article = NewsArticle(
                    id=self.generate_article_id(title, url),
                    title=title,
                    content=title,  # Would fetch full content in production
                    source=source['name'],
                    url=url,
                    published_date=datetime.now(),
                    language=source['language'],
                    category=source['category'],
                    scraped_at=datetime.now()
                )
                articles.append(article)
                
                if len(articles) >= 20:
                    break
                    
            except Exception as e:
                continue
        
        return articles
    
    async def scrape_source(self, source: Dict) -> List[NewsArticle]:
        """Scrape a single news source"""
        logger.info(f"Scraping {source['name']}...")
        
        html = await self.fetch_page(source['url'])
        if not html:
            return []
        
        # Route to appropriate extractor
        if 'adaderana' in source['url']:
            articles = self.extract_ada_derana(html, source)
        elif 'dailymirror' in source['url']:
            articles = self.extract_daily_mirror(html, source)
        else:
            articles = self.generic_extractor(html, source)
        
        logger.info(f"Extracted {len(articles)} articles from {source['name']}")
        return articles
    
    async def scrape_all(self) -> List[NewsArticle]:
        """Scrape all sources concurrently"""
        tasks = [self.scrape_source(source) for source in self.sources]
        
        # Execute with concurrency limit
        results = []
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Scraping error: {str(result)}")
                else:
                    results.extend(result)
        
        self.scraped_articles = results
        return results
    
    def save_to_json(self, output_path: Path):
        """Save scraped articles to JSON"""
        data = [article.to_dict() for article in self.scraped_articles]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} articles to {output_path}")


async def main():
    """Test the scraper"""
    from config import DATA_CONFIG, RAW_DATA_DIR
    
    async with NewsScraperEngine(DATA_CONFIG.NEWS_SOURCES) as scraper:
        articles = await scraper.scrape_all()
        
        output_file = RAW_DATA_DIR / f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        scraper.save_to_json(output_file)
        
        print(f"\nScraped {len(articles)} articles")
        print(f"Saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())