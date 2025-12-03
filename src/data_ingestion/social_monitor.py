"""
Social Media Monitoring Module
Tracks trends and discussions on Twitter and Reddit
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """Data model for social media posts"""
    id: str
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement: int  # likes, retweets, upvotes
    url: str
    tags: List[str]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TwitterMonitor:
    """
    Monitor Twitter/X for Sri Lankan trends
    Note: Requires Twitter API v2 bearer token
    """
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        
    async def get_trending_topics(self, location_id: str = "23424778") -> List[Dict]:
        """
        Get trending topics for Sri Lanka
        location_id: 23424778 is Sri Lanka's WOEID
        """
        # Simulated response for demo (API requires authentication)
        trends = [
            {"name": "Fuel Prices", "volume": 15000, "sentiment": -0.4},
            {"name": "Tourism", "volume": 8500, "sentiment": 0.6},
            {"name": "Cricket", "volume": 12000, "sentiment": 0.8},
            {"name": "Power Cuts", "volume": 6500, "sentiment": -0.7},
            {"name": "Colombo", "volume": 5000, "sentiment": 0.2},
        ]
        
        logger.info(f"Retrieved {len(trends)} trending topics from Twitter")
        return trends
    
    async def search_tweets(self, query: str, max_results: int = 100) -> List[SocialPost]:
        """Search for tweets matching query"""
        # Simulated for demo - in production, use actual API
        posts = []
        
        for i in range(min(max_results, 20)):
            post = SocialPost(
                id=f"tweet_{i}",
                platform="twitter",
                content=f"Sample tweet about {query} #{i}",
                author=f"user_{i}",
                timestamp=datetime.now() - timedelta(hours=i),
                engagement=100 + i * 10,
                url=f"https://twitter.com/user_{i}/status/{i}",
                tags=[query, "SriLanka"]
            )
            posts.append(post)
        
        logger.info(f"Retrieved {len(posts)} tweets for query: {query}")
        return posts


class RedditMonitor:
    """
    Monitor Reddit for Sri Lankan discussions
    Focuses on r/srilanka and related subreddits
    """
    
    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = "PulseX/1.0"
        
    async def get_hot_posts(self, subreddit: str = "srilanka", limit: int = 50) -> List[SocialPost]:
        """Get hot posts from subreddit"""
        # Simulated for demo
        posts = []
        
        topics = [
            "Discussion about economic situation",
            "Tourism recommendations thread",
            "Power cut schedules today",
            "New infrastructure project announced",
            "Cost of living megathread"
        ]
        
        for i, topic in enumerate(topics):
            post = SocialPost(
                id=f"reddit_{i}",
                platform="reddit",
                content=topic,
                author=f"redditor_{i}",
                timestamp=datetime.now() - timedelta(hours=i*2),
                engagement=50 + i * 20,
                url=f"https://reddit.com/r/{subreddit}/comments/{i}",
                tags=[subreddit, "discussion"]
            )
            posts.append(post)
        
        logger.info(f"Retrieved {len(posts)} posts from r/{subreddit}")
        return posts
    
    async def search_comments(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Search for comments matching query"""
        # Simulated for demo
        posts = []
        
        for i in range(min(limit, 15)):
            post = SocialPost(
                id=f"comment_{i}",
                platform="reddit",
                content=f"Comment discussing {query}",
                author=f"user_{i}",
                timestamp=datetime.now() - timedelta(hours=i),
                engagement=10 + i * 2,
                url=f"https://reddit.com/comments/{i}",
                tags=[query]
            )
            posts.append(post)
        
        return posts


class SocialMediaAggregator:
    """Aggregates data from multiple social platforms"""
    
    def __init__(self, twitter_token: str = "", reddit_id: str = "", reddit_secret: str = ""):
        self.twitter = TwitterMonitor(twitter_token) if twitter_token else None
        self.reddit = RedditMonitor(reddit_id, reddit_secret)
        
    async def collect_all(self, topics: List[str]) -> Dict[str, List[SocialPost]]:
        """Collect posts from all platforms for given topics"""
        results = {
            "twitter": [],
            "reddit": []
        }
        
        # Collect from Twitter
        if self.twitter:
            for topic in topics:
                posts = await self.twitter.search_tweets(topic)
                results["twitter"].extend(posts)
        
        # Collect from Reddit
        reddit_posts = await self.reddit.get_hot_posts()
        results["reddit"].extend(reddit_posts)
        
        for topic in topics:
            comments = await self.reddit.search_comments(topic)
            results["reddit"].extend(comments)
        
        total = sum(len(posts) for posts in results.values())
        logger.info(f"Collected {total} social media posts across all platforms")
        
        return results
    
    def analyze_sentiment_distribution(self, posts: List[SocialPost]) -> Dict:
        """Analyze overall sentiment from social posts"""
        import random
        
        if not posts:
            return {"positive": 0, "neutral": 0, "negative": 0}
        
        # Simulated sentiment analysis
        distribution = {
            "positive": random.randint(20, 40),
            "neutral": random.randint(30, 50),
            "negative": random.randint(20, 40)
        }
        
        total = sum(distribution.values())
        return {k: v/total for k, v in distribution.items()}


async def main():
    """Test social monitoring"""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RAW_DATA_DIR
    
    aggregator = SocialMediaAggregator()
    
    topics = ["fuel prices", "tourism", "economy", "infrastructure"]
    
    results = await aggregator.collect_all(topics)
    
    # Save results
    output_file = RAW_DATA_DIR / f"social_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json_data = {
            platform: [post.to_dict() for post in posts]
            for platform, posts in results.items()
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSocial Media Monitoring Results:")
    print(f"Twitter posts: {len(results['twitter'])}")
    print(f"Reddit posts: {len(results['reddit'])}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())