"""
News Classification Engine
Categorizes news articles to filter out noise (e.g. Crime) vs. Signal (e.g. Economy).
"""

import re

class NewsClassifier:
    def __init__(self):
        # Define keywords for each category
        self.categories = {
            "CRIME": [
                "arrest", "suspect", "police", "jail", "prison", "murder", "kill", "death", 
                "dead", "injure", "wound", "accident", "court", "drug", "heroin", "ice", 
                "cannabis", "smuggle", "raid", "seize", "shot", "shooting", "stab", "clubb",
                "crime", "criminal", "theft", "robbery", "bribe"
            ],
            "ECONOMY": [
                "inflation", "economy", "economic", "bank", "imf", "debt", "loan", "tax", 
                "rupee", "dollar", "forex", "market", "stock", "trade", "export", "import", 
                "price", "cost", "fuel", "gas", "power", "energy", "tourism", "tourist"
            ],
            "POLITICS": [
                "president", "minister", "parliament", "election", "vote", "party", "mp", 
                "cabinet", "government", "opposition", "policy", "bill", "act", "gazette"
            ],
            "DISASTER": [
                "flood", "landslide", "rain", "storm", "weather", "warning", "disaster", 
                "earthquake", "cyclone", "dam"
            ]
        }

    def classify(self, text: str) -> str:
        """Assigns a category to a news headline"""
        text = text.lower()
        
        # Check specific categories
        for category, keywords in self.categories.items():
            for word in keywords:
                if word in text:
                    return category
        
        return "GENERAL"  # Default if no keywords match