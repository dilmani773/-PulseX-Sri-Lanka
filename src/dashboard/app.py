"""
PulseX Sri Lanka - Interactive Dashboard
Uses Custom Components for a polished UI
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

# Import your custom components
import components as ui

st.set_page_config(page_title="PulseX Sri Lanka", layout="wide", page_icon="🇱🇰")

# Load Data
def load_data():
    path = PROCESSED_DATA_DIR / "dashboard_data.json"
    if not path.exists():
        ui.alert_box("No data found. Run pipeline first.", "error")
        return None
    with open(path, 'r') as f: return json.load(f)

def main():
    st.title("🇱🇰 PulseX Sri Lanka")
    st.caption(f"Situational Awareness Platform | Updated: {datetime.now().strftime('%H:%M')}")
    
    data = load_data()
    if not data: return

    # --- 1. Context Alerts ---
    if data['context']['weather_risk_score'] > 0.6:
        ui.alert_box(f"High Weather Risk: {data['context']['weather_summary'].splitlines()[0]}", "warning")

    # --- 2. Key Metrics Row ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.metric_card("Overall Risk", f"{data['overall_risk']:.0%}", 
                      delta="Critical" if data['overall_risk'] > 0.7 else "Stable", 
                      icon="🛡️")
    with c2:
        ui.metric_card("Sentiment", f"{data['sentiment_avg']:.2f}", icon="💬")
    with c3:
        ui.metric_card("Inflation", f"{data['context']['inflation_rate']}%", icon="💰")
    with c4:
        ui.metric_card("Articles", str(data['total_articles']), icon="📰")

    st.markdown("---")

    # --- 3. Main Content ---
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📡 Intelligence Feed")
        
        # News Feed using Custom Component
        for item in data.get('recent_news', []):
            ui.news_card(
                title=item['title'],
                source=item['source'],
                time_ago=item['published_date'][:10],
                sentiment=item.get('sentiment_score', 0),
                category=item.get('type', 'General').title()
            )

    with col_right:
        st.subheader("🤖 Strategic Actions")
        
        # Recommendations using Custom Component
        for rec in data.get('recommendations', []):
            ui.recommendation_card(
                priority=rec['priority'],
                action=rec['action'],
                reason=rec['reason'],
                impact=rec['impact']
            )
            
        st.subheader("📊 Risk Factors")
        breakdown = data.get('risk_breakdown', {})
        for k, v in breakdown.items():
            ui.progress_ring(v, k.replace('_', ' ').title())

if __name__ == "__main__":
    main()
