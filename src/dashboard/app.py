"""
PulseX Sri Lanka - Interactive Dashboard
User-friendly real-time business intelligence platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DASHBOARD_CONFIG, MODEL_CONFIG
from models.risk_scorer import BayesianRiskScorer, RiskLevel

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG.TITLE,
    page_icon=DASHBOARD_CONFIG.PAGE_ICON,
    layout=DASHBOARD_CONFIG.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    /* Force header colors to be visible across Streamlit themes */
    .main-header, .main-header * { color: #0f172a !important; }
    .subtitle, .subtitle * { color: #374151 !important; }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #3B82F6;
        color: #111827; /* Ensure dark text on white cards */
    }
    .risk-critical { border-left-color: #EF4444 !important; }
    .risk-high { border-left-color: #F59E0B !important; }
    .risk-medium { border-left-color: #F59E0B !important; }
    .risk-low { border-left-color: #10B981 !important; }
    .recommendation-box {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3B82F6;
        color: #111827; /* Force readable text color inside recommendation boxes */
    }
    /* Helper classes for consistent foreground on light/dark backgrounds */
    .white-bg { background: #FFFFFF !important; color: #0f172a !important; }
    .dark-bg { background: #0f172a !important; color: #ffffff !important; }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-critical { background: #FEE2E2; border-left: 4px solid #EF4444; }
    .alert-warning { background: #FEF3C7; border-left: 4px solid #F59E0B; }
    .alert-info { background: #DBEAFE; border-left: 4px solid #3B82F6; }
</style>
""", unsafe_allow_html=True)


class DashboardController:
    """Main controller for dashboard logic"""
    
    def __init__(self):
        self.risk_scorer = BayesianRiskScorer()
        self.load_data()
    
    def load_data(self):
        """Load or simulate real-time data"""
        # In production, this would load from database/cache
        # For demo, we'll simulate data
        
        if 'data_loaded' not in st.session_state:
            st.session_state.news_data = self.generate_sample_news()
            st.session_state.metrics_data = self.generate_sample_metrics()
            st.session_state.trends_data = self.generate_sample_trends()
            st.session_state.data_loaded = True
            st.session_state.last_update = datetime.now()
    
    def generate_sample_news(self) -> pd.DataFrame:
        """Generate sample news data"""
        categories = ['Political', 'Economic', 'Social', 'Infrastructure', 'Health']
        sources = ['Ada Derana', 'Daily Mirror', 'Hiru News', 'News First']
        
        news_items = []
        for i in range(50):
            hours_ago = np.random.exponential(5)
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            
            news_items.append({
                'title': f"Breaking: Important development in {np.random.choice(categories)} sector",
                'source': np.random.choice(sources),
                'category': np.random.choice(categories),
                'sentiment': np.random.uniform(-1, 1),
                'impact_score': np.random.uniform(0, 1),
                'timestamp': timestamp,
                'engagement': np.random.randint(100, 10000)
            })
        
        return pd.DataFrame(news_items).sort_values('timestamp', ascending=False)
    
    def generate_sample_metrics(self) -> Dict:
        """Generate sample metrics"""
        return {
            'overall_risk': np.random.uniform(0.3, 0.8),
            'sentiment_avg': np.random.uniform(-0.5, 0.5),
            'active_alerts': np.random.randint(0, 5),
            'data_sources': 8,
            'articles_today': np.random.randint(100, 500),
            'trending_topics': 12
        }
    
    def generate_sample_trends(self) -> pd.DataFrame:
        """Generate sample trending topics"""
        topics = [
            'Fuel prices', 'Power cuts', 'Tourism growth', 'Export performance',
            'Infrastructure development', 'Agricultural output', 'Tech sector',
            'Education reforms', 'Healthcare access', 'Transport strikes'
        ]
        
        trends = []
        for topic in np.random.choice(topics, 10, replace=False):
            trends.append({
                'topic': topic,
                'volume': np.random.randint(50, 1000),
                'sentiment': np.random.uniform(-1, 1),
                'trend_direction': np.random.choice(['↑', '↓', '→']),
                'risk_level': np.random.uniform(0, 1)
            })
        
        return pd.DataFrame(trends).sort_values('volume', ascending=False)


def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="main-header" style="color:white !important;">🇱🇰 PulseX Sri Lanka</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle" style="color:white !important;">Real-Time Business Intelligence & Situational Awareness</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Last Updated**")
        st.markdown(f"{st.session_state.last_update.strftime('%H:%M:%S')}")
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()


def render_key_metrics(metrics: Dict):
    """Render key performance indicators"""
    st.markdown("### 📊 Key Indicators")
    
    cols = st.columns(4)
    
    # Overall Risk
    with cols[0]:
        risk_score = metrics['overall_risk']
        risk_color = DASHBOARD_CONFIG.COLORS['high_risk'] if risk_score > 0.7 else \
                     DASHBOARD_CONFIG.COLORS['medium_risk'] if risk_score > 0.4 else \
                     DASHBOARD_CONFIG.COLORS['low_risk']
        
        st.metric(
            label="Overall Risk Level",
            value=f"{risk_score:.0%}",
            delta=f"{np.random.randint(-10, 10)}% vs yesterday",
            delta_color="inverse"
        )
    
    # Sentiment
    with cols[1]:
        sentiment = metrics['sentiment_avg']
        st.metric(
            label="Public Sentiment",
            value=f"{sentiment:+.2f}",
            delta="Stable" if abs(sentiment) < 0.2 else "Volatile"
        )
    
    # Active Alerts
    with cols[2]:
        st.metric(
            label="Active Alerts",
            value=metrics['active_alerts'],
            delta=f"{np.random.randint(-2, 3)} new"
        )
    
    # Data Coverage
    with cols[3]:
        st.metric(
            label="Articles Analyzed Today",
            value=metrics['articles_today'],
            delta=f"+{np.random.randint(10, 50)} in last hour"
        )


def render_ai_recommendations():
    """Render AI-powered recommendations"""
    st.markdown("### 🤖 AI-Powered Recommendations")
    
    recommendations = [
        {
            'priority': 'HIGH',
            'action': 'Monitor fuel price discussions closely',
            'reason': 'Sentiment declining rapidly (-15% in 2 hours)',
            'impact': 'Could affect transportation & logistics sectors'
        },
        {
            'priority': 'MEDIUM',
            'action': 'Review supply chain contingencies',
            'reason': 'Weather alerts for Western Province',
            'impact': 'Potential distribution delays'
        },
        {
            'priority': 'LOW',
            'action': 'Opportunity: Tourism sentiment positive',
            'reason': '+23% positive mentions vs last week',
            'impact': 'Consider marketing push'
        }
    ]
    
    for rec in recommendations:
        priority_color = {'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'}[rec['priority']]
        
        st.markdown(f"""
        <div class="recommendation-box" style="border-left-color: {priority_color};">
            <strong style="color: {priority_color};">[{rec['priority']}]</strong> {rec['action']}<br/>
            <small>📋 {rec['reason']}</small><br/>
            <small>💡 {rec['impact']}</small>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_timeline():
    """Public mood over time (plain English).

    - Title: 'Public Mood' (not technical)
    - Badge shows current mood: Positive / Neutral / Negative with emoji and color
    - Line chart kept but no technical axis labels
    - Short instruction in plain English below the chart
    """
    st.markdown("### Public Mood — How people feel now")

    # Sample time series (replace with real series when available)
    hours = pd.date_range(end=datetime.now(), periods=48, freq='H')
    sentiment = np.cumsum(np.random.randn(48) * 0.05)
    df = pd.DataFrame({'time': hours, 'mood': sentiment}).set_index('time')

    # Show a large current-mood badge + small explanation
    latest = df['mood'].iloc[-1]
    if latest > 0.2:
        mood_label = 'Positive'
        mood_color = '#10B981'  # green
        mood_emoji = '😊'
    elif latest < -0.2:
        mood_label = 'Negative'
        mood_color = '#EF4444'  # red
        mood_emoji = '😟'
    else:
        mood_label = 'Neutral'
        mood_color = '#F59E0B'  # amber
        mood_emoji = '😐'

    col_big, col_small = st.columns([3, 1])
    with col_big:
        # Simple, non-technical line chart
        st.line_chart(df['mood'])
    with col_small:
        st.markdown(f"""
            <div style='background:{mood_color}; padding:14px; border-radius:8px; text-align:center;'>
                <div style='font-size:24px; font-weight:700; color:white'>{mood_emoji} {mood_label}</div>
                <div style='color:white; font-size:12px'>Now</div>
            </div>
        """, unsafe_allow_html=True)

    st.caption('Read this chart: the line goes up when people feel better and down when they feel worse. If mood is Negative, open Recent News.')


def render_trending_topics(trends_df: pd.DataFrame):
    """Trending topics in very simple words.

    - Show top 3 topics as big bullets with word labels like Good/Okay/Bad
    - Keep sentences very short and actionable
    """
    st.markdown("### Top Issues Right Now")

    if trends_df is None or len(trends_df) == 0:
        st.info("No trending topics right now")
        return

    top = trends_df.sort_values('volume', ascending=False).head(3).copy()

    # Display each top topic as a clear line with simple sentiment word
    for _, row in top.iterrows():
        sentiment_val = row.get('sentiment', 0.0)
        if sentiment_val > 0.2:
            sentiment_word = 'Good'
            color = '#10B981'
            emoji = '😊'
        elif sentiment_val < -0.2:
            sentiment_word = 'Bad'
            color = '#EF4444'
            emoji = '😟'
        else:
            sentiment_word = 'Okay'
            color = '#F59E0B'
            emoji = '😐'

        st.markdown(f"""
            <div class='white-bg' style='padding:10px; border-radius:6px; margin-bottom:6px;'>
                <strong style='font-size:16px'>{row['topic']}</strong><br/>
                <span style='color:{color}; font-weight:700'>{emoji} {sentiment_word}</span> — {int(row['volume'])} mentions
            </div>
        """, unsafe_allow_html=True)

    st.caption('Start with the top item. If it is Bad, open Recent News to read short summaries.')


def render_news_feed(news_df: pd.DataFrame):
    """Render recent news feed"""
    st.markdown("### 📰 Recent News")
    
    for _, news in news_df.head(10).iterrows():
        sentiment_emoji = '😊' if news['sentiment'] > 0.3 else '😐' if news['sentiment'] > -0.3 else '😟'
        
        time_ago = (datetime.now() - news['timestamp']).total_seconds() / 60
        if time_ago < 60:
            time_str = f"{int(time_ago)}m ago"
        else:
            time_str = f"{int(time_ago/60)}h ago"
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.5rem;">
            <strong>{sentiment_emoji} {news['title']}</strong><br/>
            <small>📰 {news['source']} • 🏷️ {news['category']} • ⏰ {time_str}</small><br/>
            <small>💬 Sentiment: {news['sentiment']:.2f} | 📊 Impact: {news['impact_score']:.0%}</small>
        </div>
        """, unsafe_allow_html=True)


def render_risk_breakdown():
    """Risk summary in plain English.

    - Show ranked list: Most important → Least important, with words High/Medium/Low
    - No percent numbers, just simple words and color cues
    """
    st.markdown("### Risk — What matters most now")

    # Example components; in real app this should come from the risk scorer
    components = {
        'Public Mood': 0.65,
        'Top Topic': 0.55,
        'Unusual Events': 0.45,
        'Data Errors': 0.30,
        'Source Trust': 0.20
    }

    # Sort by importance
    ordered = sorted(components.items(), key=lambda x: x[1], reverse=True)

    for name, score in ordered:
        if score >= 0.6:
            level = 'High'
            color = '#EF4444'
        elif score >= 0.4:
            level = 'Medium'
            color = '#F59E0B'
        else:
            level = 'Low'
            color = '#10B981'

        st.markdown(f"""
            <div class='white-bg' style='padding:8px; border-radius:6px; margin-bottom:6px;'>
                <strong>{name}</strong> — <span style='color:{color}; font-weight:700'>{level}</span>
            </div>
        """, unsafe_allow_html=True)

    st.caption('If an item is High, check Recent News and AI Recommendations now.')


def main():
    """Main dashboard application"""
    
    # Initialize controller
    controller = DashboardController()
    
    # Header
    render_header()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### Data Sources")
        sources = st.multiselect(
            "Select sources",
            ['Ada Derana', 'Daily Mirror', 'Hiru News', 'News First', 'Social Media', 'Economic APIs'],
            default=['Ada Derana', 'Daily Mirror']
        )
        
        st.markdown("### Time Range")
        time_range = st.selectbox("Range", ['Last Hour', 'Last 6 Hours', 'Last 24 Hours', 'Last Week'])
        
        st.markdown("### Risk Threshold")
        risk_threshold = st.slider("Alert when risk exceeds", 0.0, 1.0, 0.7, 0.05)
        
        st.markdown("---")
        st.markdown("**Auto-Refresh**")
        auto_refresh = st.checkbox("Enable", value=True)
        if auto_refresh:
            st.markdown(f"Refreshing every {DASHBOARD_CONFIG.AUTO_REFRESH_INTERVAL}s")
    
    # Main content
    render_key_metrics(st.session_state.metrics_data)
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_sentiment_timeline()
        render_trending_topics(st.session_state.trends_data)
        render_news_feed(st.session_state.news_data)
    
    with col2:
        render_ai_recommendations()
        render_risk_breakdown()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        <strong>PulseX Sri Lanka</strong> • Real-Time Business Intelligence Platform<br/>
        Powered by Advanced ML & Bayesian Analytics
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()