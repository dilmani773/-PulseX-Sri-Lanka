"""
Reusable UI Components for PulseX Dashboard
Custom Streamlit components for consistent design
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def metric_card(title: str, value: str, delta: Optional[str] = None, 
                delta_color: str = "normal", icon: str = "📊"):
    """
    Display a metric card with title, value, and optional delta
    
    Args:
        title: Metric title
        value: Main value to display
        delta: Optional change indicator
        delta_color: 'normal', 'inverse', or 'off'
        icon: Emoji icon
    """
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"<div style='font-size: 2rem;'>{icon}</div>", 
                   unsafe_allow_html=True)
    with col2:
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def alert_box(message: str, alert_type: str = "info", dismissible: bool = False):
    """
    Display an alert box
    
    Args:
        message: Alert message
        alert_type: 'success', 'info', 'warning', 'error'
        dismissible: Whether alert can be dismissed
    """
    colors = {
        'success': ('#D1FAE5', '#065F46', '#10B981'),
        'info': ('#DBEAFE', '#1E40AF', '#3B82F6'),
        'warning': ('#FEF3C7', '#92400E', '#F59E0B'),
        'error': ('#FEE2E2', '#991B1B', '#EF4444')
    }
    
    icons = {
        'success': '✅',
        'info': 'ℹ️',
        'warning': '⚠️',
        'error': '❌'
    }
    
    bg_color, text_color, border_color = colors.get(alert_type, colors['info'])
    icon = icons.get(alert_type, 'ℹ️')
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: {text_color};
        font-weight: 500;
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: str, score: float):
    """
    Display a risk level badge
    
    Args:
        risk_level: 'critical', 'high', 'medium', 'low', 'minimal'
        score: Risk score 0-1
    """
    colors = {
        'critical': ('#EF4444', 'white'),
        'high': ('#F59E0B', 'white'),
        'medium': ('#F59E0B', 'white'),
        'low': ('#10B981', 'white'),
        'minimal': ('#10B981', 'white')
    }
    
    bg_color, text_color = colors.get(risk_level, ('#6B7280', 'white'))
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        color: {text_color};
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        display: inline-block;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
    ">
        {risk_level} ({score:.0%})
    </div>
    """, unsafe_allow_html=True)


def progress_ring(value: float, label: str, color: str = "#3B82F6"):
    """
    Display a circular progress indicator
    
    Args:
        value: Progress value 0-1
        label: Label text
        color: Ring color
    """
    percentage = int(value * 100)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'bgcolor': "lightgray",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#FEE2E2'},
                {'range': [40, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#D1FAE5'}
            ],
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        title={'text': label, 'x': 0.5, 'xanchor': 'center'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def timeline_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                   title: str = "Timeline", color: str = "#3B82F6"):
    """
    Display a timeline chart
    
    Args:
        data: DataFrame with time series data
        x_col: Column name for x-axis (time)
        y_col: Column name for y-axis (values)
        title: Chart title
        color: Line color
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        name=y_col,
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_col,
        hovermode='x unified',
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def news_card(title: str, source: str, time_ago: str, 
              sentiment: float, category: str, url: str = "#"):
    """
    Display a news article card
    
    Args:
        title: Article title
        source: News source
        time_ago: Time since published
        sentiment: Sentiment score -1 to 1
        category: Article category
        url: Article URL
    """
    # Sentiment emoji
    if sentiment > 0.3:
        sentiment_emoji = "😊"
        sentiment_color = "#10B981"
    elif sentiment < -0.3:
        sentiment_emoji = "😟"
        sentiment_color = "#EF4444"
    else:
        sentiment_emoji = "😐"
        sentiment_color = "#6B7280"
    
    st.markdown(f"""
    <div style="
        border: 1px solid #E5E7EB;
        border-left: 4px solid {sentiment_color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <strong style="font-size: 1rem; color: #1F2937;">{title}</strong>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #6B7280;">
                    📰 {source} • 🏷️ {category} • ⏰ {time_ago}
                </div>
            </div>
            <div style="font-size: 1.5rem; margin-left: 1rem;">
                {sentiment_emoji}
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.875rem;">
            <span style="color: {sentiment_color}; font-weight: 600;">
                Sentiment: {sentiment:+.2f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_box(value: str, label: str, icon: str = "📊", color: str = "#3B82F6"):
    """
    Display a statistic box
    
    Args:
        value: Main value to display
        label: Label text
        icon: Emoji icon
        color: Accent color
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
        border: 2px solid {color};
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {color}; margin-bottom: 0.25rem;">
            {value}
        </div>
        <div style="font-size: 0.875rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def recommendation_card(priority: str, action: str, reason: str, impact: str):
    """
    Display an AI recommendation card
    
    Args:
        priority: 'HIGH', 'MEDIUM', 'LOW'
        action: Recommended action
        reason: Reason for recommendation
        impact: Expected impact
    """
    colors = {
        'HIGH': '#EF4444',
        'MEDIUM': '#F59E0B',
        'LOW': '#10B981'
    }
    
    color = colors.get(priority, '#6B7280')
    
    st.markdown(f"""
    <div style="
        background-color: #F9FAFB;
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong style="color: {color}; font-size: 0.875rem; letter-spacing: 0.05em;">
                [{priority}]
            </strong>
        </div>
        <div style="font-size: 1rem; font-weight: 600; color: #1F2937; margin-bottom: 0.5rem;">
            {action}
        </div>
        <div style="font-size: 0.875rem; color: #6B7280; margin-bottom: 0.25rem;">
            📋 {reason}
        </div>
        <div style="font-size: 0.875rem; color: #6B7280;">
            💡 {impact}
        </div>
    </div>
    """, unsafe_allow_html=True)


def data_table(df: pd.DataFrame, title: str = "Data Table", max_rows: int = 10):
    """
    Display a styled data table
    
    Args:
        df: DataFrame to display
        title: Table title
        max_rows: Maximum rows to show
    """
    st.markdown(f"### {title}")
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        hide_index=True,
        height=400
    )


def trend_indicator(value: float, threshold: float = 0):
    """
    Display a trend indicator arrow
    
    Args:
        value: Trend value
        threshold: Threshold for neutral
    """
    if value > threshold:
        return "📈 ↑", "#10B981"
    elif value < -threshold:
        return "📉 ↓", "#EF4444"
    else:
        return "→", "#6B7280"


def loading_animation(message: str = "Loading..."):
    """Display a loading animation"""
    with st.spinner(message):
        import time
        time.sleep(0.5)


# Export all components
__all__ = [
    'metric_card',
    'alert_box',
    'risk_badge',
    'progress_ring',
    'timeline_chart',
    'news_card',
    'stat_box',
    'recommendation_card',
    'data_table',
    'trend_indicator',
    'loading_animation'
]