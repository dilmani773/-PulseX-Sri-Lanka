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
    """
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"<div style='font-size: 2rem;'>{icon}</div>", 
                   unsafe_allow_html=True)
    with col2:
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def alert_box(message: str, alert_type: str = "info", dismissible: bool = False):
    """
    Display an alert box with High Contrast Colors
    """
    colors = {
        'success': ('#D1FAE5', '#065F46', '#059669'), # Green: BG, Text, Border
        'info':    ('#DBEAFE', '#1E40AF', '#2563EB'), # Blue
        'warning': ('#FEF3C7', '#92400E', '#D97706'), # Orange
        'error':   ('#FEE2E2', '#991B1B', '#DC2626')  # Red
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
        border-left: 6px solid {border_color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: {text_color};
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: str, score: float):
    """
    Display a risk level badge with Stronger Colors
    """
    colors = {
        'critical': ('#DC2626', 'white'), # Red 600
        'high':     ('#EA580C', 'white'), # Orange 600
        'medium':   ('#D97706', 'white'), # Amber 600
        'low':      ('#059669', 'white'), # Emerald 600
        'minimal':  ('#059669', 'white')
    }
    
    bg_color, text_color = colors.get(risk_level, ('#4B5563', 'white'))
    
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    ">
        {risk_level} ({score:.0%})
    </div>
    """, unsafe_allow_html=True)


def progress_ring(value: float, label: str, color: str = "#3B82F6"):
    """
    Display a circular progress indicator
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
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#D1D5DB",
            'steps': [
                {'range': [0, 100], 'color': '#F3F4F6'}
            ],
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        title={'text': label, 'font': {'size': 14, 'color': '#374151', 'weight': 'bold'}}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def timeline_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                   title: str = "Timeline", color: str = "#3B82F6"):
    """
    Display a timeline chart
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
    Display a news article card with VISIBLE Backgrounds
    """
    # Logic to determine Color and Background (High Contrast Mode)
    if sentiment > 0.05:
        # POSITIVE: Emerald Green
        sentiment_emoji = "😊"
        border_color = "#059669" # Darker Green Border
        bg_color = "#D1FAE5"     # Visible Mint Green BG
        text_color = "#064E3B"   # Dark Green Text
        sentiment_text = f"+{sentiment:.2f}"
    elif sentiment < -0.05:
        # NEGATIVE: Strong Red
        sentiment_emoji = "😟"
        border_color = "#DC2626" # Darker Red Border
        bg_color = "#FEE2E2"     # Visible Red BG
        text_color = "#7F1D1D"   # Dark Red Text
        sentiment_text = f"{sentiment:.2f}"
    else:
        # NEUTRAL: Clear Grey
        sentiment_emoji = "😐"
        border_color = "#6B7280" # Dark Grey Border
        bg_color = "#E5E7EB"     # Visible Grey BG
        text_color = "#1F2937"   # Dark Grey Text
        sentiment_text = "Neutral"
    
    st.markdown(f"""
    <div style="
        border: 1px solid {border_color}40;
        border-left: 6px solid {border_color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        background-color: {bg_color}; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <strong style="font-size: 1.05rem; color: #111827;">{title}</strong>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #4B5563;">
                    📰 <b>{source}</b> • {category} • {time_ago}
                </div>
            </div>
            <div style="font-size: 1.8rem; margin-left: 1rem;">
                {sentiment_emoji}
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem;">
            <span style="color: {text_color}; font-weight: 800; background-color: rgba(255,255,255,0.5); padding: 2px 6px; border-radius: 4px;">
                Sentiment: {sentiment_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_box(value: str, label: str, icon: str = "📊", color: str = "#3B82F6"):
    """
    Display a statistic box
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20 0%, {color}05 100%);
        border: 2px solid {color};
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {color}; margin-bottom: 0.25rem;">
            {value}
        </div>
        <div style="font-size: 0.875rem; color: #4B5563; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def recommendation_card(priority: str, action: str, reason: str, impact: str):
    """
    Display an AI recommendation card with High Contrast
    """
    colors = {
        'HIGH':   ('#DC2626', '#FEF2F2'), # Red Border, Red BG
        'MEDIUM': ('#D97706', '#FFFBEB'), # Amber Border, Amber BG
        'LOW':    ('#059669', '#ECFDF5')  # Emerald Border, Green BG
    }
    
    # Default to Grey if unknown
    border_color, bg_color = colors.get(priority, ('#4B5563', '#F3F4F6'))
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 6px solid {border_color};
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong style="color: {border_color}; font-size: 0.9rem; letter-spacing: 0.05em; text-transform: uppercase; background: white; padding: 2px 8px; border-radius: 4px; border: 1px solid {border_color};">
                {priority} PRIORITY
            </strong>
        </div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
            {action}
        </div>
        <div style="font-size: 0.95rem; color: #374151; margin-bottom: 0.25rem;">
            <span style="font-weight: 600;">📋 Reason:</span> {reason}
        </div>
        <div style="font-size: 0.95rem; color: #374151;">
            <span style="font-weight: 600;">💡 Impact:</span> {impact}
        </div>
    </div>
    """, unsafe_allow_html=True)


def data_table(df: pd.DataFrame, title: str = "Data Table", max_rows: int = 10):
    """
    Display a styled data table
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