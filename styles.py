"""
Professional UI styles for the Farmer Assistant Bot
Inspired by PRAGATI design - modern, clean, agricultural theme
"""

import streamlit as st

def load_css():
    """Load custom CSS for professional styling"""
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fffe 0%, #f0f9f5 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(46, 125, 50, 0.15);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Card Styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(46, 125, 50, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.1);
        border: 2px solid rgba(46, 125, 50, 0.1);
        text-align: center;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(46, 125, 50, 0.15);
        border-color: #2E7D32;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #2E7D32;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1B5E20;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #4A5568;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #E8F5E8;
        padding: 12px 16px;
        background: white;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E7D32;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #E8F5E8;
        padding: 12px 16px;
        background: white;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #2E7D32;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fffe 0%, #f0f9f5 100%);
        border-right: 2px solid rgba(46, 125, 50, 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(46, 125, 50, 0.05);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background: transparent;
        border: none;
        color: #2E7D32;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2E7D32;
        color: white;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.2);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border: 1px solid #2E7D32;
        border-radius: 12px;
        color: #1B5E20;
    }
    
    .stError {
        background: linear-gradient(135deg, #FFEBEE 0%, #FCE4EC 100%);
        border: 1px solid #D32F2F;
        border-radius: 12px;
        color: #B71C1C;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFFDE7 100%);
        border: 1px solid #F57C00;
        border-radius: 12px;
        color: #E65100;
    }
    
    /* Metrics Styling */
    .css-1xarl3l {
        background: white;
        border-radius: 12px;
        border: 1px solid rgba(46, 125, 50, 0.1);
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #E8F5E8;
        background: white;
    }
    
    /* Upload Widget Styling */
    .css-1cpxqw2 {
        border: 2px dashed #2E7D32;
        border-radius: 12px;
        background: rgba(46, 125, 50, 0.02);
        transition: all 0.3s ease;
    }
    
    .css-1cpxqw2:hover {
        border-color: #1B5E20;
        background: rgba(46, 125, 50, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        border-radius: 10px;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #E8F5E8 100%);
        border: 1px solid rgba(46, 125, 50, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 2px;
    }
    
    .status-success {
        background: rgba(76, 175, 80, 0.1);
        color: #2E7D32;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .status-warning {
        background: rgba(255, 152, 0, 0.1);
        color: #E65100;
        border: 1px solid rgba(255, 152, 0, 0.3);
    }
    
    .status-error {
        background: rgba(244, 67, 54, 0.1);
        color: #B71C1C;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid rgba(46, 125, 50, 0.1);
        border-top: 3px solid #2E7D32;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_header():
    """Create beautiful header section"""
    st.markdown("""
    <div class="main-header">
        <h1>🌾 KisanSaathi Assistant</h1>
        <p>Your Intelligent Agricultural Companion Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(icon, title, description, key=None):
    """Create a feature card"""
    return f"""
    <div class="feature-card" onclick="window.location.href='#{key or title.lower().replace(' ', '-')}'">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-description">{description}</div>
    </div>
    """

def create_info_box(content, type_="info"):
    """Create an info box"""
    return f"""
    <div class="info-box">
        {content}
    </div>
    """

def create_status_indicator(text, status="success"):
    """Create a status indicator"""
    return f"""
    <span class="status-indicator status-{status}">
        {text}
    </span>
    """

def create_card(content, title=None):
    """Create a custom card"""
    if title:
        return f"""
        <div class="custom-card">
            <h3 style="color: #2E7D32; margin-bottom: 1rem;">{title}</h3>
            {content}
        </div>
        """
    else:
        return f"""
        <div class="custom-card">
            {content}
        </div>
        """
