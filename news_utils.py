import yfinance as yf
import feedparser
import requests
from datetime import datetime, timedelta
import streamlit as st
from bs4 import BeautifulSoup

def get_stock_news(symbol, max_articles=10):
    """Get news for a specific stock symbol"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        articles = []
        for item in news[:max_articles]:
            # Parse timestamp
            timestamp = item.get('providerPublishTime', 0)
            if timestamp:
                publish_time = datetime.fromtimestamp(timestamp)
            else:
                publish_time = datetime.now() - timedelta(hours=1)
            
            articles.append({
                'title': item.get('title', f'{symbol} News'),
                'summary': item.get('summary', 'Click to read more...'),
                'link': item.get('link', ''),
                'publisher': item.get('publisher', 'Yahoo Finance'),
                'publish_time': publish_time,
                'category': f'{symbol} News'
            })
        
        return articles, None
    except Exception as e:
        return [], str(e)

def get_general_news(category="General Market", max_articles=10):
    """Get general market news"""
    articles = []
    
    # Sample news data (replace with real API calls)
    sample_articles = [
        {
            'title': 'Stock Market Shows Mixed Performance Today',
            'summary': 'Major indices traded in mixed territory as investors digest economic data and corporate earnings.',
            'link': 'https://finance.yahoo.com',
            'publisher': 'Yahoo Finance',
            'publish_time': datetime.now() - timedelta(hours=1),
            'category': category
        },
        {
            'title': 'Federal Reserve Considers Interest Rate Changes',
            'summary': 'The Federal Reserve is weighing monetary policy options amid changing economic conditions.',
            'link': 'https://finance.yahoo.com',
            'publisher': 'Reuters',
            'publish_time': datetime.now() - timedelta(hours=2),
            'category': category
        },
        {
            'title': 'Technology Stocks Lead Market Gains',
            'summary': 'Tech sector outperforms broader market on positive earnings and innovation news.',
            'link': 'https://finance.yahoo.com',
            'publisher': 'Bloomberg',
            'publish_time': datetime.now() - timedelta(hours=3),
            'category': category
        },
        {
            'title': 'Oil Prices Rise on Supply Concerns',
            'summary': 'Crude oil futures climb as geopolitical tensions affect global supply chains.',
            'link': 'https://finance.yahoo.com',
            'publisher': 'CNBC',
            'publish_time': datetime.now() - timedelta(hours=4),
            'category': category
        },
        {
            'title': 'Cryptocurrency Market Volatility Continues',
            'summary': 'Digital assets experience significant price swings amid regulatory developments.',
            'link': 'https://finance.yahoo.com',
            'publisher': 'CoinDesk',
            'publish_time': datetime.now() - timedelta(hours=5),
            'category': category
        }
    ]
    
    # Try to get real RSS news first
    try:
        rss_urls = {
            "General Market": "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "Technology": "https://feeds.finance.yahoo.com/rss/2.0/category-technology",
            "Crypto": "https://feeds.finance.yahoo.com/rss/2.0/category-crypto",
            "Economy": "https://feeds.finance.yahoo.com/rss/2.0/category-economy"
        }
        
        url = rss_urls.get(category, rss_urls["General Market"])
        feed = feedparser.parse(url)
        
        if feed.entries:
            for entry in feed.entries[:max_articles]:
                # Parse time
                pub_time = datetime.now() - timedelta(hours=1)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        pub_time = datetime(*entry.published_parsed[:6])
                    except:
                        pass
                
                # Clean summary
                summary = entry.get('summary', entry.get('description', 'No summary available'))
                if summary and len(summary) > 10:
                    try:
                        summary = BeautifulSoup(summary, 'html.parser').get_text()[:200]
                    except:
                        summary = summary[:200]
                
                articles.append({
                    'title': entry.get('title', 'Market News'),
                    'summary': summary,
                    'link': entry.get('link', ''),
                    'publisher': 'Yahoo Finance',
                    'publish_time': pub_time,
                    'category': category
                })
        else:
            # Use sample data if RSS fails
            articles = sample_articles[:max_articles]
            
    except Exception as e:
        # Fallback to sample data
        articles = sample_articles[:max_articles]
    
    return articles, None

def display_news_article(article, index):
    """Display a single news article"""
    # Calculate time ago
    time_diff = datetime.now() - article['publish_time']
    if time_diff.days > 0:
        time_ago = f"{time_diff.days} days ago"
    elif time_diff.seconds > 3600:
        time_ago = f"{time_diff.seconds // 3600} hours ago"
    else:
        time_ago = f"{time_diff.seconds // 60} minutes ago"
    
    with st.container():
        # Article card
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="color: #1f77b4; margin: 0 0 10px 0;">{article['title']}</h4>
            <p style="color: #666; margin: 10px 0;">{article['summary']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <small style="color: #888;">📰 {article['publisher']} • 🕒 {time_ago}</small>
                <span style="background: #1f77b4; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                    {article['category']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Read more button
        if article.get('link'):
            if st.button(f"📖 Read Full Article", key=f"read_{index}"):
                st.markdown(f"🔗 [Open Article]({article['link']})")
        
        st.markdown("---")
