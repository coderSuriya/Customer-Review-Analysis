import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyBtpPsAHdro5ZPVjc0SRT4k7jl_rmvXytc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Set page configuration
st.set_page_config(
    page_title="Customer Review Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to classify sentiment with TextBlob
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive Review"
    elif polarity == 0:
        return "Neutral Review"
    else:
        return "Negative Review"

# Function to call Gemini API for review analysis
def analyze_reviews_with_gemini(reviews, prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": f"{prompt}\n\nReviews:\n{reviews}"}]}]}
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        st.error(f"Gemini API Error (Analysis): {str(e)}")
        return "Error: Unable to analyze reviews."

# Enhanced chatbot response function with conversation context
def chatbot_response(user_input, context="", chat_history=None):
    if chat_history is None:
        chat_history = []
    
    conversation = "Conversation history:\n"
    for entry in chat_history[-5:]:
        conversation += f"User: {entry['user']}\nAI: {entry['response']}\n"
    
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    You are an interactive AI assistant for a customer review analysis app. Your tasks:
    
    1. If the user says 'This review should be positive: [text]' or 'This review should be negative: [text]', respond with 'Updating sentiment to [positive/negative] for: [text].'
    
    2. For questions about the data or analysis, provide helpful insights based on the context.
    
    3. If asked for specific recommendations or product insights, extract relevant points from the data.
    
    4. Maintain a conversational tone and refer to previous exchanges when relevant.
    
    5. Be concise in your responses but thorough in your analysis.
    
    6. If the user asks how to use the app, provide a brief tutorial.
    
    Data Context: {context}
    
    {conversation}
    
    User: {user_input}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        st.error(f"Gemini API Error (Chatbot): {str(e)}")
        return "Sorry, I couldn't process your request."

# Function to generate HTML report
def generate_html_report(filtered_df, strengths, improvements, complaints, recommendations, asin_filter, sentiment_filter, analysis_style):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer Review Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #1E6091;
                text-align: center;
            }}
            h2 {{
                color: #1E6091;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 5px;
            }}
            .section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            ul {{
                padding-left: 20px;
            }}
            .summary {{
                background-color: #e9ecef;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>Customer Review Analysis Report</h1>
        <div class="summary">
            <p><strong>Generated on:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Product ASIN(s):</strong> {', '.join(asin_filter)}</p>
            <p><strong>Sentiment Filter:</strong> {sentiment_filter}</p>
            <p><strong>Analysis Style:</strong> {analysis_style}</p>
            <p><strong>Total Reviews Analyzed:</strong> {len(filtered_df)}</p>
        </div>
        
        <div class="section">
            <h2>Key Strengths</h2>
            {strengths if strengths else '<p>No positive reviews available to analyze.</p>'}
        </div>
        
        <div class="section">
            <h2>Improvement Areas</h2>
            {improvements if improvements else '<p>No negative reviews available to analyze.</p>'}
        </div>
        
        <div class="section">
            <h2>Customer Complaints</h2>
            {complaints if complaints else '<p>No negative reviews available to analyze.</p>'}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {recommendations if recommendations else '<p>No negative reviews available for recommendations.</p>'}
        </div>
        
        <div class="footer">
            Generated by Customer Review Analytics Platform • Powered by Streamlit & Gemini API • © 2025
        </div>
    </body>
    </html>
    """
    return html_content

# Function to create downloadable HTML link
def get_html_download_link(html_content, filename="review_analysis_report.html"):
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Analysis Report as HTML</a>'
    return href

# Custom CSS (unchanged from original)
st.markdown("""
    <style>
    /* [Your existing CSS remains unchanged] */
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Sidebar (unchanged)
with st.sidebar:
    st.markdown('<div class="sidebar-branding">', unsafe_allow_html=True)
    st.markdown('# 📊 Customer Review Analytics', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df['Sentiment'] = st.session_state.df['Customer Review Description'].apply(get_sentiment)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown('<div class="section-header">Filter Controls</div>', unsafe_allow_html=True)
        
        unique_asins = df['Asin'].unique()
        asin_filter = st.multiselect("Product ASIN", unique_asins, default=unique_asins[0] if len(unique_asins) > 0 else [])
        
        sentiment_options = ["All", "Positive Review", "Neutral Review", "Negative Review"]
        sentiment_filter = st.radio("Sentiment Filter", sentiment_options)
        
        analysis_style = st.radio(
            "Analysis Depth",
            ["Simple Crisp", "In-Depth Review"],
            help="Choose between concise bullet points or detailed analysis"
        )
        
        filtered_df = df[df['Asin'].isin(asin_filter)]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_filter]
        
        st.markdown('<div class="section-header">Data Stats</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(filtered_df)}</div>
                <div class="metric-label">Filtered Reviews</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            positive_count = len(filtered_df[filtered_df['Sentiment'] == "Positive Review"])
            positive_percent = round((positive_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{positive_percent}%</div>
                <div class="metric-label">Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)

# Main content area
if st.session_state.df is None:
    # Welcome screen (unchanged)
    st.markdown('<div class="dashboard-title">Customer Review Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Upload a CSV file to analyze customer sentiment and gain actionable insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Upload your customer review CSV file using the sidebar uploader
        2. Filter reviews by product ASIN and sentiment
        3. Explore the interactive visualizations
        4. Chat with our AI assistant for deeper insights
        5. Review automatically generated analysis reports
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Expected CSV Format</div>', unsafe_allow_html=True)
        sample_data = {
            'Asin': ['B001XXXXX', 'B001XXXXX', 'B002YYYYY'],
            'Customer Review Description': [
                'Great product, exactly as described!', 
                'Disappointed with the quality.', 
                'Works well but delivery was delayed.'
            ]
        }
        st.dataframe(pd.DataFrame(sample_data))
        st.markdown('</div>', unsafe_allow_html=True)
        
else:
    df = st.session_state.df
    filtered_df = df[df['Asin'].isin(asin_filter)]
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_filter]
    
    main_tabs = st.tabs(["Dashboard", "Review Explorer", "Analysis", "AI Assistant"])
    
    # --- Dashboard Tab --- (unchanged)
    with main_tabs[0]:
        st.markdown('<div class="dashboard-title">Customer Review Dashboard</div>', unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(filtered_df)}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metrics_col2:
            pos_count = len(filtered_df[filtered_df['Sentiment'] == "Positive Review"])
            pos_percent = round((pos_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pos_count} ({pos_percent}%)</div>
                <div class="metric-label">Positive Reviews</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metrics_col3:
            neg_count = len(filtered_df[filtered_df['Sentiment'] == "Negative Review"])
            neg_percent = round((neg_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{neg_count} ({neg_percent}%)</div>
                <div class="metric-label">Negative Reviews</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metrics_col4:
            neutral_count = len(filtered_df[filtered_df['Sentiment'] == "Neutral Review"])
            neutral_percent = round((neutral_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{neutral_count} ({neutral_percent}%)</div>
                <div class="metric-label">Neutral Reviews</div>
            </div>
            """, unsafe_allow_html=True)
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
            
            sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            colors = {'Positive Review': '#4CAF50', 'Neutral Review': '#FFC107', 'Negative Review': '#F44336'}
            
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts['Sentiment'],
                values=sentiment_counts['Count'],
                hole=.4,
                marker=dict(colors=[colors.get(label, '#999') for label in sentiment_counts['Sentiment']]),
                textinfo='percent+label'
            )])
            
            fig.update_layout(
                margin=dict(t=0, b=0, l=10, r=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with viz_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Key Insights Summary</div>', unsafe_allow_html=True)
            
            if 'analysis_cache' in st.session_state:
                cache_key = f"{','.join(asin_filter)}_{sentiment_filter}_{analysis_style}"
                if cache_key+"_improvements" in st.session_state.analysis_cache:
                    improvements = st.session_state.analysis_cache[cache_key+"_improvements"]
                    st.markdown(f"<div class='big-bullets'>{improvements}</div>", unsafe_allow_html=True)
                else:
                    st.info("Filter reviews and visit the Analysis tab to generate insights.")
            else:
                st.info("Filter reviews and visit the Analysis tab to generate insights.")
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Review Explorer Tab --- (unchanged)
    with main_tabs[1]:
        st.markdown('<div class="section-header">Review Explorer</div>', unsafe_allow_html=True)
        
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.text_input("Search in reviews", placeholder="Enter keywords...")
        with search_col2:
            sort_by = st.selectbox("Sort by", ["Sentiment", "Asin"])
        
        if search_term:
            filtered_df = filtered_df[filtered_df['Customer Review Description'].str.contains(search_term, case=False, na=False)]
        
        if sort_by == "Sentiment":
            filtered_df = filtered_df.sort_values(by=['Sentiment'])
        else:
            filtered_df = filtered_df.sort_values(by=['Asin'])
        
        styled_reviews = []
        for _, row in filtered_df.iterrows():
            sentiment_class = ""
            if row['Sentiment'] == "Positive Review":
                sentiment_class = "status-positive"
            elif row['Sentiment'] == "Negative Review":
                sentiment_class = "status-negative"
            else:
                sentiment_class = "status-neutral"
                
            styled_reviews.append({
                "ASIN": row['Asin'],
                "Review": row['Customer Review Description'],
                "Sentiment": f"<span class='{sentiment_class}'>{row['Sentiment']}</span>"
            })
        
        if styled_reviews:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for i, review in enumerate(styled_reviews):
                st.markdown(f"""
                <div style="padding: 10px 0; border-bottom: 1px solid #eee;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <div><strong>ASIN:</strong> {review['ASIN']}</div>
                        <div>{review['Sentiment']}</div>
                    </div>
                    <div style="font-size: 15px;">{review['Review']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No reviews match your current filters.")
    
    # --- Analysis Tab --- (Modified with Export Option)
    with main_tabs[2]:
        st.markdown('<div class="section-header">Review Analysis</div>', unsafe_allow_html=True)
        
        positive_reviews = filtered_df[filtered_df['Sentiment'] == "Positive Review"]['Customer Review Description'].tolist()
        negative_reviews = filtered_df[filtered_df['Sentiment'] == "Negative Review"]['Customer Review Description'].tolist()
        positive_reviews_text = "\n".join(positive_reviews)
        negative_reviews_text = "\n".join(negative_reviews)
        
        if analysis_style == "Simple Crisp":
            strength_prompt = "Analyze these positive reviews for the selected ASIN(s) and summarize all key strengths in exactly 5-6 short, crisp bullet points covering all essential information."
            improvement_prompt = "Analyze these negative reviews for the selected ASIN(s) and summarize all key improvement areas in exactly 5-6 short, crisp bullet points covering all essential information."
            complaints_prompt = "Analyze these negative reviews for the selected ASIN(s) and summarize all common customer complaints in exactly 5-6 short, crisp bullet points covering all essential information."
            recommendation_prompt = "Based on these negative reviews for the selected ASIN(s), provide exactly 5-6 concise improvement recommendations in short, crisp bullet points covering all essential information."
        else:  # In-Depth Review
            strength_prompt = "Analyze these positive reviews for the selected ASIN(s) and provide a detailed breakdown of key strengths in comprehensive bullet points with examples."
            improvement_prompt = "Analyze these negative reviews for the selected ASIN(s) and provide a detailed breakdown of key improvement areas in comprehensive bullet points with examples."
            complaints_prompt = "Analyze these negative reviews for the selected ASIN(s) and provide a detailed list of common customer complaints in comprehensive bullet points with examples."
            recommendation_prompt = "Based on these negative reviews for the selected ASIN(s), provide detailed improvement recommendations in comprehensive bullet points with specific suggestions."
        
        cache_key = f"{','.join(asin_filter)}_{sentiment_filter}_{analysis_style}"
        
        analysis_tabs = st.tabs(["Key Strengths", "Improvement Areas", "Customer Complaints", "Recommendations"])
        
        # Variables to store analysis results for export
        strengths, improvements, complaints, recommendations = "", "", "", ""
        
        with analysis_tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if positive_reviews:
                if cache_key+"_strengths" not in st.session_state.analysis_cache:
                    with st.spinner("Analyzing strengths..."):
                        strengths = analyze_reviews_with_gemini(positive_reviews_text, strength_prompt)
                        st.session_state.analysis_cache[cache_key+"_strengths"] = strengths
                else:
                    strengths = st.session_state.analysis_cache[cache_key+"_strengths"]
                
                st.markdown(f"<div class='big-bullets'>{strengths}</div>", unsafe_allow_html=True)
            else:
                strengths = ""
                st.info("No positive reviews available to analyze.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with analysis_tabs[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if negative_reviews:
                if cache_key+"_improvements" not in st.session_state.analysis_cache:
                    with st.spinner("Analyzing improvement areas..."):
                        improvements = analyze_reviews_with_gemini(negative_reviews_text, improvement_prompt)
                        st.session_state.analysis_cache[cache_key+"_improvements"] = improvements
                else:
                    improvements = st.session_state.analysis_cache[cache_key+"_improvements"]
                
                st.markdown(f"<div class='big-bullets'>{improvements}</div>", unsafe_allow_html=True)
            else:
                improvements = ""
                st.info("No negative reviews available to analyze.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with analysis_tabs[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if negative_reviews:
                if cache_key+"_complaints" not in st.session_state.analysis_cache:
                    with st.spinner("Analyzing customer complaints..."):
                        complaints = analyze_reviews_with_gemini(negative_reviews_text, complaints_prompt)
                        st.session_state.analysis_cache[cache_key+"_complaints"] = complaints
                else:
                    complaints = st.session_state.analysis_cache[cache_key+"_complaints"]
                
                st.markdown(f"<div class='big-bullets'>{complaints}</div>", unsafe_allow_html=True)
            else:
                complaints = ""
                st.info("No negative reviews available to analyze complaints.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with analysis_tabs[3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if negative_reviews:
                if cache_key+"_recommendations" not in st.session_state.analysis_cache:
                    with st.spinner("Generating recommendations..."):
                        recommendations = analyze_reviews_with_gemini(negative_reviews_text, recommendation_prompt)
                        st.session_state.analysis_cache[cache_key+"_recommendations"] = recommendations
                else:
                    recommendations = st.session_state.analysis_cache[cache_key+"_recommendations"]
                
                st.markdown(f"<div class='big-bullets'>{recommendations}</div>", unsafe_allow_html=True)
            else:
                recommendations = ""
                st.info("No negative reviews available for recommendations.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export button
        if st.button("Export Analysis as HTML"):
            with st.spinner("Generating HTML report..."):
                html_report = generate_html_report(
                    filtered_df, strengths, improvements, complaints, recommendations,
                    asin_filter, sentiment_filter, analysis_style
                )
                st.markdown(get_html_download_link(html_report), unsafe_allow_html=True)
                st.success("Report generated! Click the link above to download.")

    # --- AI Assistant Tab --- (unchanged)
    with main_tabs[3]:
        st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
        
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-message">{chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">{chat["response"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_message = st.text_input("Ask about your data or update reviews", placeholder="Type your message here...")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                btn1 = st.form_submit_button("📊 Top Complaints", 
                                           help="Get the most common customer complaints")
            with col2:
                btn2 = st.form_submit_button("💡 Improvement Ideas", 
                                          help="Get ideas to improve product ratings")
            with col3:
                btn3 = st.form_submit_button("🔍 Product Strengths", 
                                          help="Find what customers love about products")
            
            submit_button = st.form_submit_button("Send")
        
        input_text = None
        if submit_button and user_message:
            input_text = user_message
        elif btn1:
            input_text = "What are the top customer complaints in the current data?"
        elif btn2:
            input_text = "What are the most important improvements that should be made based on these reviews?"
        elif btn3:
            input_text = "What are the key strengths of the products according to customers?"
        
        if input_text:
            context = f"Current filtered data for ASIN(s) {', '.join(asin_filter)}:\n"
            context += f"Total reviews: {len(filtered_df)}\n"
            context += f"Positive reviews: {len(positive_reviews)} examples: {positive_reviews_text[:500]}...\n"
            context += f"Negative reviews: {len(negative_reviews)} examples: {negative_reviews_text[:500]}...\n"
            
            sentiment_summary = f"Sentiment distribution: Positive: {pos_percent}%, Negative: {neg_percent}%, Neutral: {neutral_percent}%\n"
            context += sentiment_summary
            
            with st.spinner("Processing your request..."):
                response = chatbot_response(input_text, context, st.session_state.chat_history)
            
            update_made = False
            if "should be positive" in input_text.lower():
                review_text = input_text.split("should be positive:", 1)[-1].strip()
                if review_text in df['Customer Review Description'].values:
                    index = df[df['Customer Review Description'] == review_text].index[0]
                    st.session_state.df.loc[index, 'Sentiment'] = "Positive Review"
                    response = f"Updating sentiment to positive for: '{review_text}'."
                    update_made = True
                else:
                    response = f"Review not found: '{review_text}'."
            elif "should be negative" in input_text.lower():
                review_text = input_text.split("should be negative:", 1)[-1].strip()
                if review_text in df['Customer Review Description'].values:
                    index = df[df['Customer Review Description'] == review_text].index[0]
                    st.session_state.df.loc[index, 'Sentiment'] = "Negative Review"
                    response = f"Updating sentiment to negative for: '{review_text}'."
                    update_made = True
                else:
                    response = f"Review not found: '{review_text}'."
            
            if not update_made and "sorry" not in response.lower():
                st.session_state.chat_history.append({"user": input_text, "response": response})
            
            with chat_container:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for chat in st.session_state.chat_history:
                    st.markdown(f'<div class="user-message">{chat["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bot-message">{chat["response"]}</div>', unsafe_allow_html=True)
                if update_made or "sorry" in response.lower():
                    st.markdown(f'<div class="user-message">{input_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; font-size: 12px;'>
        Powered by Streamlit & Gemini API • Built for Customer Insights • © 2025
    </div>
    """, unsafe_allow_html=True)