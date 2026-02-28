import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import RetailDataGenerator
from models import RetailMLModels
from recommendation_engine import RetailRecommendationEngine
from utils import RetailAnalyticsUtils

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Retail Analytics Platform",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def set_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stSidebar {
        background-color: #1E2139;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background-color: #1E2139;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .recommendation-card {
        background-color: #1E2139;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #667eea;
    }
    .high-priority {
        border-left: 4px solid #ff6b6b;
    }
    .medium-priority {
        border-left: 4px solid #ffd93d;
    }
    .low-priority {
        border-left: 4px solid #6bcf7f;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = RetailMLModels()
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RetailRecommendationEngine()
    if 'utils' not in st.session_state:
        st.session_state.utils = RetailAnalyticsUtils()
    if 'cluster_info' not in st.session_state:
        st.session_state.cluster_info = None

# Sidebar navigation
def create_sidebar():
    with st.sidebar:
        st.markdown('<div class="header-title">🏪 Retail AI</div>', unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Home",
                "📊 Dataset",
                "🤖 Predictions",
                "👥 Customer Segmentation",
                "📈 Visualizations & Insights",
                "💡 AI Recommendations"
            ]
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Generate New Data", help="Generate fresh synthetic dataset"):
            generate_new_data()
        
        if st.button("📥 Load Sample Data", help="Load pre-generated sample data"):
            load_sample_data()
        
        if st.button("💾 Save Current Session", help="Save current models and data"):
            save_session()
        
        st.markdown("---")
        st.markdown("### System Status")
        
        if st.session_state.data is not None:
            st.success(f"✅ Dataset loaded: {len(st.session_state.data)} records")
        else:
            st.warning("⚠️ No dataset loaded")
        
        if hasattr(st.session_state.models, 'models') and st.session_state.models.models:
            st.success(f"✅ {len(st.session_state.models.models)} models trained")
        else:
            st.warning("⚠️ No models trained")
    
    return page

# Home page
def show_home():
    st.markdown('<div class="header-title">Smart Retail Analytics Platform</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Sales Forecasting</h3>
            <p>Predict monthly sales using advanced Linear Regression algorithms with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Customer Classification</h3>
            <p>Classify purchase decisions and predict loyalty categories using ML models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡 AI Recommendations</h3>
            <p>Get intelligent business recommendations powered by AI and customer segmentation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🔍 Problem Statement</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>Business Challenge</h4>
    <p>Retail businesses struggle with understanding customer behavior, predicting sales trends, and making data-driven decisions. 
    Traditional methods often fail to capture complex patterns in customer data, leading to missed opportunities and inefficient marketing strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎯 Solution Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🤖 Machine Learning Models</h4>
        <ul>
            <li><strong>Linear Regression:</strong> Sales prediction</li>
            <li><strong>Decision Tree:</strong> Purchase decision classification</li>
            <li><strong>KNN:</strong> Loyalty category prediction</li>
            <li><strong>K-Means:</strong> Customer segmentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📈 Key Features</h4>
        <ul>
            <li>Real-time predictions and insights</li>
            <li>Interactive visualizations</li>
            <li>AI-powered recommendations</li>
            <li>Comprehensive analytics dashboard</li>
            <li>Export capabilities for reports</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">👥 Team & Technology</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🔧 Technology Stack</h4>
        <p>Python, Streamlit, Scikit-learn, Plotly, Pandas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Science</h4>
        <p>Advanced ML algorithms and statistical analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
        <h4>🎨 User Experience</h4>
        <p>Modern, responsive, and intuitive interface</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow diagram
    st.markdown('<div class="section-header">🔄 Workflow Process</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add workflow steps
    steps = [
        ("Data Generation", 1, 1),
        ("Model Training", 2, 1),
        ("Predictions", 3, 1),
        ("Segmentation", 4, 1),
        ("Recommendations", 5, 1)
    ]
    
    for step, x, y in steps:
        fig.add_shape(
            type="rect",
            x0=x-0.4, y0=y-0.2, x1=x+0.4, y1=y+0.2,
            line=dict(color="#667eea", width=2),
            fillcolor="#1E2139"
        )
        fig.add_annotation(
            x=x, y=y, text=step,
            font=dict(color="white", size=10),
            showarrow=False
        )
    
    # Add arrows
    for i in range(len(steps)-1):
        fig.add_annotation(
            x=steps[i][1] + 0.4, y=steps[i][2],
            ax=steps[i+1][1] - 0.4, ay=steps[i+1][2],
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor="#667eea"
        )
    
    fig.update_layout(
        title="Smart Retail Analytics Workflow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Dataset page
def show_dataset():
    st.markdown('<div class="header-title">📊 Dataset Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🎲 Data Generation</div>', unsafe_allow_html=True)
        
        num_records = st.number_input(
            "Number of records to generate:",
            min_value=100,
            max_value=10000,
            value=1500,
            step=100
        )
        
        if st.button("🚀 Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic retail data..."):
                generator = RetailDataGenerator(num_records=num_records)
                df = generator.generate_dataset()
                st.session_state.data = df
                st.success(f"✅ Generated {len(df)} records successfully!")
        
        st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your CSV file:",
            type=['csv'],
            help="Upload a CSV file with retail customer data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ Uploaded {len(df)} records successfully!")
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")
    
    with col2:
        st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show sample data
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="retail_data.csv">📥 Download Dataset</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("📝 No dataset loaded. Generate synthetic data or upload a CSV file.")
    
    # Data summary section
    if st.session_state.data is not None:
        st.markdown('<div class="section-header">📈 Data Summary Statistics</div>', unsafe_allow_html=True)
        
        df = st.session_state.data
        utils = st.session_state.utils
        
        # Generate summary table
        summary_table = utils.generate_data_summary_table(df)
        st.dataframe(summary_table, use_container_width=True)
        
        # Data quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        with col3:
            st.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
        
        with col4:
            st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))

# Predictions page
def show_predictions():
    st.markdown('<div class="header-title">🤖 ML Predictions</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the Dataset page.")
        return
    
    df = st.session_state.data
    models = st.session_state.models
    
    # Prediction type selection
    prediction_type = st.selectbox(
        "Select Prediction Type:",
        ["💰 Sales Prediction", "🎯 Purchase Decision", "⭐ Loyalty Category"]
    )
    
    if prediction_type == "💰 Sales Prediction":
        st.markdown('<div class="section-header">💰 Monthly Sales Prediction</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Customer Data")
            
            # Create input form
            with st.form("sales_prediction_form"):
                age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                income = st.number_input("Annual Income (₹)", min_value=25000, max_value=200000, value=75000)
                tenure = st.number_input("Customer Tenure (months)", min_value=1, max_value=120, value=36)
                purchase_freq = st.number_input("Purchase Frequency", min_value=1, max_value=15, value=5)
                prev_purchase = st.number_input("Previous Purchase Amount (₹)", min_value=50, max_value=5000, value=1000)
                category = st.selectbox("Product Category", 
                    ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys", "Beauty", "Food", "Health", "Automotive"])
                location = st.selectbox("Store Location", ["Downtown", "Suburban", "Mall", "Airport", "Online"])
                marketing_spend = st.number_input("Marketing Spend (₹)", min_value=10, max_value=500, value=100)
                discount = st.number_input("Discount Offered (%)", min_value=0, max_value=40, value=10)
                seasonal_demand = st.number_input("Seasonal Demand Index", min_value=0, max_value=100, value=50)
                
                submitted = st.form_submit_button("🔮 Predict Sales", type="primary")
            
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame([{
                    'Customer_Age': age,
                    'Gender': gender,
                    'Annual_Income': income,
                    'Customer_Tenure_Months': tenure,
                    'Purchase_Frequency': purchase_freq,
                    'Previous_Purchase_Amount': prev_purchase,
                    'Product_Category': category,
                    'Store_Location': location,
                    'Marketing_Spend': marketing_spend,
                    'Discount_Offered': discount,
                    'Seasonal_Demand_Index': seasonal_demand
                }])
                
                # Train model if not already trained
                if 'linear_regression' not in models.models:
                    with st.spinner("Training Linear Regression model..."):
                        model, metrics = models.train_linear_regression(df)
                
                # Make prediction
                try:
                    prediction = models.predict_sales(input_data)[0]
                    st.success(f"🎯 Predicted Monthly Sales: ₹{prediction:,.2f}")
                    
                    # Feature importance
                    importance = models.get_feature_importance('decision_tree')
                    if importance:
                        fig = st.session_state.utils.create_feature_importance_plot(importance)
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        with col2:
            st.markdown("#### Model Performance")
            
            if 'linear_regression' in models.performance_metrics:
                metrics = models.performance_metrics['linear_regression']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{metrics['R2_Score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            
            # Model training button
            if st.button("🎯 Train/Retrain Model"):
                with st.spinner("Training Linear Regression model..."):
                    model, metrics = models.train_linear_regression(df)
                    st.success("✅ Model trained successfully!")
                    st.json(metrics)
    
    elif prediction_type == "🎯 Purchase Decision":
        st.markdown('<div class="section-header">🎯 Purchase Decision Classification</div>', unsafe_allow_html=True)
        
        # Similar implementation for purchase decision prediction
        st.info("Purchase Decision prediction interface - similar to sales prediction")
        
        # Train model button
        if st.button("🎯 Train Decision Tree Model"):
            with st.spinner("Training Decision Tree model..."):
                model, metrics = models.train_decision_tree(df)
                st.success("✅ Model trained successfully!")
                
                if 'confusion_matrix' in metrics:
                    fig = st.session_state.utils.create_confusion_matrix_plot(
                        metrics['confusion_matrix'], ['No', 'Yes']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:  # Loyalty Category
        st.markdown('<div class="section-header">⭐ Loyalty Category Prediction</div>', unsafe_allow_html=True)
        
        # Similar implementation for loyalty category prediction
        st.info("Loyalty Category prediction interface - similar to sales prediction")
        
        # Train model button
        if st.button("🎯 Train KNN Model"):
            with st.spinner("Training KNN model..."):
                model, metrics = models.train_knn(df)
                st.success("✅ Model trained successfully!")
                st.json(metrics)

# Customer Segmentation page
def show_segmentation():
    st.markdown('<div class="header-title">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the Dataset page.")
        return
    
    df = st.session_state.data
    models = st.session_state.models
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🎯 K-Means Clustering</div>', unsafe_allow_html=True)
        
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
        
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Running K-Means clustering..."):
                model, cluster_info = models.train_kmeans(df, n_clusters)
                st.session_state.cluster_info = cluster_info
                
                st.success(f"✅ Clustering completed with {n_clusters} clusters!")
        
        if st.session_state.cluster_info:
            cluster_info = st.session_state.cluster_info
            
            # Elbow method plot
            fig = st.session_state.utils.create_elbow_method_plot(
                cluster_info['inertias'], cluster_info['k_range']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D cluster visualization
            fig_2d = st.session_state.utils.create_2d_cluster_plot(
                cluster_info['scaled_data'], cluster_info['cluster_labels']
            )
            st.plotly_chart(fig_2d, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Cluster Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.cluster_info:
            df_with_clusters = st.session_state.cluster_info['df_with_clusters']
            
            # Cluster distribution
            fig = st.session_state.utils.create_cluster_distribution_plot(df_with_clusters)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            cluster_stats = df_with_clusters.groupby('Cluster').agg({
                'Annual_Income': 'mean',
                'Monthly_Sales': 'mean',
                'Purchase_Frequency': 'mean',
                'Customer_Age': 'mean'
            }).round(2)
            
            st.write("#### Cluster Statistics")
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Download clustered data
            csv = df_with_clusters.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="clustered_customers.csv">📥 Download Clustered Data</a>'
            st.markdown(href, unsafe_allow_html=True)

# Visualizations page
def show_visualizations():
    st.markdown('<div class="header-title">📈 Visualizations & Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the Dataset page.")
        return
    
    df = st.session_state.data
    utils = st.session_state.utils
    
    # Visualization selection
    viz_type = st.selectbox(
        "Select Visualization:",
        ["💰 Sales vs Marketing Spend", "💳 Income vs Purchase Frequency", "🎁 Discount vs Sales", "📊 Model Performance", "📋 Data Overview"]
    )
    
    if viz_type == "💰 Sales vs Marketing Spend":
        st.markdown('<div class="section-header">💰 Sales vs Marketing Spend Analysis</div>', unsafe_allow_html=True)
        fig = utils.create_sales_vs_marketing_plot(df)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "💳 Income vs Purchase Frequency":
        st.markdown('<div class="section-header">💳 Income vs Purchase Frequency</div>', unsafe_allow_html=True)
        fig = utils.create_income_vs_frequency_plot(df)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🎁 Discount vs Sales":
        st.markdown('<div class="section-header">🎁 Discount Impact on Sales</div>', unsafe_allow_html=True)
        fig = utils.create_discount_vs_sales_plot(df)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "📊 Model Performance":
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state.models, 'performance_metrics') and st.session_state.models.performance_metrics:
            fig = utils.create_model_performance_comparison(st.session_state.models.performance_metrics)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 Train models first to see performance comparison.")
    
    else:  # Data Overview
        st.markdown('<div class="section-header">📋 Data Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sales = df['Monthly_Sales'].mean()
            st.metric("Avg Monthly Sales", f"₹{avg_sales:,.2f}")
        
        with col2:
            avg_income = df['Annual_Income'].mean()
            st.metric("Avg Annual Income", f"₹{avg_income:,.2f}")
        
        with col3:
            avg_purchase_freq = df['Purchase_Frequency'].mean()
            st.metric("Avg Purchase Frequency", f"{avg_purchase_freq:.1f}")
        
        with col4:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        # Data distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Monthly_Sales', title='Monthly Sales Distribution')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Annual_Income', title='Annual Income Distribution')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

# AI Recommendations page
def show_recommendations():
    st.markdown('<div class="header-title">💡 AI-Powered Recommendations</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the Dataset page.")
        return
    
    df = st.session_state.data
    engine = st.session_state.recommendation_engine
    
    # Recommendation type selection
    rec_type = st.selectbox(
        "Select Recommendation Type:",
        ["👤 Individual Customer", "👥 Customer Segment", "📊 Cluster Analysis"]
    )
    
    if rec_type == "👤 Individual Customer":
        st.markdown('<div class="section-header">👤 Individual Customer Recommendations</div>', unsafe_allow_html=True)
        
        # Customer selection
        customer_ids = range(min(5, len(df)))  # Show first 5 customers for demo
        selected_customer = st.selectbox("Select Customer:", [f"Customer {i+1}" for i in customer_ids])
        
        if selected_customer:
            customer_idx = int(selected_customer.split()[1]) - 1
            customer_data = df.iloc[customer_idx]
            
            # Generate recommendations
            recommendations = engine.generate_recommendations(customer_data)
            
            # Display customer info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Income", f"₹{customer_data['Annual_Income']:,.2f}")
            with col2:
                st.metric("Purchase Freq", customer_data['Purchase_Frequency'])
            with col3:
                st.metric("Loyalty", customer_data['Loyalty_Category'])
            
            # Display recommendations
            st.markdown(f"#### 🎯 Customer Segment: {recommendations['customer_segment'].replace('_', ' ').title()}")
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                priority_class = rec['priority'].lower() + '-priority'
                st.markdown(f"""
                <div class="recommendation-card {priority_class}">
                    <h4>{i}. {rec['action']} (Priority: {rec['priority']})</h4>
                    <p><strong>Description:</strong> {rec['description']}</p>
                    <p><strong>Expected Impact:</strong> {rec['expected_impact']}</p>
                    <p><strong>Implementation Cost:</strong> {rec['implementation_cost']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Business insights
            st.markdown("#### 💡 Business Insights")
            for insight in recommendations['business_insights']:
                st.markdown(f"- {insight}")
            
            # ROI projection
            roi = recommendations['roi_projection']
            st.markdown("#### 📈 ROI Projection")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Implementation Cost", f"₹{roi['total_implementation_cost']:,.2f}")
            with col2:
                st.metric("Annual Revenue Increase", f"₹{roi['projected_annual_revenue_increase']:,.2f}")
            with col3:
                st.metric("ROI", f"{roi['roi_percentage']:.1f}%")
    
    elif rec_type == "👥 Customer Segment":
        st.markdown('<div class="section-header">👥 Customer Segment Analysis</div>', unsafe_allow_html=True)
        
        # Segment analysis
        segments = ['Budget Customers', 'Premium Customers', 'Seasonal Buyers', 'High-Value Customers', 'At-Risk Customers']
        selected_segment = st.selectbox("Select Segment:", segments)
        
        if selected_segment:
            # Create sample customer for segment
            segment_key = selected_segment.lower().replace(' ', '_').replace('-', '_')
            sample_customer = {
                'Annual_Income': 75000,
                'Purchase_Frequency': 5,
                'Monthly_Sales': 3000,
                'Loyalty_Category': 'Silver',
                'Customer_Age': 35
            }
            
            recommendations = engine.generate_recommendations(sample_customer)
            
            st.markdown(f"#### 🎯 {selected_segment}")
            
            for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{i}. {rec['action']}</h4>
                    <p>{rec['description']}</p>
                    <p><strong>Impact:</strong> {rec['expected_impact']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Cluster Analysis
        st.markdown('<div class="section-header">📊 Cluster-Based Recommendations</div>', unsafe_allow_html=True)
        
        if st.session_state.cluster_info:
            df_with_clusters = st.session_state.cluster_info['df_with_clusters']
            
            # Generate segment summary
            segment_summary = engine.generate_segment_summary(df_with_clusters)
            
            for cluster_name, cluster_data in segment_summary.items():
                with st.expander(f"📊 {cluster_name} - {cluster_data['cluster_size']} customers"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Income", f"₹{cluster_data['average_income']:,.2f}")
                    with col2:
                        st.metric("Avg Sales", f"₹{cluster_data['average_monthly_sales']:,.2f}")
                    with col3:
                        st.metric("Segment", cluster_data['customer_segment'].replace('_', ' ').title())
                    
                    st.markdown("**Top Recommendations:**")
                    for i, rec in enumerate(cluster_data['top_recommendations'], 1):
                        st.markdown(f"{i}. {rec['action']} - {rec['description']}")
        else:
            st.info("📝 Run clustering first to see cluster-based recommendations.")

# Helper functions
def generate_new_data():
    generator = RetailDataGenerator(num_records=1500)
    df = generator.generate_dataset()
    st.session_state.data = df
    st.success(f"✅ Generated {len(df)} records successfully!")

def load_sample_data():
    # Generate sample data for demo
    generator = RetailDataGenerator(num_records=1000)
    df = generator.generate_dataset()
    st.session_state.data = df
    st.success(f"✅ Loaded {len(df)} sample records!")

def save_session():
    if st.session_state.data is not None:
        filename = st.session_state.utils.export_to_csv(st.session_state.data)
        st.success(f"✅ Session saved to {filename}")
    else:
        st.warning("⚠️ No data to save")

# Main application
def main():
    set_custom_css()
    initialize_session_state()
    
    page = create_sidebar()
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Dataset":
        show_dataset()
    elif page == "🤖 Predictions":
        show_predictions()
    elif page == "👥 Customer Segmentation":
        show_segmentation()
    elif page == "📈 Visualizations & Insights":
        show_visualizations()
    elif page == "💡 AI Recommendations":
        show_recommendations()

if __name__ == "__main__":
    main()
