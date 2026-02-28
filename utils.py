import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetailAnalyticsUtils:
    """
    Utility functions for retail analytics, data visualization, and reporting.
    Provides helper functions for the Streamlit application.
    """
    
    @staticmethod
    def create_animated_metric_card(title, value, subtitle=None, delta=None, color="blue"):
        """
        Create HTML for animated metric card.
        
        Args:
            title: Card title
            value: Main value to display
            subtitle: Optional subtitle
            delta: Optional change value
            color: Theme color
        
        Returns:
            HTML string for the metric card
        """
        delta_html = ""
        if delta is not None:
            delta_color = "green" if delta > 0 else "red"
            delta_sign = "+" if delta > 0 else ""
            delta_html = f'<div style="color: {delta_color}; font-size: 14px; margin-top: 5px;">{delta_sign}{delta}% vs last period</div>'
        
        subtitle_html = f'<div style="color: #666; font-size: 12px; margin-top: 5px;">{subtitle}</div>' if subtitle else ""
        
        html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            margin: 10px 0;
        ">
            <div style="font-size: 14px; opacity: 0.9;">{title}</div>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">{value}</div>
            {subtitle_html}
            {delta_html}
        </div>
        """
        return html
    
    @staticmethod
    def create_sales_vs_marketing_plot(df):
        """
        Create interactive sales vs marketing spend plot.
        
        Args:
            df: Input dataframe
        
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            df, 
            x='Marketing_Spend', 
            y='Monthly_Sales',
            color='Loyalty_Category',
            size='Purchase_Frequency',
            hover_data=['Customer_Age', 'Annual_Income'],
            title='Sales vs Marketing Spend Analysis',
            labels={
                'Marketing_Spend': 'Marketing Spend (₹)',
                'Monthly_Sales': 'Monthly Sales (₹)',
                'Loyalty_Category': 'Loyalty Category'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True
        )

        return fig
    
    @staticmethod
    def create_income_vs_frequency_plot(df):
        """
        Create income vs purchase frequency scatter plot.
        
        Args:
            df: Input dataframe
        
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            df,
            x='Annual_Income',
            y='Purchase_Frequency',
            color='Gender',
            size='Monthly_Sales',
            facet_col='Store_Location',
            hover_data=['Customer_Age', 'Loyalty_Category'],
            title='Income vs Purchase Frequency by Store Location',
            labels={
                'Annual_Income': 'Annual Income (₹)',
                'Purchase_Frequency': 'Purchase Frequency (per month)',
                'Gender': 'Gender',
                'Store_Location': 'Store Location'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_discount_vs_sales_plot(df):
        """
        Create discount vs sales relationship plot.
        
        Args:
            df: Input dataframe
        
        Returns:
            Plotly figure
        """
        # Create bins for discount analysis
        df['Discount_Bins'] = pd.cut(df['Discount_Offered'], 
                                     bins=[0, 10, 20, 30, 40], 
                                     labels=['0-10%', '11-20%', '21-30%', '31-40%'])
        
        fig = px.box(
            df,
            x='Discount_Bins',
            y='Monthly_Sales',
            color='Purchase_Decision',
            title='Impact of Discount on Sales by Purchase Decision',
            labels={
                'Discount_Bins': 'Discount Range',
                'Monthly_Sales': 'Monthly Sales (₹)',
                'Purchase_Decision': 'Purchase Decision'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_cluster_distribution_plot(df_with_clusters):
        """
        Create cluster distribution pie chart.
        
        Args:
            df_with_clusters: DataFrame with cluster labels
        
        Returns:
            Plotly figure
        """
        cluster_counts = df_with_clusters['Cluster'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title='Customer Cluster Distribution',
            template='plotly_dark',
            height=400,
            showlegend=True,
            annotations=[dict(text='Clusters', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    @staticmethod
    def create_elbow_method_plot(inertias, k_range):
        """
        Create elbow method visualization for optimal K selection.
        
        Args:
            inertias: List of inertia values
            k_range: List of K values
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='cyan', width=3),
            marker=dict(size=8)
        ))
        
        # Find elbow point (simplified)
        if len(inertias) > 2:
            elbow_point = 2  # Default to K=2 for simplicity
            fig.add_trace(go.Scatter(
                x=[k_range[elbow_point]],
                y=[inertias[elbow_point]],
                mode='markers',
                name='Optimal K',
                marker=dict(color='red', size=15, symbol='star')
            ))
        
        fig.update_layout(
            title='Elbow Method for Optimal Cluster Selection',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_2d_cluster_plot(scaled_data, cluster_labels):
        """
        Create 2D visualization of clusters using PCA.
        
        Args:
            scaled_data: Scaled feature data
            cluster_labels: Cluster labels
        
        Returns:
            Plotly figure
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(scaled_data)
        
        df_plot = pd.DataFrame({
            'PC1': data_2d[:, 0],
            'PC2': data_2d[:, 1],
            'Cluster': cluster_labels
        })
        
        fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster',
            title='Customer Segments (2D PCA Visualization)',
            labels={
                'PC1': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                'PC2': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_model_performance_comparison(performance_metrics):
        """
        Create model performance comparison chart.
        
        Args:
            performance_metrics: Dictionary of model performance metrics
        
        Returns:
            Plotly figure
        """
        models = []
        accuracies = []
        r2_scores = []
        
        for model_name, metrics in performance_metrics.items():
            models.append(model_name.replace('_', ' ').title())
            if 'Accuracy' in metrics:
                accuracies.append(metrics['Accuracy'] * 100)
            else:
                accuracies.append(0)
            
            if 'R2_Score' in metrics:
                r2_scores.append(metrics['R2_Score'] * 100)
            else:
                r2_scores.append(0)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy (%)', 'Model R² Score (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Performance Comparison',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_plot(feature_importance):
        """
        Create feature importance visualization.
        
        Args:
            feature_importance: Dictionary of feature importance scores
        
        Returns:
            Plotly figure
        """
        if not feature_importance:
            return go.Figure()
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker=dict(color='rgba(55, 128, 191, 0.7)')
            )
        ])
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_plot(conf_matrix, class_names=None):
        """
        Create confusion matrix heatmap.
        
        Args:
            conf_matrix: Confusion matrix array
            class_names: List of class names
        
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def generate_data_summary_table(df):
        """
        Generate comprehensive data summary table.
        
        Args:
            df: Input dataframe
        
        Returns:
            Formatted summary dataframe
        """
        summary_data = []
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                summary_data.append({
                    'Feature': column,
                    'Type': 'Numerical',
                    'Count': df[column].count(),
                    'Mean': round(df[column].mean(), 2),
                    'Std': round(df[column].std(), 2),
                    'Min': df[column].min(),
                    'Max': df[column].max(),
                    'Missing': df[column].isnull().sum()
                })
            else:
                summary_data.append({
                    'Feature': column,
                    'Type': 'Categorical',
                    'Count': df[column].count(),
                    'Unique': df[column].nunique(),
                    'Most Common': df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A',
                    'Missing': df[column].isnull().sum()
                })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def export_to_csv(df, filename=None):
        """
        Export dataframe to CSV with timestamp.
        
        Args:
            df: Dataframe to export
            filename: Optional filename
        
        Returns:
            Filename used
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retail_data_export_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def create_download_link(data, filename, link_text):
        """
        Create HTML download link for data.
        
        Args:
            data: Data to download (DataFrame or dict)
            filename: Filename for download
            link_text: Text for the link
        
        Returns:
            HTML string for download link
        """
        if isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
        else:
            import json
            json_str = json.dumps(data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    
    @staticmethod
    def format_currency(value):
        """Format value as currency."""
        return f"₹{value:,.2f}"
    
    @staticmethod
    def format_percentage(value):
        """Format value as percentage."""
        return f"{value:.1f}%"
    
    @staticmethod
    def get_seasonal_insights(df):
        """
        Generate seasonal insights from data.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dictionary of seasonal insights
        """
        insights = {}
        
        # Seasonal demand analysis
        high_demand = df[df['Seasonal_Demand_Index'] > 70]
        low_demand = df[df['Seasonal_Demand_Index'] < 30]
        
        insights['high_demand_customers'] = len(high_demand)
        insights['low_demand_customers'] = len(low_demand)
        insights['avg_sales_high_demand'] = high_demand['Monthly_Sales'].mean() if len(high_demand) > 0 else 0
        insights['avg_sales_low_demand'] = low_demand['Monthly_Sales'].mean() if len(low_demand) > 0 else 0
        
        # Discount effectiveness
        discount_effectiveness = df.groupby('Discount_Offered')['Monthly_Sales'].mean().to_dict()
        insights['discount_effectiveness'] = discount_effectiveness
        
        return insights

if __name__ == "__main__":
    # Example usage
    from data_generator import RetailDataGenerator
    
    # Generate sample data
    generator = RetailDataGenerator(num_records=100)
    df = generator.generate_dataset()
    
    # Test utility functions
    utils = RetailAnalyticsUtils()
    
    # Create sample plots
    sales_plot = utils.create_sales_vs_marketing_plot(df)
    income_plot = utils.create_income_vs_frequency_plot(df)
    
    print("Utility functions loaded successfully!")
    print(f"Sample data shape: {df.shape}")
    print(f"Data summary created with {len(utils.generate_data_summary_table(df))} features")
