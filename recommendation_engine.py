import pandas as pd
import numpy as np
from datetime import datetime
import json

class RetailRecommendationEngine:
    """
    AI-powered recommendation engine for retail business decisions.
    Generates intelligent business suggestions based on customer segments,
    purchase patterns, and predictive analytics.
    """
    
    def __init__(self):
        self.recommendation_rules = self._initialize_recommendation_rules()
        self.business_impact_scores = self._initialize_impact_scores()
    
    def _initialize_recommendation_rules(self):
        """Initialize rule-based recommendation logic."""
        return {
            'budget_customers': {
                'characteristics': {
                    'income_threshold': 50000,
                    'purchase_frequency_low': True,
                    'loyalty_bronze': True
                },
                'recommendations': [
                    {
                        'action': 'Targeted Discount Campaigns',
                        'description': 'Launch personalized discount offers (15-25%) during weekends',
                        'priority': 'High',
                        'expected_impact': 'Increase purchase frequency by 30%',
                        'implementation_cost': 'Low'
                    },
                    {
                        'action': 'Bundle Deals',
                        'description': 'Create product bundles to increase average order value',
                        'priority': 'Medium',
                        'expected_impact': 'Increase basket size by 20%',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Loyalty Entry Program',
                        'description': 'Offer easy-to-achieve loyalty milestones with small rewards',
                        'priority': 'High',
                        'expected_impact': 'Improve customer retention by 25%',
                        'implementation_cost': 'Low'
                    }
                ]
            },
            'premium_customers': {
                'characteristics': {
                    'income_threshold': 100000,
                    'purchase_frequency_high': True,
                    'loyalty_gold': True
                },
                'recommendations': [
                    {
                        'action': 'Exclusive Membership Offers',
                        'description': 'Provide VIP access to new products and exclusive events',
                        'priority': 'High',
                        'expected_impact': 'Increase customer lifetime value by 40%',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Personalized Shopping Experience',
                        'description': 'Assign dedicated personal shoppers and concierge services',
                        'priority': 'Medium',
                        'expected_impact': 'Increase satisfaction score by 35%',
                        'implementation_cost': 'High'
                    },
                    {
                        'action': 'Early Access Privileges',
                        'description': 'Grant early access to sales and new product launches',
                        'priority': 'High',
                        'expected_impact': 'Increase repeat purchases by 25%',
                        'implementation_cost': 'Low'
                    }
                ]
            },
            'seasonal_buyers': {
                'characteristics': {
                    'seasonal_demand_high': True,
                    'purchase_frequency_medium': True,
                    'discount_sensitive': True
                },
                'recommendations': [
                    {
                        'action': 'Seasonal Marketing Push',
                        'description': 'Launch targeted campaigns aligned with seasonal demand peaks',
                        'priority': 'High',
                        'expected_impact': 'Increase seasonal sales by 45%',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Inventory Optimization',
                        'description': 'Adjust inventory levels based on seasonal demand forecasts',
                        'priority': 'High',
                        'expected_impact': 'Reduce stockouts by 60%',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Cross-Seasonal Promotions',
                        'description': 'Create promotions that bridge seasonal gaps',
                        'priority': 'Medium',
                        'expected_impact': 'Smooth revenue throughout the year',
                        'implementation_cost': 'Low'
                    }
                ]
            },
            'high_value_customers': {
                'characteristics': {
                    'high_monthly_sales': True,
                    'high_purchase_amount': True,
                    'loyalty_silver_gold': True
                },
                'recommendations': [
                    {
                        'action': 'Premium Loyalty Rewards',
                        'description': 'Offer tiered rewards with exclusive benefits',
                        'priority': 'High',
                        'expected_impact': 'Increase retention rate by 50%',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Referral Programs',
                        'description': 'Implement customer referral programs with generous incentives',
                        'priority': 'Medium',
                        'expected_impact': 'Acquire new high-value customers',
                        'implementation_cost': 'Low'
                    },
                    {
                        'action': 'Customized Product Recommendations',
                        'description': 'Use AI to provide personalized product suggestions',
                        'priority': 'High',
                        'expected_impact': 'Increase conversion rate by 30%',
                        'implementation_cost': 'High'
                    }
                ]
            },
            'at_risk_customers': {
                'characteristics': {
                    'declining_purchase_frequency': True,
                    'low_recent_spend': True,
                    'loyalty_bronze': True
                },
                'recommendations': [
                    {
                        'action': 'Re-engagement Campaign',
                        'description': 'Launch targeted win-back campaigns with special offers',
                        'priority': 'High',
                        'expected_impact': 'Recover 40% of at-risk customers',
                        'implementation_cost': 'Medium'
                    },
                    {
                        'action': 'Customer Feedback Initiative',
                        'description': 'Conduct surveys to understand pain points and preferences',
                        'priority': 'Medium',
                        'expected_impact': 'Improve service quality and satisfaction',
                        'implementation_cost': 'Low'
                    },
                    {
                        'action': 'Flexible Payment Options',
                        'description': 'Offer installment plans and flexible payment methods',
                        'priority': 'Medium',
                        'expected_impact': 'Reduce purchase barriers',
                        'implementation_cost': 'Low'
                    }
                ]
            }
        }
    
    def _initialize_impact_scores(self):
        """Initialize business impact scoring system."""
        return {
            'revenue_impact': {
                'High': 0.8,
                'Medium': 0.5,
                'Low': 0.2
            },
            'implementation_cost': {
                'High': 0.2,
                'Medium': 0.5,
                'Low': 0.8
            },
            'time_to_implement': {
                'Immediate': 1.0,
                'Short_term': 0.7,
                'Medium_term': 0.4,
                'Long_term': 0.1
            }
        }
    
    def classify_customer_segment(self, customer_data):
        """
        Classify customer into segments based on their characteristics.
        
        Args:
            customer_data: Dictionary or DataFrame row with customer features
        
        Returns:
            String representing customer segment
        """
        if isinstance(customer_data, pd.Series):
            customer = customer_data.to_dict()
        else:
            customer = customer_data
        
        # Extract key features
        income = customer.get('Annual_Income', 0)
        purchase_freq = customer.get('Purchase_Frequency', 0)
        loyalty = customer.get('Loyalty_Category', 'Bronze')
        monthly_sales = customer.get('Monthly_Sales', 0)
        prev_purchase = customer.get('Previous_Purchase_Amount', 0)
        seasonal_demand = customer.get('Seasonal_Demand_Index', 50)
        
        # Classification logic
        if income >= 100000 and purchase_freq >= 8 and loyalty == 'Gold':
            return 'premium_customers'
        elif income < 50000 and purchase_freq <= 3 and loyalty == 'Bronze':
            return 'budget_customers'
        elif monthly_sales >= 5000 and prev_purchase >= 2000 and loyalty in ['Silver', 'Gold']:
            return 'high_value_customers'
        elif seasonal_demand >= 70:
            return 'seasonal_buyers'
        elif purchase_freq <= 2 and loyalty == 'Bronze':
            return 'at_risk_customers'
        else:
            return 'standard_customers'
    
    def generate_recommendations(self, customer_data, cluster_info=None):
        """
        Generate personalized recommendations for a customer or segment.
        
        Args:
            customer_data: Customer data dictionary or DataFrame
            cluster_info: Optional cluster information from K-Means
        
        Returns:
            Dictionary with recommendations and business insights
        """
        # Classify customer segment
        segment = self.classify_customer_segment(customer_data)
        
        # Get base recommendations
        if segment in self.recommendation_rules:
            base_recommendations = self.recommendation_rules[segment]['recommendations']
        else:
            # Default recommendations for standard customers
            base_recommendations = [
                {
                    'action': 'General Engagement',
                    'description': 'Focus on improving customer experience and satisfaction',
                    'priority': 'Medium',
                    'expected_impact': 'Increase overall satisfaction by 15%',
                    'implementation_cost': 'Medium'
                }
            ]
        
        # Enhance recommendations with cluster information if available
        if cluster_info is not None:
            enhanced_recommendations = self._enhance_with_cluster_data(
                base_recommendations, cluster_info, customer_data
            )
        else:
            enhanced_recommendations = base_recommendations
        
        # Calculate recommendation scores
        scored_recommendations = self._score_recommendations(enhanced_recommendations)
        
        # Sort by priority and score
        scored_recommendations.sort(key=lambda x: (
            0 if x['priority'] == 'High' else 1 if x['priority'] == 'Medium' else 2,
            -x['recommendation_score']
        ))
        
        return {
            'customer_segment': segment,
            'recommendations': scored_recommendations[:5],  # Top 5 recommendations
            'business_insights': self._generate_business_insights(customer_data, segment),
            'roi_projection': self._calculate_roi_projection(scored_recommendations[:3])
        }
    
    def _enhance_with_cluster_data(self, base_recommendations, cluster_info, customer_data):
        """Enhance recommendations with cluster-specific insights."""
        enhanced = base_recommendations.copy()
        
        # Add cluster-specific recommendations
        if cluster_info.get('cluster_size', 0) > 100:  # Large cluster
            enhanced.append({
                'action': 'Scale Cluster Strategy',
                'description': f"Implement strategies for large customer group ({cluster_info.get('cluster_size')} customers)",
                'priority': 'High',
                'expected_impact': 'Economies of scale in marketing efforts',
                'implementation_cost': 'Low'
            })
        
        # Add demographic-specific recommendations
        age = customer_data.get('Customer_Age', 35)
        if age < 30:
            enhanced.append({
                'action': 'Digital-First Approach',
                'description': 'Focus on mobile app and social media engagement',
                'priority': 'Medium',
                'expected_impact': 'Increase engagement among younger customers',
                'implementation_cost': 'Medium'
            })
        elif age > 50:
            enhanced.append({
                'action': 'Traditional Marketing Integration',
                'description': 'Combine digital with traditional marketing channels',
                'priority': 'Medium',
                'expected_impact': 'Improve reach across age groups',
                'implementation_cost': 'Medium'
            })
        
        return enhanced
    
    def _score_recommendations(self, recommendations):
        """Score recommendations based on impact, cost, and implementation time."""
        for rec in recommendations:
            impact_score = self.business_impact_scores['revenue_impact'].get(
                self._extract_impact_level(rec['expected_impact']), 0.5
            )
            cost_score = self.business_impact_scores['implementation_cost'].get(
                rec['implementation_cost'], 0.5
            )
            
            # Calculate composite score (higher is better)
            rec['recommendation_score'] = (impact_score * 0.6) + (cost_score * 0.4)
        
        return recommendations
    
    def _extract_impact_level(self, impact_text):
        """Extract impact level from impact description text."""
        impact_text = impact_text.lower()
        if any(word in impact_text for word in ['increase by 40%', 'increase by 45%', 'increase by 50%']):
            return 'High'
        elif any(word in impact_text for word in ['increase by 20%', 'increase by 25%', 'increase by 30%']):
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_business_insights(self, customer_data, segment):
        """Generate business insights based on customer analysis."""
        insights = []
        
        income = customer_data.get('Annual_Income', 0)
        purchase_freq = customer_data.get('Purchase_Frequency', 0)
        monthly_sales = customer_data.get('Monthly_Sales', 0)
        
        # Income-based insights
        if income > 100000:
            insights.append("High-income customer with premium purchasing power")
        elif income < 40000:
            insights.append("Price-sensitive customer requiring value-focused strategies")
        
        # Frequency-based insights
        if purchase_freq > 8:
            insights.append("Highly engaged customer suitable for loyalty programs")
        elif purchase_freq < 3:
            insights.append("Low engagement requiring re-engagement strategies")
        
        # Sales-based insights
        if monthly_sales > 5000:
            insights.append("High-value customer contributing significantly to revenue")
        
        # Segment-specific insights
        if segment == 'premium_customers':
            insights.append("Ideal candidate for exclusive VIP programs")
        elif segment == 'budget_customers':
            insights.append("Responds well to discount and promotional strategies")
        
        return insights
    
    def _calculate_roi_projection(self, top_recommendations):
        """Calculate projected ROI for top recommendations."""
        total_cost = 0
        projected_revenue_increase = 0
        
        cost_mapping = {'Low': 1000, 'Medium': 5000, 'High': 15000}
        
        for rec in top_recommendations:
            cost = cost_mapping.get(rec['implementation_cost'], 5000)
            total_cost += cost
            
            # Extract percentage increase from expected impact
            impact_text = rec['expected_impact'].lower()
            if '%' in impact_text:
                try:
                    percentage = int(''.join(filter(str.isdigit, impact_text.split('%')[0])))
                    # Assume average monthly revenue of $10,000 for calculation
                    projected_revenue_increase += (10000 * percentage / 100) * 12  # Annual projection
                except:
                    projected_revenue_increase += 5000  # Default increase
        
        roi = ((projected_revenue_increase - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_implementation_cost': total_cost,
            'projected_annual_revenue_increase': projected_revenue_increase,
            'roi_percentage': round(roi, 2),
            'payback_period_months': round(total_cost / (projected_revenue_increase / 12), 1) if projected_revenue_increase > 0 else 'N/A'
        }
    
    def generate_segment_summary(self, df_with_clusters):
        """
        Generate summary for all customer segments.
        
        Args:
            df_with_clusters: DataFrame with cluster labels
        
        Returns:
            Dictionary with segment summaries and recommendations
        """
        segment_summary = {}
        
        for cluster_id in df_with_clusters['Cluster'].unique():
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            # Calculate cluster statistics
            avg_income = cluster_data['Annual_Income'].mean()
            avg_purchase_freq = cluster_data['Purchase_Frequency'].mean()
            avg_monthly_sales = cluster_data['Monthly_Sales'].mean()
            cluster_size = len(cluster_data)
            
            # Get dominant loyalty category
            dominant_loyalty = cluster_data['Loyalty_Category'].mode().iloc[0]
            
            # Create representative customer profile
            representative_customer = {
                'Annual_Income': avg_income,
                'Purchase_Frequency': avg_purchase_freq,
                'Monthly_Sales': avg_monthly_sales,
                'Loyalty_Category': dominant_loyalty,
                'Customer_Age': cluster_data['Customer_Age'].mean()
            }
            
            # Generate recommendations for this segment
            recommendations = self.generate_recommendations(representative_customer)
            
            segment_summary[f'Cluster_{cluster_id}'] = {
                'cluster_size': cluster_size,
                'average_income': round(avg_income, 2),
                'average_purchase_frequency': round(avg_purchase_freq, 2),
                'average_monthly_sales': round(avg_monthly_sales, 2),
                'dominant_loyalty': dominant_loyalty,
                'customer_segment': recommendations['customer_segment'],
                'top_recommendations': recommendations['recommendations'][:3],
                'business_insights': recommendations['business_insights']
            }
        
        return segment_summary
    
    def export_recommendations(self, recommendations, filename='recommendations.json'):
        """Export recommendations to JSON file."""
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        print(f"Recommendations exported to {filename}")

if __name__ == "__main__":
    # Example usage
    engine = RetailRecommendationEngine()
    
    # Sample customer data
    sample_customer = {
        'Annual_Income': 75000,
        'Purchase_Frequency': 6,
        'Monthly_Sales': 3000,
        'Loyalty_Category': 'Silver',
        'Customer_Age': 35,
        'Seasonal_Demand_Index': 60
    }
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(sample_customer)
    
    print("=== AI-Powered Retail Recommendations ===")
    print(f"Customer Segment: {recommendations['customer_segment']}")
    print("\nTop Recommendations:")
    for i, rec in enumerate(recommendations['recommendations'][:3], 1):
        print(f"\n{i}. {rec['action']} (Priority: {rec['priority']})")
        print(f"   Description: {rec['description']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print(f"   Implementation Cost: {rec['implementation_cost']}")
    
    print(f"\nBusiness Insights:")
    for insight in recommendations['business_insights']:
        print(f"- {insight}")
    
    print(f"\nROI Projection:")
    roi = recommendations['roi_projection']
    print(f"- Implementation Cost: ${roi['total_implementation_cost']:,.2f}")
    print(f"- Projected Annual Revenue Increase: ${roi['projected_annual_revenue_increase']:,.2f}")
    print(f"- ROI: {roi['roi_percentage']}%")
    print(f"- Payback Period: {roi['payback_period_months']} months")
