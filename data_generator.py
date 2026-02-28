import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class RetailDataGenerator:
    """
    Generates synthetic retail dataset for machine learning models.
    Creates realistic customer data with logical relationships between features.
    """
    
    def __init__(self, num_records=1500):
        self.num_records = num_records
        np.random.seed(42)
        random.seed(42)
    
    def generate_customer_demographics(self):
        """Generate customer demographic data with realistic distributions."""
        ages = np.random.normal(38, 12, self.num_records)
        ages = np.clip(ages, 18, 80).astype(int)
        
        genders = np.random.choice(['Male', 'Female'], self.num_records, p=[0.48, 0.52])
        
        # Income distribution with realistic skew
        income_base = np.random.lognormal(10.5, 0.5, self.num_records)
        annual_income = np.clip(income_base, 25000, 200000).astype(int)
        
        # Customer tenure in months
        tenure = np.random.exponential(36, self.num_records)
        tenure = np.clip(tenure, 1, 120).astype(int)
        
        return pd.DataFrame({
            'Customer_Age': ages,
            'Gender': genders,
            'Annual_Income': annual_income,
            'Customer_Tenure_Months': tenure
        })
    
    def generate_behavioral_features(self, demographics):
        """Generate behavioral features based on demographics."""
        df = demographics.copy()
        
        # Purchase frequency influenced by age and income
        base_frequency = 2 + (df['Annual_Income'] / 50000) + (df['Customer_Age'] / 30)
        purchase_frequency = np.random.poisson(base_frequency).astype(int)
        purchase_frequency = np.clip(purchase_frequency, 1, 15)
        df['Purchase_Frequency'] = purchase_frequency
        
        # Previous purchase amount correlated with income and frequency
        prev_purchase_base = (df['Annual_Income'] * 0.02) * (df['Purchase_Frequency'] / 5)
        previous_purchase = np.random.normal(prev_purchase_base, prev_purchase_base * 0.3)
        previous_purchase = np.clip(previous_purchase, 50, 5000).astype(int)
        df['Previous_Purchase_Amount'] = previous_purchase
        
        return df
    
    def generate_business_features(self, df):
        """Generate business-related features."""
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
                     'Toys', 'Beauty', 'Food', 'Health', 'Automotive']
        df['Product_Category'] = np.random.choice(categories, self.num_records)
        
        # Store locations
        locations = ['Downtown', 'Suburban', 'Mall', 'Airport', 'Online']
        df['Store_Location'] = np.random.choice(locations, self.num_records)
        
        # Marketing spend based on customer value
        marketing_spend = (df['Annual_Income'] * 0.001) + np.random.normal(0, 50)
        marketing_spend = np.clip(marketing_spend, 10, 500).astype(int)
        df['Marketing_Spend'] = marketing_spend
        
        # Discount offered (seasonal and customer-based)
        discount_base = np.random.beta(2, 5, self.num_records) * 30
        discount = np.clip(discount_base, 0, 40).astype(int)
        df['Discount_Offered'] = discount
        
        # Seasonal demand index
        seasonal_demand = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, self.num_records)) + \
                         np.random.normal(0, 10, self.num_records)
        seasonal_demand = np.clip(seasonal_demand, 0, 100).astype(int)
        df['Seasonal_Demand_Index'] = seasonal_demand
        
        return df
    
    def generate_target_variables(self, df):
        """Generate target variables with logical relationships."""
        # Monthly Sales (target for regression)
        sales_base = (df['Annual_Income'] * 0.05) + \
                    (df['Previous_Purchase_Amount'] * 0.3) + \
                    (df['Marketing_Spend'] * 2) + \
                    (df['Purchase_Frequency'] * 50) - \
                    (df['Discount_Offered'] * 10)
        
        monthly_sales = np.random.normal(sales_base, sales_base * 0.2)
        monthly_sales = np.clip(monthly_sales, 100, 10000).astype(int)
        df['Monthly_Sales'] = monthly_sales
        
        # Purchase Decision (target for classification)
        purchase_prob = 1 / (1 + np.exp(-(sales_base - 500) / 1000))
        purchase_decision = np.random.random(self.num_records) < purchase_prob
        df['Purchase_Decision'] = np.where(purchase_decision, 'Yes', 'No')
        
        # Loyalty Category (target for classification)
        loyalty_score = (df['Customer_Tenure_Months'] / 12) + \
                       (df['Purchase_Frequency'] * 2) + \
                       (df['Monthly_Sales'] / 1000)
        
        loyalty_categories = []
        for score in loyalty_score:
            if score > 50:
                loyalty_categories.append('Gold')
            elif score > 25:
                loyalty_categories.append('Silver')
            else:
                loyalty_categories.append('Bronze')
        
        df['Loyalty_Category'] = loyalty_categories
        
        return df
    
    def generate_dataset(self):
        """Generate complete synthetic dataset."""
        print(f"Generating {self.num_records} synthetic retail records...")
        
        # Step 1: Generate demographics
        demographics = self.generate_customer_demographics()
        
        # Step 2: Generate behavioral features
        df = self.generate_behavioral_features(demographics)
        
        # Step 3: Generate business features
        df = self.generate_business_features(df)
        
        # Step 4: Generate target variables
        df = self.generate_target_variables(df)
        
        # Reorder columns for better readability
        column_order = [
            'Customer_Age', 'Gender', 'Annual_Income', 'Customer_Tenure_Months',
            'Purchase_Frequency', 'Previous_Purchase_Amount', 'Product_Category',
            'Store_Location', 'Marketing_Spend', 'Discount_Offered',
            'Seasonal_Demand_Index', 'Monthly_Sales', 'Purchase_Decision',
            'Loyalty_Category'
        ]
        
        df = df[column_order]
        
        print(f"Dataset generated successfully with {len(df)} records and {len(df.columns)} features.")
        return df
    
    def save_dataset(self, df, filename='retail_data.csv'):
        """Save dataset to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
    def get_data_summary(self, df):
        """Generate data summary statistics."""
        summary = {
            'total_records': len(df),
            'features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'target_variables': {
                'regression': 'Monthly_Sales',
                'classification_binary': 'Purchase_Decision',
                'classification_multiclass': 'Loyalty_Category'
            }
        }
        return summary

if __name__ == "__main__":
    # Example usage
    generator = RetailDataGenerator(num_records=1500)
    dataset = generator.generate_dataset()
    
    # Display summary
    summary = generator.get_data_summary(dataset)
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save dataset
    generator.save_dataset(dataset)
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(dataset.head())
