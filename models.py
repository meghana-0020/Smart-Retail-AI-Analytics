import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class RetailMLModels:
    """
    Machine Learning models for retail analytics:
    - Linear Regression for sales prediction
    - Decision Tree for purchase decision classification
    - KNN for loyalty category prediction
    - K-Means for customer segmentation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.performance_metrics = {}
    
    def preprocess_data(self, df, target_column, task_type='regression'):
        """
        Preprocess data for ML models.
        
        Args:
            df: Input dataframe
            target_column: Target variable name
            task_type: 'regression', 'classification_binary', or 'classification_multiclass'
        
        Returns:
            X, y: Features and target variables
        """
        # Make a copy to avoid modifying original data
        data = df.copy()

        # For regression tasks, ensure other target columns are not used as features
        # This prevents issues where models are trained with classification targets
        # like Purchase_Decision or Loyalty_Category as input features, which will
        # not be present at prediction time.
        if task_type == 'regression':
            data = data.drop(columns=['Purchase_Decision', 'Loyalty_Category'], errors='ignore')
        
        # Handle categorical variables
        categorical_columns = data.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col])
            else:
                data[col] = self.encoders[col].transform(data[col])
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
        
        # Encode target if it's categorical
        if task_type != 'regression' and y is not None:
            if target_column not in self.encoders:
                self.encoders[target_column] = LabelEncoder()
                y = self.encoders[target_column].fit_transform(y)
            else:
                y = self.encoders[target_column].transform(y)
        
        # Store feature column names
        self.feature_columns[target_column] = X.columns.tolist()
        
        return X, y
    
    def train_linear_regression(self, df, target_column='Monthly_Sales'):
        """
        Train Linear Regression model for sales prediction.
        
        Args:
            df: Input dataframe
            target_column: Target variable for regression
        
        Returns:
            Trained model and performance metrics
        """
        print("Training Linear Regression model...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, target_column, 'regression')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_column] = scaler
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store model and metrics
        self.models['linear_regression'] = model
        self.performance_metrics['linear_regression'] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2,
            'Target_Variable': target_column
        }
        
        print(f"Linear Regression trained successfully!")
        print(f"R² Score: {r2:.4f}, RMSE: {rmse:.2f}")
        
        return model, {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def train_decision_tree(self, df, target_column='Purchase_Decision'):
        """
        Train Decision Tree model for purchase decision classification.
        
        Args:
            df: Input dataframe
            target_column: Target variable for classification
        
        Returns:
            Trained model and performance metrics
        """
        print("Training Decision Tree model...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, target_column, 'classification_binary')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_column] = scaler
        
        # Train model
        model = DecisionTreeClassifier(random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store model and metrics
        self.models['decision_tree'] = model
        self.performance_metrics['decision_tree'] = {
            'Accuracy': accuracy,
            'Target_Variable': target_column,
            'Classification_Report': class_report
        }
        
        print(f"Decision Tree trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        return model, {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def train_knn(self, df, target_column='Loyalty_Category'):
        """
        Train KNN model for loyalty category prediction.
        
        Args:
            df: Input dataframe
            target_column: Target variable for classification
        
        Returns:
            Trained model and performance metrics
        """
        print("Training KNN model...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, target_column, 'classification_multiclass')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_column] = scaler
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store model and metrics
        self.models['knn'] = model
        self.performance_metrics['knn'] = {
            'Accuracy': accuracy,
            'Target_Variable': target_column,
            'Classification_Report': class_report
        }
        
        print(f"KNN trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        return model, {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def train_kmeans(self, df, n_clusters=4):
        """
        Train K-Means model for customer segmentation.
        
        Args:
            df: Input dataframe
            n_clusters: Number of clusters
        
        Returns:
            Trained model and cluster information
        """
        print("Training K-Means model...")
        
        # Preprocess data (exclude target variables)
        exclude_columns = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
        feature_data = df.drop(columns=exclude_columns, errors='ignore')
        
        # Handle categorical variables
        categorical_columns = feature_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            # Ensure all categorical columns are label-encoded before scaling
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                feature_data[col] = self.encoders[col].fit_transform(feature_data[col])
            else:
                feature_data[col] = self.encoders[col].transform(feature_data[col])
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        self.scalers['kmeans'] = scaler
        
        # Find optimal number of clusters using elbow method
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(scaled_data)
            inertias.append(kmeans_temp.inertia_)
        
        # Train final model
        model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = model.fit_predict(scaled_data)
        
        # Store model
        self.models['kmeans'] = model
        
        # Add cluster labels to original data
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        print(f"K-Means trained successfully with {n_clusters} clusters!")
        
        return model, {
            'cluster_labels': cluster_labels,
            'inertias': inertias,
            'k_range': list(k_range),
            'df_with_clusters': df_with_clusters,
            'scaled_data': scaled_data
        }
    
    def predict_sales(self, input_data):
        """Predict monthly sales using trained linear regression model."""
        if 'linear_regression' not in self.models:
            raise ValueError("Linear Regression model not trained yet!")
        
        # Preprocess input data
        processed_data = input_data.copy()
        for col, encoder in self.encoders.items():
            if col in processed_data.columns and col != 'Monthly_Sales':
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Scale features
        scaler = self.scalers['Monthly_Sales']
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.models['linear_regression'].predict(scaled_data)
        return prediction
    
    def predict_purchase_decision(self, input_data):
        """Predict purchase decision using trained decision tree model."""
        if 'decision_tree' not in self.models:
            raise ValueError("Decision Tree model not trained yet!")
        
        # Preprocess input data
        processed_data = input_data.copy()
        for col, encoder in self.encoders.items():
            if col in processed_data.columns and col != 'Purchase_Decision':
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Scale features
        scaler = self.scalers['Purchase_Decision']
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.models['decision_tree'].predict(scaled_data)
        
        # Convert back to original labels
        prediction_labels = self.encoders['Purchase_Decision'].inverse_transform(prediction)
        return prediction_labels
    
    def predict_loyalty_category(self, input_data):
        """Predict loyalty category using trained KNN model."""
        if 'knn' not in self.models:
            raise ValueError("KNN model not trained yet!")
        
        # Preprocess input data
        processed_data = input_data.copy()
        for col, encoder in self.encoders.items():
            if col in processed_data.columns and col != 'Loyalty_Category':
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Scale features
        scaler = self.scalers['Loyalty_Category']
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.models['knn'].predict(scaled_data)
        
        # Convert back to original labels
        prediction_labels = self.encoders['Loyalty_Category'].inverse_transform(prediction)
        return prediction_labels
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models."""
        if model_name == 'decision_tree' and model_name in self.models:
            model = self.models[model_name]
            feature_names = self.feature_columns.get('Purchase_Decision', [])
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        return {}
    
    def save_models(self, filepath='models/'):
        """Save trained models to disk."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for name, model in self.models.items():
            with open(f"{filepath}{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        with open(f"{filepath}encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(f"{filepath}scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models from disk."""
        import os
        
        if not os.path.exists(filepath):
            print(f"Models directory {filepath} not found!")
            return
        
        # Load models
        model_files = [f for f in os.listdir(filepath) if f.endswith('.pkl') and f != 'encoders.pkl' and f != 'scalers.pkl']
        for file in model_files:
            model_name = file.replace('.pkl', '')
            with open(f"{filepath}{file}", 'rb') as f:
                self.models[model_name] = pickle.load(f)
        
        # Load encoders and scalers
        if os.path.exists(f"{filepath}encoders.pkl"):
            with open(f"{filepath}encoders.pkl", 'rb') as f:
                self.encoders = pickle.load(f)
        
        if os.path.exists(f"{filepath}scalers.pkl"):
            with open(f"{filepath}scalers.pkl", 'rb') as f:
                self.scalers = pickle.load(f)
        
        print(f"Models loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_generator import RetailDataGenerator
    
    # Generate sample data
    generator = RetailDataGenerator(num_records=1000)
    df = generator.generate_dataset()
    
    # Initialize and train models
    ml_models = RetailMLModels()
    
    # Train all models
    lr_model, lr_metrics = ml_models.train_linear_regression(df)
    dt_model, dt_metrics = ml_models.train_decision_tree(df)
    knn_model, knn_metrics = ml_models.train_knn(df)
    kmeans_model, kmeans_metrics = ml_models.train_kmeans(df)
    
    # Display performance metrics
    print("\n=== Model Performance Summary ===")
    for model_name, metrics in ml_models.performance_metrics.items():
        print(f"\n{model_name.upper()}:")
        for key, value in metrics.items():
            if key != 'Classification_Report':
                print(f"  {key}: {value}")
    
    # Save models
    ml_models.save_models()
