import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
from datetime import datetime
import logging
from app.data_processing import prepare_features_for_ml, clean_dataset
from io import BytesIO
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'regression': {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgboost': XGBRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svm': SVR(),
                'decision_tree': DecisionTreeRegressor(random_state=42)
            },
            'classification': {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': XGBClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(random_state=42),
                'decision_tree': DecisionTreeClassifier(random_state=42)
            }
        }
    
    def train_model(self, dataset_data, features, target, model_type, algorithm, test_size=0.2, user_id=None):
        """Train a machine learning model"""
        try:
            logger.info(f"Starting model training: {model_type}, {algorithm}")
            
            # Load and clean data
            df = pd.read_csv(BytesIO(dataset_data))
            df_clean = clean_dataset(df)
            
            # Validate features and target
            missing_features = [f for f in features if f not in df_clean.columns]
            if missing_features:
                return {'success': False, 'error': f'Features não encontradas: {missing_features}'}
            
            if target not in df_clean.columns:
                return {'success': False, 'error': f'Variável alvo não encontrada: {target}'}
            
            # Prepare features
            X, y = prepare_features_for_ml(df_clean, features, target)
            
            # Handle missing target values
            if y.isnull().any():
                valid_indices = y.notnull()
                X = X[valid_indices]
                y = y[valid_indices]
            
            if len(X) == 0:
                return {'success': False, 'error': 'Dados insuficientes após limpeza'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Get model
            if model_type not in self.models or algorithm not in self.models[model_type]:
                return {'success': False, 'error': 'Tipo de modelo ou algoritmo não suportado'}
            
            model = self.models[model_type][algorithm]
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            if model_type == 'regression':
                metrics = self._evaluate_regression(y_test, y_pred)
            else:
                metrics = self._evaluate_classification(y_test, y_pred)
            
            metrics['training_time'] = training_time
            
            # Save model
            model_filename = f"model_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join('instance/ml_models', model_filename)
            
            # Create directory if it doesn't exist
            os.makedirs('instance/ml_models', exist_ok=True)
            
            # Save model and feature names
            model_data = {
                'model': model,
                'feature_names': features,
                'target_name': target,
                'model_type': model_type,
                'algorithm': algorithm,
                'training_date': datetime.now(),
                'metrics': metrics
            }
            
            joblib.dump(model_data, model_path)
            
            return {
                'success': True,
                'model_path': model_path,
                **metrics
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_regression(self, y_true, y_pred):
        """Evaluate regression model"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate accuracy as percentage (simplified)
        accuracy = max(0, min(1, 1 - (rmse / (y_true.max() - y_true.min()))))
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae
        }
    
    def _evaluate_classification(self, y_true, y_pred):
        """Evaluate classification model"""
        accuracy = accuracy_score(y_true, y_pred)
        
        # Additional metrics for classification
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate precision, recall, f1 from macro averages
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': cm.tolist()
        }
    
    def cross_validate(self, file_path, features, target, model_type, algorithm, cv=5):
        """Perform cross-validation"""
        try:
            df = pd.read_csv(file_path)
            df_clean = clean_dataset(df)
            
            X, y = prepare_features_for_ml(df_clean, features, target)
            
            if model_type not in self.models or algorithm not in self.models[model_type]:
                return {'success': False, 'error': 'Model type or algorithm not supported'}
            
            model = self.models[model_type][algorithm]
            
            if model_type == 'regression':
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            return {
                'success': True,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'cv_scores': scores.tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def train_model(file_path, features, target, model_type, algorithm, test_size=0.2, user_id=None):
    """Train a machine learning model"""
    trainer = ModelTrainer()
    return trainer.train_model(file_path, features, target, model_type, algorithm, test_size, user_id)

def make_prediction(model_path, input_data):
    """Make prediction using trained model"""
    try:
        # Load model data
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Convert input data to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Ensure all features are present
        missing_features = [f for f in feature_names if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features correctly
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return prediction.tolist()
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

def evaluate_model(model_path, test_data_path):
    """Evaluate model on test data"""
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        target_name = model_data['target_name']
        model_type = model_data['model_type']
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        X_test, y_test = prepare_features_for_ml(test_df, feature_names, target_name)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        trainer = ModelTrainer()
        if model_type == 'regression':
            metrics = trainer._evaluate_regression(y_test, y_pred)
        else:
            metrics = trainer._evaluate_classification(y_test, y_pred)
        
        return {'success': True, 'metrics': metrics}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_model_info(model_path):
    """Get information about trained model"""
    try:
        model_data = joblib.load(model_path)
        return {
            'success': True,
            'model_type': model_data.get('model_type'),
            'algorithm': model_data.get('algorithm'),
            'feature_names': model_data.get('feature_names'),
            'target_name': model_data.get('target_name'),
            'training_date': model_data.get('training_date'),
            'metrics': model_data.get('metrics', {})
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}