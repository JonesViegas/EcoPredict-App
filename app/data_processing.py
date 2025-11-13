import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import logging

warnings.filterwarnings('ignore')

def clean_dataset(df):
    """Clean and preprocess dataset"""
    df_clean = df.copy()
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median
    if len(numeric_columns) > 0:
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    
    # Impute categorical columns with mode
    if len(categorical_columns) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_columns] = imputer.fit_transform(df_clean[categorical_columns])
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Remove columns with too many missing values (>50%)
    missing_percentage = df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = missing_percentage[missing_percentage > 0.5].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean

def calculate_correlations(df, method='pearson'):
    """Calculate correlation matrix"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method)

def generate_statistics(df):
    """Generate descriptive statistics for dataset"""
    stats = {}
    
    # Numeric columns statistics
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats['numeric'] = {
            'count': numeric_df.count().to_dict(),
            'mean': numeric_df.mean().to_dict(),
            'std': numeric_df.std().to_dict(),
            'min': numeric_df.min().to_dict(),
            'max': numeric_df.max().to_dict(),
            'median': numeric_df.median().to_dict()
        }
    
    # Categorical columns statistics
    categorical_df = df.select_dtypes(include=['object'])
    if not categorical_df.empty:
        stats['categorical'] = {
            'count': categorical_df.count().to_dict(),
            'unique': categorical_df.nunique().to_dict(),
            'top': categorical_df.mode().iloc[0].to_dict() if not categorical_df.empty else {},
            'freq': {col: categorical_df[col].value_counts().iloc[0] 
                    for col in categorical_df.columns}
        }
    
    # Overall dataset info
    stats['overall'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return stats

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def prepare_features_for_ml(df, features, target):
    """Prepare features for machine learning"""
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Scale numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 0:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    return X, y

def calculate_feature_importance(model, feature_names):
    """Calculate feature importance for tree-based models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
    except:
        pass
    return None

def validate_air_quality_data(df):
    """
    Valida dados de qualidade do ar verificando valores ausentes e ranges
    """
    try:
        validation_results = {
            'is_valid': True,
            'issues': [],
            'missing_data': {},
            'out_of_range': {}
        }
        
        # Verificar valores ausentes
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_counts.sum() / total_cells) * 100
        
        if missing_percentage > 0:
            validation_results['missing_data'] = {
                'total_missing': missing_counts.sum(),
                'missing_percentage': missing_percentage,
                'columns_missing': missing_counts[missing_counts > 0].to_dict()
            }
            validation_results['issues'].append(f"Dados ausentes: {missing_percentage:.1f}%")
        
        # Verificar ranges para colunas conhecidas
        expected_ranges = {
            'pm25': (0, 500),
            'pm10': (0, 600),
            'o3': (0, 200),
            'co': (0, 50),
            'so2': (0, 100),
            'no2': (0, 200),
            'temperature': (-50, 60),
            'humidity': (0, 100),
            'pressure': (800, 1100),
            'wind_speed': (0, 100)
        }
        
        for column, (min_val, max_val) in expected_ranges.items():
            if column in df.columns:
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                if not out_of_range.empty:
                    validation_results['out_of_range'][column] = len(out_of_range)
                    validation_results['issues'].append(f"Valores fora do range em {column}: {len(out_of_range)} registros")
        
        # Se houver muitos problemas, marcar como inválido
        if missing_percentage > 50 or len(validation_results['issues']) > 5:
            validation_results['is_valid'] = False
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Erro na validação de dados: {e}")
        return {
            'is_valid': False,
            'issues': [f"Erro na validação: {str(e)}"],
            'missing_data': {},
            'out_of_range': {}
        }