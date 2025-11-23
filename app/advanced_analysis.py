import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg') # Define o backend não interativo
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- NOVAS IMPORTAÇÕES DO SCIKIT-LEARN ---
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

plt.set_loglevel('warning')

def plot_to_base64(plt_figure):
    """Converte uma figura do Matplotlib para uma string Base64."""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(plt_figure)
    return img_str

# --- NOVA FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def prepare_data_for_modeling(df, features, target, is_classification=False):
    """
    Função robusta para limpar, pré-processar e separar os dados.
    """
    # 1. Seleciona colunas e remove linhas com valores ausentes no alvo
    all_cols = features + [target]
    data = df[all_cols].dropna(subset=[target]).copy()

    if len(data) < 10:
        raise ValueError("Dados insuficientes após a remoção de valores ausentes no alvo.")

    X = data[features]
    y = data[target]

    # 2. Codifica a variável alvo se for classificação
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_labels = le.classes_ # Guardar os nomes das classes originais
    else:
        target_labels = None

    # 3. Identifica tipos de features (numéricas vs. categóricas)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Cria um pipeline de pré-processamento
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Aplica o pré-processamento em X
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, target_labels, preprocessor


# --- FUNÇÕES DE ANÁLISE NÃO SUPERVISIONADA (SEM ALTERAÇÃO) ---
def run_kmeans_clustering(df, features, n_clusters):
    if len(features) < 2:
        return {'error': 'Selecione pelo menos duas features para a clusterização.'}
    data = df[features].dropna()
    if len(data) < n_clusters:
        return {'error': f'Dados insuficientes ({len(data)} linhas) para criar {n_clusters} clusters.'}
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(data_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    ax.set_title(f'Resultado da Clusterização K-Means (k={n_clusters})')
    ax.set_xlabel(f'Feature: {features[0]} (Padronizado)')
    ax.set_ylabel(f'Feature: {features[1]} (Padronizado)')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    cluster_stats = pd.Series(clusters).value_counts().reset_index()
    cluster_stats.columns = ['Cluster', 'Número de Pontos']
    return {
        'plot': plot_to_base64(fig),
        'stats_table': cluster_stats.to_html(classes='table table-sm table-striped', index=False)
    }

def run_pca_analysis(df, features):
    """Executa PCA e retorna gráficos explicativos, incluindo os loadings."""
    data = df[features].dropna()
    if len(data) < 2 or len(features) < 2:
        return {'error': 'Dados ou features insuficientes para análise PCA.'}

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Limita o número de componentes ao mínimo entre features e amostras
    n_components = min(len(features), len(data))
    pca = PCA(n_components=n_components)
    
    principal_components = pca.fit_transform(data_scaled)
    
    # --- Gráfico 1: Variância Explicada (Scree Plot) ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    explained_variance = pca.explained_variance_ratio_ * 100
    component_labels = [f'PC{i+1}' for i in range(n_components)]
    ax1.bar(component_labels, explained_variance, alpha=0.7)
    ax1.set_title('Variância Explicada por Componente Principal (PCA)')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Porcentagem de Variância (%)')
    
    # --- Gráfico 2: Projeção dos Dados ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.6)
    ax2.set_title('Dados Projetados nos 2 Primeiros Componentes Principais')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    ax2.grid(True)

    # --- NOVO GRÁFICO 3: Mapa de Calor das Contribuições (Loadings) ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    loadings_df = pd.DataFrame(pca.components_.T, columns=component_labels, index=features)
    sns.heatmap(loadings_df, annot=True, cmap='vlag', fmt='.2f', ax=ax3)
    ax3.set_title('Contribuição das Features nos Componentes Principais')

    return {
        'scree_plot': plot_to_base64(fig1),
        'components_plot': plot_to_base64(fig2),
        'loadings_heatmap': plot_to_base64(fig3) # <-- Novo item retornado
    }
# --- FUNÇÕES DE ANÁLISE SUPERVISIONADA (ATUALIZADAS) ---

def run_classification_analysis(df, features, target):
    """Treina um modelo de classificação usando o pipeline de pré-processamento."""
    try:
        X_processed, y, target_labels, _ = prepare_data_for_modeling(df, features, target, is_classification=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)
        
        model = SVC(kernel='rbf') # Usando SVC como exemplo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_labels, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=target_labels, yticklabels=target_labels)
        ax.set_title(f'Matriz de Confusão (Acurácia: {accuracy:.2%})')
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        
        return {
            'confusion_matrix_plot': plot_to_base64(fig),
            'accuracy': f"{accuracy:.2%}",
            'report_table': report_df.to_html(classes='table table-sm table-bordered')
        }
    except Exception as e:
        return {'error': f'Erro na análise de classificação: {str(e)}'}


def run_regression_analysis(df, features, target):
    """Treina um modelo de regressão usando o pipeline de pré-processamento."""
    try:
        X_processed, y, _, _ = prepare_data_for_modeling(df, features, target, is_classification=False)

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2)
        ax.set_title(f'Valores Reais vs. Previstos (R² = {r2:.2f})')
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Valores Previstos')
        ax.grid(True)
        
        return {
            'prediction_plot': plot_to_base64(fig),
            'r2_score': f"{r2:.4f}",
            'rmse': f"{rmse:.4f}"
        }
    except Exception as e:
        return {'error': f'Erro na análise de regressão: {str(e)}'}