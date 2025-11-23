import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Desativa avisos de Matplotlib sobre layout, que são comuns em ambientes de backend
plt.set_loglevel('warning') 

def plot_to_base64(plt_figure):
    """Converte uma figura do Matplotlib para uma string Base64 para embutir no HTML."""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(plt_figure) # Fecha a figura para liberar memória
    return img_str

def generate_openaq_analysis(df):
    """Gera análises específicas para dados da OpenAQ."""
    analysis = {}
    
    # 1. Gráfico de Série Temporal do AQI
    if 'datetime' in df.columns and 'Overall_AQI' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['datetime'], df['Overall_AQI'], label='AQI Geral', color='blue')
        ax.set_title('Evolução do Índice de Qualidade do Ar (AQI)')
        ax.set_xlabel('Data')
        ax.set_ylabel('AQI')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        analysis['timeseries_plot'] = plot_to_base64(fig)

    # 2. Matriz de Correlação
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Matriz de Correlação de Variáveis')
        analysis['correlation_heatmap'] = plot_to_base64(fig)
        
    return analysis

def generate_inpe_analysis(df):
    """Gera análises específicas para dados do INPE."""
    analysis = {}
    
    # 1. Gráfico de Focos por Bioma
    if 'bioma' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df['bioma'].value_counts().plot(kind='bar', ax=ax, color='orangered')
        ax.set_title('Número de Focos de Queimada por Bioma')
        ax.set_xlabel('Bioma')
        ax.set_ylabel('Contagem de Focos')
        plt.xticks(rotation=45)
        analysis['fires_by_biome_plot'] = plot_to_base64(fig)

    # 2. Tabela de Focos por Estado
    if 'estado' in df.columns:
        state_counts = df['estado'].value_counts().reset_index()
        state_counts.columns = ['Estado', 'Número de Focos']
        analysis['fires_by_state_table'] = state_counts.to_html(classes='table table-striped', index=False)

    return analysis

def generate_inmet_analysis(df):
    """Gera análises específicas para dados do INMET."""
    analysis = {}
    
    # 1. Gráfico de Temperatura e Umidade
    if 'datetime' in df.columns and 'temperature' in df.columns and 'humidity' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Temperatura (°C)', color='red')
        ax1.plot(df['datetime'], df['temperature'], color='red', label='Temperatura')
        ax1.tick_params(axis='y', labelcolor='red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Umidade (%)', color='blue')
        ax2.plot(df['datetime'], df['humidity'], color='blue', label='Umidade')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax1.set_title('Variação de Temperatura e Umidade')
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        analysis['temp_humidity_plot'] = plot_to_base64(fig)
        
    return analysis

def generate_generic_analysis(df):
    """Gera uma análise padrão para uploads genéricos."""
    analysis = {}
    
    # 1. Estatísticas Descritivas
    try:
        analysis['descriptive_stats'] = df.describe().to_html(classes='table table-striped')
    except Exception:
        analysis['descriptive_stats'] = "<p>Não foi possível gerar estatísticas descritivas.</p>"
        
    # 2. Matriz de Correlação (se houver colunas numéricas)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", ax=ax)
        ax.set_title('Matriz de Correlação')
        analysis['correlation_heatmap'] = plot_to_base64(fig)
        
    return analysis