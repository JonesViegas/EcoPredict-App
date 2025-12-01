import os
import pandas as pd
from werkzeug.utils import secure_filename
from functools import wraps
from flask import abort
from flask_login import current_user
from datetime import datetime
from flask import current_app

# Decorator para rotas administrativas
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# Funções para upload de arquivos
def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida"""
    allowed_extensions = {'csv', 'xlsx', 'xls'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_uploaded_file(file, user_id, upload_folder):
    try:
        # 1. Salvar o arquivo com segurança
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{user_id}_{timestamp}_{original_filename}"
        
        # Garante que o diretório de uploads exista
        #os.makedirs(upload_folder, exist_ok=True)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # 2. Ler e processar o arquivo com Pandas
        df = pd.read_csv(file_path) # Ou pd.read_excel se você suportar

        # 3. Calcular métricas de qualidade (exatamente como nas outras funções)
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        quality_score = max(0, 100 - missing_percentage)

        # 4. Retornar um dicionário em caso de SUCESSO
        return {
            'success': True,
            'filename': filename,
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'rows_count': len(df),
            'columns_count': len(df.columns),
            'quality_score': quality_score,
            'missing_percentage': missing_percentage
        }

    except Exception as e:
        # 5. Retornar um dicionário em caso de FALHA
        # O log é importante para você saber o que deu errado
        print(f"ERRO AO PROCESSAR UPLOAD: {e}") 
        return {
            'success': False,
            'error': str(e)
        }
# Função para calcular uso de disco (para admin)
def calculate_disk_usage():
    """Calcular uso de disco dos datasets"""
    try:
        from app.models import Dataset
        import os
        
        total_size = 0
        datasets = Dataset.query.all()
        
        for dataset in datasets:
            if os.path.exists(dataset.file_path):
                total_size += os.path.getsize(dataset.file_path)
        
        return {
            'total_mb': round(total_size / (1024 * 1024), 2),
            'total_datasets': len(datasets)
        }
    except Exception as e:
        print(f"Erro ao calcular uso de disco: {e}")
        return {'total_mb': 0, 'total_datasets': 0}
# Função para obter informações do sistema
def get_system_info():
    """Obter informações do sistema"""
    try:
        import platform
        import flask
        from datetime import datetime
        
        return {
            'python_version': platform.python_version(),
            'flask_version': flask.__version__,
            'system': platform.system(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'start_time': datetime.now()
        }
    except Exception as e:
        print(f"Erro ao obter info do sistema: {e}")
        return {
            'python_version': 'N/A',
            'flask_version': 'N/A', 
            'system': 'N/A',
            'processor': 'N/A',
            'hostname': 'N/A',
            'start_time': datetime.now()
        }