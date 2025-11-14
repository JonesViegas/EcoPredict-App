import os
import pandas as pd
from werkzeug.utils import secure_filename
from functools import wraps
from flask import abort
from flask_login import current_user

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
    """Processa o arquivo enviado e retorna informações"""
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Criar pasta do usuário se não existir
            user_folder = os.path.join(upload_folder, str(user_id))
            os.makedirs(user_folder, exist_ok=True)
            
            # Salvar arquivo
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            
            # Ler arquivo para obter informações
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            file_info = {
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'rows_count': len(df),
                'columns_count': len(df.columns),
                'columns': df.columns.tolist()
            }
            
            return file_info
        return None
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return None

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