import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from flask import current_app, url_for
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime
import re
import secrets

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Verifica se o arquivo é permitido"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config.get('ALLOWED_EXTENSIONS', {'csv', 'xlsx', 'xls'})

def send_reset_email(user, token):
    """Simula envio de email de reset de senha"""
    # Em produção, implemente com servidor SMTP real
    reset_url = url_for('auth.reset_password', token=token, _external=True)
    
    logger.info(f"EMAIL SIMULADO - Reset de senha para {user.email}")
    logger.info(f"URL de reset: {reset_url}")
    logger.info(f"Token: {token}")
    
    # Em produção, descomente e configure:
    """
    msg = MIMEMultipart()
    msg['From'] = current_app.config['MAIL_USERNAME']
    msg['To'] = user.email
    msg['Subject'] = "EcoPredict - Reset de Senha"
    
    body = f"""
    # Para resetar sua senha, clique no link abaixo:
    # {reset_url}
    #
    # Este link expira em 1 hora.
    # Se você não solicitou este reset, ignore este email.
    # """
    # 
    # msg.attach(MIMEText(body, 'plain'))
    # 
    # server = smtplib.SMTP(current_app.config['MAIL_SERVER'], current_app.config['MAIL_PORT'])
    # server.starttls()
    # server.login(current_app.config['MAIL_USERNAME'], current_app.config['MAIL_PASSWORD'])
    # server.send_message(msg)
    # server.quit()
    # """
    
    return True

def log_security_event(event, user_id=None, email=None):
    """Log de eventos de segurança"""
    timestamp = datetime.utcnow().isoformat()
    user_info = f"user_id: {user_id}" if user_id else f"email: {email}" if email else "unknown user"
    
    logger.warning(f"SECURITY - {timestamp} - {event} - {user_info}")

def validate_password_strength(password):
    """Valida a força da senha"""
    if len(password) < 8:
        return {'valid': False, 'message': 'A senha deve ter pelo menos 8 caracteres'}
    
    if not re.search(r"[A-Z]", password):
        return {'valid': False, 'message': 'A senha deve conter pelo menos uma letra maiúscula'}
    
    if not re.search(r"[a-z]", password):
        return {'valid': False, 'message': 'A senha deve conter pelo menos uma letra minúscula'}
    
    if not re.search(r"\d", password):
        return {'valid': False, 'message': 'A senha deve conter pelo menos um número'}
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return {'valid': False, 'message': 'A senha deve conter pelo menos um caractere especial'}
    
    # Verificar senhas comuns
    common_passwords = ['12345678', 'password', 'senha123', 'admin123', 'qwerty']
    if password.lower() in common_passwords:
        return {'valid': False, 'message': 'Esta senha é muito comum. Escolha uma senha mais forte'}
    
    return {'valid': True, 'message': 'Senha forte'}

def validate_air_quality_data(form_data):
    """Validate air quality data before saving"""
    try:
        # Check required fields
        if not form_data.get('location') or not form_data.get('latitude') or not form_data.get('longitude'):
            return {'valid': False, 'error': 'Localização, latitude e longitude são obrigatórios'}
        
        # Validate coordinate ranges
        lat = form_data.get('latitude')
        lon = form_data.get('longitude')
        
        if not (-90 <= lat <= 90):
            return {'valid': False, 'error': 'Latitude deve estar entre -90 e 90'}
        if not (-180 <= lon <= 180):
            return {'valid': False, 'error': 'Longitude deve estar entre -180 e 180'}
        
        # Validate pollutant ranges (simplified)
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        for pollutant in pollutants:
            value = form_data.get(pollutant)
            if value is not None and value < 0:
                return {'valid': False, 'error': f'{pollutant.upper()} não pode ser negativo'}
        
        return {'valid': True}
        
    except Exception as e:
        return {'valid': False, 'error': f'Erro na validação: {str(e)}'}

def process_uploaded_file(file, user_id):
    """Processa arquivo uploadado com verificações de segurança"""
    try:
        if not allowed_file(file.filename):
            return {'success': False, 'error': 'Tipo de arquivo não permitido'}
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'], 
            f"{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
        )
        
        file.save(file_path)
        
        # Verificar se é um arquivo CSV/Excel válido
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            os.remove(file_path)  # Remover arquivo inválido
            return {'success': False, 'error': f'Arquivo corrompido ou formato inválido: {str(e)}'}
        
        # Calcular métricas de qualidade
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        quality_score = max(0, 100 - missing_percentage * 2)
        
        return {
            'success': True,
            'filename': filename,
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'rows_count': len(df),
            'columns_count': len(df.columns),
            'quality_score': round(quality_score, 2),
            'missing_percentage': round(missing_percentage, 2)
        }
        
    except Exception as e:
        logger.error(f"Erro no processamento do arquivo: {str(e)}")
        return {'success': False, 'error': f'Erro interno no processamento: {str(e)}'}

def generate_secure_token(length=32):
    """Gera token seguro"""
    return secrets.token_urlsafe(length)

def classify_aqi(df):
    """
    Calcula o Índice de Qualidade do Ar (IQA) para múltiplos poluentes com base nos padrões da EPA.
    Adiciona colunas de IQA para cada poluente, um IQA geral e uma categoria de classificação.
    """
    import pandas as pd
    import numpy as np
    
    # Tabela de breakpoints do IQA (μg/m³ para particulados, ppb para gases)
    breakpoints = {
        'pm25': [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), 
                (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
        'pm10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), 
                (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
        'o3':   [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), 
                (86, 105, 151, 200), (106, 200, 201, 300)],
        'co':   [(0, 4400, 0, 50), (4401, 9400, 51, 100), (9401, 12400, 101, 150), 
                (12401, 15400, 151, 200), (15401, 30400, 201, 300)],
        'so2':  [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), 
                (186, 304, 151, 200)],
        'no2':  [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), 
                (361, 649, 151, 200)]
    }
    
    categories = {
        (0, 50): 'Boa', 
        (51, 100): 'Moderada', 
        (101, 150): 'Ruim para Grupos Sensíveis',
        (151, 200): 'Ruim', 
        (201, 300): 'Muito Ruim', 
        (301, 501): 'Perigosa'
    }

    def calculate_sub_index(concentration, pollutant):
        if pd.isna(concentration) or pollutant not in breakpoints: 
            return np.nan
        for bp_low, bp_high, i_low, i_high in breakpoints[pollutant]:
            if bp_low <= concentration <= bp_high:
                return ((i_high - i_low) / (bp_high - bp_low)) * (concentration - bp_low) + i_low
        return np.nan

    # Calcula AQI para cada poluente
    for poll in breakpoints.keys():
        if poll in df.columns:
            df[f'AQI_{poll}'] = df[poll].apply(lambda x: calculate_sub_index(x, poll))

    # Calcula AQI geral (máximo entre todos)
    aqi_cols = [f'AQI_{poll}' for poll in breakpoints.keys() if f'AQI_{poll}' in df.columns]
    if not aqi_cols:
        df['Overall_AQI'] = np.nan
        df['AQI_Category'] = 'Indisponível'
        return df

    df['Overall_AQI'] = df[aqi_cols].max(axis=1)
    
    # Classifica a categoria
    def get_category(aqi):
        if pd.isna(aqi): 
            return 'Indisponível'
        for (i_low, i_high), cat in categories.items():
            if i_low <= aqi <= i_high: 
                return cat
        return 'Perigosa'
        
    df['AQI_Category'] = df['Overall_AQI'].apply(get_category)
    return df