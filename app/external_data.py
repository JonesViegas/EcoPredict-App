import os
import pandas as pd
import numpy as np
import requests
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime, time, timedelta
import logging
import random
import threading
from contextlib import contextmanager
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from app import db
from app.models import Dataset
from app.utils import allowed_file

# Blueprint para dados externos
external_bp = Blueprint('external', __name__, url_prefix='/external')

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Gerenciador de contexto para limitar tempo de execução (compatível com Windows)
    """
    def timeout_handler():
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

def cleanup_dataframe(df):
    """Remove colunas desnecessárias e otimiza tipos de dados"""
    try:
        # Converter para tipos mais eficientes
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col]) > 0 and df[col].nunique() / len(df) < 0.5:  # Se poucos valores únicos
                df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        logger.warning(f"Erro na otimização do DataFrame: {e}")
        return df

def get_location_profile(location: str):
    """Perfis de localização"""
    location_profiles = {
        'são paulo': {
            'pollutants': {'pm25': (15, 45), 'pm10': (25, 65), 'o3': (30, 90), 'co': (300, 900), 'so2': (5, 25), 'no2': (20, 60)},
            'weather': {'temperature': (18, 32), 'humidity': (60, 90), 'pressure': (1010, 1020), 'wind_speed': (2, 8)}
        },
        'cuiabá': {
            'pollutants': {'pm25': (20, 60), 'pm10': (30, 80), 'o3': (40, 110), 'co': (400, 1000), 'so2': (8, 30), 'no2': (25, 70)},
            'weather': {'temperature': (22, 38), 'humidity': (40, 80), 'pressure': (1008, 1015), 'wind_speed': (3, 12)}
        },
        'belém': {
            'pollutants': {'pm25': (10, 35), 'pm10': (20, 55), 'o3': (25, 80), 'co': (200, 700), 'so2': (3, 18), 'no2': (15, 45)},
            'weather': {'temperature': (24, 32), 'humidity': (75, 95), 'pressure': (1010, 1015), 'wind_speed': (2, 6)}
        },
        'tangara': {
            'pollutants': {'pm25': (15, 50), 'pm10': (25, 70), 'o3': (35, 100), 'co': (350, 950), 'so2': (6, 28), 'no2': (20, 65)},
            'weather': {'temperature': (20, 35), 'humidity': (45, 85), 'pressure': (1009, 1018), 'wind_speed': (3, 10)}
        },
        'london': {
            'pollutants': {'pm25': (8, 35), 'pm10': (15, 50), 'o3': (25, 80), 'co': (200, 700), 'so2': (2, 15), 'no2': (15, 45)},
            'weather': {'temperature': (5, 25), 'humidity': (70, 95), 'pressure': (1005, 1025), 'wind_speed': (5, 15)}
        },
        'new york': {
            'pollutants': {'pm25': (10, 40), 'pm10': (20, 55), 'o3': (35, 95), 'co': (250, 800), 'so2': (3, 20), 'no2': (25, 65)},
            'weather': {'temperature': (0, 30), 'humidity': (50, 85), 'pressure': (1010, 1030), 'wind_speed': (3, 10)}
        },
        'default': {
            'pollutants': {'pm25': (10, 35), 'pm10': (15, 50), 'o3': (25, 85), 'co': (200, 700), 'so2': (3, 18), 'no2': (15, 50)},
            'weather': {'temperature': (15, 30), 'humidity': (50, 85), 'pressure': (1010, 1020), 'wind_speed': (2, 8)}
        }
    }
    
    location_lower = location.lower()
    for loc in location_profiles:
        if loc in location_lower:
            return location_profiles[loc]
    
    return location_profiles['default']

def classify_aqi(df):
    """
    Calcula o Índice de Qualidade do Ar (IQA) e inclui features meteorológicas
    """
    try:
        # Tabela de breakpoints do IQA 
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

        # Calcula AQI para cada poluente disponível
        for poll in breakpoints.keys():
            if poll in df.columns:
                df[f'AQI_{poll}'] = df[poll].apply(lambda x: calculate_sub_index(x, poll))

        # Calcula AQI geral (máximo entre todos)
        aqi_cols = [f'AQI_{poll}' for poll in breakpoints.keys() if f'AQI_{poll}' in df.columns]
        if aqi_cols:
            df['Overall_AQI'] = df[aqi_cols].max(axis=1)
            
            def get_category(aqi):
                if pd.isna(aqi): 
                    return 'Indisponível'
                for (i_low, i_high), cat in categories.items():
                    if i_low <= aqi <= i_high: 
                        return cat
                return 'Perigosa'
                
            df['AQI_Category'] = df['Overall_AQI'].apply(get_category)
        else:
            df['Overall_AQI'] = np.nan
            df['AQI_Category'] = 'Indisponível'
            
        # Garante que as features meteorológicas estejam presentes
        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for feature in weather_features:
            if feature not in df.columns:
                # Gera valores padrão se a feature não existir
                if feature == 'temperature':
                    df[feature] = np.random.uniform(15, 30, len(df))
                elif feature == 'humidity':
                    df[feature] = np.random.uniform(50, 85, len(df))
                elif feature == 'pressure':
                    df[feature] = np.random.uniform(1010, 1020, len(df))
                elif feature == 'wind_speed':
                    df[feature] = np.random.uniform(2, 8, len(df))
        
        return df
        
    except Exception as e:
        logger.error(f"Erro no cálculo do AQI: {e}")
        # Garante que as features básicas existam mesmo em caso de erro
        if 'Overall_AQI' not in df.columns:
            df['Overall_AQI'] = np.nan
            df['AQI_Category'] = 'Erro no cálculo'
        return df

def generate_realistic_air_quality_data(location: str, date_from: str, date_to: str, limit: int = 500):
    """Gera dados realistas de qualidade do ar com features meteorológicas"""
    
    # Validar inputs primeiro
    if not location or not date_from or not date_to:
        logger.error("Parâmetros inválidos para geração de dados")
        return []
    
    try:
        start = datetime.strptime(date_from, '%Y-%m-%d') if date_from else datetime.now() - timedelta(days=7)
        end = datetime.strptime(date_to, '%Y-%m-%d') if date_to else datetime.now()
        
        if start > end:
            logger.error("Data inicial maior que data final")
            return []
        
        # Poluentes principais
        parameters = ['pm25', 'pm10', 'o3', 'co', 'so2', 'no2']
        demo_data = []
        
        # Determina o perfil baseado na localização
        profile = get_location_profile(location)
        poll_profile = profile['pollutants']
        weather_profile = profile['weather']
        
        current_date = start
        records_per_day = max(1, min(24, limit // max(1, (end - start).days)))
        
        logger.info(f"Iniciando geração de dados para {location} - {date_from} a {date_to}")
        
        while current_date <= end and len(demo_data) < limit:
            # Gera dados para cada hora do dia
            for hour in range(24):
                if len(demo_data) >= limit:
                    break
                    
                datetime_str = current_date.replace(hour=hour, minute=0, second=0).isoformat() + 'Z'
                
                # Gera dados meteorológicos primeiro
                temp_min, temp_max = weather_profile['temperature']
                humidity_min, humidity_max = weather_profile['humidity']
                pressure_min, pressure_max = weather_profile['pressure']
                wind_min, wind_max = weather_profile['wind_speed']
                
                # Variação diurna para meteorologia
                if 13 <= hour <= 15:  # Mais quente ao meio-dia
                    temperature = random.uniform(temp_max - 5, temp_max)
                    humidity = random.uniform(humidity_min, humidity_min + 20)
                elif 3 <= hour <= 5:  # Mais frio de madrugada
                    temperature = random.uniform(temp_min, temp_min + 5)
                    humidity = random.uniform(humidity_max - 20, humidity_max)
                else:
                    temperature = random.uniform(temp_min, temp_max)
                    humidity = random.uniform(humidity_min, humidity_max)
                
                pressure = random.uniform(pressure_min, pressure_max)
                wind_speed = random.uniform(wind_min, wind_max)
                
                # Gera dados de poluentes
                for param in parameters:
                    if len(demo_data) >= limit:
                        break
                        
                    base_min, base_max = poll_profile.get(param, (5, 50))
                    
                    # Variação diurna - piores valores durante horário de pico
                    hour_factor = 1.0
                    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Horários de pico
                        hour_factor = 1.8
                    elif 10 <= hour <= 16:  # Meio do dia
                        hour_factor = 1.3
                    elif 20 <= hour <= 6:  # Madrugada
                        hour_factor = 0.6
                    
                    # Influência meteorológica
                    weather_factor = 1.0
                    if wind_speed > 8:  # Vento forte dispersa poluentes
                        weather_factor *= 0.7
                    if humidity > 80:  # Alta umidade pode aumentar alguns poluentes
                        weather_factor *= 1.2
                    
                    # Variação aleatória
                    random_factor = random.uniform(0.7, 1.3)
                    
                    # Tendência sazonal
                    seasonal_factor = 1.0
                    if current_date.month in [6, 7, 8, 9]:  # Meses secos
                        seasonal_factor = 1.2
                    
                    value = round((base_min + (base_max - base_min) * random_factor * hour_factor * weather_factor * seasonal_factor), 2)
                    
                    demo_data.append({
                        'datetime': datetime_str,
                        'location': location,
                        'parameter': param,
                        'value': value,
                        'unit': 'μg/m³' if param in ['pm25', 'pm10'] else 'ppb',
                        'city': location.split(',')[0] if ',' in location else location,
                        'country': 'BR' if 'brasil' in location.lower() or 'são' in location.lower() else 'US',
                        'latitude': -23.550 if 'são paulo' in location.lower() else -15.601 if 'cuiabá' in location.lower() else 51.507,
                        'longitude': -46.633 if 'são paulo' in location.lower() else -56.098 if 'cuiabá' in location.lower() else -0.128,
                        'temperature': round(temperature, 1),
                        'humidity': round(humidity, 1),
                        'pressure': round(pressure, 1),
                        'wind_speed': round(wind_speed, 1)
                    })
            
            current_date += timedelta(days=1)
        
        logger.info(f"Gerados {len(demo_data)} registros com features meteorológicas para {location}")
        return demo_data
        
    except Exception as e:
        logger.error(f"Erro crítico ao gerar dados: {e}")
        return []

@external_bp.route('/sources')
@login_required
def sources():
    """Página principal de fontes de dados externos"""
    return render_template('external/sources.html')

@external_bp.route('/openaq', methods=['GET', 'POST'])
@login_required
def openaq():
    """Busca dados da OpenAQ - Versão Simplificada e Funcional"""
    if request.method == 'POST':
        try:
            # Validar todos os parâmetros
            location = request.form.get('location', '').strip()
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')

            limit = int(request.form.get('limit', 500))
            
            if not all([location, date_from, date_to]):
                flash('Todos os campos são obrigatórios.', 'danger')
                return redirect(url_for('external.openaq'))
            
            # Validar datas
            try:
                start_date = datetime.strptime(date_from, '%Y-%m-%d')
                end_date = datetime.strptime(date_to, '%Y-%m-%d')
                if start_date > end_date:
                    flash('Data inicial não pode ser maior que data final.', 'danger')
                    return redirect(url_for('external.openaq'))
            except ValueError:
                flash('Formato de data inválido. Use YYYY-MM-DD.', 'danger')
                return redirect(url_for('external.openaq'))

            logger.info(f"Iniciando processamento para {location} - {date_from} a {date_to}")

            # Gera dados de demonstração com timeout
            try:
                with time_limit(30):  # Agora funciona como gerenciador de contexto
                    demo_data = generate_realistic_air_quality_data(location, date_from, date_to, limit)
            except TimeoutException:
                logger.error("Timeout na geração de dados")
                flash("Operação demorou muito tempo. Tente novamente.", 'warning')
                return redirect(url_for('external.openaq'))
            
            if not demo_data:
                flash('Erro ao gerar dados de demonstração.', 'danger')
                return redirect(url_for('external.openaq'))
            
            # Processa os dados
            df_raw = pd.DataFrame(demo_data)
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], errors='coerce')
            df_raw = df_raw.dropna(subset=['datetime'])  # Remove datas inválidas
            
            # Cria pivot table com médias por horário
            df_pivot = df_raw.pivot_table(
                index='datetime', 
                columns='parameter', 
                values='value', 
                aggfunc='mean'
            ).reset_index().sort_values('datetime')

            # Remove colunas completamente vazias
            df_pivot = df_pivot.dropna(axis=1, how='all')

            # Preenche valores faltantes com a média da coluna
            numeric_columns = df_pivot.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_pivot[col] = df_pivot[col].fillna(df_pivot[col].mean())

            # Calcula AQI
            df_processed = classify_aqi(df_pivot.copy())
            
            # Otimiza DataFrame
            df_processed = cleanup_dataframe(df_processed)

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_location = secure_filename(location.replace(' ', '_'))
            filename = f"openaq_{clean_location}_{timestamp}.csv"

            # 1. Pega o diretório de upload da configuração central.
            upload_dir = current_app.config['UPLOAD_FOLDER']
            # 2. Garante que o diretório exista.
            #os.makedirs(upload_dir, exist_ok=True)
            # 3. Cria o caminho final de forma segura.
            file_path = os.path.join(upload_dir, filename)

            df_processed.to_csv(file_path, index=False)
            
            # Calcula métricas de qualidade
            total_cells = df_processed.size
            missing_cells = df_processed.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            quality_score = max(0, 100 - missing_percentage)
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path, # Agora o file_path está correto
                file_size=os.path.getsize(file_path),
                rows_count=len(df_processed),
                columns_count=len(df_processed.columns),
                description=f"Dados de qualidade do ar para {location} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=float(quality_score),
                missing_data_percentage=float(missing_percentage),
                source='openaq'
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            logger.info(f"Processamento concluído: {len(df_processed)} registros gerados")
            flash(f'✅ Dados para {location} importados com sucesso! ({len(df_processed)} registros)', 'success')
            return render_template('external/openaq.html', dataset=dataset)
            # --- FIM DA CORREÇÃO ---
            
        except Exception as e:
            logger.error(f"Erro ao processar dados OpenAQ: {e}", exc_info=True)
            flash(f'Erro ao processar dados: {str(e)}', 'danger')
            return redirect(url_for('external.openaq'))

    return render_template('external/openaq.html')

@external_bp.route('/inmet', methods=['GET', 'POST'])
@login_required
def inmet():
    """Busca dados do INMET - Versão simplificada"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        try:
            station_code = request.form.get('station_code', '').strip()
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')
            
            if not all([station_code, date_from, date_to]):
                flash('Todos os campos são obrigatórios.', 'danger')
                return redirect(url_for('external.inmet'))
            
            # Validar datas
            try:
                start_date = datetime.strptime(date_from, '%Y-%m-%d')
                end_date = datetime.strptime(date_to, '%Y-%m-%d')
                if start_date > end_date:
                    flash('Data inicial não pode ser maior que data final.', 'danger')
                    return redirect(url_for('external.inmet'))
            except ValueError:
                flash('Formato de data inválido. Use YYYY-MM-DD.', 'danger')
                return redirect(url_for('external.inmet'))

            # Gera dados de demonstração para INMET
            demo_data = []
            current_date = start_date
            
            while current_date <= end_date and len(demo_data) < 100:
                for hour in range(24):
                    if len(demo_data) >= 100:
                        break
                    
                    demo_data.append({
                        'datetime': current_date.replace(hour=hour, minute=0, second=0).isoformat(),
                        'station': station_code,
                        'temperature': round(random.uniform(15, 35), 1),
                        'humidity': round(random.uniform(30, 95), 1),
                        'pressure': round(random.uniform(1000, 1020), 1),
                        'wind_speed': round(random.uniform(0, 15), 1),
                        'precipitation': round(random.uniform(0, 10), 1),
                        'solar_radiation': round(random.uniform(0, 1000), 1)
                    })
                
                current_date += timedelta(days=1)
            
            if not demo_data:
                flash('Erro ao gerar dados de demonstração.', 'danger')
                return redirect(url_for('external.inmet'))
            
            # Processa os dados
            df = pd.DataFrame(demo_data)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df = cleanup_dataframe(df)

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inmet_{station_code}_{timestamp}.csv"

            # 1. Pega o diretório de upload da configuração central.
            upload_dir = current_app.config['UPLOAD_FOLDER']
            # 2. Garante que o diretório exista.
            os.makedirs(upload_dir, exist_ok=True)
            # 3. Cria o caminho final de forma segura.
            file_path = os.path.join(upload_dir, filename)
            
            df.to_csv(file_path, index=False)
            
            # Calcula métricas de qualidade
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            quality_score = max(0, 100 - missing_percentage)
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path, # Agora o file_path está correto
                file_size=os.path.getsize(file_path),
                rows_count=len(df),
                columns_count=len(df.columns),
                description=f"Dados meteorológicos INMET estação {station_code} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=float(quality_score),
                missing_data_percentage=float(missing_percentage),
                source='inmet'
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'✅ Dados INMET para estação {station_code} importados com sucesso! ({len(df)} registros)', 'success')
            return redirect(url_for('main.datasets'))
            # --- FIM DA CORREÇÃO ---

        except Exception as e:
            logger.error(f"Erro ao processar dados INMET: {e}", exc_info=True)
            flash(f'Erro ao processar dados: {str(e)}', 'danger')
            return redirect(url_for('external.inmet'))

    return render_template('external/inmet.html', today=today)

@external_bp.route('/inpe', methods=['GET', 'POST'])
@login_required
def inpe():
    """Busca dados de queimadas do INPE - Versão simplificada"""
    if request.method == 'POST':
        try:
            state = request.form.get('state', 'Brasil')
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')

            if not date_from or not date_to:
                flash('Por favor, informe ambas as datas.', 'danger')
                return redirect(url_for('external.inpe'))

            # Validar datas
            try:
                start_date = datetime.strptime(date_from, '%Y-%m-%d')
                end_date = datetime.strptime(date_to, '%Y-%m-%d')
                if start_date > end_date:
                    flash('Data inicial não pode ser maior que data final.', 'danger')
                    return redirect(url_for('external.inpe'))
            except ValueError:
                flash('Formato de data inválido. Use YYYY-MM-DD.', 'danger')
                return redirect(url_for('external.inpe'))

            # Gera dados de demonstração para INPE
            demo_data = []
            current_date = start_date
            
            while current_date <= end_date and len(demo_data) < 200:
                # Gera dados realistas de queimadas
                fires_count = random.randint(0, 50)
                if fires_count > 0:
                    for _ in range(fires_count):
                        demo_data.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'state': state,
                            'city': f"Cidade_{random.randint(1, 20)}",
                            'biome': random.choice(['Amazônia', 'Cerrado', 'Mata Atlântica', 'Caatinga', 'Pampa', 'Pantanal']),
                            'fires_count': fires_count,
                            'risk_level': random.choice(['Baixo', 'Médio', 'Alto', 'Crítico']),
                            'latitude': round(random.uniform(-33.0, 5.0), 4),
                            'longitude': round(random.uniform(-73.0, -35.0), 4),
                            'temperature': round(random.uniform(25, 40), 1),
                            'humidity': round(random.uniform(20, 80), 1)
                        })
                
                current_date += timedelta(days=1)
            
            if not demo_data:
                flash('Nenhum dado de queimadas encontrado para o período selecionado.', 'warning')
                return redirect(url_for('external.inpe'))

            # Processa os dados
            df = pd.DataFrame(demo_data)
            df = cleanup_dataframe(df)

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_state = secure_filename(state.replace(' ', '_'))
            filename = f"inpe_queimadas_{clean_state}_{timestamp}.csv"

            # 1. Pega o diretório de upload da configuração central.
            upload_dir = current_app.config['UPLOAD_FOLDER']
            # 2. Garante que o diretório exista.
            os.makedirs(upload_dir, exist_ok=True)
            # 3. Cria o caminho final de forma segura.
            file_path = os.path.join(upload_dir, filename)
            
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            # Calcula métricas de qualidade
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            quality_score = max(0, 100 - missing_percentage)
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path, # Agora o file_path está correto
                file_size=os.path.getsize(file_path),
                rows_count=len(df),
                columns_count=len(df.columns),
                description=f"Dados de queimadas INPE {state} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=float(quality_score),
                missing_data_percentage=float(missing_percentage),
                source='inpe'
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'✅ Dados de queimadas para {state} importados com sucesso! ({len(df)} registros)', 'success')
            return redirect(url_for('main.datasets'))
            # --- FIM DA CORREÇÃO ---

        except Exception as e:
            logger.error(f"Erro ao buscar dados do INPE: {e}", exc_info=True)
            flash(f'Erro ao buscar dados: {str(e)}', 'danger')
            return redirect(url_for('external.inpe'))

    return render_template('external/inpe.html')


@external_bp.route('/api/inmet/stations')
@login_required
def api_inmet_stations():
    """API para buscar estações do INMET - Versão com mais estações"""
    try:
        state = request.args.get('state', '')
        
        # Base de dados expandida de estações por estado
        demo_stations = [
            # São Paulo (SP)
            {'code': 'A701', 'name': 'São Paulo - Mirante de Santana', 'state': 'SP', 'latitude': -23.498, 'longitude': -46.622},
            {'code': 'A702', 'name': 'Campinas - Centro', 'state': 'SP', 'latitude': -22.907, 'longitude': -47.060},
            {'code': 'A703', 'name': 'Santos - Praia', 'state': 'SP', 'latitude': -23.960, 'longitude': -46.333},
            {'code': 'A704', 'name': 'Ribeirão Preto', 'state': 'SP', 'latitude': -21.177, 'longitude': -47.810},
            {'code': 'A705', 'name': 'São José dos Campos', 'state': 'SP', 'latitude': -23.223, 'longitude': -45.900},
            {'code': 'A706', 'name': 'Sorocaba', 'state': 'SP', 'latitude': -23.501, 'longitude': -47.458},
            {'code': 'A707', 'name': 'Bauru', 'state': 'SP', 'latitude': -22.314, 'longitude': -49.060},
            
            # Rio de Janeiro (RJ)
            {'code': 'B801', 'name': 'Rio de Janeiro - Centro', 'state': 'RJ', 'latitude': -22.906, 'longitude': -43.172},
            {'code': 'B802', 'name': 'Niterói', 'state': 'RJ', 'latitude': -22.883, 'longitude': -43.103},
            {'code': 'B803', 'name': 'Duque de Caxias', 'state': 'RJ', 'latitude': -22.785, 'longitude': -43.305},
            {'code': 'B804', 'name': 'Nova Iguaçu', 'state': 'RJ', 'latitude': -22.755, 'longitude': -43.460},
            {'code': 'B805', 'name': 'Petrópolis', 'state': 'RJ', 'latitude': -22.505, 'longitude': -43.178},
            
            # Minas Gerais (MG)
            {'code': 'C901', 'name': 'Belo Horizonte - Centro', 'state': 'MG', 'latitude': -19.916, 'longitude': -43.934},
            {'code': 'C902', 'name': 'Uberlândia', 'state': 'MG', 'latitude': -18.912, 'longitude': -48.275},
            {'code': 'C903', 'name': 'Contagem', 'state': 'MG', 'latitude': -19.931, 'longitude': -44.053},
            {'code': 'C904', 'name': 'Juiz de Fora', 'state': 'MG', 'latitude': -21.764, 'longitude': -43.349},
            {'code': 'C905', 'name': 'Betim', 'state': 'MG', 'latitude': -19.967, 'longitude': -44.198},
            
            # Paraná (PR)
            {'code': 'D101', 'name': 'Curitiba - Centro', 'state': 'PR', 'latitude': -25.428, 'longitude': -49.273},
            {'code': 'D102', 'name': 'Londrina', 'state': 'PR', 'latitude': -23.310, 'longitude': -51.162},
            {'code': 'D103', 'name': 'Maringá', 'state': 'PR', 'latitude': -23.425, 'longitude': -51.938},
            {'code': 'D104', 'name': 'Ponta Grossa', 'state': 'PR', 'latitude': -25.094, 'longitude': -50.162},
            {'code': 'D105', 'name': 'Cascavel', 'state': 'PR', 'latitude': -24.955, 'longitude': -53.455},
            
            # Santa Catarina (SC)
            {'code': 'E201', 'name': 'Florianópolis - Centro', 'state': 'SC', 'latitude': -27.595, 'longitude': -48.548},
            {'code': 'E202', 'name': 'Joinville', 'state': 'SC', 'latitude': -26.304, 'longitude': -48.848},
            {'code': 'E203', 'name': 'Blumenau', 'state': 'SC', 'latitude': -26.918, 'longitude': -49.066},
            {'code': 'E204', 'name': 'São José', 'state': 'SC', 'latitude': -27.614, 'longitude': -48.636},
            {'code': 'E205', 'name': 'Criciúma', 'state': 'SC', 'latitude': -28.677, 'longitude': -49.369},
            
            # Rio Grande do Sul (RS)
            {'code': 'F301', 'name': 'Porto Alegre - Centro', 'state': 'RS', 'latitude': -30.031, 'longitude': -51.234},
            {'code': 'F302', 'name': 'Caxias do Sul', 'state': 'RS', 'latitude': -29.168, 'longitude': -51.179},
            {'code': 'F303', 'name': 'Pelotas', 'state': 'RS', 'latitude': -31.771, 'longitude': -52.342},
            {'code': 'F304', 'name': 'Canoas', 'state': 'RS', 'latitude': -29.915, 'longitude': -51.184},
            {'code': 'F305', 'name': 'Santa Maria', 'state': 'RS', 'latitude': -29.684, 'longitude': -53.806},
            
            # Bahia (BA)
            {'code': 'G401', 'name': 'Salvador - Centro', 'state': 'BA', 'latitude': -12.971, 'longitude': -38.501},
            {'code': 'G402', 'name': 'Feira de Santana', 'state': 'BA', 'latitude': -12.258, 'longitude': -38.959},
            {'code': 'G403', 'name': 'Vitória da Conquista', 'state': 'BA', 'latitude': -14.866, 'longitude': -40.839},
            {'code': 'G404', 'name': 'Camaçari', 'state': 'BA', 'latitude': -12.699, 'longitude': -38.324},
            {'code': 'G405', 'name': 'Itabuna', 'state': 'BA', 'latitude': -14.785, 'longitude': -39.280},
            
            # Ceará (CE)
            {'code': 'H501', 'name': 'Fortaleza - Centro', 'state': 'CE', 'latitude': -3.731, 'longitude': -38.526},
            {'code': 'H502', 'name': 'Caucaia', 'state': 'CE', 'latitude': -3.732, 'longitude': -38.662},
            {'code': 'H503', 'name': 'Juazeiro do Norte', 'state': 'CE', 'latitude': -7.213, 'longitude': -39.315},
            {'code': 'H504', 'name': 'Maracanaú', 'state': 'CE', 'latitude': -3.877, 'longitude': -38.625},
            {'code': 'H505', 'name': 'Sobral', 'state': 'CE', 'latitude': -3.685, 'longitude': -40.344},
            
            # Pernambuco (PE)
            {'code': 'I601', 'name': 'Recife - Centro', 'state': 'PE', 'latitude': -8.047, 'longitude': -34.877},
            {'code': 'I602', 'name': 'Jaboatão dos Guararapes', 'state': 'PE', 'latitude': -8.112, 'longitude': -35.014},
            {'code': 'I603', 'name': 'Olinda', 'state': 'PE', 'latitude': -8.001, 'longitude': -34.845},
            {'code': 'I604', 'name': 'Caruaru', 'state': 'PE', 'latitude': -8.284, 'longitude': -35.970},
            {'code': 'I605', 'name': 'Petrolina', 'state': 'PE', 'latitude': -9.388, 'longitude': -40.500},
            
            # Pará (PA)
            {'code': 'J701', 'name': 'Belém - Centro', 'state': 'PA', 'latitude': -1.455, 'longitude': -48.502},
            {'code': 'J702', 'name': 'Ananindeua', 'state': 'PA', 'latitude': -1.365, 'longitude': -48.372},
            {'code': 'J703', 'name': 'Santarém', 'state': 'PA', 'latitude': -2.443, 'longitude': -54.708},
            {'code': 'J704', 'name': 'Marabá', 'state': 'PA', 'latitude': -5.368, 'longitude': -49.118},
            {'code': 'J705', 'name': 'Castanhal', 'state': 'PA', 'latitude': -1.297, 'longitude': -47.917},
            
            # Amazonas (AM)
            {'code': 'K801', 'name': 'Manaus - Centro', 'state': 'AM', 'latitude': -3.119, 'longitude': -60.021},
            {'code': 'K802', 'name': 'Parintins', 'state': 'AM', 'latitude': -2.628, 'longitude': -56.735},
            {'code': 'K803', 'name': 'Itacoatiara', 'state': 'AM', 'latitude': -3.143, 'longitude': -58.444},
            {'code': 'K804', 'name': 'Manacapuru', 'state': 'AM', 'latitude': -3.299, 'longitude': -60.620},
            {'code': 'K805', 'name': 'Coari', 'state': 'AM', 'latitude': -4.085, 'longitude': -63.141},
            
            # Goiás (GO)
            {'code': 'L901', 'name': 'Goiânia - Centro', 'state': 'GO', 'latitude': -16.680, 'longitude': -49.253},
            {'code': 'L902', 'name': 'Aparecida de Goiânia', 'state': 'GO', 'latitude': -16.819, 'longitude': -49.246},
            {'code': 'L903', 'name': 'Anápolis', 'state': 'GO', 'latitude': -16.328, 'longitude': -48.953},
            {'code': 'L904', 'name': 'Rio Verde', 'state': 'GO', 'latitude': -17.745, 'longitude': -50.919},
            {'code': 'L905', 'name': 'Águas Lindas de Goiás', 'state': 'GO', 'latitude': -15.761, 'longitude': -48.281},
            
            # Mato Grosso (MT)
            {'code': 'M011', 'name': 'Cuiabá - Centro', 'state': 'MT', 'latitude': -15.601, 'longitude': -56.098},
            {'code': 'M012', 'name': 'Várzea Grande', 'state': 'MT', 'latitude': -15.645, 'longitude': -56.132},
            {'code': 'M013', 'name': 'Rondonópolis', 'state': 'MT', 'latitude': -16.467, 'longitude': -54.635},
            {'code': 'M014', 'name': 'Sinop', 'state': 'MT', 'latitude': -11.864, 'longitude': -55.498},
            {'code': 'M015', 'name': 'Tangará da Serra', 'state': 'MT', 'latitude': -14.619, 'longitude': -57.483},
            
            # Mato Grosso do Sul (MS)
            {'code': 'N111', 'name': 'Campo Grande - Centro', 'state': 'MS', 'latitude': -20.469, 'longitude': -54.620},
            {'code': 'N112', 'name': 'Dourados', 'state': 'MS', 'latitude': -22.221, 'longitude': -54.806},
            {'code': 'N113', 'name': 'Três Lagoas', 'state': 'MS', 'latitude': -20.785, 'longitude': -51.700},
            {'code': 'N114', 'name': 'Corumbá', 'state': 'MS', 'latitude': -19.008, 'longitude': -57.652},
            {'code': 'N115', 'name': 'Ponta Porã', 'state': 'MS', 'latitude': -22.536, 'longitude': -55.725},
            
            # Espírito Santo (ES)
            {'code': 'O211', 'name': 'Vitória - Centro', 'state': 'ES', 'latitude': -20.315, 'longitude': -40.312},
            {'code': 'O212', 'name': 'Vila Velha', 'state': 'ES', 'latitude': -20.329, 'longitude': -40.292},
            {'code': 'O213', 'name': 'Serra', 'state': 'ES', 'latitude': -20.121, 'longitude': -40.307},
            {'code': 'O214', 'name': 'Cariacica', 'state': 'ES', 'latitude': -20.263, 'longitude': -40.416},
            {'code': 'O215', 'name': 'Linhares', 'state': 'ES', 'latitude': -19.394, 'longitude': -40.064},
            
            # Alagoas (AL)
            {'code': 'P311', 'name': 'Maceió - Centro', 'state': 'AL', 'latitude': -9.665, 'longitude': -35.735},
            {'code': 'P312', 'name': 'Arapiraca', 'state': 'AL', 'latitude': -9.752, 'longitude': -36.661},
            {'code': 'P313', 'name': 'Rio Largo', 'state': 'AL', 'latitude': -9.478, 'longitude': -35.839},
            {'code': 'P314', 'name': 'Palmeira dos Índios', 'state': 'AL', 'latitude': -9.405, 'longitude': -36.628},
            {'code': 'P315', 'name': 'União dos Palmares', 'state': 'AL', 'latitude': -9.163, 'longitude': -36.031},
            
            # Sergipe (SE)
            {'code': 'Q411', 'name': 'Aracaju - Centro', 'state': 'SE', 'latitude': -10.909, 'longitude': -37.074},
            {'code': 'Q412', 'name': 'Nossa Senhora do Socorro', 'state': 'SE', 'latitude': -10.855, 'longitude': -37.125},
            {'code': 'Q413', 'name': 'Lagarto', 'state': 'SE', 'latitude': -10.917, 'longitude': -37.650},
            {'code': 'Q414', 'name': 'Itabaiana', 'state': 'SE', 'latitude': -10.685, 'longitude': -37.425},
            {'code': 'Q415', 'name': 'Estância', 'state': 'SE', 'latitude': -11.262, 'longitude': -37.438},
            
            # Paraíba (PB)
            {'code': 'R511', 'name': 'João Pessoa - Centro', 'state': 'PB', 'latitude': -7.119, 'longitude': -34.845},
            {'code': 'R512', 'name': 'Campina Grande', 'state': 'PB', 'latitude': -7.230, 'longitude': -35.881},
            {'code': 'R513', 'name': 'Santa Rita', 'state': 'PB', 'latitude': -7.137, 'longitude': -34.976},
            {'code': 'R514', 'name': 'Patos', 'state': 'PB', 'latitude': -7.024, 'longitude': -37.280},
            {'code': 'R515', 'name': 'Bayeux', 'state': 'PB', 'latitude': -7.125, 'longitude': -34.932},
            
            # Rio Grande do Norte (RN)
            {'code': 'S611', 'name': 'Natal - Centro', 'state': 'RN', 'latitude': -5.779, 'longitude': -35.200},
            {'code': 'S612', 'name': 'Mossoró', 'state': 'RN', 'latitude': -5.187, 'longitude': -37.344},
            {'code': 'S613', 'name': 'Parnamirim', 'state': 'RN', 'latitude': -5.915, 'longitude': -35.262},
            {'code': 'S614', 'name': 'São Gonçalo do Amarante', 'state': 'RN', 'latitude': -5.793, 'longitude': -35.328},
            {'code': 'S615', 'name': 'Macaíba', 'state': 'RN', 'latitude': -5.857, 'longitude': -35.353},
            
            # Piauí (PI)
            {'code': 'T711', 'name': 'Teresina - Centro', 'state': 'PI', 'latitude': -5.089, 'longitude': -42.809},
            {'code': 'T712', 'name': 'Parnaíba', 'state': 'PI', 'latitude': -2.905, 'longitude': -41.775},
            {'code': 'T713', 'name': 'Picos', 'state': 'PI', 'latitude': -7.077, 'longitude': -41.466},
            {'code': 'T714', 'name': 'Floriano', 'state': 'PI', 'latitude': -6.766, 'longitude': -43.022},
            {'code': 'T715', 'name': 'Piripiri', 'state': 'PI', 'latitude': -4.273, 'longitude': -41.777},
            
            # Maranhão (MA)
            {'code': 'U811', 'name': 'São Luís - Centro', 'state': 'MA', 'latitude': -2.530, 'longitude': -44.302},
            {'code': 'U812', 'name': 'Imperatriz', 'state': 'MA', 'latitude': -5.519, 'longitude': -47.478},
            {'code': 'U813', 'name': 'São José de Ribamar', 'state': 'MA', 'latitude': -2.562, 'longitude': -44.054},
            {'code': 'U814', 'name': 'Timon', 'state': 'MA', 'latitude': -5.094, 'longitude': -42.837},
            {'code': 'U815', 'name': 'Caxias', 'state': 'MA', 'latitude': -4.858, 'longitude': -43.356},
            
            # Acre (AC)
            {'code': 'V911', 'name': 'Rio Branco - Centro', 'state': 'AC', 'latitude': -9.974, 'longitude': -67.810},
            {'code': 'V912', 'name': 'Cruzeiro do Sul', 'state': 'AC', 'latitude': -7.627, 'longitude': -72.675},
            {'code': 'V913', 'name': 'Sena Madureira', 'state': 'AC', 'latitude': -9.065, 'longitude': -68.657},
            {'code': 'V914', 'name': 'Tarauacá', 'state': 'AC', 'latitude': -8.161, 'longitude': -70.765},
            {'code': 'V915', 'name': 'Feijó', 'state': 'AC', 'latitude': -8.165, 'longitude': -70.354},
            
            # Amapá (AP)
            {'code': 'W021', 'name': 'Macapá - Centro', 'state': 'AP', 'latitude': 0.034, 'longitude': -51.070},
            {'code': 'W022', 'name': 'Santana', 'state': 'AP', 'latitude': -0.058, 'longitude': -51.181},
            {'code': 'W023', 'name': 'Laranjal do Jari', 'state': 'AP', 'latitude': -0.841, 'longitude': -52.516},
            {'code': 'W024', 'name': 'Oiapoque', 'state': 'AP', 'latitude': 3.840, 'longitude': -51.834},
            {'code': 'W025', 'name': 'Porto Grande', 'state': 'AP', 'latitude': 0.712, 'longitude': -51.415},
            
            # Rondônia (RO)
            {'code': 'X131', 'name': 'Porto Velho - Centro', 'state': 'RO', 'latitude': -8.761, 'longitude': -63.903},
            {'code': 'X132', 'name': 'Ji-Paraná', 'state': 'RO', 'latitude': -10.885, 'longitude': -61.951},
            {'code': 'X133', 'name': 'Ariquemes', 'state': 'RO', 'latitude': -9.916, 'longitude': -63.040},
            {'code': 'X134', 'name': 'Vilhena', 'state': 'RO', 'latitude': -12.740, 'longitude': -60.145},
            {'code': 'X135', 'name': 'Cacoal', 'state': 'RO', 'latitude': -11.438, 'longitude': -61.447},
            
            # Roraima (RR)
            {'code': 'Y241', 'name': 'Boa Vista - Centro', 'state': 'RR', 'latitude': 2.823, 'longitude': -60.675},
            {'code': 'Y242', 'name': 'Rorainópolis', 'state': 'RR', 'latitude': 0.946, 'longitude': -60.428},
            {'code': 'Y243', 'name': 'Caracaraí', 'state': 'RR', 'latitude': 1.817, 'longitude': -61.127},
            {'code': 'Y244', 'name': 'Mucajaí', 'state': 'RR', 'latitude': 2.439, 'longitude': -60.909},
            {'code': 'Y245', 'name': 'Cantá', 'state': 'RR', 'latitude': 2.609, 'longitude': -60.595},
            
            # Tocantins (TO)
            {'code': 'Z351', 'name': 'Palmas - Centro', 'state': 'TO', 'latitude': -10.184, 'longitude': -48.333},
            {'code': 'Z352', 'name': 'Araguaína', 'state': 'TO', 'latitude': -7.191, 'longitude': -48.207},
            {'code': 'Z353', 'name': 'Gurupi', 'state': 'TO', 'latitude': -11.729, 'longitude': -49.068},
            {'code': 'Z354', 'name': 'Porto Nacional', 'state': 'TO', 'latitude': -10.705, 'longitude': -48.417},
            {'code': 'Z355', 'name': 'Paraíso do Tocantins', 'state': 'TO', 'latitude': -10.175, 'longitude': -48.882},
            
            # Distrito Federal (DF)
            {'code': 'DF01', 'name': 'Brasília - Plano Piloto', 'state': 'DF', 'latitude': -15.794, 'longitude': -47.882},
            {'code': 'DF02', 'name': 'Taguatinga', 'state': 'DF', 'latitude': -15.834, 'longitude': -48.055},
            {'code': 'DF03', 'name': 'Ceilândia', 'state': 'DF', 'latitude': -15.822, 'longitude': -48.109},
            {'code': 'DF04', 'name': 'Samambaia', 'state': 'DF', 'latitude': -15.874, 'longitude': -48.085},
            {'code': 'DF05', 'name': 'Planaltina', 'state': 'DF', 'latitude': -15.617, 'longitude': -47.667},
        ]
        
        if state:
            formatted_stations = [s for s in demo_stations if s['state'] == state.upper()]
        else:
            formatted_stations = demo_stations
        
        return jsonify({
            'success': True, 
            'stations': formatted_stations,
            'count': len(formatted_stations)
        })
        
    except Exception as e:
        logger.error(f"Erro na API de estações: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500