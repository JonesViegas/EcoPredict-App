from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, send_file
from flask import current_app
from flask_login import login_required, current_user
from app.utils import admin_required, allowed_file, process_uploaded_file 
from app import db
from app.models import User, Dataset, MLModel, Alert, SystemLog, AirQualityData, log_system_event
from app.forms import DatasetUploadForm, MLModelForm, AirQualityDataForm
from app.utils import allowed_file, process_uploaded_file
from app.ml_models import train_model, make_prediction, evaluate_model
from app.data_processing import calculate_correlations, generate_statistics, clean_dataset
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
import logging
import flask 
from app.analysis import (
    generate_openaq_analysis, 
    generate_inpe_analysis, 
    generate_inmet_analysis, 
    generate_generic_analysis
)
from app.advanced_analysis import (
    run_kmeans_clustering,
    run_pca_analysis,
    run_classification_analysis,
    run_regression_analysis
)

main_bp = Blueprint('main', __name__)

logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal usando dados dos datasets"""
    try:
        import pandas as pd
        import os
        from datetime import datetime, timedelta
        
        # Get basic statistics
        user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
        total_datasets = len(user_datasets)
        
        # Tentar buscar modelos ML (se a tabela existir)
        try:
            total_models = MLModel.query.filter_by(user_id=current_user.id).count()
            active_models = MLModel.query.filter_by(user_id=current_user.id, is_active=True).count()
        except:
            total_models = 0
            active_models = 0
            
        public_datasets = Dataset.query.filter_by(is_public=True).count()
        
        # Analisar datasets para extrair dados de qualidade do ar
        all_aqi_values = []
        all_pm25_values = []
        recent_data = []
        pollutant_levels = {
            'pm25': 0, 'pm10': 0, 'o3': 0, 'no2': 0,
            'so2': 0, 'co': 0
        }
        
        # Processar cada dataset do usuário
        for dataset in user_datasets[-5:]:  # Últimos 5 datasets
            try:
                if os.path.exists(dataset.file_path):
                    df = pd.read_csv(dataset.file_path)
                    
                    # Extrair AQI
                    if 'Overall_AQI' in df.columns:
                        aqi_vals = df['Overall_AQI'].dropna().tolist()
                        all_aqi_values.extend(aqi_vals)
                    
                    # Extrair PM2.5
                    if 'pm25' in df.columns:
                        pm25_vals = df['pm25'].dropna().tolist()
                        all_pm25_values.extend(pm25_vals)
                        pollutant_levels['pm25'] = np.mean(pm25_vals) if pm25_vals else 0
                    
                    # Extrair outros poluentes
                    for pollutant in ['pm10', 'o3', 'no2', 'so2', 'co']:
                        if pollutant in df.columns:
                            vals = df[pollutant].dropna().tolist()
                            if vals:
                                pollutant_levels[pollutant] = np.mean(vals)
                    
                    # Criar dados recentes a partir do dataset
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        location = dataset.description or "Dataset Importado"
                        
                        # Tentar extrair data/hora
                        timestamp = datetime.now()
                        if 'datetime' in df.columns:
                            try:
                                timestamp = pd.to_datetime(last_row['datetime']).to_pydatetime()
                            except:
                                pass
                        
                        recent_data.append({
                            'location': location,
                            'aqi': last_row['Overall_AQI'] if 'Overall_AQI' in df.columns else 50,
                            'pm25': last_row['pm25'] if 'pm25' in df.columns else 25,
                            'temperature': last_row['temperature'] if 'temperature' in df.columns else 22,
                            'timestamp': timestamp,
                            'alert': False
                        })
                        
            except Exception as e:
                logger.warning(f"Erro ao processar dataset {dataset.id}: {e}")
                continue
        
        # Se não conseguiu dados dos datasets, usar dados de demonstração
        if not all_aqi_values and not recent_data:
            logger.info("Usando dados de demonstração para dashboard")
            all_aqi_values = [45, 68, 52, 78, 34, 89, 152, 67, 43, 91]
            all_pm25_values = [12, 25, 18, 30, 10, 35, 65, 20, 15, 40]
            recent_data = [
                {
                    'location': 'São Paulo - Dataset Demo',
                    'aqi': 45.2,
                    'pm25': 12.5,
                    'temperature': 22.5,
                    'timestamp': datetime.now(),
                    'alert': False
                },
                {
                    'location': 'Rio de Janeiro - Dataset Demo',
                    'aqi': 68.7,
                    'pm25': 25.3,
                    'temperature': 28.1,
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'alert': False
                },
                {
                    'location': 'Belo Horizonte - Dataset Demo',
                    'aqi': 152.3,
                    'pm25': 65.8,
                    'temperature': 24.2,
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'alert': True
                }
            ]
            pollutant_levels = {
                'pm25': 25.3, 'pm10': 45.2, 'o3': 35.1, 
                'no2': 18.7, 'so2': 8.4, 'co': 2.1
            }
        
        # Calcular estatísticas AQI
        avg_aqi = np.mean(all_aqi_values) if all_aqi_values else 0
        max_aqi = max(all_aqi_values) if all_aqi_values else 0
        min_aqi = min(all_aqi_values) if all_aqi_values else 0
        
        # Get AQI distribution for chart
        aqi_categories = {
            'Excelente (0-50)': len([x for x in all_aqi_values if 0 <= x <= 50]),
            'Bom (51-100)': len([x for x in all_aqi_values if 51 <= x <= 100]),
            'Moderado (101-150)': len([x for x in all_aqi_values if 101 <= x <= 150]),
            'Ruim (151-200)': len([x for x in all_aqi_values if 151 <= x <= 200]),
            'Muito Ruim (201-300)': len([x for x in all_aqi_values if 201 <= x <= 300]),
            'Perigoso (301+)': len([x for x in all_aqi_values if x > 300])
        }
        
        # Calcular PM2.5 médio
        avg_pm25 = np.mean(all_pm25_values) if all_pm25_values else 0
        
        # Alertas baseados nos dados
        active_alerts = []
        high_aqi_locations = []
        
        for data in recent_data:
            if data['aqi'] > 150:  # AQI acima de 150 = alerta
                high_aqi_locations.append(data['location'])
        
        if high_aqi_locations:
            active_alerts.append({
                'id': 1,
                'location': ', '.join(high_aqi_locations[:2]),
                'message': f'AQI crítico em {len(high_aqi_locations)} localidade(s)',
                'severity': 'high',
                'timestamp': datetime.now()
            })
        
        # Precisão de previsão (baseada na qualidade dos dados)
        if all_aqi_values:
            data_quality_score = min(95, (len(all_aqi_values) / 100) * 100)  # Simulação
            prediction_accuracy = max(75, data_quality_score - 10)
        else:
            prediction_accuracy = 0
        
        # Dados para gráfico de tendência
        trend_labels = []
        trend_data = []
        
        # Gerar tendência baseada nos dados reais
        hours = 6
        for i in range(hours):
            hour = (datetime.now() - timedelta(hours=hours-1-i)).strftime('%H:%M')
            trend_labels.append(hour)
            
            if all_aqi_values:
                # Usar variação baseada nos dados reais
                base_value = avg_aqi
                variation = (i - (hours/2)) * (max_aqi - min_aqi) / 20
                trend_value = max(0, base_value + variation)
                trend_data.append(round(trend_value, 1))
            else:
                # Dados de demonstração
                demo_data = [45, 52, 48, 65, 70, 55]
                trend_data.append(demo_data[i])
        
        # Limitar dados recentes
        recent_data = recent_data[:10]
        
        return render_template('dashboard.html',
                            total_datasets=total_datasets,
                            total_models=total_models,
                            active_models=active_models,
                            recent_data=recent_data,
                            avg_aqi=round(avg_aqi, 1),
                            max_aqi=round(max_aqi, 1),
                            min_aqi=round(min_aqi, 1),
                            aqi_categories=aqi_categories,
                            avg_pm25=round(avg_pm25, 1),
                            pollutant_levels=pollutant_levels,
                            active_alerts=active_alerts,
                            prediction_accuracy=round(prediction_accuracy, 1),
                            trend_labels=trend_labels,
                            trend_data=trend_data,
                            public_datasets=public_datasets)
    
    except Exception as e:
        logger.error(f"Erro no dashboard: {e}")
        
        # Fallback com dados de demonstração
        return render_template('dashboard.html',
                            total_datasets=0,
                            total_models=0,
                            active_models=0,
                            recent_data=[
                                {
                                    'location': 'Sistema em Configuração',
                                    'aqi': 45,
                                    'pm25': 25,
                                    'temperature': 22,
                                    'timestamp': datetime.now(),
                                    'alert': False
                                }
                            ],
                            avg_aqi=45.0,
                            max_aqi=65.0,
                            min_aqi=25.0,
                            aqi_categories={
                                'Excelente (0-50)': 3,
                                'Bom (51-100)': 2,
                                'Moderado (101-150)': 1,
                                'Ruim (151-200)': 0,
                                'Muito Ruim (201-300)': 0,
                                'Perigoso (301+)': 0
                            },
                            avg_pm25=25.0,
                            pollutant_levels={
                                'pm25': 25.0, 'pm10': 45.0, 'o3': 35.0, 
                                'no2': 18.0, 'so2': 8.0, 'co': 2.0
                            },
                            active_alerts=[],
                            prediction_accuracy=80.0,
                            trend_labels=['08:00', '10:00', '12:00', '14:00', '16:00', '18:00'],
                            trend_data=[45, 52, 48, 65, 70, 55],
                            public_datasets=0)
    
@main_bp.route('/map')
@login_required
def map_view():
    """Mapa interativo da qualidade do ar"""
    try:
        # Buscar dados das cidades dos datasets do usuário
        user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
        
        # Dados padrão de TODAS as capitais brasileiras
        cities_data = {
            # Região Norte
            'Manaus': {'lat': -3.119, 'lon': -60.021, 'aqi': 28, 'pm25': 7.3, 'temp': 32, 'humidity': 85, 'pressure': 1009},
            'Belém': {'lat': -1.455, 'lon': -48.502, 'aqi': 35, 'pm25': 12.1, 'temp': 28, 'humidity': 88, 'pressure': 1011},
            'Porto Velho': {'lat': -8.761, 'lon': -63.903, 'aqi': 32, 'pm25': 9.8, 'temp': 30, 'humidity': 82, 'pressure': 1010},
            'Rio Branco': {'lat': -9.974, 'lon': -67.807, 'aqi': 25, 'pm25': 6.5, 'temp': 29, 'humidity': 80, 'pressure': 1011},
            'Boa Vista': {'lat': 2.823, 'lon': -60.675, 'aqi': 22, 'pm25': 5.2, 'temp': 31, 'humidity': 75, 'pressure': 1012},
            'Macapá': {'lat': 0.034, 'lon': -51.069, 'aqi': 30, 'pm25': 8.7, 'temp': 27, 'humidity': 86, 'pressure': 1011},
            'Palmas': {'lat': -10.184, 'lon': -48.333, 'aqi': 26, 'pm25': 7.1, 'temp': 29, 'humidity': 70, 'pressure': 1013},
            
            # Região Nordeste
            'Salvador': {'lat': -12.971, 'lon': -38.501, 'aqi': 42, 'pm25': 15.8, 'temp': 29, 'humidity': 75, 'pressure': 1011},
            'Fortaleza': {'lat': -3.731, 'lon': -38.526, 'aqi': 35, 'pm25': 8.9, 'temp': 30, 'humidity': 80, 'pressure': 1010},
            'Recife': {'lat': -8.047, 'lon': -34.877, 'aqi': 55, 'pm25': 20.5, 'temp': 27, 'humidity': 78, 'pressure': 1011},
            'São Luís': {'lat': -2.539, 'lon': -44.282, 'aqi': 38, 'pm25': 13.2, 'temp': 28, 'humidity': 83, 'pressure': 1012},
            'Maceió': {'lat': -9.665, 'lon': -35.735, 'aqi': 45, 'pm25': 16.3, 'temp': 26, 'humidity': 77, 'pressure': 1013},
            'Natal': {'lat': -5.779, 'lon': -35.200, 'aqi': 33, 'pm25': 11.4, 'temp': 28, 'humidity': 79, 'pressure': 1012},
            'João Pessoa': {'lat': -7.119, 'lon': -34.845, 'aqi': 40, 'pm25': 14.7, 'temp': 27, 'humidity': 76, 'pressure': 1013},
            'Teresina': {'lat': -5.089, 'lon': -42.801, 'aqi': 48, 'pm25': 18.9, 'temp': 32, 'humidity': 65, 'pressure': 1011},
            'Aracaju': {'lat': -10.947, 'lon': -37.073, 'aqi': 37, 'pm25': 12.8, 'temp': 26, 'humidity': 78, 'pressure': 1014},
            
            # Região Centro-Oeste
            'Brasília': {'lat': -15.794, 'lon': -47.882, 'aqi': 38, 'pm25': 10.2, 'temp': 26, 'humidity': 55, 'pressure': 1015},
            'Goiânia': {'lat': -16.680, 'lon': -49.253, 'aqi': 52, 'pm25': 19.3, 'temp': 25, 'humidity': 60, 'pressure': 1014},
            'Campo Grande': {'lat': -20.469, 'lon': -54.620, 'aqi': 44, 'pm25': 15.1, 'temp': 23, 'humidity': 68, 'pressure': 1013},
            'Cuiabá': {'lat': -15.601, 'lon': -56.097, 'aqi': 58, 'pm25': 22.7, 'temp': 31, 'humidity': 62, 'pressure': 1010},
            
            # Região Sudeste
            'São Paulo': {'lat': -23.550, 'lon': -46.633, 'aqi': 45, 'pm25': 12.5, 'temp': 22, 'humidity': 65, 'pressure': 1013},
            'Rio de Janeiro': {'lat': -22.906, 'lon': -43.172, 'aqi': 68, 'pm25': 25.3, 'temp': 28, 'humidity': 70, 'pressure': 1012},
            'Belo Horizonte': {'lat': -19.916, 'lon': -43.934, 'aqi': 52, 'pm25': 18.7, 'temp': 24, 'humidity': 60, 'pressure': 1014},
            'Vitória': {'lat': -20.315, 'lon': -40.312, 'aqi': 41, 'pm25': 14.2, 'temp': 26, 'humidity': 72, 'pressure': 1015},
            
            # Região Sul
            'Curitiba': {'lat': -25.428, 'lon': -49.273, 'aqi': 48, 'pm25': 14.1, 'temp': 18, 'humidity': 70, 'pressure': 1016},
            'Porto Alegre': {'lat': -30.031, 'lon': -51.234, 'aqi': 41, 'pm25': 13.2, 'temp': 20, 'humidity': 65, 'pressure': 1014},
            'Florianópolis': {'lat': -27.595, 'lon': -48.548, 'aqi': 36, 'pm25': 11.8, 'temp': 22, 'humidity': 74, 'pressure': 1015}
        }
        
        # Mapeamento de siglas e nomes alternativos para cidades
        city_mappings = {
            'são paulo': 'São Paulo', 'sao paulo': 'São Paulo', 'sp': 'São Paulo',
            'rio de janeiro': 'Rio de Janeiro', 'rio': 'Rio de Janeiro', 'rj': 'Rio de Janeiro',
            'belo horizonte': 'Belo Horizonte', 'bh': 'Belo Horizonte', 'mg': 'Belo Horizonte',
            'brasília': 'Brasília', 'brasilia': 'Brasília', 'df': 'Brasília',
            'salvador': 'Salvador', 'ba': 'Salvador',
            'fortaleza': 'Fortaleza', 'ce': 'Fortaleza',
            'recife': 'Recife', 'pe': 'Recife',
            'curitiba': 'Curitiba', 'pr': 'Curitiba',
            'porto alegre': 'Porto Alegre', 'poa': 'Porto Alegre', 'rs': 'Porto Alegre',
            'manaus': 'Manaus', 'am': 'Manaus',
            'belém': 'Belém', 'belem': 'Belém', 'pa': 'Belém',
            'porto velho': 'Porto Velho', 'ro': 'Porto Velho',
            'rio branco': 'Rio Branco', 'ac': 'Rio Branco',
            'boa vista': 'Boa Vista', 'rr': 'Boa Vista',
            'macapá': 'Macapá', 'macapa': 'Macapá', 'ap': 'Macapá',
            'palmas': 'Palmas', 'to': 'Palmas',
            'são luís': 'São Luís', 'sao luis': 'São Luís', 'slz': 'São Luís', 'ma': 'São Luís',
            'maceió': 'Maceió', 'maceio': 'Maceió', 'al': 'Maceió',
            'natal': 'Natal', 'rn': 'Natal',
            'joão pessoa': 'João Pessoa', 'joao pessoa': 'João Pessoa', 'jp': 'João Pessoa', 'pb': 'João Pessoa',
            'teresina': 'Teresina', 'pi': 'Teresina',
            'aracaju': 'Aracaju', 'se': 'Aracaju',
            'goiânia': 'Goiânia', 'goiania': 'Goiânia', 'go': 'Goiânia',
            'campo grande': 'Campo Grande', 'cg': 'Campo Grande', 'ms': 'Campo Grande',
            'cuiabá': 'Cuiabá', 'cuiaba': 'Cuiabá', 'mt': 'Cuiabá',
            'vitória': 'Vitória', 'vitoria': 'Vitória', 'es': 'Vitória',
            'florianópolis': 'Florianópolis', 'florianopolis': 'Florianópolis', 'floripa': 'Florianópolis', 'sc': 'Florianópolis'
        }
        
        # Tentar carregar dados reais dos datasets do usuário
        dataset_cities_processed = 0
        for dataset in user_datasets:
            try:
                if os.path.exists(dataset.file_path):
                    df = pd.read_csv(dataset.file_path)
                    
                    if len(df) > 0:
                        # Pegar a última linha (dado mais recente)
                        last_row = df.iloc[-1]
                        
                        # Extrair dados disponíveis
                        aqi = last_row.get('Overall_AQI', last_row.get('aqi', 50))
                        pm25 = last_row.get('pm25', 25)
                        pm10 = last_row.get('pm10', 45)
                        o3 = last_row.get('o3', 35)
                        temp = last_row.get('temperature', 22)
                        humidity = last_row.get('humidity', 65)
                        pressure = last_row.get('pressure', 1013)
                        
                        # Determinar qual cidade baseado na descrição ou nome do arquivo
                        location = dataset.description or dataset.original_filename or ""
                        location_lower = location.lower()
                        
                        # Encontrar a cidade correspondente
                        matched_city = None
                        for key, city_name in city_mappings.items():
                            if key in location_lower:
                                matched_city = city_name
                                break
                        
                        # Se encontrou uma cidade correspondente, atualizar os dados
                        if matched_city and matched_city in cities_data:
                            cities_data[matched_city].update({
                                'aqi': float(aqi), 
                                'pm25': float(pm25), 
                                'pm10': float(pm10),
                                'o3': float(o3), 
                                'temp': float(temp), 
                                'humidity': float(humidity),
                                'pressure': float(pressure),
                                'data_source': 'user_dataset',  # Marcar como dado do usuário
                                'dataset_id': dataset.id
                            })
                            dataset_cities_processed += 1
                            logger.info(f"Dados atualizados para {matched_city} do dataset {dataset.id}")
                            
            except Exception as e:
                logger.warning(f"Erro ao processar dataset {dataset.id} para mapa: {e}")
                continue
        
        logger.info(f"Processados {dataset_cities_processed} datasets para o mapa")
        
        # Calcular estatísticas para exibir no template
        total_cities = len(cities_data)
        cities_with_user_data = sum(1 for city in cities_data.values() if city.get('data_source') == 'user_dataset')
        
        return render_template('map.html', 
                             cities_data=cities_data,
                             total_cities=total_cities,
                             cities_with_user_data=cities_with_user_data,
                             mapbox_token=current_app.config.get('MAPBOX_ACCESS_TOKEN', ''),
                             now=datetime.now())
        
    except Exception as e:
        logger.error(f"Erro no mapa: {e}")
        # Fallback com dados de demonstração de algumas cidades principais
        fallback_data = {
            'São Paulo': {'lat': -23.550, 'lon': -46.633, 'aqi': 45, 'pm25': 12.5, 'temp': 22, 'humidity': 65, 'pressure': 1013},
            'Rio de Janeiro': {'lat': -22.906, 'lon': -43.172, 'aqi': 68, 'pm25': 25.3, 'temp': 28, 'humidity': 70, 'pressure': 1012},
            'Belo Horizonte': {'lat': -19.916, 'lon': -43.934, 'aqi': 52, 'pm25': 18.7, 'temp': 24, 'humidity': 60, 'pressure': 1014},
            'Brasília': {'lat': -15.794, 'lon': -47.882, 'aqi': 38, 'pm25': 10.2, 'temp': 26, 'humidity': 55, 'pressure': 1015}
        }
        return render_template('map.html', 
                             cities_data=fallback_data,
                             total_cities=len(fallback_data),
                             cities_with_user_data=0,
                             mapbox_token='',
                             now=datetime.now())

@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    form = DatasetUploadForm()
    
    if form.validate_on_submit():
        file = form.dataset_file.data
        
        if file and allowed_file(file.filename):
            try:
                # Process uploaded file
                result = process_uploaded_file(file, current_user.id, current_app.config['UPLOAD_FOLDER'])
                
                if result['success']:
                    # Create dataset record
                    dataset = Dataset(
                        filename=result['filename'],
                        original_filename=file.filename,
                        file_path=result['file_path'],
                        file_size=result['file_size'],
                        rows_count=result['rows_count'],
                        columns_count=result['columns_count'],
                        description=form.description.data,
                        is_public=form.is_public.data,
                        user_id=current_user.id,
                        data_quality_score=float(result['quality_score']),
                        missing_data_percentage=float(result['missing_percentage']),
                        source='user_upload'
                    )
                    
                    db.session.add(dataset)
                    db.session.commit()
                    
                    flash(f'Dataset "{file.filename}" carregado com sucesso!', 'success')
                    return redirect(url_for('main.datasets'))
                else:
                    flash(f'Erro ao processar arquivo: {result["error"]}', 'danger')
                    
            except Exception as e:
                flash(f'Erro ao fazer upload: {str(e)}', 'danger')
        else:
            flash('Tipo de arquivo não permitido. Use CSV ou Excel.', 'warning')
    
    return render_template('upload.html', form=form)

@main_bp.route('/datasets')
@login_required
def datasets():
    """Página principal de datasets"""
    try:
        # Use 'uploaded_at' em vez de 'upload_date' - verifique qual campo existe no seu modelo
        user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.uploaded_at.desc()).all()
        
        # Buscar datasets públicos (se aplicável)
        public_datasets = Dataset.query.filter_by(is_public=True).filter(Dataset.user_id != current_user.id).all()
        
        return render_template('datasets.html', 
                             user_datasets=user_datasets,
                             public_datasets=public_datasets,
                             title='Meus Datasets')
        
    except Exception as e:
        logger.error(f"Erro ao carregar página de datasets: {e}")
        flash('Erro ao carregar datasets.', 'error')
        return redirect(url_for('main.dashboard'))

@main_bp.route('/ml-models', methods=['GET', 'POST'])
@login_required
def ml_models():
    form = MLModelForm()
    user_models = MLModel.query.filter_by(user_id=current_user.id).order_by(MLModel.created_at.desc()).all()
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.original_filename).all() # Adicionado order_by para consistência
    
    if form.validate_on_submit():
        dataset_id = request.form.get('dataset_id')
        features = request.form.getlist('features')
        
        if not dataset_id or not features:
            flash('Selecione um dataset e pelo menos uma feature.', 'warning')
            return render_template('ml_models.html', form=form, models=user_models, datasets=user_datasets)
        
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            flash('Dataset não encontrado.', 'danger')
            return render_template('ml_models.html', form=form, models=user_models, datasets=user_datasets)
        
        try:
            # Chama a função de treinamento do modelo
            model_result = train_model(
                dataset.file_path,
                features,
                form.target_variable.data,
                form.model_type.data,
                form.algorithm.data,
                form.test_size.data,
                current_user.id
            )
            
            if model_result['success']:
                # Garante a conversão para float nativo, tratando valores None para
                # evitar erros com o banco de dados.
                
                accuracy = float(model_result['accuracy']) if model_result.get('accuracy') is not None else 0.0
                precision = float(model_result.get('precision')) if model_result.get('precision') is not None else None
                recall = float(model_result.get('recall')) if model_result.get('recall') is not None else None
                f1_score = float(model_result.get('f1_score')) if model_result.get('f1_score') is not None else None
                training_time = float(model_result['training_time']) if model_result.get('training_time') is not None else 0.0

                # Cria o registro do modelo no banco de dados com os valores convertidos
                ml_model = MLModel(
                    name=form.name.data,
                    model_type=form.model_type.data,
                    algorithm=form.algorithm.data,
                    model_path=model_result['model_path'],
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    training_time=training_time,
                    user_id=current_user.id,
                    target_variable=form.target_variable.data
                )
                ml_model.set_features(features)
                
                db.session.add(ml_model)
                db.session.commit()
                
                # A lógica para ativar o modelo e mostrar a mensagem flash agora usa
                # a variável 'accuracy' já convertida e segura.
                if accuracy > 0.85:
                    ml_model.is_active = True
                    db.session.commit()
                    flash(f'Modelo treinado com sucesso! Precisão: {accuracy:.2%} - Modelo ativado automaticamente.', 'success')
                else:
                    flash(f'Modelo treinado com sucesso! Precisão: {accuracy:.2%} - Necessita melhorias para ativação.', 'warning')
                    
                return redirect(url_for('main.ml_models'))
            else:
                flash(f'Erro no treinamento: {model_result["error"]}', 'danger')
                
        except Exception as e:
            # Captura qualquer outro erro inesperado durante o processo
            logger.error(f"Erro inesperado ao treinar modelo: {e}", exc_info=True)
            flash(f'Erro inesperado ao treinar modelo: {str(e)}', 'danger')
    
    return render_template('ml_models.html', form=form, models=user_models, datasets=user_datasets)

@main_bp.route('/toggle-model/<int:model_id>')
@login_required
def toggle_model(model_id):
    model = MLModel.query.get_or_404(model_id)
    
    if model.user_id != current_user.id:
        flash('Acesso negado.', 'danger')
        return redirect(url_for('main.ml_models'))
    
    if model.accuracy >= 0.85:
        model.is_active = not model.is_active
        db.session.commit()
        
        status = "ativado" if model.is_active else "desativado"
        flash(f'Modelo {status} com sucesso!', 'success')
    else:
        flash('Não é possível ativar modelos com precisão inferior a 85%.', 'warning')
    
    return redirect(url_for('main.ml_models'))

@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    model_id = data.get('model_id')
    input_data = data.get('input_data')
    
    model = MLModel.query.get_or_404(model_id)
    
    if model.user_id != current_user.id and not model.is_public:
        return jsonify({'error': 'Acesso negado'}), 403
    
    try:
        prediction = make_prediction(model.model_path, input_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/reports')
@login_required
def reports():
    # Generate correlation matrix for user's datasets
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
    correlations = {}
    statistics = {}
    
    for dataset in user_datasets:
        try:
            df = pd.read_csv(dataset.file_path)
            corr_matrix = calculate_correlations(df)
            stats = generate_statistics(df)
            
            correlations[dataset.id] = corr_matrix.to_dict()
            statistics[dataset.id] = stats
        except Exception as e:
            flash(f'Erro ao processar dataset {dataset.original_filename}: {str(e)}', 'warning')
    
    # Get prediction alerts
    active_models = MLModel.query.filter_by(user_id=current_user.id, is_active=True).all()
    alerts = []
    
    for model in active_models:
        # Simulate some alerts based on model predictions
        if model.accuracy > 0.90:
            alerts.append({
                'type': 'success',
                'message': f'Modelo {model.name} está com excelente performance',
                'timestamp': datetime.utcnow()
            })
        elif model.accuracy < 0.80:
            alerts.append({
                'type': 'warning',
                'message': f'Modelo {model.name} precisa de retreinamento',
                'timestamp': datetime.utcnow()
            })
    
    return render_template('reports.html',
                         correlations=correlations,
                         statistics=statistics,
                         alerts=alerts,
                         datasets=user_datasets)


@main_bp.route('/api/air-quality-data')
def api_air_quality_data():
    # API endpoint for air quality data
    location = request.args.get('location')
    
    query = AirQualityData.query
    
    if location:
        query = query.filter(AirQualityData.location.ilike(f'%{location}%'))
    
    data = query.order_by(AirQualityData.timestamp.desc()).limit(100).all()
    
    result = []
    for item in data:
        result.append({
            'id': item.id,
            'location': item.location,
            'latitude': item.latitude,
            'longitude': item.longitude,
            'aqi': item.aqi,
            'pm25': item.pm25,
            'pm10': item.pm10,
            'no2': item.no2,
            'so2': item.so2,
            'co': item.co,
            'o3': item.o3,
            'temperature': item.temperature,
            'humidity': item.humidity,
            'wind_speed': item.wind_speed,
            'timestamp': item.timestamp.isoformat()
        })
    
    return jsonify(result)

@main_bp.route('/add-air-quality-data', methods=['GET', 'POST'])
@login_required
def add_air_quality_data():
    form = AirQualityDataForm()
    
    if form.validate_on_submit():
        try:
            # Validate the data
            validation_result = validate_air_quality_data(form.data)
            
            if validation_result['valid']:
                air_data = AirQualityData(
                    location=form.location.data,
                    latitude=form.latitude.data,
                    longitude=form.longitude.data,
                    pm25=form.pm25.data,
                    pm10=form.pm10.data,
                    no2=form.no2.data,
                    so2=form.so2.data,
                    co=form.co.data,
                    o3=form.o3.data,
                    temperature=form.temperature.data,
                    humidity=form.humidity.data,
                    wind_speed=form.wind_speed.data
                )
                
                # Calculate AQI
                air_data.calculate_aqi()
                
                db.session.add(air_data)
                db.session.commit()
                
                flash('Dados de qualidade do ar adicionados com sucesso!', 'success')
                return redirect(url_for('main.map_view'))
            else:
                flash(f'Dados inválidos: {validation_result["error"]}', 'danger')
                
        except Exception as e:
            flash(f'Erro ao adicionar dados: {str(e)}', 'danger')
    
    return render_template('add_air_quality_data.html', form=form)


@main_bp.route('/api/dataset-features/<int:dataset_id>')
@login_required
def api_dataset_features(dataset_id):
    """API para buscar features e seus tipos em um dataset."""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.user_id != current_user.id and not dataset.is_public:
            return jsonify({'success': False, 'error': 'Acesso negado'}), 403
        
        if not os.path.exists(dataset.file_path):
            return jsonify({'success': False, 'error': 'Arquivo não encontrado'}), 404

        # NOVA LÓGICA DE ANÁLISE DE TIPOS
        # ----------------------------------------------------
        df = pd.read_csv(dataset.file_path, nrows=500) # Lê algumas linhas para inferir tipos

        columns_with_types = []
        for col in df.columns:
            # Pula colunas de data/id que não servem como feature nem alvo
            if 'date' in col.lower() or 'time' in col.lower() or 'id' in col.lower() or col.startswith('Unnamed'):
                continue
            
            column_type = ''
            # Verifica se a coluna é numérica
            if is_numeric_dtype(df[col]):
                # Se for numérica, mas tiver poucos valores únicos (ex: 0, 1, 2), é categórica
                if df[col].nunique() < 25: # Um limiar: menos de 25 valores únicos = categoria
                    column_type = 'categorical'
                else:
                    column_type = 'numeric'
            else:
                # Se não for numérica, é categórica
                column_type = 'categorical'

            columns_with_types.append({'name': col, 'type': column_type})
        # ----------------------------------------------------

        return jsonify({
            'success': True, 
            'columns': columns_with_types, # <-- Retorna a nova estrutura
            'dataset_name': dataset.original_filename,
            'total_rows': dataset.rows_count
        })
            
    except Exception as e:
        logger.error(f"Erro ao buscar features do dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'error': 'Ocorreu um erro inesperado no servidor.'}), 500
    
@main_bp.route('/datasets/<int:dataset_id>/export')
@login_required
def export_dataset(dataset_id):
    """Exportar dataset"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        # Verificar se o caminho do arquivo é absoluto ou relativo
        file_path = dataset.file_path
        
        # Se o caminho não for absoluto, construir o caminho completo
        if not os.path.isabs(file_path):
            file_path = os.path.join(current_app.root_path, file_path)
        
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            logger.error(f"Caminho atual de trabalho: {os.getcwd()}")
            flash('Arquivo não encontrado.', 'error')
            return redirect(url_for('main.datasets'))
        
        # Nome do arquivo para download
        download_name = f"export_{dataset.original_filename}"
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Erro ao exportar dataset {dataset_id}: {e}")
        flash('Erro ao exportar dataset.', 'error')
        return redirect(url_for('main.datasets'))
    
@main_bp.route('/datasets/<int:dataset_id>/preview')
@login_required
def dataset_preview(dataset_id):
    """Retorna preview do dataset para visualização"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        # Verificar se o arquivo existe
        if not os.path.exists(dataset.file_path):
            return jsonify({'success': False, 'error': 'Arquivo não encontrado'}), 404
        
        # Ler o arquivo CSV
        df = pd.read_csv(dataset.file_path)
        
        # Preparar dados para preview (primeiras 10 linhas)
        preview_data = {
            'headers': df.columns.tolist(),
            'rows': df.head(10).fillna('N/A').values.tolist()
        }
        
        return jsonify({
            'success': True,
            'dataset': {
                'id': dataset.id,
                'original_filename': dataset.original_filename,
                'rows_count': dataset.rows_count,
                'columns_count': dataset.columns_count,
                'data_quality_score': dataset.data_quality_score,
                'description': dataset.description
            },
            'preview_data': preview_data
        })
        
    except Exception as e:
        logger.error(f"Erro ao fazer preview do dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/download-dataset/<int:dataset_id>')
@login_required
def download_dataset(dataset_id):
    """Download do dataset original"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        # Verificar se o caminho do arquivo é absoluto ou relativo
        file_path = dataset.file_path
        
        # Se o caminho não for absoluto, construir o caminho completo
        if not os.path.isabs(file_path):
            file_path = os.path.join(current_app.root_path, file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado para download: {file_path}")
            flash('Arquivo não encontrado.', 'error')
            return redirect(url_for('main.datasets'))
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=dataset.original_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Erro ao fazer download do dataset {dataset_id}: {e}")
        flash('Erro ao fazer download do dataset.', 'error')
        return redirect(url_for('main.datasets'))

@main_bp.route('/api/datasets/<int:dataset_id>/preview')
@login_required
def api_dataset_preview(dataset_id):
    """API para preview do dataset"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        # Ler o arquivo CSV
        df = pd.read_csv(dataset.file_path)
        
        # Preparar preview (primeiras 10 linhas)
        preview_data = {
            'headers': df.columns.tolist(),
            'rows': df.head(10).fillna('').values.tolist(),
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
        return jsonify({
            'success': True,
            'dataset': {
                'id': dataset.id,
                'filename': dataset.original_filename,
                'description': dataset.description,
                'rows_count': dataset.rows_count,
                'columns_count': dataset.columns_count,
                'quality_score': dataset.data_quality_score,
                'file_size': dataset.file_size,
                'uploaded_at': dataset.uploaded_at.isoformat() if dataset.uploaded_at else None
            },
            'preview': preview_data
        })
        
    except Exception as e:
        logger.error(f"Erro no preview do dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/api/datasets/<int:dataset_id>/info')
@login_required
def api_dataset_info(dataset_id):
    """API para informações detalhadas do dataset"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        # Ler o arquivo para análise detalhada
        df = pd.read_csv(dataset.file_path)
        
        # Estatísticas básicas
        numeric_columns = df.select_dtypes(include=['number']).columns
        stats = {}
        for col in numeric_columns:
            stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        # Informações de missing data
        missing_data = df.isnull().sum().to_dict()
        missing_percentage = {col: (count / len(df)) * 100 for col, count in missing_data.items()}
        
        return jsonify({
            'success': True,
            'dataset': {
                'id': dataset.id,
                'filename': dataset.original_filename,
                'description': dataset.description,
                'rows_count': dataset.rows_count,
                'columns_count': dataset.columns_count,
                'quality_score': dataset.data_quality_score,
                'file_size_mb': round(dataset.file_size / (1024 * 1024), 2),
                'uploaded_at': dataset.uploaded_at.strftime('%d/%m/%Y %H:%M') if dataset.uploaded_at else 'N/A'
            },
            'analysis': {
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_data': missing_data,
                'missing_percentage': missing_percentage,
                'stats': stats
            }
        })
        
    except Exception as e:
        logger.error(f"Erro nas informações do dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/datasets/<int:dataset_id>/download')
@login_required
def download_user_dataset(dataset_id):  # Este nome deve corresponder ao usado no template
    """Download do dataset original"""
    try:
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        if not os.path.exists(dataset.file_path):
            flash('Arquivo não encontrado.', 'error')
            return redirect(url_for('main.datasets'))
        
        return send_file(
            dataset.file_path,
            as_attachment=True,
            download_name=dataset.original_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Erro ao fazer download do dataset {dataset_id}: {e}")
        flash('Erro ao fazer download do dataset.', 'error')
        return redirect(url_for('main.datasets'))
    
def calculate_aqi_stats(user_id):
    """Calcula estatísticas de AQI baseadas nos datasets do usuário"""
    try:
        datasets = Dataset.query.filter_by(user_id=user_id).all()
        all_aqi_values = []
        
        for dataset in datasets:
            try:
                df = pd.read_csv(dataset.file_path)
                if 'Overall_AQI' in df.columns:
                    aqi_values = df['Overall_AQI'].dropna().tolist()
                    all_aqi_values.extend(aqi_values)
            except Exception as e:
                logger.warning(f"Erro ao ler dataset {dataset.id}: {e}")
                continue
        
        if not all_aqi_values:
            return {
                'avg_aqi': 45.2,
                'categories': {
                    'Bom (0-50)': 35,
                    'Moderado (51-100)': 25,
                    'Ruim (101-150)': 15,
                    'Muito Ruim (151-200)': 8,
                    'Perigoso (201-300)': 5,
                    'Emergência (>300)': 2
                }
            }
        
        avg_aqi = sum(all_aqi_values) / len(all_aqi_values)
        
        # Calcular distribuição por categoria
        categories = {
            'Bom (0-50)': len([x for x in all_aqi_values if x <= 50]),
            'Moderado (51-100)': len([x for x in all_aqi_values if 51 <= x <= 100]),
            'Ruim (101-150)': len([x for x in all_aqi_values if 101 <= x <= 150]),
            'Muito Ruim (151-200)': len([x for x in all_aqi_values if 151 <= x <= 200]),
            'Perigoso (201-300)': len([x for x in all_aqi_values if 201 <= x <= 300]),
            'Emergência (>300)': len([x for x in all_aqi_values if x > 300])
        }
        
        return {
            'avg_aqi': avg_aqi,
            'categories': categories
        }
        
    except Exception as e:
        logger.error(f"Erro no cálculo de estatísticas AQI: {e}")
        return {
            'avg_aqi': 45.2,
            'categories': {
                'Bom (0-50)': 35,
                'Moderado (51-100)': 25,
                'Ruim (101-150)': 15,
                'Muito Ruim (151-200)': 8,
                'Perigoso (201-300)': 5,
                'Emergência (>300)': 2
            }
        }

def fetch_real_air_quality_data():
    """Busca dados reais de qualidade do ar de APIs públicas"""
    real_data = {}
    
    try:
        # API 1: OpenAQ (dados globais de qualidade do ar - sem API key necessária)
        logger.info("Buscando dados do OpenAQ...")
        try:
            openaq_url = "https://api.openaq.org/v2/latest?limit=50&country=BR&order_by=lastUpdated&sort=desc"
            headers = {
                'User-Agent': 'EcoPredict/1.0 (https://github.com/seu-repositorio)',
                'Accept': 'application/json'
            }
            response = requests.get(openaq_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"OpenAQ retornou {len(data.get('results', []))} resultados")
                
                for result in data.get('results', []):
                    location = result.get('location', '')
                    city = extract_city_name(location)
                    
                    if city and city not in real_data:
                        # Processar medições
                        measurements = {}
                        for measurement in result.get('measurements', []):
                            parameter = measurement.get('parameter', '')
                            value = measurement.get('value', 0)
                            unit = measurement.get('unit', '')
                            measurements[parameter] = value
                        
                        # Calcular AQI aproximado
                        aqi = calculate_aqi_from_measurements(measurements)
                        
                        real_data[city] = {
                            'aqi': aqi,
                            'pm25': measurements.get('pm25', 0),
                            'pm10': measurements.get('pm10', 0),
                            'o3': measurements.get('o3', 0),
                            'no2': measurements.get('no2', 0),
                            'so2': measurements.get('so2', 0),
                            'co': measurements.get('co', 0),
                            'source': 'OpenAQ',
                            'last_updated': result.get('lastUpdated', ''),
                            'location': location
                        }
                        logger.info(f"Dados OpenAQ para {city}: AQI {aqi}")
            else:
                logger.warning(f"OpenAQ retornou status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            logger.warning("Timeout ao buscar dados do OpenAQ")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Erro de conexão com OpenAQ: {e}")
        except Exception as e:
            logger.warning(f"Erro inesperado no OpenAQ: {e}")
        
        # API 2: WAQI (World Air Quality Index) - com sua API key
        logger.info("Buscando dados do WAQI...")
        try:
            waqi_token = current_app.config.get('WAQI_API_TOKEN', '')
            if waqi_token:
                # Estações WAQI para cidades brasileiras
                waqi_stations = {
                    'São Paulo': 'sao-paulo',
                    'Rio de Janeiro': 'rio-de-janeiro', 
                    'Belo Horizonte': 'minas-gerais',
                    'Brasília': 'brasilia',
                    'Curitiba': 'curitiba',
                    'Salvador': 'salvador',
                    'Fortaleza': 'fortaleza'
                }
                
                for city, station in waqi_stations.items():
                    if city in real_data:
                        continue  # Já tem dados do OpenAQ
                        
                    waqi_url = f"https://api.waqi.info/feed/{station}/?token={waqi_token}"
                    response = requests.get(waqi_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'ok':
                            aqi_data = data.get('data', {})
                            aqi = aqi_data.get('aqi', 0)
                            
                            if aqi > 0:
                                iaqi = aqi_data.get('iaqi', {})
                                
                                real_data[city] = {
                                    'aqi': aqi,
                                    'pm25': iaqi.get('pm25', {}).get('v', 0),
                                    'pm10': iaqi.get('pm10', {}).get('v', 0),
                                    'o3': iaqi.get('o3', {}).get('v', 0),
                                    'no2': iaqi.get('no2', {}).get('v', 0),
                                    'so2': iaqi.get('so2', {}).get('v', 0),
                                    'temp': iaqi.get('t', {}).get('v', 0),
                                    'humidity': iaqi.get('h', {}).get('v', 0),
                                    'pressure': iaqi.get('p', {}).get('v', 0),
                                    'source': 'WAQI',
                                    'last_updated': aqi_data.get('time', {}).get('s', '')
                                }
                                logger.info(f"Dados WAQI para {city}: AQI {aqi}")
                        else:
                            logger.warning(f"WAQI retornou status: {data.get('status')}")
                    else:
                        logger.warning(f"WAQI retornou HTTP {response.status_code}")
            else:
                logger.warning("Token WAQI não configurado")
                                
        except Exception as e:
            logger.warning(f"Erro ao buscar dados WAQI: {e}")
        
        logger.info(f"Total de cidades com dados reais: {len(real_data)}")
        
        # Se não conseguiu dados reais, usar fallback
        if not real_data:
            logger.info("Nenhum dado real encontrado, usando fallback")
            real_data = get_data_from_user_datasets()
            
    except Exception as e:
        logger.error(f"Erro geral ao buscar dados reais: {e}")
        real_data = get_data_from_user_datasets()
    
    return real_data

def extract_city_name(location):
    """Extrai o nome da cidade da string de localização"""
    city_mapping = {
        'sao paulo': 'São Paulo',
        'rio de janeiro': 'Rio de Janeiro',
        'belo horizonte': 'Belo Horizonte',
        'brasilia': 'Brasília',
        'salvador': 'Salvador',
        'fortaleza': 'Fortaleza',
        'manaus': 'Manaus',
        'curitiba': 'Curitiba',
        'recife': 'Recife',
        'porto alegre': 'Porto Alegre'
    }
    
    location_lower = location.lower()
    for key, city in city_mapping.items():
        if key in location_lower:
            return city
    
    return None

def calculate_aqi_from_measurements(measurements):
    """Calcula AQI aproximado baseado nas medições de poluentes"""
    try:
        pm25 = measurements.get('pm25', 0)
        pm10 = measurements.get('pm10', 0)
        o3 = measurements.get('o3', 0)
        
        # Usar o maior valor entre os poluentes principais
        if pm25 > 0:
            # Escala simplificada do AQI para PM2.5
            if pm25 <= 12: aqi_pm25 = (50 / 12) * pm25
            elif pm25 <= 35: aqi_pm25 = 50 + (50 / 23) * (pm25 - 12)
            elif pm25 <= 55: aqi_pm25 = 100 + (50 / 20) * (pm25 - 35)
            elif pm25 <= 150: aqi_pm25 = 150 + (150 / 95) * (pm25 - 55)
            else: aqi_pm25 = 300 + (200 / 150) * (pm25 - 150)
        else:
            aqi_pm25 = 0
            
        if pm10 > 0:
            # Escala para PM10
            if pm10 <= 54: aqi_pm10 = (50 / 54) * pm10
            elif pm10 <= 154: aqi_pm10 = 50 + (50 / 100) * (pm10 - 54)
            elif pm10 <= 254: aqi_pm10 = 100 + (50 / 100) * (pm10 - 154)
            elif pm10 <= 354: aqi_pm10 = 150 + (50 / 100) * (pm10 - 254)
            elif pm10 <= 424: aqi_pm10 = 200 + (100 / 70) * (pm10 - 354)
            else: aqi_pm10 = 300 + (200 / 176) * (pm10 - 424)
        else:
            aqi_pm10 = 0
            
        # Retornar o maior valor de AQI
        aqi_values = [aqi_pm25, aqi_pm10]
        aqi_values = [v for v in aqi_values if v > 0]
        
        return round(max(aqi_values)) if aqi_values else 50
        
    except Exception as e:
        logger.warning(f"Erro no cálculo do AQI: {e}")
        return 50
    
def get_data_from_user_datasets():
    """Fallback: Busca dados dos datasets do usuário"""
    from flask_login import current_user
    import pandas as pd
    import os
    
    user_data = {}
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
    for dataset in user_datasets:
        try:
            if os.path.exists(dataset.file_path):
                df = pd.read_csv(dataset.file_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    location = dataset.description or dataset.original_filename or ""
                    
                    # Mapear para cidades conhecidas
                    city = extract_city_name(location)
                    if city and city not in user_data:
                        user_data[city] = {
                            'aqi': last_row.get('Overall_AQI', 50),
                            'pm25': last_row.get('pm25', 25),
                            'pm10': last_row.get('pm10', 45),
                            'o3': last_row.get('o3', 35),
                            'temp': last_row.get('temperature', 22),
                            'humidity': last_row.get('humidity', 65),
                            'pressure': last_row.get('pressure', 1013),
                            'source': 'User Dataset'
                        }
        except Exception as e:
            logger.warning(f"Erro ao processar dataset do usuário: {e}")
            continue
    
    return user_data

@main_bp.route('/api/map/data')
@login_required
def api_map_data():
    """API para dados do mapa, baseada nos datasets do usuário (versão estável)."""
    try:
        # 1. BUSCA DADOS DA FONTE CONFIÁVEL: OS DATASETS DO USUÁRIO
        # Esta função já existe no seu código e lê os arquivos locais.
        source_data = get_data_from_user_datasets()
        
        # 2. MANTÉM A LISTA DE CIDADES E COORDENADAS PARA EXIBIÇÃO NO MAPA
        cities_coordinates = {
            'São Paulo': {'lat': -23.550, 'lon': -46.633},
            'Rio de Janeiro': {'lat': -22.906, 'lon': -43.172},
            'Belo Horizonte': {'lat': -19.916, 'lon': -43.934},
            'Brasília': {'lat': -15.794, 'lon': -47.882},
            'Salvador': {'lat': -12.971, 'lon': -38.501},
            'Fortaleza': {'lat': -3.731, 'lon': -38.526},
            'Manaus': {'lat': -3.119, 'lon': -60.021},
            'Curitiba': {'lat': -25.428, 'lon': -49.273},
            'Recife': {'lat': -8.047, 'lon': -34.877},
            'Porto Alegre': {'lat': -30.031, 'lon': -51.234}
        }
        
        # 3. CONSTRÓI A RESPOSTA COMBINANDO OS DADOS DO USUÁRIO COM AS COORDENADAS
        response_data = {}
        for city, coords in cities_coordinates.items():
            if city in source_data:
                # Se o usuário tem dados para esta cidade, use-os
                response_data[city] = {**coords, **source_data[city]}
            else:
                # Caso contrário, mostra dados simulados/padrão
                response_data[city] = {
                    **coords,
                    'aqi': 50,
                    'pm25': 25,
                    'source': 'Simulado'
                }
        
        # 4. RETORNA O JSON PARA O FRONTEND
        return jsonify({
            'success': True,
            'data': response_data,
            'timestamp': datetime.now().isoformat(),
            'sources_used': ['User Datasets'] # A fonte de dados agora é clara e única
        })
        
    except Exception as e:
        logger.error(f"Erro na API do mapa: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/admin')
@login_required
@admin_required
def admin_panel():
    """Painel administrativo do sistema"""
    try:
        print("=== INICIANDO PAINEL ADMIN ===")
        
        # Estatísticas gerais do sistema
        total_users = User.query.count()
        total_datasets = Dataset.query.count()
        total_models = MLModel.query.count()
        
        print(f"Total users: {total_users}")
        print(f"Total datasets: {total_datasets}")
        print(f"Total models: {total_models}")
        
        # Tentar contar alertas (com fallback)
        try:
            total_alerts = Alert.query.count()
            print(f"Total alerts: {total_alerts}")
        except Exception as e:
            print(f"Erro em alertas: {e}")
            total_alerts = 0
        
        # Usuários recentes (últimos 7 dias)
        recent_users = User.query.filter(
            User.created_at >= datetime.now() - timedelta(days=7)
        ).count()
        
        print(f"Recent users: {recent_users}")
        
        # Datasets por status
        datasets_public = Dataset.query.filter_by(is_public=True).count()
        datasets_private = Dataset.query.filter_by(is_public=False).count()
        
        print(f"Datasets - Public: {datasets_public}, Private: {datasets_private}")
        
        # Modelos por status
        models_active = MLModel.query.filter_by(is_active=True).count()
        models_training = MLModel.query.filter_by(is_active=False).count()
        models_error = 0
        
        print(f"Models - Active: {models_active}, Training: {models_training}")
        
        # Alertas ativos (com fallback)
        try:
            active_alerts = Alert.query.filter_by(is_active=True).count()
        except:
            active_alerts = 0
            
        # Alertas não lidos (com fallback)  
        try:
            unread_alerts = Alert.query.filter_by(is_read=False).count()
        except:
            unread_alerts = 0
        
        # Uso do sistema
        recent_activity = total_datasets + total_models
        
        # Espaço em disco usado
        disk_usage = calculate_disk_usage()
        print(f"Disk usage: {disk_usage}")
        
        # Logs recentes do sistema (com fallback)
        try:
            system_logs = SystemLog.query.order_by(SystemLog.created_at.desc()).limit(10).all()
        except Exception as e:
            print(f"Erro em system logs: {e}")
            system_logs = []
        
        # Alertas recentes para exibir (com fallback)
        try:
            recent_alerts = Alert.query.order_by(Alert.created_at.desc()).limit(5).all()
        except:
            recent_alerts = []
        
        print("=== DADOS COLETADOS COM SUCESSO ===")
        
        return render_template('admin/admin_panel.html',
                             total_users=total_users,
                             total_datasets=total_datasets,
                             total_models=total_models,
                             total_alerts=total_alerts,
                             recent_users=recent_users,
                             datasets_public=datasets_public,
                             datasets_private=datasets_private,
                             models_active=models_active,
                             models_training=models_training,
                             models_error=models_error,
                             active_alerts=active_alerts,
                             unread_alerts=unread_alerts,
                             recent_activity=recent_activity,
                             disk_usage=disk_usage,
                             system_logs=system_logs,
                             recent_alerts=recent_alerts)
        
    except Exception as e:
        logger.error(f"Erro no painel admin: {e}")
        flash('Erro ao carregar painel administrativo', 'danger')
        return redirect(url_for('main.dashboard'))
    
# Atualize também as outras rotas administrativas
@main_bp.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Gerenciamento de usuários"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/admin_users.html', users=users)

@main_bp.route('/admin/datasets')
@login_required
@admin_required
def admin_datasets():
    """Gerenciamento de datasets"""
    page = request.args.get('page', 1, type=int)
    per_page = 15
    
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/admin_datasets.html', datasets=datasets)

@main_bp.route('/admin/models')
@login_required
@admin_required
def admin_models():
    """Gerenciamento de modelos ML"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    models = MLModel.query.order_by(MLModel.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/admin_models.html', models=models)

@main_bp.route('/admin/system')
@login_required
@admin_required
def admin_system():
    """Configurações do sistema"""
    system_info = get_system_info()
    return render_template('admin/admin_system.html', system_info=system_info)

# API Routes para ações administrativas
@main_bp.route('/admin/api/delete_dataset/<int:dataset_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_dataset(dataset_id):
    """Deletar dataset (admin)"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Deletar arquivo físico
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        db.session.delete(dataset)
        db.session.commit()
        
        logger.info(f"Dataset {dataset_id} deletado por admin {current_user.email}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset deletado com sucesso'
        })
    except Exception as e:
        logger.error(f"Erro ao deletar dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/admin/api/system_stats')
@login_required
@admin_required
def system_stats():
    """API para estatísticas do sistema em tempo real"""
    try:
        import psutil
        
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_users': User.query.filter_by(is_active=True).count(),
            'total_datasets_today': Dataset.query.filter(
                Dataset.uploaded_at >= datetime.now().date()
            ).count(),
            'active_alerts': Alert.query.filter_by(is_active=True).count() if hasattr(Alert, 'query') else 0
        }
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        logger.error(f"Erro ao obter stats do sistema: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'stats': {
                'cpu_percent': 0,
                'memory_percent': 0, 
                'disk_usage': 0,
                'active_users': 0,
                'total_datasets_today': 0,
                'active_alerts': 0
            }
        }), 500
    
# Funções auxiliares
def calculate_disk_usage():
    """Calcular uso de disco dos datasets"""
    try:
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
        logger.error(f"Erro ao calcular uso de disco: {e}")
        return {'total_mb': 0, 'total_datasets': 0}

def get_system_info():
    """Obter informações do sistema"""
    try:
        import platform
        return {
            'python_version': platform.python_version(),
            'flask_version': flask.__version__,
            'system': platform.system(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'start_time': datetime.now() - timedelta(hours=2)  # Exemplo
        }
    except Exception as e:
        logger.error(f"Erro ao obter info do sistema: {e}")
        return {}
    
# routes.py - Adicione estas rotas após as rotas existentes

# =============================================
# ROTAS DE AÇÕES ADMINISTRATIVAS
# =============================================

@main_bp.route('/admin/api/backup', methods=['POST'])
@login_required
@admin_required
def admin_backup():
    """Criar backup do sistema"""
    try:
        from datetime import datetime
        import shutil
        import os
        
        # Criar nome do backup com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'ecopredict_backup_{timestamp}'
        backup_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'backups', backup_name)
        
        # Criar diretório de backup
        os.makedirs(backup_path, exist_ok=True)
        
        # Copiar arquivos importantes (exemplo)
        important_folders = ['instance', 'uploads']
        
        for folder in important_folders:
            if os.path.exists(folder):
                shutil.copytree(folder, os.path.join(backup_path, folder))
        
        # Registrar no log do sistema
        log_system_event(
            level='INFO',
            message=f'Backup do sistema criado: {backup_name}',
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': f'Backup criado com sucesso: {backup_name}',
            'backup_name': backup_name
        })
        
    except Exception as e:
        logger.error(f"Erro ao criar backup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/clear_cache', methods=['POST'])
@login_required
@admin_required
def admin_clear_cache():
    """Limpar cache do sistema"""
    try:
        import glob
        import os
        
        # Limpar cache de templates (exemplo)
        cache_dirs = [
            os.path.join(current_app.root_path, '..', '__pycache__'),
            os.path.join(current_app.instance_path, 'cache')
        ]
        
        cleared_files = 0
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                # Encontrar arquivos .pyc
                pyc_files = glob.glob(os.path.join(cache_dir, '**', '*.pyc'), recursive=True)
                for pyc_file in pyc_files:
                    try:
                        os.remove(pyc_file)
                        cleared_files += 1
                    except:
                        pass
        
        # Registrar no log
        log_system_event(
            level='INFO',
            message=f'Cache do sistema limpo. {cleared_files} arquivos removidos.',
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': f'Cache limpo com sucesso. {cleared_files} arquivos removidos.',
            'cleared_files': cleared_files
        })
        
    except Exception as e:
        logger.error(f"Erro ao limpar cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/optimize_db', methods=['POST'])
@login_required
@admin_required
def admin_optimize_db():
    """Otimizar base de dados"""
    try:
        from sqlalchemy import text
        
        # Executar comandos de otimização (exemplos para PostgreSQL)
        optimization_commands = [
            "VACUUM ANALYZE;",
            "REINDEX DATABASE current;"
        ]
        
        results = []
        for command in optimization_commands:
            try:
                db.session.execute(text(command))
                results.append(f"Comando executado: {command}")
            except Exception as cmd_error:
                results.append(f"Erro no comando {command}: {str(cmd_error)}")
        
        db.session.commit()
        
        # Registrar no log
        log_system_event(
            level='INFO',
            message='Otimização do banco de dados executada',
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': 'Otimização do banco de dados concluída',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Erro ao otimizar banco: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/restart_system', methods=['POST'])
@login_required
@admin_required
def admin_restart_system():
    """Reiniciar sistema (simulado)"""
    try:
        # Em produção, isso reiniciaria o serviço
        # Aqui é apenas uma simulação
        
        # Registrar no log
        log_system_event(
            level='WARNING',
            message='Sistema reiniciado pelo administrador',
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': 'Comando de reinicialização enviado. Sistema será reiniciado.',
            'note': 'Em ambiente de produção, isso reiniciaria o serviço.'
        })
        
    except Exception as e:
        logger.error(f"Erro ao reiniciar sistema: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/download_logs', methods=['GET'])
@login_required
@admin_required
def admin_download_logs():
    """Download dos logs do sistema"""
    try:
        import tempfile
        import zipfile
        from flask import send_file
        
        # Criar arquivo ZIP temporário com logs
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                # Adicionar logs do sistema (últimos 1000 registros)
                logs = SystemLog.query.order_by(SystemLog.created_at.desc()).limit(1000).all()
                
                # Criar arquivo de texto com logs
                log_content = "LOGS DO SISTEMA ECOPREDICT\n"
                log_content += "=" * 50 + "\n\n"
                
                for log in reversed(logs):  # Ordem cronológica
                    log_content += f"[{log.created_at.strftime('%Y-%m-%d %H:%M:%S')}] "
                    log_content += f"{log.level}: {log.message}"
                    if log.module:
                        log_content += f" (Módulo: {log.module})"
                    if log.user_id:
                        log_content += f" (Usuário ID: {log.user_id})"
                    log_content += "\n"
                
                zipf.writestr('system_logs.txt', log_content)
                
                # Adicionar logs do aplicativo se existirem
                app_log_file = 'app.log'
                if os.path.exists(app_log_file):
                    zipf.write(app_log_file, 'application.log')
            
            # Registrar no log
            log_system_event(
                level='INFO',
                message='Logs do sistema exportados',
                module='admin',
                user_id=current_user.id
            )
            
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f'ecopredict_logs_{datetime.now().strftime("%Y%m%d_%H%M")}.zip',
                mimetype='application/zip'
            )
            
    except Exception as e:
        logger.error(f"Erro ao baixar logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/clear_old_logs', methods=['POST'])
@login_required
@admin_required
def admin_clear_old_logs():
    """Limpar logs antigos"""
    try:
        # Manter apenas logs dos últimos 30 dias
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Contar logs que serão deletados
        old_logs_count = SystemLog.query.filter(
            SystemLog.created_at < cutoff_date
        ).count()
        
        # Deletar logs antigos
        SystemLog.query.filter(
            SystemLog.created_at < cutoff_date
        ).delete()
        
        db.session.commit()
        
        # Registrar no log
        log_system_event(
            level='INFO',
            message=f'Logs antigos removidos: {old_logs_count} registros',
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': f'{old_logs_count} logs antigos removidos (anteriores a {cutoff_date.strftime("%d/%m/%Y")})',
            'deleted_count': old_logs_count
        })
        
    except Exception as e:
        logger.error(f"Erro ao limpar logs antigos: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_bp.route('/admin/api/toggle_maintenance', methods=['POST'])
@login_required
@admin_required
def admin_toggle_maintenance():
    """Ativar/desativar modo manutenção"""
    try:
        maintenance_mode = request.json.get('maintenance_mode', False)
        
        # Aqui você implementaria a lógica real de modo manutenção
        # Por exemplo, criar/remover um arquivo de flag
        
        flag_file = os.path.join(current_app.instance_path, 'maintenance.flag')
        
        if maintenance_mode:
            # Ativar modo manutenção
            with open(flag_file, 'w') as f:
                f.write('maintenance')
            message = 'Modo manutenção ativado'
            log_level = 'WARNING'
        else:
            # Desativar modo manutenção
            if os.path.exists(flag_file):
                os.remove(flag_file)
            message = 'Modo manutenção desativado'
            log_level = 'INFO'
        
        # Registrar no log
        log_system_event(
            level=log_level,
            message=message,
            module='admin',
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'message': message,
            'maintenance_mode': maintenance_mode
        })
        
    except Exception as e:
        logger.error(f"Erro ao alterar modo manutenção: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@main_bp.route('/admin/api/delete_model/<int:model_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_model(model_id):
    """Deleta um modelo de ML (admin)."""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        # Deletar o arquivo físico do modelo
        if model.model_path and os.path.exists(model.model_path):
            try:
                os.remove(model.model_path)
            except Exception as e:
                logger.warning(f"Não foi possível deletar o arquivo do modelo {model_id}: {e}")

        db.session.delete(model)
        db.session.commit()
        
        logger.info(f"Modelo {model_id} deletado pelo admin {current_user.email}")
        
        return jsonify({
            'success': True,
            'message': 'Modelo de ML deletado com sucesso.'
        })
    except Exception as e:
        logger.error(f"Erro ao deletar modelo de ML: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/admin/api/toggle_model_status/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_model_status(model_id):
    """Ativa ou desativa um modelo de ML (admin)."""
    try:
        model = MLModel.query.get_or_404(model_id)
        model.is_active = not model.is_active
        db.session.commit()
        
        status = "ativado" if model.is_active else "desativado"
        logger.info(f"Modelo {model_id} {status} pelo admin {current_user.email}")
        
        return jsonify({
            'success': True,
            'message': f'Modelo {status} com sucesso.',
            'is_active': model.is_active
        })
    except Exception as e:
        logger.error(f"Erro ao alterar status do modelo: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/admin/api/cleanup_datasets', methods=['POST'])
@login_required
@admin_required
def admin_cleanup_datasets():
    """Deleta todos os datasets do sistema (ação de limpeza)."""
    try:
        num_datasets = Dataset.query.count()

        all_datasets = Dataset.query.all()
        for dataset in all_datasets:
            if dataset.file_path and os.path.exists(dataset.file_path):
                try:
                    os.remove(dataset.file_path)
                except Exception as e:
                    logger.warning(f"Não foi possível deletar o arquivo do dataset {dataset.id}: {e}")

        Dataset.query.delete()
        db.session.commit()

        logger.warning(f"LIMPEZA GERAL: {num_datasets} datasets foram deletados pelo admin {current_user.email}")

        return jsonify({
            'success': True,
            'message': f'{num_datasets} datasets foram removidos com sucesso.'
        })
    except Exception as e:
        logger.error(f"Erro durante a limpeza de datasets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/admin/api/model_details/<int:model_id>')
@login_required
@admin_required
def admin_model_details(model_id):
    """Retorna detalhes completos de um modelo de ML."""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        # Formata os dados para serem enviados como JSON
        details = {
            'id': model.id,
            'name': model.name,
            'user': model.creator.username,
            'model_type': model.model_type,
            'algorithm': model.algorithm,
            'model_path': model.model_path,
            'accuracy': f"{model.accuracy * 100:.2f}%" if model.accuracy is not None else "N/A",
            'precision': f"{model.precision * 100:.2f}%" if model.precision is not None else "N/A",
            'recall': f"{model.recall * 100:.2f}%" if model.recall is not None else "N/A",
            'f1_score': f"{model.f1_score * 100:.2f}%" if model.f1_score is not None else "N/A",
            'training_time': f"{model.training_time:.2f} segundos" if model.training_time is not None else "N/A",
            'is_active': model.is_active,
            'created_at': model.created_at.strftime('%d/%m/%Y às %H:%M:%S'),
            'features_used': model.get_features(), # Usa o método do modelo para obter a lista
            'target_variable': model.target_variable
        }
        
        return jsonify({'success': True, 'details': details})
        
    except Exception as e:
        logger.error(f"Erro ao buscar detalhes do modelo {model_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@main_bp.route('/admin/api/user_details/<int:user_id>')
@login_required
@admin_required
def admin_user_details(user_id):
    """Retorna detalhes de um usuário para exibição no modal."""
    try:
        user = User.query.get_or_404(user_id)
        
        details = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
            'is_active': user.is_active,
            'created_at': user.created_at.strftime('%d/%m/%Y %H:%M:%S'),
            'last_login': user.last_login.strftime('%d/%m/%Y %H:%M:%S') if user.last_login else "Nunca",
            'datasets_count': user.datasets.count(),
            'models_count': user.models.count()
        }
        return jsonify({'success': True, 'details': details})
    except Exception as e:
        logger.error(f"Erro ao buscar detalhes do usuário {user_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- Rota para deletar um usuário ---
@main_bp.route('/admin/api/delete_user/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Deleta um usuário (admin)."""
    try:
        # Impede que o admin se auto-delete
        if user_id == current_user.id:
            return jsonify({'success': False, 'error': 'Você não pode deletar sua própria conta.'}), 403

        user = User.query.get_or_404(user_id)
        
        # Opcional: Reatribuir ou deletar dados do usuário (datasets, modelos)
        # Por segurança, vamos apenas deletar o usuário por enquanto.
        # Em um sistema real, você precisaria de uma lógica mais complexa aqui.
        
        db.session.delete(user)
        db.session.commit()
        
        logger.warning(f"Usuário {user.email} (ID: {user_id}) foi DELETADO pelo admin {current_user.email}")
        
        return jsonify({
            'success': True,
            'message': f'Usuário "{user.username}" deletado com sucesso.'
        })
    except Exception as e:
        logger.error(f"Erro ao deletar usuário: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- Rota para ativar/desativar (você já deve ter, verifique se está assim) ---
@main_bp.route('/admin/api/toggle_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_user_status(user_id):
    """Ativar/desativar usuário."""
    try:
        # Impede que o admin se auto-desative
        if user_id == current_user.id:
            return jsonify({'success': False, 'error': 'Você não pode desativar sua própria conta.'}), 403

        user = User.query.get_or_404(user_id)
        user.is_active = not user.is_active
        db.session.commit()
        
        action = "ativado" if user.is_active else "desativado"
        logger.info(f"Usuário {user.email} {action} pelo admin {current_user.email}")
        
        return jsonify({
            'success': True,
            'message': f'Usuário {action} com sucesso.',
            'is_active': user.is_active
        })
    except Exception as e:
        logger.error(f"Erro ao alterar status do usuário: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@main_bp.route('/analysis/<int:dataset_id>')
@login_required
def analysis_page(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    # Verificação de permissão
    if dataset.user_id != current_user.id:
        flash('Você не tem permissão para analisar este dataset.', 'danger')
        return redirect(url_for('main.datasets'))

    try:
        if not os.path.exists(dataset.file_path):
            flash(f'Arquivo do dataset "{dataset.original_filename}" não encontrado.', 'danger')
            return redirect(url_for('main.datasets'))
            
        df = pd.read_csv(dataset.file_path)
        
        analysis_results = {}
        # Decide qual função de análise chamar com base na origem do dataset
        if dataset.source == 'openaq':
            analysis_results = generate_openaq_analysis(df)
        elif dataset.source == 'inpe':
            analysis_results = generate_inpe_analysis(df)
        elif dataset.source == 'inmet':
            analysis_results = generate_inmet_analysis(df)
        else: # 'user_upload' ou qualquer outro
            analysis_results = generate_generic_analysis(df)

        return render_template(
            'analysis_page.html', 
            dataset=dataset, 
            analysis_results=analysis_results
        )

    except Exception as e:
        logger.error(f"Erro ao analisar o dataset {dataset_id}: {e}")
        flash(f'Ocorreu um erro ao tentar analisar o dataset: {e}', 'danger')
        return redirect(url_for('main.datasets'))
    
@main_bp.route('/lab/<int:dataset_id>')
@login_required
def analysis_lab(dataset_id):
    """Página principal do Laboratório de Análise."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id:
        abort(403)
    
    data_preview_html = None
    try:
        df = pd.read_csv(dataset.file_path) # Carrega o DF completo para análise
        
        # Pega as colunas para os menus de seleção
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        all_features = df.columns.tolist() # Usaremos todas para o alvo de classificação

        # Gera o HTML da prévia das primeiras 10 linhas
        data_preview_html = df.head(10).to_html(
            classes='table table-sm table-striped table-bordered', 
            index=False
        )

    except Exception as e:
        flash(f"Não foi possível ler o dataset: {e}", "danger")
        numeric_features = []
        all_features = []
        
    return render_template(
        'analysis_lab.html', 
        dataset=dataset,
        numeric_features=numeric_features,
        all_features=all_features, # Passando todas as colunas
        data_preview=data_preview_html # Passando a prévia em HTML
    )
@main_bp.route('/lab/run_clustering', methods=['POST'])
@login_required
def run_clustering_endpoint():
    """Endpoint da API para executar a clusterização."""
    data = request.json
    dataset_id = data.get('dataset_id')
    features = data.get('features')
    n_clusters = int(data.get('n_clusters', 3))
    
    dataset = Dataset.query.get_or_404(dataset_id)
    df = pd.read_csv(dataset.file_path)
    
    result = run_kmeans_clustering(df, features, n_clusters)
    return jsonify(result)

@main_bp.route('/lab/run_pca', methods=['POST'])
@login_required
def run_pca_endpoint():
    """Endpoint da API para executar PCA."""
    data = request.json
    dataset = Dataset.query.get_or_404(data.get('dataset_id'))
    features = data.get('features')
    
    df = pd.read_csv(dataset.file_path)
    result = run_pca_analysis(df, features)
    return jsonify(result)

@main_bp.route('/lab/run_classification', methods=['POST'])
@login_required
def run_classification_endpoint():
    """Endpoint da API para executar Classificação."""
    data = request.json
    dataset = Dataset.query.get_or_404(data.get('dataset_id'))
    features = data.get('features')
    target = data.get('target')
    
    df = pd.read_csv(dataset.file_path)
    result = run_classification_analysis(df, features, target)
    return jsonify(result)

@main_bp.route('/lab/run_regression', methods=['POST'])
@login_required
def run_regression_endpoint():
    """Endpoint da API para executar Regressão."""
    data = request.json
    dataset = Dataset.query.get_or_404(data.get('dataset_id'))
    features = data.get('features')
    target = data.get('target')
    
    df = pd.read_csv(dataset.file_path)
    result = run_regression_analysis(df, features, target)
    return jsonify(result)