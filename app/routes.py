from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, send_file
from flask_login import login_required, current_user
from app import db
from app.models import User, Dataset, MLModel, AirQualityData
from app.forms import DatasetUploadForm, MLModelForm, AirQualityDataForm
from app.utils import allowed_file, process_uploaded_file
from app.ml_models import train_model, make_prediction, evaluate_model
from app.data_processing import calculate_correlations, generate_statistics, clean_dataset
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import folium
import joblib

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get basic statistics
    total_datasets = Dataset.query.filter_by(user_id=current_user.id).count()
    total_models = MLModel.query.filter_by(user_id=current_user.id).count()
    active_models = MLModel.query.filter_by(user_id=current_user.id, is_active=True).count()
    
    # Get recent air quality data
    recent_data = AirQualityData.query.order_by(AirQualityData.timestamp.desc()).limit(10).all()
    
    # Calculate AQI statistics
    aqi_data = AirQualityData.query.with_entities(AirQualityData.aqi).filter(AirQualityData.aqi.isnot(None)).all()
    aqi_values = [data[0] for data in aqi_data] if aqi_data else [0]
    
    avg_aqi = np.mean(aqi_values) if aqi_values else 0
    max_aqi = max(aqi_values) if aqi_values else 0
    min_aqi = min(aqi_values) if aqi_values else 0
    
    # Get AQI distribution for chart
    aqi_categories = {
        'Excelente (0-50)': len([x for x in aqi_values if 0 <= x <= 50]),
        'Bom (51-100)': len([x for x in aqi_values if 51 <= x <= 100]),
        'Moderado (101-150)': len([x for x in aqi_values if 101 <= x <= 150]),
        'Ruim (151-200)': len([x for x in aqi_values if 151 <= x <= 200]),
        'Muito Ruim (201-300)': len([x for x in aqi_values if 201 <= x <= 300]),
        'Perigoso (301+)': len([x for x in aqi_values if x > 300])
    }
    
    return render_template('dashboard.html',
                         total_datasets=total_datasets,
                         total_models=total_models,
                         active_models=active_models,
                         recent_data=recent_data,
                         avg_aqi=round(avg_aqi, 1),
                         max_aqi=round(max_aqi, 1),
                         min_aqi=round(min_aqi, 1),
                         aqi_categories=aqi_categories)

@main_bp.route('/map')
@login_required
def map_view():
    # Get all air quality data for the map
    air_quality_data = AirQualityData.query.all()
    
    # Create base map centered on Brazil
    m = folium.Map(location=[-15.7797, -47.9297], zoom_start=4)
    
    # Add markers for each data point
    for data in air_quality_data:
        # Determine color based on AQI
        if data.aqi <= 50:
            color = 'green'
        elif data.aqi <= 100:
            color = 'yellow'
        elif data.aqi <= 150:
            color = 'orange'
        elif data.aqi <= 200:
            color = 'red'
        elif data.aqi <= 300:
            color = 'purple'
        else:
            color = 'black'
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h5>{data.location}</h5>
            <p><strong>AQI:</strong> {data.aqi:.1f}</p>
            <p><strong>PM2.5:</strong> {data.pm25 or 'N/A'} μg/m³</p>
            <p><strong>PM10:</strong> {data.pm10 or 'N/A'} μg/m³</p>
            <p><strong>NO2:</strong> {data.no2 or 'N/A'} ppb</p>
            <p><strong>Temp:</strong> {data.temperature or 'N/A'} °C</p>
            <p><strong>Última atualização:</strong> {data.timestamp.strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        """
        
        folium.Marker(
            [data.latitude, data.longitude],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=data.location,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Save map to HTML string
    map_html = m._repr_html_()
    
    return render_template('map.html', map_html=map_html)

@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    form = DatasetUploadForm()
    
    if form.validate_on_submit():
        file = form.dataset_file.data
        
        if file and allowed_file(file.filename):
            try:
                # Process uploaded file
                result = process_uploaded_file(file, current_user.id)
                
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
                        data_quality_score=result['quality_score'],
                        missing_data_percentage=result['missing_percentage']
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
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.uploaded_at.desc()).all()
    public_datasets = Dataset.query.filter_by(is_public=True).filter(Dataset.user_id != current_user.id).all()
    
    return render_template('datasets.html', 
                         user_datasets=user_datasets, 
                         public_datasets=public_datasets)

@main_bp.route('/download-dataset/<int:dataset_id>')
@login_required
def download_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not dataset.is_public:
        flash('Acesso negado a este dataset.', 'danger')
        return redirect(url_for('main.datasets'))
    
    return send_file(dataset.file_path, as_attachment=True, download_name=dataset.original_filename)

@main_bp.route('/ml-models', methods=['GET', 'POST'])
@login_required
def ml_models():
    form = MLModelForm()
    user_models = MLModel.query.filter_by(user_id=current_user.id).order_by(MLModel.created_at.desc()).all()
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
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
            # Train the model
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
                # Create ML model record
                ml_model = MLModel(
                    name=form.name.data,
                    model_type=form.model_type.data,
                    algorithm=form.algorithm.data,
                    model_path=model_result['model_path'],
                    accuracy=model_result['accuracy'],
                    precision=model_result.get('precision'),
                    recall=model_result.get('recall'),
                    f1_score=model_result.get('f1_score'),
                    training_time=model_result['training_time'],
                    user_id=current_user.id,
                    target_variable=form.target_variable.data
                )
                ml_model.set_features(features)
                
                db.session.add(ml_model)
                db.session.commit()
                
                # Set as active if accuracy > 85%
                if model_result['accuracy'] > 0.85:
                    ml_model.is_active = True
                    db.session.commit()
                    flash(f'Modelo treinado com sucesso! Precisão: {model_result["accuracy"]:.2%} - Modelo ativado automaticamente.', 'success')
                else:
                    flash(f'Modelo treinado com sucesso! Precisão: {model_result["accuracy"]:.2%} - Necessita melhorias para ativação.', 'warning')
                    
                return redirect(url_for('main.ml_models'))
            else:
                flash(f'Erro no treinamento: {model_result["error"]}', 'danger')
                
        except Exception as e:
            flash(f'Erro ao treinar modelo: {str(e)}', 'danger')
    
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

@main_bp.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Acesso negado. Apenas administradores.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Admin statistics
    total_users = User.query.count()
    total_datasets = Dataset.query.count()
    total_models = MLModel.query.count()
    system_health = 'healthy'  # Simplified health check
    
    users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    return render_template('admin.html',
                         total_users=total_users,
                         total_datasets=total_datasets,
                         total_models=total_models,
                         system_health=system_health,
                         users=users)

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
    """API para buscar features disponíveis em um dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Verifica se o usuário tem acesso ao dataset
        if dataset.user_id != current_user.id and not dataset.is_public:
            return jsonify({'success': False, 'error': 'Acesso negado'}), 403
        
        # Lê o dataset para extrair as colunas
        df = pd.read_csv(dataset.file_path)
        features = df.columns.tolist()
        
        # Remove colunas não úteis para ML
        excluded_columns = ['datetime', 'date', 'time', 'location', 'city', 'country', 'latitude', 'longitude', 'unit']
        features = [f for f in features if f not in excluded_columns and not f.startswith('Unnamed')]
        
        return jsonify({
            'success': True, 
            'features': features,
            'dataset_name': dataset.original_filename
        })
        
    except Exception as e:
        logger.error(f"Erro ao buscar features do dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500