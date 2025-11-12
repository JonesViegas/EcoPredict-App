import os
import pandas as pd
import numpy as np
import requests
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

from app import db
from app.models import Dataset
from app.services.api_client import OpenAQClient, INMETClient, INPEClient
from app.utils import classify_aqi, allowed_file

# Blueprint para dados externos
external_bp = Blueprint('external', __name__, url_prefix='/external')

logger = logging.getLogger(__name__)

@external_bp.route('/sources')
@login_required
def sources():
    """Página principal de fontes de dados externos"""
    return render_template('external/sources.html')

@external_bp.route('/openaq', methods=['GET', 'POST'])
@login_required
def openaq():
    """Busca dados da OpenAQ"""
    if request.method == 'POST':
        try:
            location = request.form.get('location', '').strip()
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')
            limit = int(request.form.get('limit', 1000))

            if not location:
                flash('Por favor, informe uma localização.', 'danger')
                return redirect(url_for('external.openaq'))

            client = OpenAQClient()
            parameters = ['pm25', 'pm10', 'o3', 'co', 'so2', 'no2']
            all_data = []

            for param in parameters:
                data = client.get_measurements(
                    location=location, 
                    parameter=param, 
                    date_from=date_from, 
                    date_to=date_to, 
                    limit=limit
                )
                if data:
                    all_data.extend(data)

            if not all_data:
                flash(f'Nenhum dado encontrado para "{location}" no período especificado.', 'warning')
                return redirect(url_for('external.openaq'))
            
            # Processa os dados
            df_raw = pd.DataFrame(all_data)
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
            
            # Cria pivot table com médias por horário
            df_pivot = df_raw.pivot_table(
                index='datetime', 
                columns='parameter', 
                values='value', 
                aggfunc='mean'
            ).reset_index().sort_values('datetime')

            # Calcula AQI
            df_processed = classify_aqi(df_pivot.copy())

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_location = secure_filename(location.replace(' ', '_'))
            filename = f"openaq_{clean_location}_{timestamp}.csv"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            df_processed.to_csv(file_path, index=False)
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                rows_count=len(df_processed),
                columns_count=len(df_processed.columns),
                description=f"Dados OpenAQ para {location} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=95.0,
                missing_data_percentage=5.0
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'Dados da OpenAQ para {location} importados com sucesso!', 'success')
            return redirect(url_for('main.datasets'))

        except Exception as e:
            logger.error(f"Erro ao buscar dados da OpenAQ: {e}")
            flash(f'Erro ao buscar dados: {str(e)}', 'danger')
            return redirect(url_for('external.openaq'))

    return render_template('external/openaq.html')

@external_bp.route('/inmet', methods=['GET', 'POST'])
@login_required
def inmet():
    """Busca dados do INMET"""
    if request.method == 'POST':
        try:
            station_code = request.form.get('station_code', '').strip()
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')

            if not station_code:
                flash('Por favor, selecione uma estação.', 'danger')
                return redirect(url_for('external.inmet'))

            client = INMETClient()
            data = client.get_weather_data(station_code, date_from, date_to)

            if not data:
                flash(f'Nenhum dado encontrado para a estação {station_code}.', 'warning')
                return redirect(url_for('external.inmet'))

            df = pd.DataFrame(data)
            
            # Processa dados
            if 'DT_MEDICAO' in df.columns and 'HR_MEDICAO' in df.columns:
                df['datetime'] = pd.to_datetime(
                    df['DT_MEDICAO'] + ' ' + df['HR_MEDICAO'].astype(str).str.zfill(4),
                    format='%Y-%m-%d %H%M'
                )
            else:
                df['datetime'] = datetime.now()

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inmet_{station_code}_{timestamp}.csv"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            df.to_csv(file_path, index=False)
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                rows_count=len(df),
                columns_count=len(df.columns),
                description=f"Dados INMET estação {station_code} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=90.0,
                missing_data_percentage=10.0
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'Dados da estação {station_code} importados com sucesso!', 'success')
            return redirect(url_for('main.datasets'))

        except Exception as e:
            logger.error(f"Erro ao buscar dados do INMET: {e}")
            flash(f'Erro ao buscar dados: {str(e)}', 'danger')
            return redirect(url_for('external.inmet'))

    return render_template('external/inmet.html')

@external_bp.route('/inpe', methods=['GET', 'POST'])
@login_required
def inpe():
    """Busca dados de queimadas do INPE"""
    if request.method == 'POST':
        try:
            state = request.form.get('state', 'Brasil')
            date_from = request.form.get('date_from')
            date_to = request.form.get('date_to')

            client = INPEClient()
            df = client.get_fire_data(state, date_from, date_to)

            if df is None or df.empty:
                flash('Nenhum dado de queimadas encontrado para o período selecionado.', 'warning')
                return redirect(url_for('external.inpe'))

            # Salva dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inpe_queimadas_{state}_{timestamp}.csv"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            dataset = Dataset(
                filename=filename,
                original_filename=filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                rows_count=len(df),
                columns_count=len(df.columns),
                description=f"Dados de queimadas INPE {state} de {date_from} a {date_to}",
                is_public=False,
                user_id=current_user.id,
                data_quality_score=85.0,
                missing_data_percentage=15.0
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'Dados de queimadas para {state} importados com sucesso!', 'success')
            return redirect(url_for('main.datasets'))

        except Exception as e:
            logger.error(f"Erro ao buscar dados do INPE: {e}")
            flash(f'Erro ao buscar dados: {str(e)}', 'danger')
            return redirect(url_for('external.inpe'))

    return render_template('external/inpe.html')

@external_bp.route('/api/inmet/stations')
@login_required
def api_inmet_stations():
    """API para buscar estações do INMET"""
    try:
        state = request.args.get('state', '')
        client = INMETClient()
        stations = client.get_stations(state)
        
        # Formata estações para o frontend
        formatted_stations = []
        for station in stations:
            formatted_stations.append({
                'code': station.get('CD_ESTACAO', ''),
                'name': station.get('DC_NOME', ''),
                'state': station.get('UF', ''),
                'latitude': station.get('VL_LATITUDE', ''),
                'longitude': station.get('VL_LONGITUDE', '')
            })
        
        return jsonify({'success': True, 'stations': formatted_stations})
        
    except Exception as e:
        logger.error(f"Erro na API de estações: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500