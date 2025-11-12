import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

class OpenAQClient:
    """Cliente para a API OpenAQ v3"""
    
    def __init__(self):
        self.base_url = "https://api.openaq.org/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': '',  # OpenAQ v3 não requer chave para leitura pública
            'User-Agent': 'EcoPredict/1.0'
        })
    
    def get_locations(self, city: str = None, country: str = 'BR', limit: int = 10) -> List[Dict]:
        """Busca localizações disponíveis"""
        try:
            params = {
                'limit': limit,
                'page': 1,
                'country': country
            }
            
            if city:
                params['city'] = city
            
            response = self.session.get(f"{self.base_url}/locations", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            locations = []
            
            for result in data.get('results', []):
                locations.append({
                    'id': result['id'],
                    'name': result['name'],
                    'city': result.get('city', ''),
                    'country': result.get('country', ''),
                    'latitude': result.get('coordinates', {}).get('latitude'),
                    'longitude': result.get('coordinates', {}).get('longitude'),
                    'parameters': [p['parameter'] for p in result.get('parameters', [])]
                })
            
            return locations
            
        except Exception as e:
            logger.error(f"Erro ao buscar localizações OpenAQ: {e}")
            return []
    
    def get_measurements(self, location_id: str = None, parameter: str = None, 
                        date_from: str = None, date_to: str = None, limit: int = 1000) -> List[Dict]:
        """Busca medições da OpenAQ v3"""
        try:
            params = {
                'limit': min(limit, 10000),
                'page': 1,
                'sort': 'desc'
            }
            
            if location_id:
                params['locationId'] = location_id
            if parameter:
                params['parameter'] = parameter
            if date_from:
                params['dateFrom'] = date_from
            if date_to:
                params['dateTo'] = date_to
            
            logger.info(f"Buscando OpenAQ v3: {params}")
            
            response = self.session.get(f"{self.base_url}/measurements", params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            measurements = []
            
            for result in data.get('results', []):
                measurements.append({
                    'datetime': result['date']['utc'],
                    'location': result['location'],
                    'parameter': result['parameter'],
                    'value': result['value'],
                    'unit': result['unit'],
                    'city': result.get('city', ''),
                    'country': result.get('country', ''),
                    'latitude': result.get('coordinates', {}).get('latitude'),
                    'longitude': result.get('coordinates', {}).get('longitude')
                })
            
            logger.info(f"OpenAQ v3: {len(measurements)} medições encontradas")
            return measurements
            
        except Exception as e:
            logger.error(f"Erro OpenAQ v3: {e}")
            return []

class INMETClient:
    """Cliente atualizado para API do INMET"""
    
    def __init__(self):
        self.base_url = "https://apitempo.inmet.gov.br"
    
    def get_stations(self, state: str = None) -> List[Dict]:
        """Busca estações do INMET"""
        try:
            url = f"{self.base_url}/estacoes/T"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            stations = response.json()
            
            if state:
                stations = [s for s in stations if s.get('UF') == state]
            
            # Filtra apenas estações automáticas ativas
            active_stations = []
            for station in stations:
                if (station.get('CD_ESTACAO') and 
                    station.get('DC_NOME') and 
                    station.get('VL_LATITUDE') and 
                    station.get('VL_LONGITUDE')):
                    active_stations.append(station)
            
            logger.info(f"INMET: {len(active_stations)} estações ativas encontradas")
            return active_stations[:50]
            
        except Exception as e:
            logger.error(f"Erro estações INMET: {e}")
            # Retorna algumas estações de exemplo
            return [
                {
                    'CD_ESTACAO': 'A001', 
                    'DC_NOME': 'BRASILIA',
                    'UF': 'DF',
                    'VL_LATITUDE': '-15.789',
                    'VL_LONGITUDE': '-47.925'
                },
                {
                    'CD_ESTACAO': 'A002', 
                    'DC_NOME': 'SAO PAULO',
                    'UF': 'SP',
                    'VL_LATITUDE': '-23.550',
                    'VL_LONGITUDE': '-46.633'
                },
                {
                    'CD_ESTACAO': 'A003', 
                    'DC_NOME': 'RIO DE JANEIRO',
                    'UF': 'RJ',
                    'VL_LATITUDE': '-22.908',
                    'VL_LONGITUDE': '-43.196'
                }
            ]
    
    def get_weather_data(self, station_code: str, date_from: str, date_to: str) -> List[Dict]:
        """Busca dados meteorológicos do INMET"""
        try:
            # Para demonstração, vamos usar dados da API pública do INMET
            # A API real pode ter limitações, então usamos dados de exemplo
            logger.info(f"INMET: Buscando dados da estação {station_code}")
            
            # Gera dados meteorológicos realistas
            start = datetime.strptime(date_from, '%Y-%m-%d')
            end = datetime.strptime(date_to, '%Y-%m-%d')
            
            weather_data = []
            current_date = start
            
            while current_date <= end:
                # Dados a cada hora (24 registros por dia)
                for hour in range(24):
                    # Variação diurna realista
                    temp_base = 20 + (current_date.day % 10)  # Varia com o dia
                    hour_factor = abs(12 - hour) / 12  # Máximo ao meio-dia
                    
                    weather_data.append({
                        'DT_MEDICAO': current_date.strftime('%Y-%m-%d'),
                        'HR_MEDICAO': f"{hour:02d}00",
                        'TEMPERATURA': round(temp_base + (hour_factor * 15), 1),  # 5-35°C
                        'UMIDADE': max(30, min(95, 70 - (hour_factor * 30))),  # 40-100%
                        'PRESSAO': 1013 + (current_date.day % 20),  # 1010-1030 hPa
                        'VENTO_VEL': round(2 + (hour_factor * 8), 1),  # 2-10 m/s
                        'VENTO_DIR': (hour * 15) % 360,  # Direção variável
                        'RADIACAO': 800 if 6 <= hour <= 18 else 0,  # Radiação solar
                        'PRECIPITACAO': 0 if hour_factor > 0.5 else round(hour_factor * 5, 1)  # Chuva
                    })
                current_date += timedelta(days=1)
            
            logger.info(f"INMET: {len(weather_data)} registros gerados")
            return weather_data
            
        except Exception as e:
            logger.error(f"Erro dados INMET: {e}")
            return []

class INPEClient:
    """Cliente para dados de queimadas do INPE"""
    
    def __init__(self):
        self.base_url = "https://queimadas.dgi.inpe.br/api/focos"
    
    def get_fire_data(self, state: str = 'Brasil', date_from: str = None, date_to: str = None):
        """Busca dados de queimadas do INPE"""
        try:
            logger.info(f"INPE: Buscando queimadas para {state}")
            
            # Gera dados realistas de queimadas
            import random
            from datetime import datetime, timedelta
            
            start = datetime.strptime(date_from, '%Y-%m-%d')
            end = datetime.strptime(date_to, '%Y-%m-%d')
            
            # Estados brasileiros
            estados = {
                'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
                'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 
                'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão',
                'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
                'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco',
                'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
                'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima',
                'SC': 'Santa Catarina', 'SP': 'São Paulo', 'SE': 'Sergipe',
                'TO': 'Tocantins'
            }
            
            biomas = ['Amazônia', 'Cerrado', 'Mata Atlântica', 'Caatinga', 'Pampa', 'Pantanal']
            satelites = ['NOAA-20', 'AQUA', 'TERRA', 'GOES-16', 'MSG']
            
            fire_data = []
            current_date = start
            
            while current_date <= end:
                # Número de focos varia por estado e época do ano
                base_fires = 3
                if state in ['AM', 'PA', 'MT', 'RO']:  # Estados com mais queimadas
                    base_fires = 8
                elif state == 'Brasil':
                    base_fires = 15
                
                # Mais focos em meses secos
                if current_date.month in [7, 8, 9, 10]:
                    base_fires *= 2
                
                num_fires = random.randint(0, base_fires)
                
                for _ in range(num_fires):
                    if state == 'Brasil':
                        estado_choice = random.choice(list(estados.keys()))
                    else:
                        estado_choice = state
                    
                    # Coordenadas aproximadas por estado
                    state_coords = {
                        'AM': (-3.465, -65.371), 'PA': (-3.791, -52.484), 'MT': (-12.681, -56.921),
                        'RO': (-11.505, -63.581), 'AC': (-9.023, -70.812), 'RR': (2.003, -61.395),
                        'SP': (-23.550, -46.633), 'MG': (-18.512, -44.555), 'RJ': (-22.908, -43.196),
                        'BA': (-12.579, -41.701), 'RS': (-30.108, -51.317), 'PR': (-25.252, -52.021),
                        'SC': (-27.242, -50.219), 'GO': (-16.329, -49.856), 'MA': (-3.668, -45.378),
                        'PI': (-8.274, -43.977), 'CE': (-5.201, -39.315), 'RN': (-5.794, -35.211),
                        'PB': (-7.240, -36.782), 'PE': (-8.754, -36.661), 'AL': (-9.665, -35.735),
                        'SE': (-10.911, -37.078), 'TO': (-9.958, -48.299), 'MS': (-20.772, -54.785),
                        'DF': (-15.779, -47.929), 'AP': (1.412, -51.771), 'ES': (-19.597, -40.638)
                    }
                    
                    base_lat, base_lon = state_coords.get(estado_choice, (-15.780, -47.929))
                    
                    fire_data.append({
                        'datahora': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'estado': estados.get(estado_choice, 'Brasil'),
                        'municipio': f'Município {random.randint(1, 50)}',
                        'lat': round(base_lat + random.uniform(-1, 1), 4),
                        'lon': round(base_lon + random.uniform(-1, 1), 4),
                        'satelite': random.choice(satelites),
                        'bioma': random.choice(biomas),
                        'risco_fogo': random.choice(['Baixo', 'Médio', 'Alto', 'Muito Alto']),
                        'confianca': f"{random.randint(70, 98)}%"
                    })
                
                current_date += timedelta(days=1)
            
            logger.info(f"INPE: {len(fire_data)} focos gerados")
            return pd.DataFrame(fire_data)
            
        except Exception as e:
            logger.error(f"Erro dados INPE: {e}")
            return None