from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import secrets # Módulo para gerar tokens seguros

# O user_loader é essencial para o Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # --- Campos para Segurança de Login ---
    login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)
    
    # --- Campos para Reset de Senha ---
    reset_token = db.Column(db.String(100), unique=True, index=True, nullable=True)
    token_expiration = db.Column(db.DateTime, nullable=True)
    
    # --- Relacionamentos com Cascade Delete ---
    datasets = db.relationship('Dataset', backref='uploader', lazy='dynamic', cascade='all, delete-orphan')
    ml_models = db.relationship('MLModel', backref='creator', lazy='dynamic', cascade='all, delete-orphan')
    alerts = db.relationship('Alert', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    system_logs = db.relationship('SystemLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'

    # --- Métodos de Gerenciamento de Senha ---
    def set_password(self, password):
        """Gera e armazena o hash da senha de forma segura."""
        if len(password) < 8:
            raise ValueError("A senha deve ter pelo menos 8 caracteres.")
        self.password_hash = generate_password_hash(
            password, 
            method='pbkdf2:sha256', 
            salt_length=16
        )
    
    def check_password(self, password):
        """Verifica se a senha fornecida corresponde ao hash armazenado."""
        return check_password_hash(self.password_hash, password)

    # --- Métodos de Segurança de Login (Bloqueio de Conta) ---
    def is_locked(self):
        """Retorna True se o usuário estiver bloqueado no momento."""
        if self.locked_until is None:
            return False
        return self.locked_until > datetime.utcnow()

    def increment_login_attempts(self):
        """Incrementa as tentativas de login falhas e bloqueia a conta se necessário."""
        if self.is_locked(): # Não faz nada se já estiver bloqueado
            return

        self.login_attempts += 1
        # Bloqueia a conta por 10 minutos após 5 tentativas falhas
        if self.login_attempts >= 5:
            self.locked_until = datetime.utcnow() + timedelta(minutes=10)
        
        db.session.commit()

    def reset_login_attempts(self):
        """Reseta a contagem de tentativas de login falhas e desbloqueia a conta."""
        if self.login_attempts > 0 or self.locked_until is not None:
            self.login_attempts = 0
            self.locked_until = None
            db.session.commit()

    # --- Métodos de Reset de Senha ---
    def get_reset_token(self, expires_in=1800):
        """Gera um token seguro para reset de senha com tempo de expiração."""
        self.reset_token = secrets.token_urlsafe(32)
        self.token_expiration = datetime.utcnow() + timedelta(seconds=expires_in)
        db.session.commit()
        return self.reset_token

    @staticmethod
    def verify_reset_token(token):
        """Verifica se um token de reset é válido e não expirou."""
        user = User.query.filter_by(reset_token=token).first()
        if user and user.token_expiration > datetime.utcnow():
            return user
        return None


class Dataset(db.Model):
    __tablename__ = 'dataset'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    rows_count = db.Column(db.Integer)
    columns_count = db.Column(db.Integer)
    description = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Data quality metrics
    data_quality_score = db.Column(db.Float, default=0.0)
    missing_data_percentage = db.Column(db.Float, default=0.0)
    source = db.Column(db.String(50), nullable=True, index=True) # Ex: 'openaq', 'inmet', 'inpe', 'user_upload'

    def __repr__(self):
        return f'<Dataset {self.original_filename}>'

class MLModel(db.Model):
    __tablename__ = 'ml_model'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    algorithm = db.Column(db.String(100), nullable=False)
    model_path = db.Column(db.String(500), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_time = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Feature information
    features_used = db.Column(db.Text)
    target_variable = db.Column(db.String(100), nullable=False)
    
    def set_features(self, features_list):
        self.features_used = json.dumps(features_list)
    
    def get_features(self):
        return json.loads(self.features_used) if self.features_used else []
    
    def __repr__(self):
        return f'<MLModel {self.name}>'

class AirQualityData(db.Model):
    __tablename__ = 'air_quality_data'
    
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    pm25 = db.Column(db.Float)
    pm10 = db.Column(db.Float)
    no2 = db.Column(db.Float)
    so2 = db.Column(db.Float)
    co = db.Column(db.Float)
    o3 = db.Column(db.Float)
    aqi = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def calculate_aqi(self):
        pollutants = [self.pm25, self.pm10, self.no2, self.so2, self.co, self.o3]
        valid_pollutants = [p for p in pollutants if p is not None]
        
        if not valid_pollutants:
            self.aqi = 0
            return self.aqi
            
        max_pollutant = max(valid_pollutants)
        self.aqi = min(max_pollutant * 2, 500)
        return self.aqi
    
    def __repr__(self):
        return f'<AirQualityData {self.location} - AQI: {self.aqi}>'

# USANDO A TABELA 'alert' QUE JÁ EXISTE
class Alert(db.Model):
    __tablename__ = 'alert'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(200))
    message = db.Column(db.Text)
    alert_type = db.Column(db.String(50))
    severity = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

    # Propriedade para acessar o usuário de forma segura
    @property
    def user(self):
        from app.models import User
        return User.query.get(self.user_id) if self.user_id else None

    def __repr__(self):
        return f'<Alert {self.id}: {self.title}>'

class SystemLog(db.Model):
    __tablename__ = 'system_log'
    
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    module = db.Column(db.String(100), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Função para adicionar logs do sistema
def log_system_event(level, message, module=None, user_id=None, ip_address=None):
    try:
        log = SystemLog(
            level=level,
            message=message,
            module=module,
            user_id=user_id,
            ip_address=ip_address
        )
        db.session.add(log)
        db.session.commit()
        return True
    except Exception as e:
        print(f"Erro ao criar log do sistema: {e}")
        return False