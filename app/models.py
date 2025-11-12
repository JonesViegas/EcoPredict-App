from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import secrets

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    
    # Password reset fields
    reset_token = db.Column(db.String(100), unique=True, index=True)
    token_expiration = db.Column(db.DateTime)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='uploader', lazy='dynamic', cascade='all, delete-orphan')
    ml_models = db.relationship('MLModel', backref='creator', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash e salva a senha"""
        if len(password) < 8:
            raise ValueError("A senha deve ter pelo menos 8 caracteres")
        self.password_hash = generate_password_hash(
            password, 
            method='pbkdf2:sha256', 
            salt_length=16
        )
    
    def check_password(self, password):
        """Verifica a senha com hash"""
        return check_password_hash(self.password_hash, password)
    
    def increment_login_attempts(self):
        """Incrementa tentativas de login falhas"""
        self.login_attempts += 1
        if self.login_attempts >= 5:  # Bloqueia após 5 tentativas
            self.locked_until = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()
    
    def reset_login_attempts(self):
        """Reseta tentativas de login"""
        self.login_attempts = 0
        self.locked_until = None
        db.session.commit()
    
    def is_locked(self):
        """Verifica se a conta está bloqueada"""
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return True
        return False
    
    def generate_reset_token(self):
        """Gera token seguro para reset de senha"""
        self.reset_token = secrets.token_urlsafe(32)
        self.token_expiration = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()
        return self.reset_token
    
    def verify_reset_token(self, token):
        """Verifica token de reset"""
        if (self.reset_token == token and 
            self.token_expiration and 
            datetime.utcnow() < self.token_expiration):
            return True
        return False
    
    def __repr__(self):
        return f'<User {self.username}>'

class Dataset(db.Model):
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
    
    def __repr__(self):
        return f'<Dataset {self.original_filename}>'

class MLModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)  # 'regression', 'classification'
    algorithm = db.Column(db.String(100), nullable=False)   # 'random_forest', 'xgboost', etc.
    model_path = db.Column(db.String(500), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_time = db.Column(db.Float)  # in seconds
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Feature information
    features_used = db.Column(db.Text)  # JSON string of features
    target_variable = db.Column(db.String(100))
    
    def set_features(self, features_list):
        self.features_used = json.dumps(features_list)
    
    def get_features(self):
        return json.loads(self.features_used) if self.features_used else []
    
    def __repr__(self):
        return f'<MLModel {self.name}>'

class AirQualityData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    pm25 = db.Column(db.Float)  # Particulate Matter 2.5
    pm10 = db.Column(db.Float)  # Particulate Matter 10
    no2 = db.Column(db.Float)   # Nitrogen Dioxide
    so2 = db.Column(db.Float)   # Sulfur Dioxide
    co = db.Column(db.Float)    # Carbon Monoxide
    o3 = db.Column(db.Float)    # Ozone
    aqi = db.Column(db.Float)   # Air Quality Index
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def calculate_aqi(self):
        # Simplified AQI calculation
        # In practice, use proper AQI calculation formulas
        pollutants = [self.pm25, self.pm10, self.no2, self.so2, self.co, self.o3]
        valid_pollutants = [p for p in pollutants if p is not None]
        
        if not valid_pollutants:
            self.aqi = 0
            return self.aqi
            
        max_pollutant = max(valid_pollutants)
        # Simplified AQI calculation - scale based on maximum pollutant
        self.aqi = min(max_pollutant * 2, 500)
        return self.aqi
    
    def __repr__(self):
        return f'<AirQualityData {self.location} - AQI: {self.aqi}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))