# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import DevelopmentConfig, ProductionConfig # <-- Importar as classes de configuração
import os

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address)

login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'warning'
login_manager.session_protection = "strong"

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Define a configuração a ser usada
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')

    if config_name == 'production':
        config_object = ProductionConfig
    else:
        # O padrão é sempre desenvolvimento se não for especificado
        config_object = DevelopmentConfig
            
    app.config.from_object(config_object)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    
    # Create directories
    # A pasta 'instance' é criada automaticamente pelo Flask quando necessário
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ML_MODELS_DIR'], exist_ok=True)
    
    # Register blueprints - IMPORTAR DENTRO DO APP CONTEXT
    with app.app_context():
        from app.routes import main_bp
        from app.auth import auth_bp
        from app.external_data import external_bp
        
        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp, url_prefix='/auth')
        app.register_blueprint(external_bp) # <-- Não se esqueça de adicionar este!
    
    return app