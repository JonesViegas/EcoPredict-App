from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app import db, limiter
from app.models import User
from app.forms import LoginForm, RegistrationForm, ChangePasswordForm, ResetPasswordRequestForm, ResetPasswordForm
from datetime import datetime
import re

auth_bp = Blueprint('auth', __name__)

# Funções temporárias para evitar import circular
def send_reset_email(user, token):
    """Simula envio de email"""
    print(f"EMAIL SIMULADO: Reset de senha para {user.email}")
    print(f"Token: {token}")
    return True

def log_security_event(event, user_id=None, email=None):
    """Log de eventos de segurança"""
    print(f"SECURITY: {event} - User: {user_id or email or 'unknown'}")

def validate_password_strength(password):
    """Valida força da senha"""
    if len(password) < 8:
        return {'valid': False, 'message': 'Senha deve ter pelo menos 8 caracteres'}
    if not re.search(r"[A-Z]", password):
        return {'valid': False, 'message': 'Senha deve conter letra maiúscula'}
    if not re.search(r"[a-z]", password):
        return {'valid': False, 'message': 'Senha deve conter letra minúscula'}
    if not re.search(r"\d", password):
        return {'valid': False, 'message': 'Senha deve conter número'}
    return {'valid': True, 'message': 'Senha forte'}

# Rate limiting para previnir brute force
@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        if user and user.is_locked():
            remaining_time = (user.locked_until - datetime.utcnow()).seconds // 60
            flash(f'Conta temporariamente bloqueada. Tente novamente em {remaining_time} minutos.', 'danger')
            return render_template('auth/login.html', form=form)
        
        if user and user.check_password(form.password.data):
            if not user.is_active:
                flash('Sua conta está desativada. Entre em contato com o administrador.', 'warning')
                return render_template('auth/login.html', form=form)
            
            user.reset_login_attempts()
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            login_user(user, remember=form.remember.data)
            log_security_event('Login realizado com sucesso', user.id)
            
            next_page = request.args.get('next')
            if next_page and not next_page.startswith('/'):
                next_page = None
            return redirect(next_page or url_for('main.dashboard'))
        else:
            if user:
                user.increment_login_attempts()
            log_security_event('Tentativa de login falhou', None, form.email.data)
            flash('Email ou senha incorretos.', 'danger')
    
    return render_template('auth/login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Validação adicional de senha
        password_validation = validate_password_strength(form.password.data)
        if not password_validation['valid']:
            flash(f'Erro na senha: {password_validation["message"]}', 'danger')
            return render_template('auth/register.html', form=form)
        
        # Verificar se usuário já existe
        if User.query.filter_by(email=form.email.data).first():
            flash('Email já está em uso. Por favor use outro email.', 'danger')
            return render_template('auth/register.html', form=form)
        
        if User.query.filter_by(username=form.username.data).first():
            flash('Nome de usuário já está em uso. Por favor escolha outro.', 'danger')
            return render_template('auth/register.html', form=form)
        
        user = User(
            username=form.username.data,
            email=form.email.data,
            is_admin=False
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        log_security_event(f'Novo usuário registrado: {user.username}', user.id)
        flash('Conta criada com sucesso! Você já pode fazer login.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    log_security_event('Logout realizado', current_user.id)
    logout_user()
    flash('Você saiu da sua conta.', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            # Validar força da nova senha
            password_validation = validate_password_strength(form.new_password.data)
            if not password_validation['valid']:
                flash(f'Erro na senha: {password_validation["message"]}', 'danger')
                return render_template('auth/change_password.html', form=form)
            
            current_user.set_password(form.new_password.data)
            db.session.commit()
            
            log_security_event('Senha alterada com sucesso', current_user.id)
            flash('Sua senha foi alterada com sucesso!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Senha atual incorreta.', 'danger')
    
    return render_template('auth/change_password.html', form=form)

@auth_bp.route('/reset-password-request', methods=['GET', 'POST'])
@limiter.limit("3 per hour")
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            token = user.generate_reset_token()
            send_reset_email(user, token)
            log_security_event('Solicitação de reset de senha', user.id)
        
        # Sempre mostrar mesma mensagem por segurança
        flash('Se o email existir em nosso sistema, enviaremos instruções para resetar sua senha.', 'info')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_password_request.html', form=form)

@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
@limiter.limit("5 per hour")
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.verify_reset_token(token):
        flash('Token inválido ou expirado.', 'warning')
        return redirect(url_for('auth.reset_password_request'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        # Validar força da nova senha
        password_validation = validate_password_strength(form.password.data)
        if not password_validation['valid']:
            flash(f'Erro na senha: {password_validation["message"]}', 'danger')
            return render_template('auth/reset_password.html', form=form)
        
        user.set_password(form.password.data)
        user.reset_token = None
        user.token_expiration = None
        user.reset_login_attempts()  # Resetar bloqueios também
        db.session.commit()
        
        log_security_event('Senha resetada via token', user.id)
        flash('Sua senha foi resetada com sucesso!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_password.html', form=form)