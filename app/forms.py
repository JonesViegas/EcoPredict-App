from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, FileField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Optional, NumberRange
from app.models import User
import pandas as pd

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Senha', validators=[DataRequired()])
    remember = BooleanField('Lembrar-me')
    submit = SubmitField('Entrar')

class RegistrationForm(FlaskForm):
    username = StringField('Nome de Usuário', 
                          validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', 
                       validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Senha', 
                            validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirmar Senha', 
                                    validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Cadastrar')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Nome de usuário já está em uso. Por favor escolha outro.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email já está em uso. Por favor use outro email.')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Senha Atual', validators=[DataRequired()])
    new_password = PasswordField('Nova Senha', 
                                validators=[DataRequired(), Length(min=8)])
    confirm_new_password = PasswordField('Confirmar Nova Senha', 
                                        validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Alterar Senha')

class ResetPasswordRequestForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Solicitar Reset de Senha')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Nova Senha', 
                            validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirmar Nova Senha', 
                                    validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Resetar Senha')

class DatasetUploadForm(FlaskForm):
    dataset_file = FileField('Arquivo de Dados', validators=[DataRequired()])
    description = TextAreaField('Descrição', 
                               validators=[Optional(), Length(max=500)])
    is_public = BooleanField('Tornar público')
    submit = SubmitField('Upload Dataset')

class MLModelForm(FlaskForm):
    name = StringField('Nome do Modelo', 
                      validators=[DataRequired(), Length(max=255)])
    model_type = SelectField('Tipo de Modelo', 
                           choices=[('regression', 'Regressão'), 
                                   ('classification', 'Classificação')],
                           validators=[DataRequired()])
    algorithm = SelectField('Algoritmo', 
                          choices=[('random_forest', 'Random Forest'),
                                  ('xgboost', 'XGBoost'),
                                  ('decision_tree', 'Decision Tree'),
                                  ('svm', 'SVM')],
                          validators=[DataRequired()])
    target_variable = StringField('Variável Alvo', 
                                 validators=[DataRequired()])
    test_size = FloatField('Tamanho do Teste', 
                          default=0.2, 
                          validators=[NumberRange(min=0.1, max=0.5)])
    submit = SubmitField('Treinar Modelo')

class AirQualityDataForm(FlaskForm):
    location = StringField('Localização', validators=[DataRequired()])
    latitude = FloatField('Latitude', validators=[DataRequired()])
    longitude = FloatField('Longitude', validators=[DataRequired()])
    pm25 = FloatField('PM2.5', validators=[Optional()])
    pm10 = FloatField('PM10', validators=[Optional()])
    no2 = FloatField('NO2', validators=[Optional()])
    so2 = FloatField('SO2', validators=[Optional()])
    co = FloatField('CO', validators=[Optional()])
    o3 = FloatField('O3', validators=[Optional()])
    temperature = FloatField('Temperatura (°C)', validators=[Optional()])
    humidity = FloatField('Umidade (%)', validators=[Optional()])
    wind_speed = FloatField('Velocidade do Vento (m/s)', validators=[Optional()])
    submit = SubmitField('Adicionar Dados')