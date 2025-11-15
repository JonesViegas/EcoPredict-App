
# 🌿 **EcoPredict - Sistema de Monitoramento da Qualidade do Ar**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)

Sistema web completo para monitoramento, análise e previsão da qualidade do ar utilizando machine learning.

---

## 🚀 **Funcionalidades Principais**

### 🔐 Autenticação e Segurança

* Sistema de login seguro com validações
* Troca de senha e recuperação de conta
* Proteção contra ataques CSRF e brute force
* Diferenciação entre usuários comuns e administradores

### 📊 Dashboard e Visualização

* Dashboard interativo com métricas em tempo real
* Mapa interativo com dados de qualidade do ar
* Gráficos e relatórios detalhados
* Indicadores de AQI (Índice de Qualidade do Ar)

### 🤖 Machine Learning

* Treinamento de modelos preditivos (Random Forest, XGBoost, SVM etc.)
* Validação automática com precisão > 85%
* Previsões em tempo real
* Análise de correlação entre variáveis

### 📁 Gerenciamento de Dados

* Upload de datasets em CSV e Excel
* Processamento automático e validação de dados
* Cálculo de métricas de qualidade
* Suporte a datasets públicos e privados

### 🌐 Fontes de Dados Externas

* **OpenAQ**: dados globais
* **INMET**: dados meteorológicos
* **INPE**: focos de calor
* Integração e processamento automático

---

## 🛠️ Tecnologias Utilizadas

### **Backend**

* Python 3.9+
* Flask
* SQLAlchemy
* PostgreSQL
* Flask-Login
* Flask-WTF

### **Machine Learning**

* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Joblib

### **Frontend**

* Bootstrap 5
* JavaScript
* Chart.js
* Font Awesome
* Folium

### **APIs Externas**

* OpenAQ API v3
* INMET API
* INPE Queimadas API

---

## 📦 Instalação e Configuração

### **Pré-requisitos**

* Python 3.9+
* PostgreSQL 13+
* pip

---

### **1. Clone o repositório**

```bash
git clone https://github.com/JonesViegas/EcoPredict-App.git
cd ecopredict
```

### **2. Configure o ambiente virtual**

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### **3. Instale as dependências**

```bash
pip install -r requirements.txt
```

### **4. Configure o arquivo `.env`**

```bash
cp .env.example .env
```

Edite com:

```
DATABASE_URL=postgresql://usuario:senha@localhost:5432/nomeprojeto_db
SECRET_KEY=sua-chave-secreta-aqui
```

### **5. Configure o PostgreSQL**

```sql
CREATE DATABASE ecopredict_db;
CREATE USER ecouser WITH PASSWORD 'suasenha';
GRANT ALL PRIVILEGES ON DATABASE ecopredict_db TO ecouser;
```

### **6. Execute as migrações**

```bash
flask db init
flask db migrate -m "Initial tables"
flask db upgrade
```

### **7. Crie o usuário administrador**

```bash
python -c "
from app import create_app, db
from app.models import User
app = create_app()
with app.app_context():
    admin = User(username='admin', email='email@projeto.com', is_admin=True)
    admin.set_password('Admin')
    db.session.add(admin)
    db.session.commit()
    print('Admin criado: aemail@projeto.com / Admin')
"
```

### **8. Execute a aplicação**

```bash
flask run
```

Acesse: [http://localhost:5000](http://localhost:5000)

---

## 🔧 Comandos Úteis

### **Desenvolvimento**

```bash
flask run --debug
flask db migrate -m "Descrição da migração"
flask db upgrade
```

### **Administração**

```bash
flask create-admin
flask shell
```

### **Backup**

```bash
python backup.py
```

---

## 🗃️ Estrutura do Projeto

```text
ecopredict/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   ├── auth.py
│   ├── external_data.py
│   ├── ml_models.py
│   ├── utils.py
│   ├── services/
│   │   └── api_client.py
│   ├── templates/
│   └── static/
├── migrations/
├── instance/
│   ├── uploads/
│   └── ml_models/
├── requirements.txt
├── config.py
└── run.py
```

---

## 📊 Modelos de Dados

### **User**

* Autenticação
* Permissões admin/user

### **Dataset**

* Metadados
* Métricas de qualidade
* Relacionamento com usuários

### **MLModel**

* Algoritmos treinados
* Métricas de performance

### **AirQualityData**

* Dados de qualidade do ar
* Coordenadas
* Timestamps e medições

---

## 🔐 Segurança

* Hash de senhas com bcrypt
* CSRF Protection
* Rate Limiting
* Headers de segurança
* Sanitização de uploads

---

## 🌐 APIs Disponíveis

### **API Interna**

```
GET /api/air-quality-data
POST /api/predict
GET /api/inmet/stations
```

### **Endpoints Principais**

* `/` – Página inicial
* `/dashboard` – Painel
* `/external/sources` – Dados externos
* `/ml-models` – Machine Learning
* `/reports` – Relatórios

---

## 🚀 Deploy

### **Render.com (recomendado)**

* Conectar GitHub
* Variáveis de ambiente
* Deploy automático

### **Ambiente de Produção**

```
SECRET_KEY=sua-chave-forte
DATABASE_URL=postgresql://usuario:senha@host:5432/nomeprojeto
SECURITY_PASSWORD_SALT=salt
FLASK_ENV=production
```

---

## 📈 Exemplos de Uso

### **1. Coleta de Dados**

* Menu **Dados Externos**
* OpenAQ, INMET, INPE

### **2. Treinamento de Modelos**

* Upload de dataset
* Seleção de features
* Treinamento e avaliação

### **3. Análise de Correlação**

* Relatórios
* Correlações
* Estatísticas

---

## 🐛 Solução de Problemas

### **Erro no banco:**

```bash
flask db downgrade base
flask db upgrade
```

### **Erro de importação:**

```bash
pip install --force-reinstall -r requirements.txt
```

### **Permissões:**

```bash
chmod 755 instance/uploads
chmod 755 instance/ml_models
```

---

## 🤝 Contribuindo

1. Faça fork
2. Crie branch
3. Commit
4. Push
5. Pull Request

---

## 📄 Licença

Projeto sob licença **MIT**.

---

## 👥 Autores

Jones Carlos Viegas – Desenvolvimento Inicial
GitHub: **(https://github.com/JonesViegas/EcoPredict-App)**

---

Se quiser, posso gerar também:

✅ Versão reduzida
✅ Versão com sumário automático
✅ README com imagens e GIFs
✅ README profissional estilo template premium

Só pedir!
