# check_models.py
from app import create_app, db
from app.models import User, Dataset, MLModel, Alert, SystemLog

app = create_app()

with app.app_context():
    print("=== VERIFICANDO MODELOS ===")
    
    try:
        # Verificar User
        users = User.query.limit(1).all()
        print("✅ Modelo User: OK")
    except Exception as e:
        print(f"❌ Erro no modelo User: {e}")
    
    try:
        # Verificar Dataset
        datasets = Dataset.query.limit(1).all()
        print("✅ Modelo Dataset: OK")
    except Exception as e:
        print(f"❌ Erro no modelo Dataset: {e}")
    
    try:
        # Verificar MLModel
        models = MLModel.query.limit(1).all()
        print("✅ Modelo MLModel: OK")
    except Exception as e:
        print(f"❌ Erro no modelo MLModel: {e}")
    
    try:
        # Verificar Alert
        alerts = Alert.query.limit(1).all()
        print("✅ Modelo Alert: OK")
        if alerts:
            print(f"   Exemplo: {alerts[0]}")
    except Exception as e:
        print(f"❌ Erro no modelo Alert: {e}")
    
    try:
        # Verificar SystemLog
        logs = SystemLog.query.limit(1).all()
        print("✅ Modelo SystemLog: OK")
    except Exception as e:
        print(f"❌ Erro no modelo SystemLog: {e}")
    
    print("=== VERIFICAÇÃO CONCLUÍDA ===")