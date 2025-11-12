from app import create_app, db
from app.models import User

def create_admin():
    app = create_app()
    
    with app.app_context():
        # Verificar se o admin já existe
        if User.query.filter_by(email='admin@ecopredict.com').first():
            print("❌ Usuário admin já existe!")
            return
        
        # Criar usuário admin
        admin = User(
            username='admin',
            email='admin@ecopredict.com', 
            is_admin=True
        )
        admin.set_password('Admin123!')
        
        db.session.add(admin)
        db.session.commit()
        
        print("✅ Usuário admin criado com sucesso!")
        print("📧 Email: admin@ecopredict.com")
        print("🔑 Senha: Admin123!")
        print("⚠️  ALTERE ESTA SENHA APÓS O PRIMEIRO LOGIN!")

if __name__ == '__main__':
    create_admin()