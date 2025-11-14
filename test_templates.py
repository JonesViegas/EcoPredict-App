# test_templates.py
from app import create_app
from app.forms import LoginForm, RegistrationForm, ChangePasswordForm, ResetPasswordRequestForm, ResetPasswordForm

app = create_app()

def test_templates():
    with app.app_context():
        with app.test_request_context():
            print("=== TESTANDO TEMPLATES ===")
            
            # Testar cada formulário
            forms = {
                'login': LoginForm(),
                'register': RegistrationForm(),
                'change_password': ChangePasswordForm(),
                'reset_password_request': ResetPasswordRequestForm(),
                'reset_password': ResetPasswordForm()
            }
            
            for form_name, form in forms.items():
                print(f"\n📋 Testando {form_name}:")
                for field_name, field in form._fields.items():
                    print(f"  ✅ {field_name}: {type(field).__name__}")
            
            print("\n=== TESTE CONCLUÍDO ===")

if __name__ == '__main__':
    test_templates()