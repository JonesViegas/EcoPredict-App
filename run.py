from app import create_app, db

app = create_app()

@app.shell_context_processor
def make_shell_context():
    # Importar modelos aqui para evitar circular imports
    from app.models import User, Dataset, MLModel, AirQualityData
    return {
        'db': db,
        'User': User,
        'Dataset': Dataset, 
        'MLModel': MLModel,
        'AirQualityData': AirQualityData
    }

if __name__ == '__main__':
    app.run(debug=True)