def validate_air_quality_data(data: dict) -> dict:
    errors = {}

    required_fields = [
        "location", "latitude", "longitude", "pm25", "pm10",
        "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"
    ]

    # Verifica campos obrigatórios
    for field in required_fields:
        if data.get(field) in [None, ""]:
            errors[field] = f"O campo '{field}' é obrigatório."

    # Verifica se campos numéricos realmente são numéricos
    numeric_fields = [
        "latitude", "longitude", "pm25", "pm10", "no2", "so2",
        "co", "o3", "temperature", "humidity", "wind_speed"
    ]

    for field in numeric_fields:
        if field in data and data[field] not in [None, ""]:
            try:
                float(data[field])
            except ValueError:
                errors[field] = f"O campo '{field}' deve ser numérico."
    
    return {
        "valid": len(errors) == 0,
        "error": errors
    }
