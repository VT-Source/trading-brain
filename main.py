from sqlalchemy import inspect, text

def run_analysis():
    if engine is None: 
        print("❌ Engine non configuré.")
        return
    
    try:
        with engine.connect() as conn:
            # On vérifie la base actuelle
            current_db = conn.execute(text("SELECT current_database()")).scalar()
            print(f"🏠 CONNECTÉ À LA BASE : {current_db}")

            # On liste absolument toutes les tables visibles
            inspector = inspect(engine)
            schemas = inspector.get_schema_names()
            print(f"📁 SCHÉMAS DISPONIBLES : {schemas}")

            for schema in schemas:
                tables = inspector.get_table_names(schema=schema)
                if tables:
                    print(f"📍 Tables dans le schéma '{schema}': {tables}")

    except Exception as e:
        print(f"❌ ERREUR DIAGNOSTIC : {e}")
