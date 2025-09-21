import os

def create_folder_structure():
    folders = [
        "ml_models",
        "ml_models/config",
        "ml_models/data",
        "ml_models/data/raw/urls",
        "ml_models/data/raw/emails", 
        "ml_models/data/raw/transactions",
        "ml_models/data/processed/urls",
        "ml_models/data/processed/emails",
        "ml_models/data/processed/transactions",
        "ml_models/data/collectors",
        "ml_models/preprocessing",
        "ml_models/models",
        "ml_models/training",
        "ml_models/inference",
        "ml_models/saved/models",
        "ml_models/saved/vectorizers",
        "ml_models/saved/scalers",
        "ml_models/evaluation",
        "ml_models/api",
        "ml_models/utils"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # Create __init__.py files
        init_file = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("")

if __name__ == "__main__":
    create_folder_structure()
    print("Project structure created successfully!")