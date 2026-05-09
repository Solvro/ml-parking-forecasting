from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / 'Data'
FEATURE_ENGINEERING_DIR = PROJECT_ROOT/ 'src' / 'feature_engineering'
MODELLING_DIR = PROJECT_ROOT / 'src' / 'modelling'
PREPROCESSING_DIR = PROJECT_ROOT / 'src' / 'preprocessing'


if __name__ == "__main__":
    # Testujemy, czy ścieżki są poprawne
    print(f"Data directory: {DATA_DIR}")
    print(f"Feature Engineering directory: {FEATURE_ENGINEERING_DIR}")
    print(f"Models directory: {MODELLING_DIR}")
    print(f"Preprocessing directory: {PREPROCESSING_DIR}")