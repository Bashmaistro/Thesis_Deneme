import os

project_name = "project_name"

# Oluşturulacak dosyalar (klasör yapısı dosya path ile tanımlanıyor)
files = [
    "configs/default.yaml",
    "configs/experiment1.yaml",
    "data/raw/data.txt",
    "data/processed/data.txt",
    "notebooks/example.ipynb",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/cnn.py",
    "src/models/transformer.py",
    "src/utils/__init__.py",
    "src/utils/metrics.py",
    "src/utils/logger.py",
    "src/utils/data_loader.py",
    "src/training/__init__.py",
    "src/training/trainer.py",
    "src/training/evaluator.py",
    "src/inference/predictor.py",
    "src/tests/test_models.py",
    "src/tests/test_utils.py",
    "experiments/exp1/results.txt",
    "experiments/exp2/results.txt",
    "scripts/train.sh",
    "scripts/run_inference.sh",
    "requirements.txt",
    "environment.yml",
    "README.md"
]

for file in files:
    path = os.path.join(project_name, file)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # klasör varsa geç
    with open(path, "w") as f:
        pass  # boş dosya oluştur

print(f"✅ '{project_name}' içinde {len(files)} dosya oluşturuldu.")
