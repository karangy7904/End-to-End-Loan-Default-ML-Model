from pathlib import Path


def test_expected_project_files_exist():
    root = Path(__file__).parents[1]
    required = ["README.md", "requirements.txt", "src/train.py", "src/predict.py"]
    assert all((root / item).exists() for item in required)

