from models.src.utils import load_joblib, TRAINING_ARTIFACTS_FILE

artifacts = load_joblib(TRAINING_ARTIFACTS_FILE)
features = artifacts["model_feature_names"]
print(features)