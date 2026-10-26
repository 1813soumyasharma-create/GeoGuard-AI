"""
Landslide Susceptibility Prediction System
Module: AI Model Trainer & Evaluator
Author: 3rd Year Engineering Project
Description: Trains and benchmarks multiple ML/AI models (Random Forest, XGBoost, MLP Neural Network, Logistic Regression)
on geotechnical and remote sensing conditioning factors. Evaluates ROC-AUC, F1, precision, recall, and extracts feature importances.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
)
from xgboost import XGBClassifier

from src.data_generator import generate_landslide_dataset, save_dataset

# Core Feature Definitions
NUMERICAL_FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "plan_curvature",
    "profile_curvature", "twi", "rainfall_72h_mm", "dist_to_fault_m",
    "dist_to_stream_m", "dist_to_road_m", "ndvi"
]

CATEGORICAL_FEATURES = [
    "lithology", "soil_type", "land_use"
]

TARGET_COLUMN = "landslide"


def build_preprocessor() -> ColumnTransformer:
    """Creates a standard preprocessing pipeline for GIS tabular conditioning factors."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extracts transformed feature names after one-hot encoding."""
    num_cols = NUMERICAL_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_cols + cat_cols


def train_and_benchmark_models(df: pd.DataFrame, models_dir: str = "models") -> Dict[str, Any]:
    """
    Trains multiple AI classifiers, evaluates comprehensive metrics, and persists models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    # Stratified 80-20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Initialize Candidate Classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss"
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=350,
            alpha=0.001,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=600,
            C=1.0,
            random_state=42
        )
    }
    
    results = {}
    trained_pipelines = {}
    
    print("\n" + "="*60)
    print("LANDSLIDE SUSCEPTIBILITY AI BENCHMARKING ENGINE")
    print("="*60)
    
    for name, clf in candidate_models.items():
        print(f"\n[Training] {name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predictions & Probabilities
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC curve points (subsampled for compact JSON storage)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 50)
        roc_data = {
            "fpr": [float(x) for x in fpr[::step]],
            "tpr": [float(x) for x in tpr[::step]]
        }
        
        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
            "roc_curve": roc_data
        }
        
        print(f" -> Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        # Save individual model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, os.path.join(models_dir, f"{clean_name}.joblib"))
        
    # Extract Global Feature Importance (from Random Forest & XGBoost)
    rf_pipeline = trained_pipelines["Random Forest"]
    rf_clf = rf_pipeline.named_steps["classifier"]
    feature_names = get_feature_names(rf_pipeline.named_steps["preprocessor"])
    
    rf_importances = rf_clf.feature_importances_
    sorted_idx = np.argsort(rf_importances)[::-1]
    
    feature_importance_list = [
        {"feature": feature_names[i], "importance": round(float(rf_importances[i]), 4)}
        for i in sorted_idx[:15]
    ]
    
    # Save benchmark metrics and metadata to JSON
    metadata = {
        "dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "models": results,
        "feature_importances": feature_importance_list,
        "best_model_by_auc": max(results.keys(), key=lambda k: results[k]["roc_auc"]),
        "best_model_by_f1": max(results.keys(), key=lambda k: results[k]["f1_score"])
    }
    
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n[Saved] Models and benchmark metrics saved to {models_dir}/")
    print(f"[Best Model] Ranked by ROC-AUC: {metadata['best_model_by_auc']} ({results[metadata['best_model_by_auc']]['roc_auc']:.4f})")
    
    return metadata


class LandslidePredictor:
    """Inference helper for loading models and executing spatial or single-point predictions."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.metrics = {}
        self._load_models()
        
    def _load_models(self):
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
                
        mapping = {
            "Random Forest": "random_forest.joblib",
            "XGBoost": "xgboost.joblib",
            "Neural Network (MLP)": "neural_network_mlp.joblib",
            "Logistic Regression": "logistic_regression.joblib"
        }
        
        for name, filename in mapping.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                
    def predict_point(self, model_name: str, input_dict: Dict[str, Any]) -> Tuple[float, int]:
        """Runs single-point prediction, returning (failure_probability, binary_prediction)."""
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
            
        model = self.models[model_name]
        df_in = pd.DataFrame([input_dict])
        prob = float(model.predict_proba(df_in)[0, 1])
        pred = int(model.predict(df_in)[0])
        return prob, pred
        
    def predict_dataframe(self, model_name: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Runs batch prediction over a pandas DataFrame."""
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
            
        model = self.models[model_name]
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)
        return probs, preds


if __name__ == "__main__":
    data_path = "data/landslide_data.csv"
    if not os.path.exists(data_path):
        df = generate_landslide_dataset(n_samples=3000)
        save_dataset(df, data_path)
    else:
        df = pd.read_csv(data_path)
        
    train_and_benchmark_models(df)
