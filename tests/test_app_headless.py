"""
Smoke test to ensure app modules and predictor load cleanly in Streamlit context.
"""
import unittest
import pandas as pd
from src.model_trainer import LandslidePredictor
from src.geotechnical_advisor import generate_geotechnical_advisory, classify_susceptibility_zone

class TestAppHeadless(unittest.TestCase):
    def test_predictor_loaded(self):
        predictor = LandslidePredictor(models_dir="models")
        self.assertIn("Random Forest", predictor.models)
        self.assertIn("XGBoost", predictor.models)
        self.assertIn("Neural Network (MLP)", predictor.models)
        self.assertIn("Logistic Regression", predictor.models)

    def test_inference_and_advisory(self):
        predictor = LandslidePredictor(models_dir="models")
        sample = {
            "elevation_m": 850.0,
            "slope_deg": 35.0,
            "aspect_deg": 180.0,
            "plan_curvature": -1.2,
            "profile_curvature": 0.5,
            "twi": 8.5,
            "rainfall_72h_mm": 210.0,
            "dist_to_fault_m": 450.0,
            "dist_to_stream_m": 120.0,
            "dist_to_road_m": 85.0,
            "ndvi": 0.32,
            "lithology": "Gneiss",
            "soil_type": "Debris/Colluvium",
            "land_use": "Tea/Coffee Plantation"
        }
        prob, pred = predictor.predict_point("XGBoost", sample)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertIn(pred, [0, 1])
        
        adv = generate_geotechnical_advisory(sample, prob)
        self.assertIn("susceptibility_zone", adv)
        self.assertIn("structural_mitigation", adv)
        self.assertIn("drainage_mitigation", adv)

if __name__ == "__main__":
    unittest.main()
