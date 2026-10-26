"""
Unit and Integration Tests for Landslide Susceptibility System
"""

import unittest
import numpy as np
import pandas as pd
from src.data_generator import generate_landslide_dataset, LITHOLOGY_TYPES, SOIL_TYPES
from src.geotechnical_advisor import (
    classify_susceptibility_zone, generate_geotechnical_advisory,
    SUSCEPTIBILITY_ZONES
)
from src.model_trainer import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.df = generate_landslide_dataset(n_samples=150)

    def test_dataset_shape_and_columns(self):
        self.assertEqual(len(self.df), 150)
        expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["location_id", "latitude", "longitude", "true_failure_prob", "landslide"]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Missing expected column: {col}")

    def test_realistic_ranges(self):
        self.assertTrue(self.df["slope_deg"].between(0.0, 70.0).all(), "Slope out of physical bounds")
        self.assertTrue(self.df["elevation_m"].between(200.0, 3000.0).all(), "Elevation out of bounds")
        self.assertTrue(self.df["rainfall_72h_mm"].between(0.0, 600.0).all(), "Rainfall out of bounds")
        self.assertTrue(self.df["ndvi"].between(-0.2, 1.0).all(), "NDVI out of bounds")
        self.assertTrue(set(self.df["landslide"].unique()).issubset({0, 1}), "Invalid binary failure label")


class TestGeotechnicalAdvisor(unittest.TestCase):
    def test_zone_classification(self):
        self.assertEqual(classify_susceptibility_zone(0.10)["zone"], "Very Low")
        self.assertEqual(classify_susceptibility_zone(0.30)["zone"], "Low")
        self.assertEqual(classify_susceptibility_zone(0.50)["zone"], "Moderate")
        self.assertEqual(classify_susceptibility_zone(0.70)["zone"], "High")
        self.assertEqual(classify_susceptibility_zone(0.90)["zone"], "Very High")

    def test_critical_trigger_detection(self):
        critical_input = {
            "slope_deg": 42.0,
            "rainfall_72h_mm": 280.0,
            "dist_to_road_m": 45.0,
            "dist_to_stream_m": 80.0,
            "dist_to_fault_m": 250.0,
            "ndvi": 0.15,
            "lithology": "Weathered Regolith",
            "soil_type": "Debris/Colluvium",
            "land_use": "Barren/Quarry",
            "twi": 11.5
        }
        advisory = generate_geotechnical_advisory(critical_input, failure_probability=0.88)
        self.assertEqual(advisory["susceptibility_zone"], "Very High")
        self.assertEqual(advisory["alert_level"], "CRITICAL EMERGENCY")
        
        # Verify mitigation plans include civil engineering interventions
        structural_texts = " ".join(advisory["structural_mitigation"])
        drainage_texts = " ".join(advisory["drainage_mitigation"])
        self.assertIn("Soil Nailing", structural_texts)
        self.assertIn("Drain Pipes", drainage_texts)


if __name__ == "__main__":
    unittest.main()
