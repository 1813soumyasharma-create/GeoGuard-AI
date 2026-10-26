"""
Landslide Susceptibility Prediction System
Module: Data Generator & GIS Synthesizer
Author: 3rd Year Engineering AI & Geotech Project
Description: Synthesizes geotechnical, morphometric, hydrological, and environmental
factors based on empirical geomorphological failure criteria (Mohr-Coulomb & GSI standards)
for training AI models and spatial susceptibility mapping.
"""

import numpy as np
import pandas as pd
import os

# Set seed for reproducible engineering benchmarks
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Geological and Soil classifications typical in mountainous landslide hotspots (e.g. Western Ghats / Himalayas)
LITHOLOGY_TYPES = ["Gneiss", "Charnockite", "Granite", "Schist", "Weathered Regolith", "Colluvium"]
SOIL_TYPES = ["Lateritic Soil", "Sandy Loam", "Clayey Loam", "Debris/Colluvium", "Silty Clay"]
LAND_USE_TYPES = ["Dense Forest", "Tea/Coffee Plantation", "Scrubland/Grassland", "Agricultural Terraces", "Settlement/Built-up", "Barren/Quarry"]

# Geotechnical vulnerability weights for categorical factors (based on empirical engineering literature)
LITHOLOGY_VULNERABILITY = {
    "Granite": 0.15,
    "Charnockite": 0.25,
    "Gneiss": 0.45,
    "Schist": 0.65,
    "Colluvium": 0.85,
    "Weathered Regolith": 0.90
}

SOIL_VULNERABILITY = {
    "Lateritic Soil": 0.35,
    "Clayey Loam": 0.45,
    "Silty Clay": 0.55,
    "Sandy Loam": 0.70,
    "Debris/Colluvium": 0.90
}

LANDUSE_VULNERABILITY = {
    "Dense Forest": 0.10,          # High root cohesion & canopy interception
    "Tea/Coffee Plantation": 0.55, # Shallow root system, slope clearing
    "Scrubland/Grassland": 0.45,
    "Agricultural Terraces": 0.60, # Irrigation saturation & altered slope
    "Settlement/Built-up": 0.75,   # Surcharge loads & altered drainage
    "Barren/Quarry": 0.92          # Completely exposed cut face, zero cohesion
}


def generate_landslide_dataset(n_samples: int = 2500, region_center: tuple = (11.60, 76.10)) -> pd.DataFrame:
    """
    Generates a realistic GIS landslide dataset containing 13 conditioning factors + 1 binary landslide label.
    Coordinates are anchored around an active landslide corridor (e.g., Wayanad / Western Ghats belt).
    
    Parameters:
        n_samples (int): Number of survey / grid locations to synthesize.
        region_center (tuple): (Latitude, Longitude) center coordinates.
        
    Returns:
        pd.DataFrame: Tabular dataset with geotechnical & remote sensing factors and ground-truth failure label.
    """
    lat_center, lon_center = region_center
    
    # 1. Spatial Coordinates (approx. 35km x 35km region)
    latitudes = lat_center + np.random.uniform(-0.18, 0.18, size=n_samples)
    longitudes = lon_center + np.random.uniform(-0.18, 0.18, size=n_samples)
    
    # 2. Topographic & Morphometric Factors
    # Elevation: Mountainous relief from 350m to 2100m
    elevation = np.random.triangular(left=350, mode=950, right=2150, size=n_samples)
    
    # Slope Angle (Degrees): 0 to 62 degrees
    # Higher elevations correlate moderately with steeper slopes
    slope_base = np.random.beta(a=2.2, b=3.5, size=n_samples) * 60.0
    slope_deg = np.clip(slope_base + (elevation - 350) / 1800.0 * 8.0 + np.random.normal(0, 3.5, size=n_samples), 1.0, 64.0)
    
    # Aspect (Degrees): 0 to 360 degrees
    aspect_deg = np.random.uniform(0.0, 360.0, size=n_samples)
    
    # Plan Curvature: Concave (<0, water converging) vs Convex (>0, water diverging)
    plan_curvature = np.random.normal(loc=-0.1, scale=1.4, size=n_samples)
    plan_curvature = np.clip(plan_curvature, -3.5, 3.5)
    
    # Profile Curvature: Deceleration/Acceleration of down-slope flow
    profile_curvature = np.random.normal(loc=0.0, scale=1.2, size=n_samples)
    profile_curvature = np.clip(profile_curvature, -3.0, 3.0)
    
    # Topographic Wetness Index (TWI): ln(a / tan(beta)). Typically 2.5 - 15.0
    # Flatter slopes have higher TWI; steep crests have low TWI
    rad_slope = np.radians(np.clip(slope_deg, 1.0, 89.0))
    twi = np.clip(np.log(np.random.uniform(80, 4500, size=n_samples) / np.tan(rad_slope)), 2.0, 16.5)
    
    # 3. Hydrological & Meteorological Trigger Factors
    # Antecedent 72-hour cumulative precipitation (mm): Critical trigger for slope failure
    rainfall_72h = np.random.triangular(left=40, mode=180, right=450, size=n_samples)
    
    # Distance to Streams / Drainage Network (m): Toe erosion & seepage line
    dist_to_stream_m = np.random.exponential(scale=350, size=n_samples)
    dist_to_stream_m = np.clip(dist_to_stream_m, 10.0, 2800.0)
    
    # 4. Anthropogenic & Geological Lineament Factors
    # Distance to Roads / Cut Slopes (m): Toe removal & blast fracturing
    dist_to_road_m = np.random.exponential(scale=450, size=n_samples)
    dist_to_road_m = np.clip(dist_to_road_m, 5.0, 3200.0)
    
    # Distance to Tectonic Faults / Lineaments (m)
    dist_to_fault_m = np.random.exponential(scale=800, size=n_samples)
    dist_to_fault_m = np.clip(dist_to_fault_m, 15.0, 5000.0)
    
    # 5. Land Cover & Ecological Factors
    # NDVI: Normalized Difference Vegetation Index (-0.1 to 0.85)
    ndvi = np.clip(np.random.normal(loc=0.55, scale=0.18, size=n_samples), -0.05, 0.88)
    
    # Categorical attributes
    lithology = np.random.choice(
        LITHOLOGY_TYPES, 
        size=n_samples, 
        p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08]
    )
    
    soil_type = np.random.choice(
        SOIL_TYPES, 
        size=n_samples, 
        p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )
    
    land_use = np.random.choice(
        LAND_USE_TYPES, 
        size=n_samples, 
        p=[0.38, 0.26, 0.12, 0.12, 0.07, 0.05]
    )
    
    # 6. Physical Geotechnical Failure Index (Ground Truth Synthesis)
    # Based on Mohr-Coulomb failure criterion & limit equilibrium principles:
    # Failure is driven by:
    # + Slope angle (gravitational shear stress component)
    # + Rainfall & TWI (pore water pressure build-up reducing effective stress)
    # + Proximity to road cut (unsupported toe) & stream (scour erosion)
    # + Proximity to faults (weak shear planes)
    # + High curvature concavity (funneling debris & runoff)
    # + Weak lithology & uncohesive debris soil
    # - Resisting factors: Root cohesion (NDVI), competent bedrock, gentle slope
    
    lith_score = np.array([LITHOLOGY_VULNERABILITY[l] for l in lithology])
    soil_score = np.array([SOIL_VULNERABILITY[s] for s in soil_type])
    lu_score = np.array([LANDUSE_VULNERABILITY[lu] for lu in land_use])
    
    # Normalized components [0, 1]
    norm_slope = np.clip((slope_deg - 10.0) / 45.0, 0.0, 1.2)
    norm_rain = np.clip((rainfall_72h - 80.0) / 320.0, 0.0, 1.2)
    norm_road = np.clip(1.0 - (dist_to_road_m / 800.0), 0.0, 1.0)
    norm_stream = np.clip(1.0 - (dist_to_stream_m / 600.0), 0.0, 1.0)
    norm_fault = np.clip(1.0 - (dist_to_fault_m / 1500.0), 0.0, 1.0)
    norm_ndvi_vulnerability = np.clip((0.75 - ndvi) / 0.75, 0.0, 1.0)
    norm_concavity = np.clip((-plan_curvature + 1.5) / 3.0, 0.0, 1.0)
    
    # Composite Geotechnical Hazard Score
    # Slope (0.28) + Rain (0.24) + Road (0.12) + Soil/Lith (0.14) + Stream (0.08) + NDVI (0.08) + Fault (0.06)
    hazard_score = (
        0.28 * norm_slope +
        0.24 * norm_rain +
        0.12 * norm_road +
        0.08 * norm_stream +
        0.08 * soil_score +
        0.06 * lith_score +
        0.06 * norm_fault +
        0.08 * norm_ndvi_vulnerability +
        0.05 * lu_score +
        0.05 * norm_concavity
    )
    
    # Convert to probability using logistic function with realistic noise
    # Adding noise to simulate unmeasured factors (groundwater fluctuations, seismic tremors)
    noise = np.random.logistic(loc=0.0, scale=0.18, size=n_samples)
    raw_logit = (hazard_score - 0.46) * 7.5 + noise
    failure_probability = 1.0 / (1.0 + np.exp(-raw_logit))
    
    # Binary classification label: 1 = Landslide occurred, 0 = Stable slope
    landslide_label = (failure_probability >= 0.50).astype(int)
    
    # Assemble comprehensive DataFrame
    df = pd.DataFrame({
        "location_id": [f"LOC_{i+1:04d}" for i in range(n_samples)],
        "latitude": np.round(latitudes, 5),
        "longitude": np.round(longitudes, 5),
        "elevation_m": np.round(elevation, 1),
        "slope_deg": np.round(slope_deg, 2),
        "aspect_deg": np.round(aspect_deg, 1),
        "plan_curvature": np.round(plan_curvature, 3),
        "profile_curvature": np.round(profile_curvature, 3),
        "twi": np.round(twi, 2),
        "rainfall_72h_mm": np.round(rainfall_72h, 1),
        "dist_to_fault_m": np.round(dist_to_fault_m, 1),
        "dist_to_stream_m": np.round(dist_to_stream_m, 1),
        "dist_to_road_m": np.round(dist_to_road_m, 1),
        "ndvi": np.round(ndvi, 3),
        "lithology": lithology,
        "soil_type": soil_type,
        "land_use": land_use,
        "true_failure_prob": np.round(failure_probability, 4),
        "landslide": landslide_label
    })
    
    return df


def save_dataset(df: pd.DataFrame, file_path: str = "data/landslide_data.csv") -> str:
    """Saves generated dataset to disk ensuring directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[Data Generator] Successfully saved {len(df)} samples to {file_path}")
    return file_path


if __name__ == "__main__":
    df = generate_landslide_dataset(n_samples=3000)
    save_dataset(df)
    print(f"Landslide distribution:\n{df['landslide'].value_counts(normalize=True)}")
