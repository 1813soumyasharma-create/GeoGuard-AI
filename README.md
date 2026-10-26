# 🛡️ GeoGuard-AI: AI-Powered Landslide Susceptibility Prediction & Early Warning System

> **Third-Year Engineering Capstone Project**  
> *Interdisciplinary Laboratory of Geotechnical Engineering, Remote Sensing (GIS), and Artificial Intelligence*

---

## 📌 1. Project Abstract & Motivation

Landslides are among the most catastrophic natural disasters in high-relief mountainous terrains across the world (e.g., Western Ghats, Wayanad 2024, Nilgiris, and the Himalayan tectonic belt), inflicting catastrophic loss of human life, community displacement, and severe highway infrastructure destruction during torrential monsoon periods.

Traditional geotechnical slope stability analysis relies heavily on the **Limit Equilibrium Method (LEM)** (such as Bishop's simplified method or Janbu's method) and Finite Element Modeling (FEM). While rigorous, these methods require extensive on-site drilling, continuous borehole extraction, and laboratory shear tests (triaxial/direct shear), making them cost-prohibitive and computationally intractable across regional watersheds ($100s \text{ km}^2$).

**GeoGuard-AI** bridges this critical engineering gap by combining **multi-source geospatial remote sensing features**, **geotechnical soil mechanics**, and **advanced Machine Learning (Tree Ensembles & Deep Learning)** into an end-to-end, real-time spatial Landslide Susceptibility Mapping (LSM) and Early Warning System (EWS).

---

## 📐 2. Geotechnical Theoretical Foundation

### 2.1 The Mohr-Coulomb Failure Criterion & Factor of Safety (FoS)
In geotechnical mechanics, slope failure occurs when the downslope gravitational shear stress ($\tau_d$) exceeds the available shear strength ($\tau_f$) along a critical slip surface:

$$\text{FoS} = \frac{\tau_f}{\tau_d} = \frac{c' + (\sigma_n - u_w) \tan \phi'}{\gamma \cdot z \cdot \sin \beta \cdot \cos \beta}$$

Where:
- $c'$: Effective soil cohesion ($\text{kN/m}^2$)
- $\phi'$: Effective internal angle of friction (degrees)
- $\sigma_n$: Total normal stress on the failure plane ($=\gamma \cdot z \cdot \cos^2 \beta$)
- $u_w$: Pore water pressure ($\text{kN/m}^2$) induced by transient water table rise and perched infiltration
- $\beta$: Slope angle (degrees)
- $z$: Depth of shear failure plane (m)
- $\gamma$: Unit weight of soil/regolith ($\text{kN/m}^3$)

### 2.2 Mechanism of Rainfall-Induced Failures
Prolonged antecedent precipitation elevates $u_w$. According to Terzaghi's effective stress principle:
$$\sigma' = \sigma_n - u_w$$
As pore pressure increases, effective normal stress $\sigma'$ approaches zero, dramatically destroying frictional shear resistance $(\sigma' \tan \phi')$ and precipitating liquefaction, debris slides, or translational block failures.

---

## 🛰️ 3. Geotechnical & Remote Sensing Conditioning Factors

GeoGuard-AI processes 14 multi-disciplinary geo-environmental features:

| Category | Factor | Unit / Range | Physical Geotechnical Significance |
| :--- | :--- | :--- | :--- |
| **Morphometric** | Slope Angle ($\beta$) | $0^\circ - 65^\circ$ | Primary driving gravitational shear stress component ($\tau_d \propto \sin \beta$). |
| **Morphometric** | Elevation | $300\text{m} - 2200\text{m}$ | Orographic rainfall gradient, freeze-thaw weathering, and vegetation zonation. |
| **Morphometric** | Aspect | $0^\circ - 360^\circ$ | Downslope azimuth, solar irradiance, evapotranspiration, and windward monsoon face. |
| **Morphometric** | Plan Curvature | $-3.5 \text{ to } +3.5$ | Contour curvature: negative (<0) indicates concave slope (runoff/sediment convergence). |
| **Morphometric** | Profile Curvature | $-3.0 \text{ to } +3.0$ | Rate of slope change: controls downslope flow velocity and acceleration/deceleration. |
| **Hydrological** | Topographic Wetness Index (TWI) | $2.0 - 18.0$ | $\ln(a / \tan \beta)$: Quantifies steady-state soil saturation and pore pressure build-up. |
| **Hydrological** | 72h Antecedent Rainfall | $10\text{mm} - 500\text{mm}$ | Primary dynamic trigger driving pore water overpressurization. |
| **Hydrological** | Distance to Streams | $0\text{m} - 3000\text{m}$ | Fluvial toe scour, undercut banks, and elevated riparian water tables. |
| **Anthropogenic**| Distance to Road Cuts | $0\text{m} - 3000\text{m}$ | Unsupported toe excavation, slope destressing, blasting fractures, and surcharge. |
| **Geological** | Distance to Faults | $0\text{m} - 5000\text{m}$ | Tectonic shear zones, joint micro-fracturing, and reduced Rock Mass Rating (RMR). |
| **Geological** | Bedrock Lithology | Categorical | Rock strength and weathering susceptibility (Granite, Gneiss, Schist, Colluvium). |
| **Geological** | Soil Texture / Regolith | Categorical | Permeability vs cohesion (Laterite, Sandy Loam, Silty Clay, Debris Colluvium). |
| **Ecological** | Land Use / Land Cover (LULC)| Categorical | Root cohesion differences (Dense Forest vs Tea Plantation vs Barren Quarry). |
| **Ecological** | NDVI Index | $-0.10 \text{ to } 0.88$ | Vegetative health, canopy rainfall kinetic dampening, and mechanical root binding. |

---

## 📊 4. AI Model Architecture & Experimental Benchmark

Four diverse algorithmic architectures were trained and evaluated on 3,000 spatial survey nodes using a stratified 80/20 train/test split:

```
+-----------------------------------------------------------------------------------------------+
| Algorithm                | Accuracy (%) | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
+-----------------------------------------------------------------------------------------------+
| XGBoost (Gradient Boost) | 89.00%       | 0.9096    | 0.8988               | 0.9041   | 0.9579  |
| Random Forest Classifier | 87.17%       | 0.8665    | 0.9191               | 0.8920   | 0.9546  |
| Logistic Regression      | 87.17%       | 0.9062    | 0.8671               | 0.8863   | 0.9482  |
| Neural Network (MLP)     | 86.17%       | 0.8706    | 0.8931               | 0.8816   | 0.9404  |
+-----------------------------------------------------------------------------------------------+
```

### 🔍 Top 10 Features Ranked by Gini / Tree Importance
1. **Antecedent Rainfall (72h)**: $21.32\%$
2. **Slope Angle**: $18.36\%$
3. **Distance to Road Cuts**: $10.93\%$
4. **Topographic Wetness Index (TWI)**: $8.16\%$
5. **Distance to Drainage / Streams**: $7.57\%$
6. **Distance to Faults / Lineaments**: $5.69\%$
7. **Plan Curvature (Concavity)**: $4.94\%$
8. **Vegetation Index (NDVI)**: $4.67\%$
9. **Aspect Azimuth**: $3.78\%$
10. **Profile Curvature**: $3.57\%$

---

## 🗺️ 5. Standardized Susceptibility Zonation (GSI & USGS Guidelines)

GeoGuard-AI discretizes continuous failure probabilities $[0.0, 1.0]$ into five standardized engineering hazard classes:

```
[0.00 - 0.20]  🟢 VERY LOW   | Static equilibrium under normal conditions. No engineering works needed.
[0.20 - 0.40]  🟡 LOW        | Stable slope; minor sheet erosion possible under cloudbursts. Routine checks.
[0.40 - 0.60]  🟠 MODERATE   | Marginal factor of safety (1.1 - 1.3). Surface drains and toe walls recommended.
[0.60 - 0.80]  🔴 HIGH       | Critical equilibrium. Structural stabilization required (Soil nailing, gabions).
[0.80 - 1.00]  🟣 VERY HIGH   | Imminent failure hazard. Immediate evacuation of downslope runout zone.
```

---

## 🛠️ 6. System Features & Dashboard Tabs

1. **🗺️ Regional GIS Spatial Zonation Map**:
   - Interactive Folium satellite/cartographic base map.
   - Dynamic marker color-coding for thousands of surveyed slope points.
   - Click-to-inspect popups showing full slope parameters and AI risk index.
2. **🧪 Point Slope Risk Evaluator & What-If Simulation**:
   - Real-time parameter sliders with pre-configured failure scenarios.
   - Instant AI probability gauge meter.
   - Automated **Civil Engineering Mitigation Advisory** (Structural, Drainage, Bio-Engineering, and Instrumentation).
3. **📊 AI Model Benchmarking & Explainable AI (XAI)**:
   - Comparative metric tables, ROC curves, confusion matrices, and feature importance bar charts.
4. **⛈️ Monsoon Cloudburst Simulator**:
   - Simulates rainfall anomalies ($+50\text{mm}$ to $+250\text{mm}$).
   - Displays real-time regional susceptibility shifts and simulated NDRF/DDMA early warning siren alerts.
5. **📁 Batch Survey CSV Predictor & GIS Export**:
   - Upload borehole/drone survey CSVs for high-throughput batch classification.
   - Export enriched CSV with predicted hazard tiers for QGIS/ArcGIS ingestion.
6. **📚 Engineering Viva Voce Reference**:
   - Comprehensive theoretical formulations and oral exam Q&A guide.

---

## 🚀 7. Quickstart & Installation Guide

### Prerequisites
- Python 3.10 or 3.11 (Recommended for scikit-learn & xgboost compatibility)
- Git

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/your-username/landslide-susceptibility-ai.git
cd landslide-susceptibility-ai

# 2. Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models & generate benchmark dataset
python src/model_trainer.py

# 5. Run unit tests
python -m unittest tests/test_pipeline.py

# 6. Launch the interactive dashboard
streamlit run app.py
```

Access the dashboard in your web browser at: **`http://localhost:8501`**

---

## 🎓 8. Viva Voce Preparation & Project Defense Guide

### Q1: Why is Recall (Sensitivity) more critical than Accuracy in this project?
> **Answer:** In natural hazard prediction, a **False Negative** (classifying a critical slope as safe) can result in casualties and destruction. A **False Positive** (classifying a stable slope as dangerous) merely leads to preventive inspection or temporary traffic diversion. Hence, maximizing Recall and Area Under the ROC Curve is prioritized over raw accuracy.

### Q2: What distinguishes this AI model from classical Slope Stability Software (e.g., GeoStudio / SLOPE/W)?
> **Answer:** SLOPE/W performs 2D deterministic slice analysis on a single slope profile where detailed soil shear strength parameters ($c, \phi$) are already known from laboratory boreholes. GeoGuard-AI operates on regional scales ($100s \text{ km}^2$), identifying high-risk hotspots from remote sensing data to determine *where* geotechnical engineers should focus detailed borehole investigations and structural mitigation.

### Q3: Why is drainage considered 80% of landslide mitigation?
> **Answer:** Gravitational shear stress is constant for a given slope angle, but shear strength is dynamically governed by effective stress: $\sigma' = \sigma_n - u_w$. By installing sub-horizontal perforated drains and catchwater ditches, groundwater levels are depressed, dissipating pore pressure $u_w$, which restores the effective normal stress and stabilizes the slope without requiring excessive structural concrete.

---

## 👥 Authors & Academic Credits
- **Project Title:** AI-Powered Prediction of Landslide Susceptibility & Early Warning System
- **Academic Level:** 3rd Year Bachelor of Technology / Bachelor of Engineering (B.Tech / B.E.)
- **Specialization:** Civil Engineering & Computer Science / AI / Data Science
#   G e o G u a r d - A I  
 