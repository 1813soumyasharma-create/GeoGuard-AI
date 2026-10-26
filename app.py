"""
GeoGuard-AI: AI-Powered Landslide Susceptibility Prediction & Early Warning System
Third-Year Engineering Capstone Prototype
Department of Civil & Computer Science Engineering
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.data_generator import (
    generate_landslide_dataset, save_dataset,
    LITHOLOGY_TYPES, SOIL_TYPES, LAND_USE_TYPES
)
from src.model_trainer import (
    LandslidePredictor, train_and_benchmark_models,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from src.geotechnical_advisor import (
    classify_susceptibility_zone, generate_geotechnical_advisory,
    SUSCEPTIBILITY_ZONES
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="GeoGuard-AI | Landslide Susceptibility Predictor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Engineering Dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-zone {
        padding: 6px 14px;
        border-radius: 20px;
        color: white;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95rem;
    }
    .advisory-box {
        background: #F1F5F9;
        border-left: 5px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_create_dataset():
    """Loads existing dataset or generates benchmark dataset if not present."""
    data_path = "data/landslide_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    df = generate_landslide_dataset(n_samples=2800)
    save_dataset(df, data_path)
    return df


@st.cache_resource
def load_predictor():
    """Initializes and caches the ML inference engine."""
    predictor = LandslidePredictor(models_dir="models")
    return predictor


# Ensure Data & Models are Ready
data_df = load_or_create_dataset()
predictor = load_predictor()

# If models haven't been trained yet, train them on the fly
if not predictor.models:
    with st.spinner("Initializing and training baseline AI models (Random Forest, XGBoost, MLP, Logistic Regression)..."):
        train_and_benchmark_models(data_df)
        predictor = load_predictor()


# ==========================================
# SIDEBAR CONTROLS & PROJECT INFO
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 🏔️ **GeoGuard-AI System**")
    st.caption("**Third-Year Engineering Capstone** | *Geotechnical AI & Remote Sensing Lab*")
    
    st.markdown("---")
    st.markdown("#### ⚙️ **Inference Engine**")
    available_models = list(predictor.models.keys()) if predictor.models else ["Random Forest"]
    selected_model = st.selectbox(
        "Active AI Classifier:",
        available_models,
        index=0,
        help="Select Machine Learning classifier for susceptibility inference."
    )
    
    st.markdown("---")
    st.markdown("#### 🎯 **Pre-Configured Scenarios**")
    scenario = st.selectbox(
        "Load Test Slope Scenario:",
        [
            "Custom (Manual Sliders)",
            "⚠️ High Monsoon Cloudburst (Rainfall 320mm, Slope 38°)",
            "🚜 Road Excavation Destabilization (Slope 42°, Road Cut 25m)",
            "🌲 Stable Forest Slope (Dense Canopy, Granite Bedrock)",
            "💧 Saturated Debris Colluvium (High TWI, Concave Basin)"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### 📌 **Project Team & Viva Info**")
    st.caption("**Domain:** Geotechnical Engg. & Deep Learning")
    st.caption("**Key Formulations:** Mohr-Coulomb Criterion & GSI Zonation")
    st.caption("**Target Terrain:** High-Relief Mountainous Corridors (Western Ghats / Himalayas)")


# Set Preset Values based on Scenario
preset_values = {
    "slope": 26.5,
    "rainfall": 125.0,
    "elevation": 920.0,
    "dist_road": 380.0,
    "dist_stream": 320.0,
    "dist_fault": 1100.0,
    "twi": 6.8,
    "ndvi": 0.52,
    "plan_curv": -0.2,
    "prof_curv": 0.1,
    "lithology": "Gneiss",
    "soil_type": "Lateritic Soil",
    "land_use": "Tea/Coffee Plantation"
}

if "High Monsoon Cloudburst" in scenario:
    preset_values.update({
        "slope": 38.0,
        "rainfall": 320.0,
        "elevation": 1250.0,
        "dist_road": 120.0,
        "dist_stream": 110.0,
        "twi": 11.2,
        "ndvi": 0.35,
        "plan_curv": -1.8,
        "soil_type": "Debris/Colluvium",
        "lithology": "Weathered Regolith"
    })
elif "Road Excavation" in scenario:
    preset_values.update({
        "slope": 44.0,
        "rainfall": 180.0,
        "elevation": 880.0,
        "dist_road": 20.0,
        "dist_stream": 450.0,
        "twi": 5.4,
        "ndvi": 0.18,
        "plan_curv": -0.4,
        "land_use": "Barren/Quarry",
        "soil_type": "Sandy Loam"
    })
elif "Stable Forest Slope" in scenario:
    preset_values.update({
        "slope": 12.0,
        "rainfall": 65.0,
        "elevation": 650.0,
        "dist_road": 1400.0,
        "dist_stream": 850.0,
        "dist_fault": 3200.0,
        "twi": 4.5,
        "ndvi": 0.82,
        "plan_curv": 0.8,
        "lithology": "Granite",
        "land_use": "Dense Forest",
        "soil_type": "Lateritic Soil"
    })
elif "Saturated Debris Colluvium" in scenario:
    preset_values.update({
        "slope": 32.0,
        "rainfall": 240.0,
        "elevation": 1100.0,
        "dist_road": 280.0,
        "dist_stream": 40.0,
        "twi": 13.8,
        "ndvi": 0.38,
        "plan_curv": -2.4,
        "soil_type": "Debris/Colluvium",
        "lithology": "Schist"
    })


# ==========================================
# HEADER BANNER
# ==========================================
st.markdown('<p class="main-title">🛡️ GeoGuard-AI: Landslide Susceptibility Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Physics-Informed Machine Learning & Spatial Remote Sensing for Regional Slope Hazard Zonation & Early Warning</p>', unsafe_allow_html=True)


# ==========================================
# NAVIGATION TABS
# ==========================================
tab_map, tab_eval, tab_benchmark, tab_ews, tab_batch, tab_viva = st.tabs([
    "🗺️ Spatial GIS Zonation",
    "🧪 Point Slope Risk & Advisory",
    "📊 AI Model Benchmarking & XAI",
    "⛈️ Monsoon Cloudburst Simulator",
    "📁 Batch Survey CSV Predictor",
    "📚 Engineering Viva & Theory"
])


# ==========================================
# TAB 1: SPATIAL GIS ZONATION MAP
# ==========================================
with tab_map:
    st.subheader("Regional Landslide Susceptibility Zonation (LSM)")
    st.caption("Interactive Spatial GIS Map displaying surveyed terrain grid nodes classified into standard GSI/USGS hazard tiers.")
    
    # Pre-calculate probabilities for a subset of the dataset using the selected model
    sample_geo = data_df.head(600).copy()
    probs, _ = predictor.predict_dataframe(selected_model, sample_geo)
    sample_geo["pred_prob"] = probs
    sample_geo["zone_info"] = [classify_susceptibility_zone(p) for p in probs]
    sample_geo["zone_name"] = [z["zone"] for z in sample_geo["zone_info"]]
    sample_geo["zone_color"] = [z["color"] for z in sample_geo["zone_info"]]
    
    # KPI Row
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown("""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">SURVEYED NODES</span>
            <div class="metric-value" style="color: #1E293B;">{}</div>
        </div>
        """.format(len(sample_geo)), unsafe_allow_html=True)
    with col_k2:
        high_crit = (sample_geo["pred_prob"] >= 0.60).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">HIGH / CRITICAL RISK</span>
            <div class="metric-value" style="color: #EF4444;">{high_crit:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        avg_prob = sample_geo["pred_prob"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">MEAN SUSCEPTIBILITY INDEX</span>
            <div class="metric-value" style="color: #3B82F6;">{avg_prob:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown("""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">ACTIVE MODEL</span>
            <div class="metric-value" style="color: #10B981; font-size: 1.4rem; padding-top: 5px;">{}</div>
        </div>
        """.format(selected_model), unsafe_allow_html=True)
        
    st.write("")
    
    # Filter Controls
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_zones = st.multiselect(
            "Filter Display by Susceptibility Zone:",
            ["Very Low", "Low", "Moderate", "High", "Very High"],
            default=["Moderate", "High", "Very High"]
        )
    with col_f2:
        map_style = st.selectbox(
            "Base Map Layer:",
            ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"],
            index=0
        )
        
    filtered_geo = sample_geo[sample_geo["zone_name"].isin(filter_zones)]
    
    # Build Folium Map
    center_lat = sample_geo["latitude"].mean()
    center_lon = sample_geo["longitude"].mean()
    
    tiles_dict = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB Positron": "CartoDB positron",
        "CartoDB Dark_Matter": "CartoDB dark_matter"
    }
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tiles_dict[map_style]
    )
    
    # Add Points to Map
    for _, row in filtered_geo.iterrows():
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
            <b style="color: #0F172A; font-size: 13px;">ID: {row['location_id']}</b><br>
            <b>Susceptibility:</b> <span style="color: {row['zone_color']}; font-weight: bold;">{row['pred_prob']:.3f} ({row['zone_name']})</span><br>
            <hr style="margin: 4px 0;">
            <b>Slope:</b> {row['slope_deg']}°<br>
            <b>Elevation:</b> {row['elevation_m']} m<br>
            <b>72h Rainfall:</b> {row['rainfall_72h_mm']} mm<br>
            <b>Lithology:</b> {row['lithology']}<br>
            <b>Soil:</b> {row['soil_type']}<br>
            <b>Dist. to Road:</b> {row['dist_to_road_m']} m<br>
            <b>Vegetation NDVI:</b> {row['ndvi']}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=row["zone_color"],
            fill=True,
            fill_color=row["zone_color"],
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['location_id']}: {row['zone_name']} ({row['pred_prob']:.2f})"
        ).add_to(m)
        
    # Render Folium map in Streamlit
    folium_static(m, width=1150, height=520)
    
    # Legend Row
    st.markdown("#### 🧭 Susceptibility Zonation Legend (USGS / GSI Classification)")
    leg_cols = st.columns(5)
    for i, zone in enumerate(SUSCEPTIBILITY_ZONES):
        with leg_cols[i]:
            st.markdown(
                f"""<div style="background-color: {zone['color']}; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">
                {zone['zone']}<br><span style="font-size: 0.8rem; font-weight: normal;">{zone['min_prob']:.2f} - {zone['max_prob']:.2f}</span>
                </div>""",
                unsafe_allow_html=True
            )


# ==========================================
# TAB 2: POINT-SLOPE RISK EVALUATOR & ADVISORY
# ==========================================
with tab_eval:
    st.subheader("🧪 Single-Slope Geotechnical Risk Evaluator & Civil Advisory")
    st.caption("Input site-specific topographic, geological, and hydrologic parameters to generate instant AI failure probability and geotechnical engineering mitigation plans.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.markdown("##### 📐 Topographic & Relief Factors")
        in_slope = st.slider("Slope Angle (°):", 0.0, 65.0, float(preset_values["slope"]), 0.5,
                             help="Critical factor in gravitational driving shear stress.")
        in_elevation = st.slider("Elevation (m):", 300.0, 2200.0, float(preset_values["elevation"]), 20.0)
        in_aspect = st.slider("Aspect (°):", 0.0, 360.0, 180.0, 5.0, help="Downslope face azimuth (0=North, 180=South).")
        in_plan_curv = st.slider("Plan Curvature (Concave/Convex):", -3.5, 3.5, float(preset_values["plan_curv"]), 0.1,
                                 help="Negative = Concave (water convergence); Positive = Convex (water divergence).")
        in_prof_curv = st.slider("Profile Curvature:", -3.0, 3.0, float(preset_values["prof_curv"]), 0.1)

    with col_in2:
        st.markdown("##### 🌧️ Hydrologic & Distance Triggers")
        in_rain = st.slider("Antecedent 72h Rainfall (mm):", 10.0, 500.0, float(preset_values["rainfall"]), 5.0,
                            help="Primary triggering event causing pore water pressure build-up.")
        in_twi = st.slider("Topographic Wetness Index (TWI):", 2.0, 18.0, float(preset_values["twi"]), 0.2,
                           help="ln(a / tan beta) representing steady-state saturation potential.")
        in_dist_road = st.slider("Distance to Road Excavation (m):", 0.0, 3000.0, float(preset_values["dist_road"]), 25.0,
                                 help="Toe cut slope destabilization distance.")
        in_dist_stream = st.slider("Distance to Stream/River (m):", 0.0, 2500.0, float(preset_values["dist_stream"]), 25.0,
                                   help="Scour erosion and perched drainage level.")
        in_dist_fault = st.slider("Distance to Fault/Lineament (m):", 0.0, 5000.0, float(preset_values["dist_fault"]), 50.0)

    with col_in3:
        st.markdown("##### 🪨 Geological & Ecological Attributes")
        in_lithology = st.selectbox("Bedrock Lithology:", LITHOLOGY_TYPES,
                                   index=LITHOLOGY_TYPES.index(preset_values["lithology"]) if preset_values["lithology"] in LITHOLOGY_TYPES else 0)
        in_soil = st.selectbox("Soil Profile / Regolith:", SOIL_TYPES,
                               index=SOIL_TYPES.index(preset_values["soil_type"]) if preset_values["soil_type"] in SOIL_TYPES else 0)
        in_landuse = st.selectbox("Land Use / Land Cover:", LAND_USE_TYPES,
                                 index=LAND_USE_TYPES.index(preset_values["land_use"]) if preset_values["land_use"] in LAND_USE_TYPES else 0)
        in_ndvi = st.slider("Vegetation Index (NDVI):", -0.1, 0.90, float(preset_values["ndvi"]), 0.02,
                            help="Indicator of root mechanical reinforcement and canopy interception.")

    # Assemble test instance
    test_instance = {
        "elevation_m": in_elevation,
        "slope_deg": in_slope,
        "aspect_deg": in_aspect,
        "plan_curvature": in_plan_curv,
        "profile_curvature": in_prof_curv,
        "twi": in_twi,
        "rainfall_72h_mm": in_rain,
        "dist_to_fault_m": in_dist_fault,
        "dist_to_stream_m": in_dist_stream,
        "dist_to_road_m": in_dist_road,
        "ndvi": in_ndvi,
        "lithology": in_lithology,
        "soil_type": in_soil,
        "land_use": in_landuse
    }
    
    # Execute AI Prediction
    prob, pred_label = predictor.predict_point(selected_model, test_instance)
    advisory = generate_geotechnical_advisory(test_instance, prob)
    
    st.markdown("---")
    st.markdown("### 🎯 **AI Prediction & Geotechnical Evaluation Result**")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
        <div style="background: white; border: 2px solid {advisory['zone_color']}; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="color: #64748B; font-weight: 600; font-size: 0.9rem;">PREDICTED SUSCEPTIBILITY INDEX</div>
            <div style="font-size: 3.2rem; font-weight: 800; color: {advisory['zone_color']}; margin: 8px 0;">{prob:.3f}</div>
            <span class="badge-zone" style="background-color: {advisory['zone_color']};">{advisory['susceptibility_zone']} SUSCEPTIBILITY</span>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #334155; font-weight: 600;">EWS Alert Level: <span style="color: {advisory['zone_color']};">{advisory['alert_level']}</span></div>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #64748B;">{advisory['zone_summary']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.progress(prob)
        
    with res_col2:
        st.markdown("#### 🔍 **Diagnosed Destabilizing Triggers**")
        if advisory["identified_triggers"]:
            for trig in advisory["identified_triggers"]:
                st.markdown(f"""
                <div style="background: #FEF2F2 if trig['severity']=='CRITICAL' else #FFFBEB; border-left: 4px solid {'#EF4444' if trig['severity']=='CRITICAL' else '#F59E0B'}; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
                    <b style="color: #0F172A;">{trig['factor']}</b> [{trig['severity']}]<br>
                    <span style="font-size: 0.85rem; color: #475569;">{trig['impact']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical geotechnical triggers detected. Slope factors within static safety envelope.")
            
        st.markdown(f"**Emergency Defense Protocol:** {advisory['emergency_protocol']}")

    # Mitigation Action Plan Tabs
    st.markdown("#### 🛠️ **Civil Engineering & Geotechnical Mitigation Plan**")
    mit_tab1, mit_tab2, mit_tab3, mit_tab4 = st.tabs([
        "🧱 Structural Reinforcement",
        "🌊 Hydrological Drainage",
        "🌱 Bio-Engineering Interventions",
        "📡 Instrumentation & Monitoring"
    ])
    
    with mit_tab1:
        st.markdown("**Structural Stabilization Measures Recommended by Geotechnical Rules Engine:**")
        for m in advisory["structural_mitigation"]:
            st.markdown(f"- 🏗️ **{m}**")
            
    with mit_tab2:
        st.markdown("**Sub-surface & Surface Drainage Controls (Pore Pressure Relief):**")
        for d in advisory["drainage_mitigation"]:
            st.markdown(f"- 💧 **{d}**")
            
    with mit_tab3:
        st.markdown("**Bio-Engineering & Vegetative Slope Armor:**")
        for b in advisory["bio_engineering"]:
            st.markdown(f"- 🌿 **{b}**")
            
    with mit_tab4:
        st.markdown("**Real-time Structural Health & Telemetry Sensors:**")
        for s in advisory["instrumentation_monitoring"]:
            st.markdown(f"- 📶 **{s}**")


# ==========================================
# TAB 3: MODEL BENCHMARKING & XAI
# ==========================================
with tab_benchmark:
    st.subheader("📊 AI Model Benchmarking & Explainable AI (XAI)")
    st.caption("Side-by-side performance comparison of Machine Learning architectures trained on identical geotechnical conditioning factors.")
    
    metrics_data = predictor.metrics.get("models", {})
    
    if metrics_data:
        # Comparison Table
        bench_rows = []
        for model_name, m in metrics_data.items():
            bench_rows.append({
                "Algorithm": model_name,
                "Accuracy (%)": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision']:.4f}",
                "Recall (Sensitivity)": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Brier Score (Calibration)": f"{m['brier_score']:.4f}"
            })
        bench_df = pd.DataFrame(bench_rows)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        
        # Best model callout
        best_auc = predictor.metrics.get("best_model_by_auc", "Random Forest")
        st.info(f"🏆 **Optimal Evaluated Model:** `{best_auc}` achieved highest Area Under ROC Curve (AUC = {metrics_data[best_auc]['roc_auc']:.4f}). High recall is critical in geohazard prediction to minimize dangerous False Negatives (missed landslides).")
        
        # Charts: Feature Importance & ROC Curves
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("##### 📌 Global Feature Importance (Gini / Tree Importance)")
            feats = predictor.metrics.get("feature_importances", [])
            if feats:
                feat_names = [f["feature"].replace("num__", "").replace("cat__", "") for f in feats][:10]
                feat_scores = [f["importance"] for f in feats][:10]
                
                fig, ax = plt.subplots(figsize=(6, 4.5))
                y_pos = np.arange(len(feat_names))
                ax.barh(y_pos, feat_scores[::-1], color="#3B82F6", edgecolor="#1D4ED8", height=0.65)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feat_names[::-1], fontsize=9)
                ax.set_xlabel("Relative Importance Score", fontsize=10)
                ax.set_title("Top 10 Landslide Conditioning Factors", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                
        with col_c2:
            st.markdown("##### 📈 Receiver Operating Characteristic (ROC) Comparison")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
            for model_name, m in metrics_data.items():
                roc_pts = m.get("roc_curve", {})
                if roc_pts:
                    ax_roc.plot(roc_pts["fpr"], roc_pts["tpr"], label=f"{model_name} (AUC={m['roc_auc']:.3f})", lw=2)
            ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC=0.50)")
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
            ax_roc.set_title("ROC Curves Comparison", fontsize=11, fontweight="bold")
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_roc)
            
        # Confusion Matrices
        st.markdown("##### 🎯 Model Confusion Matrices (Test Evaluation)")
        cm_cols = st.columns(len(metrics_data))
        for idx, (m_name, m) in enumerate(metrics_data.items()):
            with cm_cols[idx]:
                cm = np.array(m["confusion_matrix"])
                st.markdown(f"**{m_name}**")
                cm_df = pd.DataFrame(
                    cm,
                    columns=["Pred: Stable", "Pred: Slide"],
                    index=["Actual: Stable", "Actual: Slide"]
                )
                st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("Benchmark metrics currently loading or not yet trained.")


# ==========================================
# TAB 4: MONSOON CLOUDBURST SIMULATOR
# ==========================================
with tab_ews:
    st.subheader("⛈️ Monsoon Cloudburst & Dynamic Early Warning Simulator")
    st.caption("Simulate real-time rainfall intensification across the entire monitored basin to observe rapid geomorphic susceptibility escalation and trigger automated emergency alarms.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("##### 🌀 Cloudburst Weather Scenario")
        rain_boost = st.slider("Additional Storm Rainfall Surge (+mm):", 0, 250, 80, 10,
                               help="Simulated extreme 24-hour precipitation anomaly over baseline.")
        sim_sample_size = st.slider("Monitored Basin Sample Nodes:", 100, 1000, 500, 50)
        
    sim_df = data_df.head(sim_sample_size).copy()
    
    # Baseline vs Storm Predictions
    baseline_probs, _ = predictor.predict_dataframe(selected_model, sim_df)
    storm_df = sim_df.copy()
    storm_df["rainfall_72h_mm"] = np.clip(storm_df["rainfall_72h_mm"] + rain_boost, 0.0, 600.0)
    storm_probs, _ = predictor.predict_dataframe(selected_model, storm_df)
    
    # Metric Delta
    base_high_pct = (baseline_probs >= 0.60).mean() * 100
    storm_high_pct = (storm_probs >= 0.60).mean() * 100
    crit_count = (storm_probs >= 0.80).sum()
    
    with sim_col2:
        k_c1, k_c2, k_c3 = st.columns(3)
        with k_c1:
            st.metric("Baseline High/Crit Slopes", f"{base_high_pct:.1f}%")
        with k_c2:
            st.metric("Post-Storm High/Crit Slopes", f"{storm_high_pct:.1f}%", f"+{storm_high_pct - base_high_pct:.1f}%")
        with k_c3:
            st.metric("Critical Red Emergency Nodes", f"{crit_count}", f"+{crit_count - (baseline_probs >= 0.80).sum()}")

    st.write("")
    
    # Visual Histogram Comparison
    fig_hist, ax_hist = plt.subplots(figsize=(10, 3.8))
    bins = np.linspace(0, 1, 21)
    ax_hist.hist(baseline_probs, bins=bins, alpha=0.55, color="#10B981", label="Baseline Monsoon Conditions", edgecolor="black")
    ax_hist.hist(storm_probs, bins=bins, alpha=0.65, color="#EF4444", label=f"Post-Cloudburst (+{rain_boost}mm)", edgecolor="black")
    ax_hist.axvline(0.60, color="#F97316", linestyle="--", lw=1.8, label="High Hazard Threshold (0.60)")
    ax_hist.axvline(0.80, color="#DC2626", linestyle="--", lw=1.8, label="Critical Emergency Threshold (0.80)")
    ax_hist.set_xlabel("AI Predicted Landslide Susceptibility Index (LSI)", fontsize=10)
    ax_hist.set_ylabel("Number of Monitored Grid Nodes", fontsize=10)
    ax_hist.set_title("Shift in Regional Slope Failure Probability Distribution Under Intense Rainfall", fontsize=11, fontweight="bold")
    ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_hist)
    
    # Emergency Broadcast Log Simulation
    st.markdown("#### 🚨 Simulated Disaster Management Dispatch Log (DDMA / NDRF API)")
    if crit_count > 10:
        st.error(f"🔴 **CRITICAL EMERGENCY RED ALERT DISPATCHED** | {crit_count} slopes breached critical failure threshold (P >= 0.80). Automated highway closure siren activated on NH/SH corridors. SDRF search-and-rescue teams mobilized.")
    elif storm_high_pct > 25.0:
        st.warning(f"🟠 **ORANGE WARNING ALERT** | Regional high susceptibility escalated to {storm_high_pct:.1f}%. Preventive evacuations recommended in vulnerable hillside settlements.")
    else:
        st.info("🟡 **YELLOW WATCH ADVISORY** | Monitor real-time rain gauge telemetry. Maintenance crews on standby.")


# ==========================================
# TAB 5: BATCH SURVEY CSV PREDICTOR & EXPORT
# ==========================================
with tab_batch:
    st.subheader("📁 Batch Slope Survey Processing & GIS Export")
    st.caption("Upload geotechnical reconnaissance borehole or drone DEM survey CSV to perform high-throughput AI susceptibility classification and export GIS-ready shape/GeoJSON data.")
    
    uploaded_file = st.file_uploader("Upload Slope Survey CSV:", type=["csv"])
    
    # Sample download template button
    sample_csv_data = data_df.head(25).drop(columns=["true_failure_prob", "landslide"]).to_csv(index=False)
    st.download_button(
        label="📥 Download Template Sample CSV (25 Locations)",
        data=sample_csv_data,
        file_name="sample_landslide_survey.csv",
        mime="text/csv"
    )
    
    if uploaded_file is not None:
        batch_input_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded survey dataset with **{len(batch_input_df)}** records.")
    else:
        st.info("Demonstration Mode: Using 150 benchmark test survey locations from repository.")
        batch_input_df = data_df.tail(150).drop(columns=["true_failure_prob", "landslide"]).copy()
        
    if st.button("🚀 Run AI Batch Inference"):
        with st.spinner("Executing neural model inference across survey coordinates..."):
            probs, preds = predictor.predict_dataframe(selected_model, batch_input_df)
            result_df = batch_input_df.copy()
            result_df["pred_susceptibility_index"] = np.round(probs, 4)
            result_df["susceptibility_zone"] = [classify_susceptibility_zone(p)["zone"] for p in probs]
            result_df["alert_level"] = [classify_susceptibility_zone(p)["alert_level"] for p in probs]
            
            st.success(f"Batch prediction completed for {len(result_df)} points!")
            st.dataframe(result_df[["location_id", "latitude", "longitude", "slope_deg", "rainfall_72h_mm", "pred_susceptibility_index", "susceptibility_zone", "alert_level"]].head(20), use_container_width=True)
            
            # Export CSV
            out_csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Enriched Predictions CSV",
                data=out_csv,
                file_name="landslide_predictions_enriched.csv",
                mime="text/csv"
            )


# ==========================================
# TAB 6: ENGINEERING VIVA & THEORY
# ==========================================
with tab_viva:
    st.subheader("📚 Engineering Project Documentation & Viva Voce Reference")
    st.caption("Comprehensive academic reference covering geotechnical mechanics, mathematical foundations, and expected project review Q&A.")
    
    with st.expander("1. 📐 Geotechnical Failure Mechanics & Limit Equilibrium (Mohr-Coulomb Criterion)"):
        st.markdown("""
        **The Factor of Safety (FoS)** of an infinite slope in soil mechanics is defined as:
        $$\\text{FoS} = \\frac{\\tau_f}{\\tau_d} = \\frac{c' + (\\sigma - u_w) \\tan \\phi'}{\\gamma \\cdot z \\cdot \\sin \\beta \\cdot \\cos \\beta}$$
        Where:
        - $c'$ = Effective cohesion of soil/regolith ($\text{kN/m}^2$)
        - $\\phi'$ = Effective internal angle of friction (degrees)
        - $\\sigma$ = Total normal stress ($=\\gamma \\cdot z \\cdot \\cos^2 \\beta$)
        - $u_w$ = Pore water pressure build-up during rainfall ($\text{kN/m}^2$)
        - $\\beta$ = Slope angle (degrees)
        - $z$ = Depth of failure slip surface (m)
        - $\\gamma$ = Unit weight of soil ($\text{kN/m}^3$)
        
        **Why AI is superior to conventional Limit Equilibrium:**
        Standard LEM requires expensive boreholes and localized geotechnical laboratory shear tests at every specific point. AI bridges this limitation by learning non-linear empirical spatial patterns across wide regions ($100s\\text{ km}^2$) using DEM remote sensing.
        """)
        
    with st.expander("2. 🛰️ Remote Sensing & GIS Conditioning Factors"):
        st.markdown("""
        - **Slope:** Governs gravitational shear stress ($\tau_d = W \\sin \\beta$).
        - **Aspect:** Solar radiation exposure, moisture evaporation rates, and prevailing monsoon windward slope face.
        - **Plan Curvature:** Contour curvature determining water convergence into gullies or divergence.
        - **Profile Curvature:** Controls flow acceleration or deceleration along the slope path.
        - **Topographic Wetness Index (TWI):** Defined as $\\text{TWI} = \\ln\\left(\\frac{a}{\\tan \\beta}\\right)$, where $a$ is specific catchment area. High TWI correlates with saturation and pore pressure build-up.
        - **NDVI:** Quantifies vegetation root network strength. Plant root matrices provide apparent cohesion $\Delta c_{root} = 2 - 15\\text{ kPa}$.
        - **Distance to Roads / Streams:** Anthropogenic hill-cutting unloads the toe of the slope, causing daylighting of potential shear planes.
        """)
        
    with st.expander("3. 💡 Viva Voce Oral Defense Questions & Answers"):
        st.markdown("""
        **Q1: Why did you benchmark multiple algorithms (Random Forest, XGBoost, MLP, Logistic Regression)?**
        *Answer:* Different models have distinct inductive biases. Tree ensembles (Random Forest & XGBoost) handle non-linear feature interactions, mixed categorical/continuous data, and spatial correlations without strict Gaussian assumptions. Logistic regression serves as a linear baseline, while MLP neural network assesses non-linear deep feature transformations.
        
        **Q2: In disaster prediction, why is Recall more critical than Accuracy?**
        *Answer:* In geotechnical hazard prediction, a **False Negative** (predicting a slope is safe when it actually fails) leads to catastrophic loss of human life and destruction. A **False Positive** merely results in precautionary inspections. Therefore, high Sensitivity/Recall and ROC-AUC are the primary optimization metrics.
        
        **Q3: How does antecedent rainfall trigger slope failures?**
        *Answer:* Rain infiltration reduces soil suction (matric suction) and raises the transient groundwater table. This increases pore water pressure $u_w$, which directly reduces effective stress $\\sigma' = \\sigma - u_w$ in Terzaghi's principle, causing shear strength $\\tau_f$ to drop below gravitational driving shear stress $\\tau_d$.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<center style='color: #64748B; font-size: 0.85rem;'>"
    "GeoGuard-AI Prototype | Developed for Third-Year Engineering Capstone | Academic & Laboratory Demonstration Only"
    "</center>",
    unsafe_allow_html=True
)
