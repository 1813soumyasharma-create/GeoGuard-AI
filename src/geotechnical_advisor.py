"""
Landslide Susceptibility Prediction System
Module: AI-Driven Geotechnical Expert System & Mitigation Advisor
Author: 3rd Year Engineering Project
Description: Evaluates AI model outputs alongside physical geotechnical failure triggers
to generate civil engineering recommendations, hazard zoning, and early warning advisories.
"""

from typing import Dict, List, Any


# Standard Landslide Susceptibility Zonation (GSI & USGS Guidelines)
SUSCEPTIBILITY_ZONES = [
    {
        "zone": "Very Low",
        "min_prob": 0.0,
        "max_prob": 0.20,
        "color": "#10B981", # Green
        "alert_level": "NORMAL",
        "description": "Slope is statically stable under standard monsoon conditions. Negligible failure probability."
    },
    {
        "zone": "Low",
        "min_prob": 0.20,
        "max_prob": 0.40,
        "color": "#84CC16", # Lime green
        "alert_level": "MONITORED",
        "description": "Low susceptibility. Minor topsoil sheet erosion or small raveling possible during intense cloudbursts."
    },
    {
        "zone": "Moderate",
        "min_prob": 0.40,
        "max_prob": 0.60,
        "color": "#F59E0B", # Amber / Yellow
        "alert_level": "WATCH",
        "description": "Moderate susceptibility. Marginal factor of safety (1.1 - 1.3). Potential for localized debris slides."
    },
    {
        "zone": "High",
        "min_prob": 0.60,
        "max_prob": 0.80,
        "color": "#F97316", # Orange
        "alert_level": "WARNING",
        "description": "High susceptibility. Slope near critical equilibrium. Failure likely under heavy antecedent rainfall or toe disturbance."
    },
    {
        "zone": "Very High",
        "min_prob": 0.80,
        "max_prob": 1.01,
        "color": "#EF4444", # Red
        "alert_level": "CRITICAL EMERGENCY",
        "description": "Imminent high-risk failure zone. Catastrophic translational or rotational slide / debris flow probable."
    }
]


def classify_susceptibility_zone(probability: float) -> Dict[str, Any]:
    """Classifies continuous failure probability [0, 1] into standardized susceptibility zones."""
    prob = max(0.0, min(1.0, float(probability)))
    for z in SUSCEPTIBILITY_ZONES:
        if z["min_prob"] <= prob < z["max_prob"]:
            return z
    return SUSCEPTIBILITY_ZONES[-1]


def generate_geotechnical_advisory(features: Dict[str, Any], failure_probability: float) -> Dict[str, Any]:
    """
    Generates a full geotechnical engineering report and mitigation action plan
    based on the AI model's susceptibility probability and input terrain features.
    
    Parameters:
        features: Dictionary containing slope_deg, rainfall_72h_mm, dist_to_road_m,
                  dist_to_stream_m, lithology, soil_type, ndvi, twi, etc.
        failure_probability: AI model continuous probability score (0.0 to 1.0).
        
    Returns:
        Dictionary with zone metadata, root triggers, engineering interventions,
        and early warning system (EWS) action protocol.
    """
    zone_info = classify_susceptibility_zone(failure_probability)
    
    slope = float(features.get("slope_deg", 25.0))
    rain = float(features.get("rainfall_72h_mm", 100.0))
    dist_road = float(features.get("dist_to_road_m", 500.0))
    dist_stream = float(features.get("dist_to_stream_m", 500.0))
    dist_fault = float(features.get("dist_to_fault_m", 1500.0))
    ndvi = float(features.get("ndvi", 0.5))
    lithology = str(features.get("lithology", "Gneiss"))
    soil_type = str(features.get("soil_type", "Lateritic Soil"))
    land_use = str(features.get("land_use", "Forest"))
    twi = float(features.get("twi", 6.0))
    
    # 1. Trigger Factor Diagnosis
    triggers = []
    if slope >= 35.0:
        triggers.append({
            "factor": "Steep Slope Angle (>35°)",
            "impact": "High gravitational tangential shear stress overcoming internal friction angle.",
            "severity": "CRITICAL" if slope >= 45 else "HIGH"
        })
    elif slope >= 25.0:
        triggers.append({
            "factor": "Moderate Slope Angle (25°-35°)",
            "impact": "Borderline shear stress; sensitive to pore water pressure triggers.",
            "severity": "MODERATE"
        })
        
    if rain >= 250.0:
        triggers.append({
            "factor": "Extreme Antecedent Rainfall (>250mm / 72h)",
            "impact": "Severe reduction of effective stress due to perched groundwater table and pore pressure spike.",
            "severity": "CRITICAL"
        })
    elif rain >= 140.0:
        triggers.append({
            "factor": "Elevated Antecedent Rainfall (140-250mm)",
            "impact": "Soil saturation and transient groundwater rise inducing seepage forces.",
            "severity": "HIGH"
        })
        
    if dist_road <= 150.0:
        triggers.append({
            "factor": "Proximity to Road Excavation (<150m)",
            "impact": "Toe slope destressing from anthropogenic hill cutting without retaining buttress.",
            "severity": "HIGH"
        })
        
    if dist_stream <= 100.0:
        triggers.append({
            "factor": "Stream Scour Proximity (<100m)",
            "impact": "Toe erosion and saturation at base of slope by drainage network.",
            "severity": "MODERATE"
        })
        
    if dist_fault <= 400.0:
        triggers.append({
            "factor": "Active Fault / Lineament Zone (<400m)",
            "impact": "High density of shear joints, micro-fracturing, and reduced rock mass rating (RMR).",
            "severity": "HIGH"
        })
        
    if ndvi < 0.25:
        triggers.append({
            "factor": "Sparse / Degraded Vegetation (NDVI < 0.25)",
            "impact": "Absence of root mechanical tensile anchoring and high rainfall kinetic impact.",
            "severity": "MODERATE"
        })
        
    if soil_type in ["Debris/Colluvium", "Sandy Loam"]:
        triggers.append({
            "factor": f"Unconsolidated Soil Matrix ({soil_type})",
            "impact": "Low apparent cohesion and high permeability conducive to rapid liquefaction and debris flows.",
            "severity": "HIGH"
        })

    # 2. Civil Engineering & Geotechnical Mitigation Recommendations
    structural_measures = []
    drainage_measures = []
    bio_measures = []
    monitoring_measures = []
    
    if failure_probability >= 0.60: # High or Very High
        # Structural Interventions
        if slope >= 30.0 or dist_road <= 200.0:
            structural_measures.append("Reinforced Concrete Gabion Retaining Walls (tiered with backfill free-draining gravel)")
            structural_measures.append("Soil Nailing with High-Tensile Steel Bars (min. length 6-9m, grid 1.5m x 1.5m) and Shotcrete Facade")
            structural_measures.append("Slope Benching / Terracing (reduce continuous slope height to steps <= 5m at 1:1.5 slope)")
        if dist_fault <= 500.0:
            structural_measures.append("Pre-stressed Rock Anchors / Cable Bolts into intact bedrock beyond shear planes")
            
        # Drainage Measures (Drainage is 80% of landslide mitigation)
        drainage_measures.append("Sub-horizontal Perforated HDPE Drain Pipes (drilled 15-25m into slope at 5° uphill inclination)")
        drainage_measures.append("Lined Trapezoidal Catchwater Interceptor Drains along the slope crest to divert runoff")
        drainage_measures.append("Herringbone French Drains filled with graded aggregate wrapped in non-woven geotextile")
        
        # Bio-Engineering
        bio_measures.append("Vetiver System (Chrysopogon zizanioides) contour hedges for massive 3m tensile root reinforcement")
        bio_measures.append("Hydro-seeding with biodegradable coir geo-mesh matting on exposed cut slopes")
        bio_measures.append("Reforestation with native deep taproot species (e.g. Ficus, Bamboo clumps along toe)")
        
        # Monitoring
        monitoring_measures.append("Deploy MEMS Inclinometers and TDR probes to track sub-surface deformation shear planes")
        monitoring_measures.append("Automated Vibrating Wire Piezometers to monitor transient pore water pressure")
        monitoring_measures.append("Rain gauge telemetry with SMS/Alert broadcast threshold set at 65mm / 24h")
        
    elif failure_probability >= 0.40: # Moderate
        structural_measures.append("Dry stone masonry or gabion toe wall (2.5m - 3.5m height)")
        structural_measures.append("Flexible wire rope drapery mesh to catch potential rock raveling")
        drainage_measures.append("Surface runoff diversion ditch along upper slope bench")
        drainage_measures.append("Clearing and desilting existing natural gullies and culverts")
        bio_measures.append("Slope brush layering and live staking on bare surfaces")
        monitoring_measures.append("Bi-weekly visual inspection of tension cracks and ground heave during monsoon")
    else: # Low or Very Low
        structural_measures.append("Maintain existing slope geometry; prevent unauthorized toe excavations")
        drainage_measures.append("Routine maintenance and clearance of roadside drainage ditches")
        bio_measures.append("Preserve indigenous forest canopy and groundcover mulch")
        monitoring_measures.append("Seasonal routine inspection following monsoon cycle")

    # 3. Emergency & Civil Defense Protocol
    evacuation_advice = ""
    if failure_probability >= 0.80:
        evacuation_advice = "IMMEDIATE EVACUATION PROTOCOL: Issue Red Alert to District Disaster Management Authority (DDMA). Restrict vehicular traffic on adjacent cut roads. Evacuate downslope dwellings within 250m runout zone."
    elif failure_probability >= 0.60:
        evacuation_advice = "HIGH ALERT PROTOCOL: Issue Orange Alert. Position emergency NDRF/SDRF response teams. Continuous 24/7 vigil of road corridor. Prepare temporary relief shelters."
    elif failure_probability >= 0.40:
        evacuation_advice = "ADVISORY PROTOCOL: Issue Yellow Alert. Restrict heavy multi-axle freight movement during intense rain spells (>20mm/hr)."
    else:
        evacuation_advice = "NORMAL PROTOCOL: Green Alert. Standard monsoon monitoring."

    return {
        "susceptibility_zone": zone_info["zone"],
        "failure_probability": float(failure_probability),
        "alert_level": zone_info["alert_level"],
        "zone_color": zone_info["color"],
        "zone_summary": zone_info["description"],
        "identified_triggers": triggers,
        "structural_mitigation": structural_measures,
        "drainage_mitigation": drainage_measures,
        "bio_engineering": bio_measures,
        "instrumentation_monitoring": monitoring_measures,
        "emergency_protocol": evacuation_advice
    }
