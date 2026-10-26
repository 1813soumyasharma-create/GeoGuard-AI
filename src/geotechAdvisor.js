/**
 * GeoGuard-AI
 * Geotechnical Expert System & Civil Engineering Advisory Module
 * Implements Mohr-Coulomb failure diagnostic rules & GSI/USGS hazard classification
 */

const SUSCEPTIBILITY_ZONES = [
  {
    zone: "Very Low",
    minProb: 0.0,
    maxProb: 0.20,
    color: "#10B981", // Green
    badgeClass: "bg-emerald-100 text-emerald-800 border-emerald-300",
    alertLevel: "NORMAL (GREEN)",
    description: "Slope is statically stable under standard seasonal conditions. Negligible failure probability."
  },
  {
    zone: "Low",
    minProb: 0.20,
    maxProb: 0.40,
    color: "#84CC16", // Lime
    badgeClass: "bg-lime-100 text-lime-800 border-lime-300",
    alertLevel: "MONITORED (LIME)",
    description: "Low susceptibility. Minor topsoil sheet wash or localized raveling possible during intense rainfall."
  },
  {
    zone: "Moderate",
    minProb: 0.40,
    maxProb: 0.60,
    color: "#F59E0B", // Amber
    badgeClass: "bg-amber-100 text-amber-800 border-amber-300",
    alertLevel: "WATCH (YELLOW)",
    description: "Moderate susceptibility. Marginal factor of safety (1.1 - 1.3). Potential for localized shallow debris slides."
  },
  {
    zone: "High",
    minProb: 0.60,
    maxProb: 0.80,
    color: "#F97316", // Orange
    badgeClass: "bg-orange-100 text-orange-800 border-orange-300",
    alertLevel: "WARNING (ORANGE)",
    description: "High susceptibility. Slope near critical equilibrium. Failure likely under heavy antecedent rainfall or toe disturbance."
  },
  {
    zone: "Very High",
    minProb: 0.80,
    maxProb: 1.01,
    color: "#EF4444", // Red
    badgeClass: "bg-rose-100 text-rose-800 border-rose-300",
    alertLevel: "CRITICAL EMERGENCY (RED)",
    description: "Imminent catastrophic failure hazard. Translational or rotational slide / debris torrent highly probable."
  }
];

function classifyZone(prob) {
  const p = Math.max(0.0, Math.min(1.0, parseFloat(prob) || 0.0));
  for (const z of SUSCEPTIBILITY_ZONES) {
    if (p >= z.minProb && p < z.maxProb) {
      return z;
    }
  }
  return SUSCEPTIBILITY_ZONES[SUSCEPTIBILITY_ZONES.length - 1];
}

function generateGeotechnicalAdvisory(features, probability) {
  const prob = parseFloat(probability) || 0.0;
  const zoneInfo = classifyZone(prob);

  const slope = parseFloat(features.slope_deg ?? features.slope ?? 25.0);
  const rain = parseFloat(features.rainfall_72h_mm ?? features.rainfall ?? 100.0);
  const distRoad = parseFloat(features.dist_to_road_m ?? features.dist_road ?? 500.0);
  const distStream = parseFloat(features.dist_to_stream_m ?? features.dist_stream ?? 500.0);
  const distFault = parseFloat(features.dist_to_fault_m ?? features.dist_fault ?? 1500.0);
  const ndvi = parseFloat(features.ndvi ?? 0.50);
  const soilType = String(features.soil_type ?? features.soil ?? "Lateritic Soil");

  // 1. Diagnose Geotechnical Destabilizing Triggers
  const triggers = [];

  if (slope >= 35.0) {
    triggers.push({
      factor: "Steep Slope Angle (" + slope.toFixed(1) + "°)",
      severity: slope >= 45.0 ? "CRITICAL" : "HIGH",
      impact: "Gravitational tangential driving shear stress overcomes internal friction angle (phi)."
    });
  } else if (slope >= 25.0) {
    triggers.push({
      factor: "Moderate Slope (" + slope.toFixed(1) + "°)",
      severity: "MODERATE",
      impact: "Borderline shear stress; susceptible to transient pore water pressure spikes."
    });
  }

  if (rain >= 250.0) {
    triggers.push({
      factor: "Extreme Antecedent Rainfall (" + rain.toFixed(1) + " mm / 72h)",
      severity: "CRITICAL",
      impact: "Pore water pressure build-up collapses effective normal stress: sigma' = sigma - u_w."
    });
  } else if (rain >= 140.0) {
    triggers.push({
      factor: "Elevated Antecedent Rainfall (" + rain.toFixed(1) + " mm / 72h)",
      severity: "HIGH",
      impact: "Soil matrix saturation and transient groundwater rise inducing seepage forces."
    });
  }

  if (distRoad <= 150.0) {
    triggers.push({
      factor: "Proximity to Road Excavation (" + distRoad.toFixed(0) + " m)",
      severity: "HIGH",
      impact: "Toe slope destressing and daylighting of planar discontinuities without retaining buttress."
    });
  }

  if (distStream <= 100.0) {
    triggers.push({
      factor: "Active Stream Scour Proximity (" + distStream.toFixed(0) + " m)",
      severity: "MODERATE",
      impact: "Fluvial toe erosion and continuous basal saturation by drainage network."
    });
  }

  if (distFault <= 400.0) {
    triggers.push({
      factor: "Tectonic Fault / Lineament Zone (" + distFault.toFixed(0) + " m)",
      severity: "HIGH",
      impact: "High fracture density, slickenside shear planes, and diminished Rock Mass Rating (RMR)."
    });
  }

  if (ndvi < 0.25) {
    triggers.push({
      factor: "Sparse Vegetative Canopy (NDVI: " + ndvi.toFixed(2) + ")",
      severity: "MODERATE",
      impact: "Absence of tensile root matrix cohesion and unchecked rainfall kinetic splash erosion."
    });
  }

  if (soilType === "Debris/Colluvium" || soilType === "Sandy Loam") {
    triggers.push({
      factor: "Unconsolidated Soil Matrix (" + soilType + ")",
      severity: "HIGH",
      impact: "Low cohesion and high permeability conducive to rapid liquefaction and debris flow torrents."
    });
  }

  // 2. Civil Engineering Mitigation Plan
  const structural = [];
  const drainage = [];
  const bio = [];
  const monitoring = [];

  if (prob >= 0.60) {
    // Structural Interventions
    if (slope >= 30.0 || distRoad <= 200.0) {
      structural.push("Reinforced Concrete Gabion Retaining Walls (tiered with backfill free-draining gravel)");
      structural.push("Soil Nailing with High-Tensile Steel Tendons (min. length 6-9m, grid 1.5m x 1.5m) and Shotcrete Facade");
      structural.push("Slope Benching / Terracing (reduce continuous slope height to steps <= 5m at 1:1.5 slope)");
    }
    if (distFault <= 500.0) {
      structural.push("Pre-stressed Rock Anchors / Cable Bolts into competent bedrock beyond shear slip plane");
    }

    // Drainage Measures (Drainage is 80% of landslide stabilization)
    drainage.push("Sub-horizontal Perforated HDPE Drain Pipes (drilled 15-25m into slope at 5° uphill inclination)");
    drainage.push("Lined Trapezoidal Catchwater Interceptor Drains along the crest to divert runoff");
    drainage.push("Herringbone French Drains filled with graded aggregate wrapped in non-woven geotextile");

    // Bio-Engineering
    bio.push("Vetiver System (Chrysopogon zizanioides) contour hedges for massive 3m tensile root reinforcement");
    bio.push("Hydro-seeding with biodegradable coir geo-mesh matting on exposed cut slopes");
    bio.push("Reforestation with native deep taproot species (e.g., Ficus, Bamboo clumps along slope toe)");

    // Instrumentation
    monitoring.push("Deploy MEMS Inclinometers and TDR probes to track sub-surface deformation shear planes");
    monitoring.push("Automated Vibrating Wire Piezometers to monitor transient pore water pressure");
    monitoring.push("Rain gauge telemetry with SMS/Alert broadcast threshold set at 65mm / 24h");
  } else if (prob >= 0.40) {
    structural.push("Dry stone masonry or gabion toe wall (2.5m - 3.5m height)");
    structural.push("Flexible wire rope drapery mesh to catch potential rock raveling");
    drainage.push("Surface runoff diversion ditch along upper slope bench");
    drainage.push("Routine clearance and desilting of existing natural gullies and culverts");
    bio.push("Slope brush layering and live staking on bare surfaces");
    monitoring.push("Bi-weekly visual inspection of tension cracks and ground heave during monsoon");
  } else {
    structural.push("Maintain existing natural slope geometry; prevent unauthorized toe excavations");
    drainage.push("Routine maintenance and clearance of roadside drainage ditches");
    bio.push("Preserve indigenous forest canopy and groundcover mulch");
    monitoring.push("Seasonal routine inspection following monsoon cycle");
  }

  // 3. Early Warning Protocol
  let emergencyProtocol = "";
  if (prob >= 0.80) {
    emergencyProtocol = "IMMEDIATE EVACUATION PROTOCOL: Issue Red Alert to District Disaster Management Authority (DDMA). Restrict vehicular traffic on adjacent cut roads. Evacuate downslope dwellings within 250m runout zone.";
  } else if (prob >= 0.60) {
    emergencyProtocol = "HIGH ALERT PROTOCOL: Issue Orange Alert. Position emergency NDRF/SDRF response teams. Continuous 24/7 vigil of road corridor. Prepare temporary relief shelters.";
  } else if (prob >= 0.40) {
    emergencyProtocol = "ADVISORY PROTOCOL: Issue Yellow Alert. Restrict heavy multi-axle freight movement during intense rain spells (>20mm/hr).";
  } else {
    emergencyProtocol = "NORMAL PROTOCOL: Green Alert. Standard seasonal monsoon monitoring.";
  }

  return {
    susceptibilityZone: zoneInfo.zone,
    failureProbability: prob,
    alertLevel: zoneInfo.alertLevel,
    zoneColor: zoneInfo.color,
    badgeClass: zoneInfo.badgeClass,
    zoneSummary: zoneInfo.description,
    identifiedTriggers: triggers,
    structuralMitigation: structural,
    drainageMitigation: drainage,
    bioEngineering: bio,
    instrumentationMonitoring: monitoring,
    emergencyProtocol: emergencyProtocol
  };
}

module.exports = {
  SUSCEPTIBILITY_ZONES,
  classifyZone,
  generateGeotechnicalAdvisory
};
