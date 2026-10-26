const assert = require("assert");
const aiEngine = require("../src/aiEngine");
const { classifyZone, generateGeotechnicalAdvisory } = require("../src/geotechAdvisor");
const dataStore = require("../src/dataStore");

console.log("==========================================");
console.log("RUNNING GEOGUARD-AI NODE.JS ENGINE TESTS");
console.log("==========================================");

// Test 1: DataStore loaded CSV
assert(dataStore.records.length > 0, "DataStore should have loaded records");
console.log(`[PASS] DataStore loaded ${dataStore.records.length} records.`);

// Test 2: AI Engine Single Point Prediction
const testSlope = {
  slope_deg: 38.0,
  rainfall_72h_mm: 280.0,
  elevation_m: 1100.0,
  twi: 10.5,
  dist_to_road_m: 60.0,
  dist_to_stream_m: 90.0,
  dist_to_fault_m: 300.0,
  ndvi: 0.25,
  plan_curvature: -1.5,
  lithology: "Weathered Regolith",
  soil_type: "Debris/Colluvium",
  land_use: "Tea/Coffee Plantation"
};

const predXGB = aiEngine.predict(testSlope, "XGBoost");
assert(predXGB.probability >= 0.0 && predXGB.probability <= 1.0, "Probability must be between 0 and 1");
assert(predXGB.probability > 0.70, "Critical slope should have high failure probability");
console.log(`[PASS] XGBoost prediction: P = ${predXGB.probability} (Label: ${predXGB.prediction})`);

// Test 3: Geotechnical Advisory
const adv = generateGeotechnicalAdvisory(testSlope, predXGB.probability);
assert.strictEqual(adv.susceptibilityZone, "Very High");
assert(adv.identifiedTriggers.length >= 3, "Should detect multiple failure triggers");
assert(adv.structuralMitigation.length > 0, "Should recommend structural interventions");
assert(adv.drainageMitigation.length > 0, "Should recommend drainage interventions");
console.log(`[PASS] Geotechnical Advisory generated: Zone = ${adv.susceptibilityZone}, Alert = ${adv.alertLevel}`);

// Test 4: Batch Prediction
const sampleNodes = dataStore.getNodes({ limit: 20, model: "XGBoost" });
assert.strictEqual(sampleNodes.length, 20);
assert(sampleNodes[0].pred_prob !== undefined);
console.log(`[PASS] Batch spatial node querying verified.`);

// Test 5: Storm Simulation
const stormResult = dataStore.simulateStorm({ rainBoost: 80, sampleSize: 100, model: "XGBoost" });
assert(stormResult.stormHighPct >= stormResult.baselineHighPct, "Storm must increase or maintain hazard percentage");
console.log(`[PASS] Cloudburst simulation: Baseline High = ${stormResult.baselineHighPct}%, Storm High = ${stormResult.stormHighPct}%`);

console.log("\nALL 5 NODE.JS TEST SUITES PASSED SUCCESSFULLY!\n");
