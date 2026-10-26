/**
 * GeoGuard-AI
 * Spatial GIS Data Store & Survey Record Manager
 */

const fs = require("fs");
const path = require("path");
const aiEngine = require("./aiEngine");
const { classifyZone } = require("./geotechAdvisor");

class DataStore {
  constructor(csvPath = path.join(process.cwd(), "data", "landslide_data.csv")) {
    this.csvPath = csvPath;
    this.records = [];
    this.loadData();
  }

  loadData() {
    if (!fs.existsSync(this.csvPath)) {
      console.warn("[DataStore] CSV not found at " + this.csvPath);
      return;
    }

    try {
      const content = fs.readFileSync(this.csvPath, "utf-8");
      const lines = content.trim().split(/\r?\n/);
      if (lines.length < 2) return;

      const headers = lines[0].split(",").map(h => h.trim());
      const parsed = [];

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const vals = line.split(",");
        const row = {};
        for (let j = 0; j < headers.length; j++) {
          const rawVal = vals[j];
          const numVal = Number(rawVal);
          row[headers[j]] = !isNaN(numVal) && rawVal !== "" ? numVal : rawVal;
        }
        parsed.push(row);
      }

      this.records = parsed;
      console.log(`[DataStore] Successfully loaded ${this.records.length} spatial survey nodes from CSV.`);
    } catch (e) {
      console.error("[DataStore] Error reading CSV:", e.message);
    }
  }

  getNodes({ limit = 600, zone = "All", model = "XGBoost" } = {}) {
    let dataset = this.records.slice(0, limit);

    const enriched = dataset.map((rec) => {
      const pred = aiEngine.predict(rec, model);
      const zoneInfo = classifyZone(pred.probability);
      return {
        ...rec,
        pred_prob: pred.probability,
        zone_name: zoneInfo.zone,
        zone_color: zoneInfo.color,
        alert_level: zoneInfo.alertLevel,
        badge_class: zoneInfo.badgeClass
      };
    });

    if (zone !== "All") {
      return enriched.filter(n => n.zone_name.toLowerCase() === zone.toLowerCase());
    }
    return enriched;
  }

  simulateStorm({ rainBoost = 80, sampleSize = 500, model = "XGBoost" } = {}) {
    const sample = this.records.slice(0, sampleSize);
    
    let baselineHighCount = 0;
    let stormHighCount = 0;
    let baselineCritCount = 0;
    let stormCritCount = 0;

    const nodes = sample.map((rec) => {
      // Baseline
      const basePred = aiEngine.predict(rec, model);
      const baseProb = basePred.probability;
      if (baseProb >= 0.60) baselineHighCount++;
      if (baseProb >= 0.80) baselineCritCount++;

      // Storm condition
      const stormRec = { ...rec, rainfall_72h_mm: Math.min(650.0, (rec.rainfall_72h_mm || 100) + rainBoost) };
      const stormPred = aiEngine.predict(stormRec, model);
      const stormProb = stormPred.probability;
      if (stormProb >= 0.60) stormHighCount++;
      if (stormProb >= 0.80) stormCritCount++;

      const zoneInfo = classifyZone(stormProb);

      return {
        id: rec.location_id,
        lat: rec.latitude,
        lon: rec.longitude,
        base_prob: baseProb,
        storm_prob: stormProb,
        delta: Math.round((stormProb - baseProb) * 1000) / 1000,
        zone_name: zoneInfo.zone,
        zone_color: zoneInfo.color
      };
    });

    return {
      sampleSize: sample.length,
      rainBoost,
      baselineHighPct: Math.round((baselineHighCount / sample.length) * 1000) / 10,
      stormHighPct: Math.round((stormHighCount / sample.length) * 1000) / 10,
      baselineCritCount,
      stormCritCount,
      nodes
    };
  }

  toGeoJSON(nodes) {
    return {
      type: "FeatureCollection",
      crs: {
        type: "name",
        properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" }
      },
      features: nodes.map(n => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [n.longitude, n.latitude]
        },
        properties: {
          location_id: n.location_id,
          elevation_m: n.elevation_m,
          slope_deg: n.slope_deg,
          rainfall_72h_mm: n.rainfall_72h_mm,
          lithology: n.lithology,
          soil_type: n.soil_type,
          land_use: n.land_use,
          susceptibility_index: n.pred_prob,
          hazard_zone: n.zone_name,
          alert_level: n.alert_level
        }
      }))
    };
  }
}

module.exports = new DataStore();
