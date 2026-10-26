/**
 * GeoGuard-AI: AI-Powered Landslide Susceptibility Prediction System
 * Full-Stack Node.js Server
 * Built with native Node.js HTTP (zero external dependencies required for 100% portability)
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");

const aiEngine = require("./src/aiEngine");
const { classifyZone, generateGeotechnicalAdvisory, SUSCEPTIBILITY_ZONES } = require("./src/geotechAdvisor");
const dataStore = require("./src/dataStore");

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME_TYPES = {
  ".html": "text/html; charset=UTF-8",
  ".css": "text/css; charset=UTF-8",
  ".js": "application/javascript; charset=UTF-8",
  ".json": "application/json; charset=UTF-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".csv": "text/csv; charset=UTF-8"
};

function sendJSON(res, statusCode, data) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  });
  res.end(JSON.stringify(data));
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", chunk => {
      body += chunk.toString();
      if (body.length > 5 * 1024 * 1024) { // 5MB limit
        reject(new Error("Payload too large"));
      }
    });
    req.on("end", () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (e) {
        resolve({ raw: body });
      }
    });
    req.on("error", reject);
  });
}

const server = http.createServer(async (req, res) => {
  // Handle CORS Preflight
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    });
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // ==========================================
  // REST API ROUTES
  // ==========================================

  // 1. GET /api/nodes -> Return spatial GIS nodes
  if (req.method === "GET" && pathname === "/api/nodes") {
    const limit = parseInt(parsedUrl.query.limit, 10) || 600;
    const zone = parsedUrl.query.zone || "All";
    const model = parsedUrl.query.model || "XGBoost";
    const nodes = dataStore.getNodes({ limit, zone, model });
    sendJSON(res, 200, { total: nodes.length, nodes });
    return;
  }

  // 2. POST /api/predict -> Single-point slope risk evaluation & advisory
  if (req.method === "POST" && pathname === "/api/predict") {
    try {
      const body = await parseBody(req);
      const model = body.model || "XGBoost";
      const features = body.features || body;
      const prediction = aiEngine.predict(features, model);
      const advisory = generateGeotechnicalAdvisory(features, prediction.probability);
      sendJSON(res, 200, {
        success: true,
        prediction,
        advisory
      });
    } catch (err) {
      sendJSON(res, 400, { success: false, error: err.message });
    }
    return;
  }

  // 3. GET /api/benchmarks -> Model evaluation metrics, ROC curves, feature importances
  if (req.method === "GET" && pathname === "/api/benchmarks") {
    sendJSON(res, 200, {
      success: true,
      metrics: aiEngine.metrics,
      zones: SUSCEPTIBILITY_ZONES
    });
    return;
  }

  // 4. POST /api/simulate-storm -> Monsoon cloudburst anomaly simulation
  if (req.method === "POST" && pathname === "/api/simulate-storm") {
    try {
      const body = await parseBody(req);
      const rainBoost = parseFloat(body.rainBoost) || 80.0;
      const sampleSize = parseInt(body.sampleSize, 10) || 500;
      const model = body.model || "XGBoost";
      const result = dataStore.simulateStorm({ rainBoost, sampleSize, model });
      sendJSON(res, 200, { success: true, ...result });
    } catch (err) {
      sendJSON(res, 400, { success: false, error: err.message });
    }
    return;
  }

  // 5. POST /api/batch-predict -> High-throughput survey batch evaluation
  if (req.method === "POST" && pathname === "/api/batch-predict") {
    try {
      const body = await parseBody(req);
      const records = Array.isArray(body.records) ? body.records : [];
      const model = body.model || "XGBoost";
      const results = records.map(r => {
        const pred = aiEngine.predict(r, model);
        const zoneInfo = classifyZone(pred.probability);
        return {
          ...r,
          pred_susceptibility: pred.probability,
          hazard_zone: zoneInfo.zone,
          alert_level: zoneInfo.alertLevel,
          zone_color: zoneInfo.color
        };
      });
      sendJSON(res, 200, { success: true, total: results.length, results });
    } catch (err) {
      sendJSON(res, 400, { success: false, error: err.message });
    }
    return;
  }

  // 6. GET /api/export-geojson -> Export nodes in standard GIS GeoJSON format
  if (req.method === "GET" && pathname === "/api/export-geojson") {
    const limit = parseInt(parsedUrl.query.limit, 10) || 1000;
    const model = parsedUrl.query.model || "XGBoost";
    const nodes = dataStore.getNodes({ limit, zone: "All", model });
    const geojson = dataStore.toGeoJSON(nodes);
    res.writeHead(200, {
      "Content-Type": "application/geo+json",
      "Content-Disposition": 'attachment; filename="landslide_susceptibility_nodes.geojson"',
      "Access-Control-Allow-Origin": "*"
    });
    res.end(JSON.stringify(geojson, null, 2));
    return;
  }

  // 7. GET /api/sample-csv -> Download demo CSV template
  if (req.method === "GET" && pathname === "/api/sample-csv") {
    const sampleNodes = dataStore.records.slice(0, 30);
    if (sampleNodes.length === 0) {
      sendJSON(res, 404, { error: "No records available" });
      return;
    }
    const headers = Object.keys(sampleNodes[0]).filter(k => k !== "true_failure_prob" && k !== "landslide");
    const csvLines = [headers.join(",")];
    for (const row of sampleNodes) {
      csvLines.push(headers.map(h => row[h]).join(","));
    }
    res.writeHead(200, {
      "Content-Type": "text/csv; charset=UTF-8",
      "Content-Disposition": 'attachment; filename="sample_slope_survey.csv"',
      "Access-Control-Allow-Origin": "*"
    });
    res.end(csvLines.join("\n"));
    return;
  }

  // ==========================================
  // STATIC FILE SERVING (Tailwind Frontend)
  // ==========================================
  let filePath = path.join(PUBLIC_DIR, pathname === "/" ? "index.html" : pathname);
  
  // Prevent directory traversal attacks
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || "application/octet-stream";

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === "ENOENT") {
        // Fallback to index.html for single-page routing
        fs.readFile(path.join(PUBLIC_DIR, "index.html"), (err2, fallback) => {
          if (err2) {
            res.writeHead(404, { "Content-Type": "text/plain" });
            res.end("404 Not Found");
          } else {
            res.writeHead(200, { "Content-Type": "text/html; charset=UTF-8" });
            res.end(fallback);
          }
        });
      } else {
        res.writeHead(500);
        res.end("Server Error: " + err.code);
      }
    } else {
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content);
    }
  });
}); } module.exports = server;

if (process.env.NODE_ENV !== 'production') { server.listen(PORT, () => {
  console.log(`\n` + "=".repeat(65));
  console.log(`🛡️  GeoGuard-AI Full-Stack Node.js Server Running`);
  console.log(`📍 URL: http://localhost:${PORT}`);
  console.log(`📦 Loaded: ${dataStore.records.length} spatial survey points`);
  console.log(`⚡ Tailwind CSS + Leaflet GIS + AI Inference Ready`);
  console.log("=".repeat(65) + `\n`);
}); } module.exports = server;
