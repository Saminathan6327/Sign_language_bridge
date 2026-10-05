/**
 * server.js - Fast Express Backend for Sign Language Gesture Classification
 * 
 * Features:
 * - Express server running on port 3000 with CORS enabled.
 * - POST /predict accepts a JSON payload of 63 hand coordinates (x, y, z * 21).
 * - Spawns a child process executing predict.py, piping coordinates via stdin.
 * - Reads Python stdout and returns the translated word and confidence to the frontend.
 */

import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_SCRIPT = path.join(__dirname, "predict.py");
const PYTHON_CMD = process.env.PYTHON_BIN || "python";

// Middleware
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// Root health check endpoint
app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "Sign Language Bridge API",
    endpoints: ["POST /predict", "POST /api/smooth-sentence"],
    port: PORT,
  });
});

/**
 * POST /api/smooth-sentence
 * LLM Smoothing placeholder endpoint:
 * Accepts: { "sentence": "HELP PLEASE WATER" } or { "text": "..." }
 * Returns natural, polished conversational English.
 */
app.post("/api/smooth-sentence", (req, res) => {
  const raw = (req.body.sentence || req.body.text || "").trim();
  if (!raw) {
    return res.status(400).json({ error: "Missing sentence in request body." });
  }

  // Example rule-based conversational normalizer (replace with OpenAI/Gemini/Ollama API call in production)
  let smoothed = raw;
  const upper = raw.toUpperCase();

  if (upper.includes("HELP") && (upper.includes("PLEASE") || upper.includes("WATER"))) {
    smoothed = "Could you please help me? I need some water.";
  } else if (upper.includes("HELP") && upper.includes("PLEASE")) {
    smoothed = "Excuse me, could you please help me with this?";
  } else if (upper.includes("I") && upper.includes("LIKE")) {
    smoothed = "I really like this, thank you!";
  } else if (upper.includes("WE") && upper.includes("LIKE")) {
    smoothed = "We really like this service!";
  } else if (upper.includes("I") && upper.includes("WATER")) {
    smoothed = "May I please have some drinking water?";
  } else if (upper.includes("WE") && upper.includes("HELP")) {
    smoothed = "We would appreciate some assistance with our inquiry, please.";
  } else if (upper.includes("WATER") || upper.includes("FOOD")) {
    smoothed = "I would like to request some water or food, please.";
  } else if (upper.includes("HELLO") || upper.includes("WELCOME")) {
    smoothed = "Hello, good day! I am here at Counter #04.";
  } else if (upper.includes("THANK") || upper.includes("YOU")) {
    smoothed = "Thank you very much for your kind assistance!";
  } else if (raw.includes("உதவி") && (raw.includes("தயவுசெய்து") || raw.includes("தண்ணீர்"))) {
    smoothed = "தயவுசெய்து எனக்கு உதவ முடியுமா? எனக்கு கொஞ்சம் தண்ணீர் தேவை.";
  } else if (raw.includes("உதவி") && raw.includes("தயவுசெய்து")) {
    smoothed = "தயவுசெய்து எனக்கு இதில் கொஞ்சம் உதவ முடியுமா?";
  } else if (raw.includes("நான்") && raw.includes("பிடிக்கும்")) {
    smoothed = "எனக்கு இது மிகவும் பிடிக்கும், மிக்க நன்றி!";
  } else if (raw.includes("நாம்") && raw.includes("பிடிக்கும்")) {
    smoothed = "எங்களுக்கு இந்த சேவை மிகவும் பிடிக்கும்!";
  } else if (raw.includes("நான்") && raw.includes("தண்ணீர்")) {
    smoothed = "தயவுசெய்து எனக்கு குடிதண்ணீர் கிடைக்குமா?";
  } else if (raw.includes("நாம்") && raw.includes("உதவி")) {
    smoothed = "எங்கள் குழுவிற்கு ஒரு சிறிய உதவி தேவைப்படுகிறது.";
  } else if (raw.includes("தண்ணீர்") || raw.includes("உணவு")) {
    smoothed = "தயவுசெய்து உணவு அல்லது குடிதண்ணீர் பெற விரும்புகிறேன்.";
  } else if (raw.includes("வணக்கம்")) {
    smoothed = "வணக்கம்! கவுண்டர் #04-ல் இருக்கிறேன்.";
  } else if (raw.includes("நன்றி")) {
    smoothed = "உங்கள் கனிவான உதவிக்கு மிக்க நன்றி!";
  } else {
    // Basic grammatical title casing and punctuation
    smoothed = raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
    if (!/[.!?]$/.test(smoothed)) smoothed += ".";
  }

  return res.json({
    original: raw,
    smoothedSentence: smoothed,
    model: "sign-bridge-llm-smoother-v1",
    timestamp: new Date().toISOString(),
  });
});

/**
 * POST /predict
 * Accepts:
 *   { "landmarks": [x0, y0, z0, ..., x20, y20, z20] }
 *   or { "features": [...] }
 *   or direct array [...]
 */
app.post("/predict", (req, res) => {
  // Extract coordinates array from varied payload formats
  let coordinates = null;

  if (Array.isArray(req.body)) {
    coordinates = req.body;
  } else if (req.body && typeof req.body === "object") {
    coordinates = req.body.landmarks || req.body.features || req.body.coordinates;
  }

  // Validate presence and length of coordinates
  if (!coordinates || !Array.isArray(coordinates)) {
    return res.status(400).json({
      error: "Invalid request payload. Expected an array of 63 or 126 hand coordinates.",
      receivedType: typeof coordinates,
    });
  }

  if (coordinates.length !== 63 && coordinates.length !== 126) {
    return res.status(422).json({
      error: `Expected 63 coordinates (21 landmarks) or 126 coordinates (42 landmarks for 2 hands), received ${coordinates.length}.`,
    });
  }

  // Spawn Python inference process
  const pyProcess = spawn(PYTHON_CMD, [PYTHON_SCRIPT], {
    cwd: __dirname,
    stdio: ["pipe", "pipe", "pipe"],
  });

  let stdoutData = "";
  let stderrData = "";

  // Capture stdout from Python
  pyProcess.stdout.on("data", (chunk) => {
    stdoutData += chunk.toString();
  });

  // Capture stderr for debugging
  pyProcess.stderr.on("data", (chunk) => {
    stderrData += chunk.toString();
  });

  // Handle process termination
  pyProcess.on("close", (code) => {
    if (code !== 0 && !stdoutData.trim()) {
      console.error(`[!] Python predict.py exited with code ${code}. Stderr:`, stderrData);
      return res.status(500).json({
        error: "Prediction failed inside Python inference script.",
        details: stderrData.trim(),
      });
    }

    try {
      const trimmed = stdoutData.trim();
      let responsePayload;

      // Try parsing JSON output from predict.py
      try {
        const parsed = JSON.parse(trimmed);
        responsePayload = {
          prediction: parsed.prediction || parsed.word || parsed.label || "—",
          confidence: parsed.confidence !== undefined ? parsed.confidence : 1.0,
          handsDetected: parsed.handsDetected || (coordinates.length === 126 ? 2 : 1),
          points: parsed.points || (coordinates.length === 126 ? 42 : 21),
          status: parsed.status,
        };
      } catch {
        // Fallback: predict.py printed plain string/word
        responsePayload = {
          prediction: trimmed || "—",
          confidence: 1.0,
          handsDetected: coordinates.length === 126 ? 2 : 1,
          points: coordinates.length === 126 ? 42 : 21,
        };
      }

      return res.json(responsePayload);
    } catch (err) {
      console.error("[!] Error parsing Python output:", err);
      return res.status(500).json({
        error: "Failed to parse inference result.",
        rawOutput: stdoutData,
      });
    }
  });

  // Handle spawn failure (e.g. Python executable not found)
  pyProcess.on("error", (err) => {
    console.error(`[!] Failed to spawn Python process ('${PYTHON_CMD}'):`, err);
    return res.status(500).json({
      error: `Failed to execute Python inference runtime ('${PYTHON_CMD}').`,
      message: err.message,
    });
  });

  // Write coordinates JSON to Python's stdin and close stream
  pyProcess.stdin.write(JSON.stringify(coordinates));
  pyProcess.stdin.end();
});

// Start Express server
const server = app.listen(PORT, () => {
  console.log("=".repeat(60));
  console.log(`  SIGN LANGUAGE BRIDGE - EXPRESS API SERVER`);
  console.log("=".repeat(60));
  console.log(`[*] Server running at    : http://localhost:${PORT}`);
  console.log(`[*] Inference endpoint   : POST http://localhost:${PORT}/predict`);
  console.log(`[*] Python interpreter   : ${PYTHON_CMD}`);
  console.log(`[*] Prediction script    : ${PYTHON_SCRIPT}`);
  console.log("=".repeat(60) + "\n");
});

export default app;
