/**
 * main.js - Real-Time Dual-Hand (42 Points) Sign Language Interpreter Kiosk Terminal
 * 
 * Powered by:
 * - MediaPipe Tasks Vision (Dual Hand Tracking: numHands=2, 42 landmarks)
 * - HTML5 <video> & <canvas> with custom dual-color skeletal rendering
 * - High-speed Real-Time Gesture Calculator Engine (Geometric & Kinematic)
 * - Public Service Accessibility Chatbox with auto-send, speech synthesis (TTS), & conversation history
 * - Fetch API -> Express Backend (http://localhost:3000/predict) transmitting 126-coordinate payloads
 */

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const canvasCtx = canvas.getContext("2d");

// Overlays & Guides
const loaderOverlay = document.getElementById("loader_overlay");
const loaderStatus = document.getElementById("loader_status");
const reticleBadge = document.getElementById("reticle_badge");
const hand1Indicator = document.getElementById("hand1_indicator");
const hand2Indicator = document.getElementById("hand2_indicator");

// Kiosk Telemetry & Badges
const kioskClock = document.getElementById("kiosk_clock");
const backendBadge = document.getElementById("backend_badge");
const backendStatusText = document.getElementById("backend_status_text");
const backendLatencyEl = document.getElementById("backend_latency");
const fpsCounter = document.getElementById("fps_counter");
const ptsCounter = document.getElementById("pts_counter");
const payloadStatus = document.getElementById("payload_status");
const payloadPreview = document.getElementById("payload_preview");

// Prediction Display
const predictionDisplay = document.getElementById("prediction_display");
const predictionSubtext = document.getElementById("prediction_subtext");
const confidencePercent = document.getElementById("confidence_percent");
const confidenceBar = document.getElementById("confidence_bar");
const confidenceLabel = document.getElementById("confidence_label");
const vocabGrid = document.getElementById("vocab_grid");

// Chatbox Elements
const chatMessages = document.getElementById("chat_messages");
const chatInput = document.getElementById("chat_input");
const btnSendChat = document.getElementById("btn_send_chat");
const btnSpace = document.getElementById("btn_space");
const btnBackspace = document.getElementById("btn_backspace");
const btnSpeakTranscript = document.getElementById("btn_speak_transcript");

const btnToggleTts = document.getElementById("btn_toggle_tts");
const btnToggleAutosend = document.getElementById("btn_toggle_autosend");
const btnExportChat = document.getElementById("btn_export_chat");
const btnClearChat = document.getElementById("btn_clear_chat");

// Staging Banner
const gestureStagingBar = document.getElementById("gesture_staging_bar");
const stagingGestureName = document.getElementById("staging_gesture_name");
const stagingConfBadge = document.getElementById("staging_conf_badge");
const stagingProgressFill = document.getElementById("staging_progress_fill");
const stagingStatusText = document.getElementById("staging_status_text");


// Controls
const btnToggleCam = document.getElementById("btn_toggle_cam");
const camBtnText = document.getElementById("cam_btn_text");
const btnToggleMirror = document.getElementById("btn_toggle_mirror");
const btnToggleLandmarks = document.getElementById("btn_toggle_landmarks");
const btnTogglePip = document.getElementById("btn_toggle_pip");
const pipBtnText = document.getElementById("pip_btn_text");
const viewportPanel = document.querySelector(".viewport-panel");

// Language Switcher & i18n DOM Elements
const btnLangEn = document.getElementById("btn_lang_en");
const btnLangTa = document.getElementById("btn_lang_ta");
const btnToggleLang = document.getElementById("btn_toggle_lang");
const btnToggleLangText = document.getElementById("btn_toggle_lang_text");

const kioskTagline = document.getElementById("kiosk_tagline");
const kioskTitle = document.getElementById("kiosk_title");
const kioskStationId = document.getElementById("kiosk_station_id");
const panelTitle = document.getElementById("panel_title");
const btnMirrorText = document.getElementById("btn_mirror_text");
const btnLandmarksText = document.getElementById("btn_landmarks_text");
const calcCardTitle = document.getElementById("calc_card_title");
const vocabDeckTitle = document.getElementById("vocab_deck_title");
const vocabDeckMeta = document.getElementById("vocab_deck_meta");
const chatboxTitle = document.getElementById("chatbox_title");
const btnToggleTtsText = document.getElementById("btn_toggle_tts_text");
const btnToggleAutosendText = document.getElementById("btn_toggle_autosend_text");
const btnExportChatText = document.getElementById("btn_export_chat_text");
const btnClearChatText = document.getElementById("btn_clear_chat_text");
const stagingTag = document.getElementById("staging_tag");
const btnSendChatText = document.getElementById("btn_send_chat_text");
const btnSpaceText = document.getElementById("btn_space_text");
const btnSpeakText = document.getElementById("btn_speak_text");
const telemetryTitle = document.getElementById("telemetry_title");
const footerCompliance = document.getElementById("footer_compliance");

// =============================================================================
// CONFIGURATION & STATE
// =============================================================================
const BACKEND_URL = "http://localhost:3000/predict";
const MEDIAPIPE_CDN_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let handLandmarker = null;
let drawingUtils = null;
let HandLandmarkerClass = null;

let webcamRunning = false;
let showLandmarks = true;
let isMirrored = true;

let lastVideoTime = -1;
let lastPredictTime = 0;
const PREDICT_INTERVAL_MS = 100; // Throttle backend fetch requests (~10 Hz)
let isRequestInFlight = false;

// FPS tracking
let frameCount = 0;
let lastFpsUpdateTime = performance.now();
let currentFps = 0;

// Chat & Auto-Send Options
let isTtsEnabled = true;
let isAutoSendEnabled = true;
const AUTO_SEND_HOLD_MS = 1100; // Hold gesture for 1.1 seconds to commit word to sentence

// Gesture Tracking & Staging State
let activeGestureObj = null;
let stagingGestureKey = "";
let stagingStartTime = 0;
let lastCommittedGestureKey = "";
let lastCommitTimestamp = 0;
const COMMIT_COOLDOWN_MS = 1600; // Prevent spamming duplicate commit within 1.6s

// Continuous Sentence Construction & Silence Timer State
let draftSentence = "";
let lastAppendedWord = "";
let hasCommittedCurrentHold = false; // Prevents holding a gesture for e.g. 3s from adding duplicate words
const SILENCE_TIMEOUT_MS = 2000; // Finalize sentence after 2.0s of zero hands
let silenceTimerId = null;
let silenceStartTime = 0;

// Backend health state
let backendOnline = false;

// Language Settings State (persisted in localStorage: "en" | "ta")
let currentLanguage = "en";
try {
  const savedLang = localStorage.getItem("bridgesign_lang");
  if (savedLang === "ta" || savedLang === "en") {
    currentLanguage = savedLang;
  }
} catch (e) {
  console.warn("localStorage unavailable:", e);
}

const UI_TRANSLATIONS = {
  en: {
    kioskTagline: "Citizen Services • Accessibility Terminal",
    kioskTitle: "Sign Language Bridge",
    kioskStationId: "Counter #04 • Public Service Hub",
    backendConnecting: "Connecting: localhost:3000",
    backendConnected: "Connected: localhost:3000 (126-Pt API)",
    backendOffline: "Standby: localhost:3000",

    viewportTitle: "Dual-Hand Vision Viewport",
    reticleAwaiting: "Awaiting Hands",
    reticleTracking1H: "1 Hand Detected (21 pts)",
    reticleTracking2H: "2 Hands Detected (42 pts)",
    reticleCameraOff: "Camera Offline",
    reticleDispatched: "Sentence Dispatched",
    hand1Name: "Hand 1: Cyan (21 pts)",
    hand2Name: "Hand 2: Amber (21 pts)",
    camActive: "Camera Active",
    camOffline: "Camera Offline",
    mirror: "Mirror",
    landmarks: "Landmarks",
    floatPip: "Float PiP",
    dockPip: "📌 Dock View",

    calcTitle: "Real-Time Gesture Calculation",
    calcSubtitle: "42-Point Real-Time Vision",
    calcPlaceholder: "Show one or both hands (42 points)",
    calcAccuracy: "Calculation Accuracy",
    vocabTitle: "Vocabulary Palette",
    vocabMeta: "34 Signs • 1H & 2H",

    chatboxTitle: "Real-Time Gesture Chatbox",
    ttsOn: "🔊 Speech ON",
    ttsOff: "🔇 Speech OFF",
    autoSend: "⚡ Auto-Send",
    export: "📥 Export",
    clear: "🗑️ Clear",
    activeGesture: "ACTIVE GESTURE:",
    waitingForSign: "Waiting for sign...",
    holdToCommit: "Hold to commit",
    holdReady: "Ready to Add",
    holdMotionActive: "⚡ Motion Detected: Continue signing...",
    holdSteady: "Hold steady to add word...",
    senderAssistant: "🤖 Counter #04 Interpreter",
    senderYou: "You (Signer)",
    systemReady: "System Ready",
    chatCleared: "Chat Cleared",
    chatClearedBody: "Chat cleared. Point one or both hands (42 points) toward the camera to begin a new real-time translation session.",
    welcomeMsg: "👋 <strong>Welcome to Sign Language Bridge!</strong> You can now use both hands (<strong>42 points</strong>) or one hand (<strong>21 points</strong>). Try gestures like <strong>Namaste 🙏</strong>, <strong>Heart ❤️</strong>, <strong>I 🙋</strong>, <strong>We 👥</strong>, <strong>Like ✨</strong>, or <strong>Hello ✋</strong>. Your gestures will be calculated in real-time and output directly here in this chatbox!",
    chatPlaceholder: "Real-time gesture buffers here (or type message)...",
    sendBtn: "Send 🚀",
    spaceBtn: "Space",
    backspaceBtn: "⌫",
    speakBtn: "🔊 Speak",

    telemetryTitle: "126-Coordinate Feature Payload (42 Points × 3D)",
    telemetryAwaiting: "[ awaiting hand detection (0/42 points)... ]",
    footerCompliance: "ADA & WCAG 2.1 Compliant Service Interface",
    langSwitchLabel: "தமிழ்",
    langSwitchedAlert: "🌐 Language set to English.",
  },
  ta: {
    kioskTagline: "குடிமக்கள் சேவைகள் • அணுகல்தன்மை முனையம்",
    kioskTitle: "சைகை மொழி பாலம்",
    kioskStationId: "கவுண்டர் #04 • பொது சேவை மையம்",
    backendConnecting: "இணைக்கிறது: localhost:3000",
    backendConnected: "இணைக்கப்பட்டது: localhost:3000 (126-புள்ளி API)",
    backendOffline: "காத்திருப்பு: localhost:3000",

    viewportTitle: "இரு கை பார்வை திரை",
    reticleAwaiting: "கைகளுக்காக காத்திருக்கிறது",
    reticleTracking1H: "1 கை கண்டறியப்பட்டது (21 புள்ளிகள்)",
    reticleTracking2H: "2 கைகள் கண்டறியப்பட்டன (42 புள்ளிகள்)",
    reticleCameraOff: "கேமரா ஆஃப் செய்யப்பட்டுள்ளது",
    reticleDispatched: "வாக்கியம் அனுப்பப்பட்டது",
    hand1Name: "கை 1: சியான் (21 புள்ளிகள்)",
    hand2Name: "கை 2: அம்பர் (21 புள்ளிகள்)",
    camActive: "கேமரா ஆன்",
    camOffline: "கேமரா ஆஃப்",
    mirror: "பிரதிபலிப்பு",
    landmarks: "புள்ளிகள்",
    floatPip: "மிதக்கும் திரை",
    dockPip: "📌 திரை பொருத்து",

    calcTitle: "நிகழ்நேர சைகை கணிப்பு",
    calcSubtitle: "42-புள்ளி நிகழ்நேர பார்வை",
    calcPlaceholder: "ஒன்று அல்லது இரு கைகளையும் காட்டுங்கள் (42 புள்ளிகள்)",
    calcAccuracy: "கணிப்பு துல்லியம்",
    vocabTitle: "சைகை அகராதி",
    vocabMeta: "34 சைகைகள் • 1 கை & 2 கைகள்",

    chatboxTitle: "நிகழ்நேர சைகை உரையாடல்",
    ttsOn: "🔊 பேச்சு ஆன்",
    ttsOff: "🔇 பேச்சு ஆஃப்",
    autoSend: "⚡ தானியங்கி அனுப்பு",
    export: "📥 பதிவிறக்கு",
    clear: "🗑️ அழி",
    activeGesture: "செயலில் உள்ள சைகை:",
    waitingForSign: "சைகைக்காக காத்திருக்கிறது...",
    holdToCommit: "உறுதிப்படுத்த பிடிக்கவும்",
    holdReady: "சேர்க்க தயார்",
    holdMotionActive: "⚡ சைகை இயக்கம் கண்டறியப்பட்டது: தொடரவும்...",
    holdSteady: "வார்த்தையை சேர்க்க கையை நிலையாக வைக்கவும்...",
    senderAssistant: "🤖 கவுண்டர் #04 மொழிபெயர்ப்பாளர்",
    senderYou: "நீங்கள் (சைகையாளர்)",
    systemReady: "அமைப்பு தயார்",
    chatCleared: "உரையாடல் அழிக்கப்பட்டது",
    chatClearedBody: "உரையாடல் அழிக்கப்பட்டது. புதிய அமர்வை தொடங்க ஒன்று அல்லது இரு கைகளையும் கேமராவின் முன் காட்டுங்கள்.",
    welcomeMsg: "👋 <strong>சைகை மொழி பாலத்திற்கு நல்வரவு!</strong> நீங்கள் இரு கைகளையும் (<strong>42 புள்ளிகள்</strong>) அல்லது ஒரு கையையும் (<strong>21 புள்ளிகள்</strong>) பயன்படுத்தலாம். <strong>வணக்கம் 🙏</strong>, <strong>இதயம் ❤️</strong>, <strong>நான் 🙋</strong>, <strong>நாம் 👥</strong>, <strong>பிடிக்கும் ✨</strong> போன்ற சைகைகளை செய்து பாருங்கள். உங்கள் சைகைகள் நிகழ்நேரத்தில் கணிக்கப்பட்டு இங்கே உரையாடலில் தோன்றும்!",
    chatPlaceholder: "சைகை வார்த்தைகள் இங்கே சேரும் (அல்லது தட்டச்சு செய்யவும்)...",
    sendBtn: "அனுப்பு 🚀",
    spaceBtn: "இடைவெளி",
    backspaceBtn: "⌫",
    speakBtn: "🔊 பேசு",

    telemetryTitle: "126-ஆயத்தொலைவு தரவு பேலோட் (42 புள்ளிகள் × 3D)",
    telemetryAwaiting: "[ கைகளின் வருகைக்காக காத்திருக்கிறது (0/42 புள்ளிகள்)... ]",
    footerCompliance: "ADA & WCAG 2.1 அணுகல்தன்மை சேவை இடைமுகம்",
    langSwitchLabel: "English",
    langSwitchedAlert: "🌐 மொழி தமிழுக்கு மாற்றப்பட்டது.",
  }
};

// =============================================================================
// 1. KIOSK CLOCK
// =============================================================================
function startKioskClock() {
  const updateClock = () => {
    const now = new Date();
    kioskClock.textContent = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };
  updateClock();
  setInterval(updateClock, 1000);
}

// =============================================================================
// 2. LOAD MEDIAPIPE TASKS VISION (DUAL HAND: numHands = 2)
// =============================================================================
async function initMediaPipeVision() {
  loaderStatus.textContent = "Loading @mediapipe/tasks-vision CDN...";

  try {
    let visionModule = window.tasksVision;
    if (!visionModule) {
      visionModule = await import(/* @vite-ignore */ MEDIAPIPE_CDN_URL);
      window.tasksVision = visionModule;
    }

    const { HandLandmarker, FilesetResolver, DrawingUtils } = visionModule;
    HandLandmarkerClass = HandLandmarker;
    drawingUtils = new DrawingUtils(canvasCtx);

    loaderStatus.textContent = "Loading Dual-Hand 42-Point Landmark Model...";

    const vision = await FilesetResolver.forVisionTasks(WASM_CDN_URL);

    // Create HandLandmarker with 2 hands support (42 points total)
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuError) {
      console.warn("[!] GPU delegate failed, falling back to CPU:", gpuError);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    }

    console.log("[*] MediaPipe HandLandmarker configured for 2 Hands (42 points).");
    loaderStatus.textContent = "Requesting Webcam Access...";
    return true;
  } catch (error) {
    console.error("[!] Failed to initialize MediaPipe Tasks Vision:", error);
    loaderStatus.textContent = "Error loading MediaPipe. Check network connection.";
    return false;
  }
}

// =============================================================================
// 3. WEBCAM STREAM SETUP
// =============================================================================
async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera API not supported by this browser.");
    return false;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadeddata = () => {
        video.play();
        webcamRunning = true;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        loaderOverlay.classList.add("hidden");
        btnToggleCam.classList.add("active");
        camBtnText.textContent = "Camera Active";

        console.log(`[*] Webcam online: ${canvas.width}x${canvas.height}`);
        resolve(true);
      };
    });
  } catch (err) {
    console.error("[!] Camera access denied:", err);
    loaderStatus.textContent = "Camera access denied. Please grant webcam permissions.";
    return false;
  }
}

function stopWebcam() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    video.srcObject = null;
  }
  webcamRunning = false;
  cancelSilenceTimer();
  btnToggleCam.classList.remove("active");
  camBtnText.textContent = "Camera Offline";
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  reticleBadge.textContent = "Camera Offline";
  reticleBadge.className = "reticle-badge searching";
  fpsCounter.textContent = "--";
  ptsCounter.textContent = "0/42";
  hand1Indicator.className = "hand-chip offline";
  hand2Indicator.className = "hand-chip offline";
}

// =============================================================================
// 4. REAL-TIME GEOMETRIC GESTURE CALCULATION ENGINE (1 & 2 HANDS / 42 PTS)
// =============================================================================
function dist2D(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function getFingerStates(landmarks) {
  // landmarks has 21 points
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];

  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const indexMcp = landmarks[5];

  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const middleMcp = landmarks[9];

  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const ringMcp = landmarks[13];

  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];
  const pinkyMcp = landmarks[17];

  // Robust orientation-invariant extension checks using Euclidean distance ratios along the kinematic finger chains
  // Does not fail when hand is tilted, horizontal, or angled toward camera/chest
  const indexExt = dist2D(indexTip, wrist) > dist2D(indexPip, wrist) * 1.12 && dist2D(indexTip, indexMcp) > dist2D(indexPip, indexMcp) * 0.95;
  const middleExt = dist2D(middleTip, wrist) > dist2D(middlePip, wrist) * 1.12 && dist2D(middleTip, middleMcp) > dist2D(middlePip, middleMcp) * 0.95;
  const ringExt = dist2D(ringTip, wrist) > dist2D(ringPip, wrist) * 1.12 && dist2D(ringTip, ringMcp) > dist2D(ringPip, ringMcp) * 0.95;
  const pinkyExt = dist2D(pinkyTip, wrist) > dist2D(pinkyPip, wrist) * 1.12 && dist2D(pinkyTip, pinkyMcp) > dist2D(pinkyPip, pinkyMcp) * 0.95;

  // Thumb extended if tip is far from palm center (MCP 9) and index MCP (MCP 5)
  const thumbExt = dist2D(thumbTip, indexMcp) > 0.075 && dist2D(thumbTip, wrist) > dist2D(thumbMcp, wrist) * 1.12;

  // Directional orientations (relative to hand's own wrist/MCPs)
  const thumbUp = thumbTip.y < thumbMcp.y - 0.035;
  const thumbDown = thumbTip.y > thumbMcp.y + 0.035;
  const fingersUp = indexTip.y < indexMcp.y - 0.02 && middleTip.y < middleMcp.y - 0.02;
  const fingersDown = indexTip.y > wrist.y + 0.04 && middleTip.y > wrist.y + 0.04;

  return {
    wrist,
    thumbTip,
    thumbMcp,
    thumbIp,
    indexTip,
    indexPip,
    indexMcp,
    middleTip,
    middlePip,
    middleMcp,
    ringTip,
    ringPip,
    ringMcp,
    pinkyTip,
    pinkyPip,
    pinkyMcp,
    thumbExt,
    indexExt,
    middleExt,
    ringExt,
    pinkyExt,
    thumbUp,
    thumbDown,
    fingersUp,
    fingersDown,
  };
}

// =============================================================================
// 4. BEGINNER SIGN VOCABULARY DICTIONARY (20 WORDS)
// =============================================================================
const BEGINNER_VOCABULARY = [
  {
    key: "HELLO",
    label: "Hello",
    labelTa: "வணக்கம்",
    wordTa: "வணக்கம்",
    emoji: "👋",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Open palm waving facing forward",
    tipTa: "திறந்த உள்ளங்கையை அசைக்கவும்",
    speech: "Hello! Welcome to citizen services.",
    speechTa: "வணக்கம்! குடிமக்கள் சேவைகளுக்கு நல்வரவு.",
    response: "Hello! Welcome to Counter #04. How can I assist you with your services today?",
    responseTa: "வணக்கம்! கவுண்டர் #04-க்கு நல்வரவு. உங்களுக்கு எவ்வாறு உதவ முடியும்?",
  },
  {
    key: "THANK_YOU",
    label: "Thank You",
    labelTa: "நன்றி",
    wordTa: "நன்றி",
    emoji: "🙏",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both palms touching together in prayer",
    tipTa: "இரு கைகளையும் குவித்து வணக்கம் தெரிவிக்கவும்",
    speech: "Thank you very much.",
    speechTa: "மிக்க நன்றி.",
    response: "You are very welcome! It is our honor to serve you.",
    responseTa: "நல்வரவு! உங்களுக்கு சேவை செய்வதில் நாங்கள் மகிழ்ச்சியடைகிறோம்.",
  },
  {
    key: "YES",
    label: "Yes",
    labelTa: "ஆம்",
    wordTa: "ஆம்",
    emoji: "👍",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb pointing upright (thumbs up)",
    tipTa: "கட்டை விரலை மேல்நோக்கி உயர்த்தவும் (தம்ப்ஸ் அப்)",
    speech: "Yes, confirmed.",
    speechTa: "ஆம், உறுதிப்படுத்தப்பட்டது.",
    response: "Understood: Confirmed. Proceeding with your application.",
    responseTa: "புரிந்துகொள்ளப்பட்டது: உறுதிப்படுத்தப்பட்டது. உங்கள் விண்ணப்பம் தொடர்கிறது.",
  },
  {
    key: "NO",
    label: "No",
    labelTa: "இல்லை",
    wordTa: "இல்லை",
    emoji: "👎",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb pointing downward (thumbs down)",
    tipTa: "கட்டை விரலை கீழ்நோக்கி காட்டவும் (தம்ப்ஸ் டவுன்)",
    speech: "No, decline.",
    speechTa: "இல்லை, நிராகரிக்கப்பட்டது.",
    response: "Noted: Cancelled. We will not proceed with this action.",
    responseTa: "குறிப்பு எடுக்கப்பட்டது: ரத்து செய்யப்பட்டது.",
  },
  {
    key: "PLEASE",
    label: "Please",
    labelTa: "தயவுசெய்து",
    wordTa: "தயவுசெய்து",
    emoji: "🤲",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open cupped palms held gently upward",
    tipTa: "இரு கைகளையும் ஏந்தி மேல்நோக்கி காட்டவும்",
    speech: "Please assist me.",
    speechTa: "தயவுசெய்து எனக்கு உதவுங்கள்.",
    response: "Certainly! We are glad to assist you with every step.",
    responseTa: "நிச்சயமாக! உங்களுக்கு உதவ நாங்கள் தயாராக உள்ளோம்.",
  },
  {
    key: "HELP",
    label: "Help",
    labelTa: "உதவி",
    wordTa: "உதவி",
    emoji: "🆘",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Fist resting on flat palm of other hand",
    tipTa: "ஒரு உள்ளங்கையில் மற்றொரு மூடிய கையை வைக்கவும்",
    speech: "I need assistance.",
    speechTa: "எனக்கு உதவி தேவை.",
    response: "Assistance alert acknowledged. A public service officer is attending to you.",
    responseTa: "உதவி எச்சரிக்கை பெறப்பட்டது. அதிகாரி ஒருவர் உங்களிடம் வருகிறார்.",
  },
  {
    key: "GOOD",
    label: "Good",
    labelTa: "நல்லது",
    wordTa: "நல்லது",
    emoji: "👌",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb and index touching in circle (OK)",
    tipTa: "கட்டை விரல் மற்றும் ஆட்காட்டி விரலை வட்டமாக்கவும் (OK)",
    speech: "Very good.",
    speechTa: "மிகவும் நல்லது.",
    response: "Great! Glad to hear everything is going smoothly.",
    responseTa: "சிறப்பானது! எல்லாம் சீராக செல்வதில் மகிழ்ச்சி.",
  },
  {
    key: "BAD",
    label: "Bad",
    labelTa: "மோசம்",
    wordTa: "மோசம்",
    emoji: "⛔",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Hand facing downward with fingers lowered",
    tipTa: "கையை கீழ்நோக்கி சாய்த்து காட்டவும்",
    speech: "There is an issue.",
    speechTa: "ஒரு சிக்கல் உள்ளது.",
    response: "We apologize for the inconvenience. Let us resolve this for you.",
    responseTa: "சிரமத்திற்கு மன்னிக்கவும். இதை நாங்கள் சரிசெய்கிறோம்.",
  },
  {
    key: "LOVE",
    label: "I Love You",
    labelTa: "அன்பு",
    wordTa: "அன்பு",
    emoji: "🤟",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb, index, and pinky extended (ASL ILY)",
    tipTa: "கட்டை, ஆட்காட்டி மற்றும் சுண்டு விரலை நீட்டவும் (ILY)",
    speech: "I love you.",
    speechTa: "அன்பும் வாழ்த்துகளும்.",
    response: "Much love, warmth, and respect right back to you! 🤟",
    responseTa: "உங்களுக்கும் எங்கள் அன்பும் மரியாதையும்! 🤟",
  },
  {
    key: "HEART",
    label: "Heart",
    labelTa: "இதயம்",
    wordTa: "இதயம்",
    emoji: "❤️",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Thumbs and index fingertips joined in heart shape",
    tipTa: "இரு கை விரல்களால் இதய வடிவம் அமைக்கவும்",
    speech: "Kindness from the heart.",
    speechTa: "இதயபூர்வமான அன்பு.",
    response: "Heartfelt kindness received! Wishing you peace and happiness.",
    responseTa: "இதயபூர்வமான வாழ்த்துகள்! அமைதியும் மகிழ்ச்சியும் பெருகட்டும்.",
  },
  {
    key: "STOP",
    label: "Stop",
    labelTa: "நில்",
    wordTa: "நில்",
    emoji: "🛑",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both wrists crossed in front forming an X",
    tipTa: "இரு மணிக்கட்டுகளையும் குறுக்காக வைத்து X அமைக்கவும்",
    speech: "Stop and pause.",
    speechTa: "நில், சற்று பொறுங்கள்.",
    response: "Process halted immediately. Take your time to review.",
    responseTa: "செயல்முறை உடனடியாக நிறுத்தப்பட்டது.",
  },
  {
    key: "SORRY",
    label: "Sorry",
    labelTa: "மன்னிக்கவும்",
    wordTa: "மன்னிக்கவும்",
    emoji: "🙇",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Closed fist held firmly over chest",
    tipTa: "மார்பின் மீது மூடிய கையை வைக்கவும்",
    speech: "I am sorry.",
    speechTa: "மன்னிக்கவும்.",
    response: "No worries at all! Everything is completely fine.",
    responseTa: "பரவாயில்லை! எந்தப் பிரச்சனையும் இல்லை.",
  },
  {
    key: "FRIEND",
    label: "Friend",
    labelTa: "நண்பர்",
    wordTa: "நண்பர்",
    emoji: "🤝",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both index fingers hooked or wrists linked",
    tipTa: "இரு ஆட்காட்டி விரல்களையும் ஒன்றோடொன்று இணைக்கவும்",
    speech: "We are friends.",
    speechTa: "நாம் நண்பர்கள்.",
    response: "Welcome, dear friend! You are always supported and valued here.",
    responseTa: "நல்வரவு அன்பு நண்பரே! நீங்கள் எப்போதும் மதிக்கப்படுகிறீர்கள்.",
  },
  {
    key: "MORE",
    label: "More",
    labelTa: "மேலும்",
    wordTa: "மேலும்",
    emoji: "🤏",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Fingertips of both hands touching together",
    tipTa: "இரு கை விரல் நுனிகளையும் ஒன்றாக குவிக்கவும்",
    speech: "I would like more.",
    speechTa: "எனக்கு மேலும் வேண்டும்.",
    response: "Displaying additional options for your request.",
    responseTa: "கூடுதல் விருப்பங்கள் திரையில் காட்டப்படுகின்றன.",
  },
  {
    key: "WATER",
    label: "Water",
    labelTa: "தண்ணீர்",
    wordTa: "தண்ணீர்",
    emoji: "💧",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "W sign: Index, middle, and ring extended upright",
    tipTa: "ஆட்காட்டி, நடு மற்றும் மோதிர விரல்களை நீட்டவும் (W)",
    speech: "Drinking water.",
    speechTa: "குடிதண்ணீர் வேண்டும்.",
    response: "Drinking water dispenser is located next to Counter #04.",
    responseTa: "குடிநீர் கவுண்டர் #04-க்கு அருகில் உள்ளது.",
  },
  {
    key: "FOOD",
    label: "Food",
    labelTa: "உணவு",
    wordTa: "உணவு",
    emoji: "🍽️",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open palms held side by side facing upward",
    tipTa: "இரு உள்ளங்கைகளையும் தட்டு போல ஏந்தவும்",
    speech: "Food and dining.",
    speechTa: "உணவு மற்றும் சிற்றுண்டி.",
    response: "Community dining and cafeteria facilities are on Level 1.",
    responseTa: "உணவக வசதிகள் தளம் 1-ல் உள்ளன.",
  },
  {
    key: "WELCOME",
    label: "Welcome",
    labelTa: "நல்வரவு",
    wordTa: "நல்வரவு",
    emoji: "👐",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open palms spread wide welcoming forward",
    tipTa: "இரு கைகளையும் விரித்து வரவேற்கவும்",
    speech: "Welcome!",
    speechTa: "நல்வரவு!",
    response: "A very warm welcome to Counter #04! How can I assist you?",
    responseTa: "கவுண்டர் #04-க்கு மனமார்ந்த நல்வரவு!",
  },
  {
    key: "PEACE",
    label: "Peace",
    labelTa: "அமைதி",
    wordTa: "அமைதி",
    emoji: "✌️",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "V sign: Index and middle fingers extended",
    tipTa: "இரண்டு விரல்களை V வடிவில் காட்டவும்",
    speech: "Peace to you.",
    speechTa: "உங்களுக்கு அமைதி உண்டாகட்டும்.",
    response: "Peace and harmony to you and your community!",
    responseTa: "அனைவருக்கும் அமைதியும் ஒற்றுமையும் நிலவட்டும்!",
  },
  {
    key: "TOGETHER",
    label: "Together",
    labelTa: "ஒன்றாக",
    wordTa: "ஒன்றாக",
    emoji: "👥",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both fists touching side by side",
    tipTa: "இரு மூடிய கைகளையும் ஒன்றாக சேர்க்கவும்",
    speech: "We work together.",
    speechTa: "நாம் ஒன்றாக செயல்படுவோம்.",
    response: "Together we accomplish more! United in service.",
    responseTa: "ஒன்றாக இணைந்து அதிக சாதனைகள் புரிவோம்!",
  },
  {
    key: "DONE",
    label: "Done",
    labelTa: "முடிந்தது",
    wordTa: "முடிந்தது",
    emoji: "✅",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Dual thumbs up / palms spread outward",
    tipTa: "இரு கைகளாலும் தம்ப்ஸ் அப் அல்லது கைகளை விரிக்கவும்",
    speech: "Finished and done.",
    speechTa: "வேலை முடிந்தது.",
    response: "Application marked as completed! Have an excellent day ahead.",
    responseTa: "விண்ணப்பம் வெற்றிகரமாக முடிந்தது! இனிய நாளாக அமையட்டும்.",
  },
  {
    key: "EMERGENCY",
    label: "Emergency",
    labelTa: "அவசரம்",
    wordTa: "அவசரம்",
    emoji: "🚨",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both hands waving urgently side to side",
    tipTa: "இரு கைகளையும் வேகமாக பக்கவாட்டில் அசைக்கவும்",
    speech: "Emergency assistance needed immediately.",
    speechTa: "உடனடி அவசர உதவி தேவை.",
    response: "Emergency alert triggered! Security and medical personnel are dispatched to Counter #04.",
    responseTa: "அவசர எச்சரிக்கை விடுக்கப்பட்டது! பாதுகாப்பு மற்றும் மருத்துவ குழுவினர் வருகிறார்கள்.",
  },
  {
    key: "DOCTOR",
    label: "Doctor",
    labelTa: "மருத்துவர்",
    wordTa: "மருத்துவர்",
    emoji: "🩺",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Tapping fingers against opposite wrist pulse",
    tipTa: "எதிர் மணிக்கட்டு நாடியை விரலால் தட்டவும்",
    speech: "I need to see a doctor.",
    speechTa: "எனக்கு மருத்துவரை பார்க்க வேண்டும்.",
    response: "Doctor requested. Healthcare liaison at Booth #01 has been notified.",
    responseTa: "மருத்துவ அதிகாரிக்கு தகவல் தெரிவிக்கப்பட்டுள்ளது.",
  },
  {
    key: "HOSPITAL",
    label: "Hospital",
    labelTa: "மருத்துவமனை",
    wordTa: "மருத்துவமனை",
    emoji: "🏥",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Index and middle tracing cross pattern on shoulder",
    tipTa: "தோள்பட்டையில் சிலுவை வடிவம் வரையவும்",
    speech: "Hospital and emergency room directions.",
    speechTa: "மருத்துவமனை வழிகாட்டல்.",
    response: "The municipal health clinic is on Ground Floor, East Wing.",
    responseTa: "மருத்துவமனை தரைத்தளம், கிழக்கு பகுதியில் உள்ளது.",
  },
  {
    key: "MEDICINE",
    label: "Medicine",
    labelTa: "மருந்து",
    wordTa: "மருந்து",
    emoji: "💊",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Middle finger pivoting in palm of open opposite hand",
    tipTa: "உள்ளங்கையில் நடுவிரலை சுழற்றவும்",
    speech: "I need medicine or pharmacy assistance.",
    speechTa: "எனக்கு மருந்து அல்லது மருந்தகம் தேவை.",
    response: "Municipal prescription pharmacy is located next to Counter #08.",
    responseTa: "மருந்தகம் கவுண்டர் #08-க்கு அருகில் உள்ளது.",
  },
  {
    key: "PAIN",
    label: "Pain",
    labelTa: "வலி",
    wordTa: "வலி",
    emoji: "🤕",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both index fingers pointing toward each other and twisting",
    tipTa: "இரு ஆட்காட்டி விரல்களையும் ஒன்றையொன்று நோக்கி திருப்பவும்",
    speech: "I am experiencing pain and need medical care.",
    speechTa: "எனக்கு உடல் வலி உள்ளது, சிகிச்சை தேவை.",
    response: "First-aid responders have been requested to Counter #04 immediately.",
    responseTa: "முதலுதவி குழு உடனடியாக அழைக்கப்பட்டுள்ளது.",
  },
  {
    key: "WAIT",
    label: "Wait",
    labelTa: "காத்திருங்கள்",
    wordTa: "காத்திருங்கள்",
    emoji: "⏳",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both hands held chest high with open palms fluttering",
    tipTa: "இரு கைகளையும் மார்பளவு உயர்த்தி விரல்களை அசைக்கவும்",
    speech: "Please wait a moment.",
    speechTa: "தயவுசெய்து சற்று காத்திருங்கள்.",
    response: "Please take a comfortable seat in the waiting lounge. We will call you momentarily.",
    responseTa: "காத்திருப்போர் அறையில் அமருங்கள். விரைவில் அழைக்கிறோம்.",
  },
  {
    key: "TIME",
    label: "Time",
    labelTa: "நேரம்",
    wordTa: "நேரம்",
    emoji: "⏰",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Index finger tapping back of opposite wrist",
    tipTa: "மணிக்கட்டில் கடிகாரம் இருக்கும் இடத்தை தட்டவும்",
    speech: "What time is it?",
    speechTa: "இப்போது நேரம் என்ன?",
    response: "Current terminal time is displayed in the live header clock.",
    responseTa: "தற்போதைய நேரம் முகப்புத் திரையில் காட்டப்படுகிறது.",
  },
  {
    key: "MONEY",
    label: "Money",
    labelTa: "பணம்",
    wordTa: "பணம்",
    emoji: "💵",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb rubbing across fingertips in coin motion",
    tipTa: "கட்டை விரலை மற்ற விரல்களில் தேய்க்கவும்",
    speech: "Payment and fee processing.",
    speechTa: "கட்டணம் மற்றும் பண பரிவர்த்தனை.",
    response: "Cash, credit cards, and transit cards are accepted at the cashier desk (Counter #02).",
    responseTa: "கட்டண கவுண்டர் #02-ல் செலுத்தலாம்.",
  },
  {
    key: "WHERE",
    label: "Where",
    labelTa: "எங்கே",
    wordTa: "எங்கே",
    emoji: "❓",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Open palm facing upward shaking side to side inquiringly",
    tipTa: "உள்ளங்கையை மேல்நோக்கி வைத்து அசைக்கவும்",
    speech: "Where do I go?",
    speechTa: "நான் எங்கு செல்ல வேண்டும்?",
    response: "Please let us know which department or office you are looking for.",
    responseTa: "நீங்கள் எந்த துறைக்கு செல்ல வேண்டும் என்று கூறுங்கள்.",
  },
  {
    key: "FAMILY",
    label: "Family",
    labelTa: "குடும்பம்",
    wordTa: "குடும்பம்",
    emoji: "👨‍👩‍👧",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both hands forming F shapes moving forward in a circle",
    tipTa: "இரு கைகளாலும் வட்டம் போல முன்னோக்கி நகர்த்தவும்",
    speech: "My family members are with me.",
    speechTa: "என் குடும்பத்தினர் என்னுடன் உள்ளனர்.",
    response: "Family seating and companion assistance are available in Lounge B.",
    responseTa: "குடும்பத்தினருக்கான இருக்கை பகுதி லவுஞ்ச் B-ல் உள்ளது.",
  },
  {
    key: "REPEAT",
    label: "Repeat",
    labelTa: "மீண்டும்",
    wordTa: "மீண்டும்",
    emoji: "🔁",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Curved open palm flipping over into opposite palm",
    tipTa: "உள்ளங்கையை எதிர் உள்ளங்கையில் புரட்டி வைக்கவும்",
    speech: "Please repeat that again.",
    speechTa: "தயவுசெய்து மீண்டும் கூறுங்கள்.",
    response: "Repeating the latest instruction and display information.",
    responseTa: "முந்தைய தகவலை மீண்டும் கூறுகிறேன்.",
  },
  {
    key: "I",
    label: "I / Me",
    labelTa: "நான்",
    wordTa: "நான்",
    emoji: "🙋",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Point index finger towards your chest (or pinky extended upright)",
    tipTa: "ஆட்காட்டி விரலை மார்பை நோக்கி சுட்டவும்",
    speech: "I",
    speechTa: "நான்",
    response: "Acknowledged for you personally (I/Me).",
    responseTa: "நீங்கள் தனிப்பட்ட முறையில் குறிப்பிட்டது ஏற்றுக்கொள்ளப்பட்டது (நான்).",
  },
  {
    key: "WE",
    label: "We",
    labelTa: "நாம்",
    wordTa: "நாம்",
    emoji: "👥",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Index finger sweeps across chest from right to left shoulder",
    tipTa: "ஆட்காட்டி விரலை மார்பின் குறுக்கே அசைக்கவும்",
    speech: "We",
    speechTa: "நாம்",
    response: "Understood: Representing your party (We/Us).",
    responseTa: "உங்கள் குழுவின் கோரிக்கை ஏற்றுக்கொள்ளப்பட்டது (நாம்).",
  },
  {
    key: "LIKE",
    label: "Like",
    labelTa: "பிடிக்கும்",
    wordTa: "பிடிக்கும்",
    emoji: "✨",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb and middle finger pinch together at chest and pull outward",
    tipTa: "கட்டை மற்றும் நடுவிரலை குவித்து வெளியே இழுக்கவும்",
    speech: "I like this.",
    speechTa: "எனக்கு இது பிடிக்கும்.",
    response: "Wonderful! We are delighted to hear that you like this.",
    responseTa: "அருமை! உங்களுக்கு இது பிடித்ததில் நாங்கள் மகிழ்ச்சியடைகிறோம்.",
  },
];

function renderVocabDeck() {
  if (!vocabGrid) return;
  vocabGrid.innerHTML = "";

  const isTa = currentLanguage === "ta";

  BEGINNER_VOCABULARY.forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "vocab-chip";
    chip.setAttribute("data-key", item.key);

    const displayLabel = isTa ? (item.labelTa || item.label) : item.label;
    const displayTip = isTa ? (item.tipTa || item.tip) : item.tip;
    const displayTag = isTa ? (item.handsCount === 2 ? "2 கைகள்" : "1 கை") : item.hands;

    chip.title = `${displayLabel} (${displayTag}) - ${displayTip}`;

    chip.innerHTML = `
      <span class="vocab-chip-emoji">${item.emoji}</span>
      <span class="vocab-chip-label">${displayLabel}</span>
      <span class="vocab-chip-tag">${displayTag}</span>
    `;

    chip.addEventListener("click", () => {
      const match = buildVocabMatch(item.key, 0.99);
      if (match) {
        highlightVocabChip(item.key);
        handleGestureCalculation(match, match.handsCount, performance.now());
        if (isTtsEnabled) {
          const spoken = isTa ? (item.speechTa || displayLabel) : `${item.label}. ${item.tip}`;
          speakText(spoken);
        }
      }
    });

    vocabGrid.appendChild(chip);
  });
}

function highlightVocabChip(vocabKey) {
  if (!vocabGrid) return;
  const chips = vocabGrid.querySelectorAll(".vocab-chip");
  chips.forEach((c) => {
    if (c.getAttribute("data-key") === vocabKey) {
      c.classList.add("active");
      c.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      c.classList.remove("active");
    }
  });
}

function buildVocabMatch(key, confidence, isMotion = false) {
  const item = BEGINNER_VOCABULARY.find((v) => v.key === key);
  if (!item) return null;

  const isTa = currentLanguage === "ta";
  const displayLabel = isTa ? (item.labelTa || item.label) : item.label;
  const displayTip = isTa ? (item.tipTa || item.tip) : item.tip;
  const displaySpeech = isTa ? (item.speechTa || item.speech) : item.speech;
  const displayResponse = isTa ? (item.responseTa || item.response) : item.response;

  return {
    ...item,
    label: displayLabel,
    labelEn: item.label,
    labelTa: item.labelTa || item.label,
    wordTa: item.wordTa || item.labelTa || item.label,
    tip: displayTip,
    speech: displaySpeech,
    response: displayResponse,
    confidence,
    isMotion,
    fullText: `${item.emoji} ${displayLabel} (${displayTip})`,
  };
}

/**
 * Workable Language Switcher Engine
 * Dynamically toggles between English ("en") and Tamil ("ta")
 */
function setLanguage(lang) {
  if (lang !== "en" && lang !== "ta") lang = "en";
  currentLanguage = lang;
  try {
    localStorage.setItem("bridgesign_lang", lang);
  } catch (e) {
    console.warn("Could not save language to localStorage:", e);
  }

  // Update switcher buttons in Header
  if (btnLangEn && btnLangTa) {
    btnLangEn.classList.toggle("active", lang === "en");
    btnLangTa.classList.toggle("active", lang === "ta");
  }

  // Update quick toggle button in Chatbox header
  if (btnToggleLangText) {
    btnToggleLangText.textContent = lang === "en" ? "🌐 தமிழ்" : "🌐 English";
  }

  const t = UI_TRANSLATIONS[lang];
  if (!t) return;

  // Header & station meta
  if (kioskTagline) kioskTagline.textContent = t.kioskTagline;
  if (kioskTitle) kioskTitle.textContent = t.kioskTitle;
  if (kioskStationId) kioskStationId.textContent = t.kioskStationId;
  if (backendStatusText) {
    backendStatusText.textContent = backendOnline ? t.backendConnected : t.backendConnecting;
  }

  // Viewport panel
  if (panelTitle) panelTitle.textContent = t.viewportTitle;
  if (camBtnText) camBtnText.textContent = webcamRunning ? t.camActive : t.camOffline;
  if (btnMirrorText) btnMirrorText.textContent = t.mirror;
  if (btnLandmarksText) btnLandmarksText.textContent = t.landmarks;
  if (pipBtnText) pipBtnText.textContent = t.floatPip;

  // Prediction card
  if (calcCardTitle) calcCardTitle.textContent = t.calcTitle;
  if (predictionSubtext && !activeGestureObj) {
    predictionSubtext.textContent = t.calcPlaceholder;
  }
  if (confidenceLabel) confidenceLabel.textContent = t.calcAccuracy;
  if (vocabDeckTitle) vocabDeckTitle.textContent = t.vocabTitle;
  if (vocabDeckMeta) vocabDeckMeta.textContent = t.vocabMeta;

  // Chatbox
  if (chatboxTitle) chatboxTitle.textContent = t.chatboxTitle;
  if (btnToggleTtsText) btnToggleTtsText.textContent = isTtsEnabled ? t.ttsOn : t.ttsOff;
  if (btnToggleAutosendText) btnToggleAutosendText.textContent = isAutoSendEnabled ? t.autoSend : (lang === "ta" ? "⏸️ கைமுறை அனுப்பு" : "⏸️ Manual Send");
  if (btnExportChatText) btnExportChatText.textContent = t.export;
  if (btnClearChatText) btnClearChatText.textContent = t.clear;
  if (stagingTag) stagingTag.textContent = t.activeGesture;
  if (!stagingGestureKey && stagingStatusText) stagingStatusText.textContent = t.holdToCommit;
  if (chatInput) chatInput.placeholder = t.chatPlaceholder;
  if (btnSendChatText) btnSendChatText.textContent = t.sendBtn;
  if (btnSpaceText) btnSpaceText.textContent = t.spaceBtn;
  if (btnSpeakText) btnSpeakText.textContent = t.speakBtn;

  // Telemetry & footer
  if (telemetryTitle) telemetryTitle.textContent = t.telemetryTitle;
  if (footerCompliance) footerCompliance.textContent = t.footerCompliance;

  // Re-render vocab deck with translated labels
  renderVocabDeck();

  // If there's an active gesture, refresh its label in prediction display & staging
  if (activeGestureObj) {
    const match = buildVocabMatch(activeGestureObj.key, activeGestureObj.confidence, activeGestureObj.isMotion);
    if (match) {
      activeGestureObj = match;
      predictionDisplay.textContent = `${match.emoji} ${match.label}`;
      stagingGestureName.textContent = `${match.emoji} ${match.label}`;
    }
  }

  // Pre-fetch voices
  if (window.speechSynthesis && window.speechSynthesis.getVoices) {
    window.speechSynthesis.getVoices();
  }
}

// =============================================================================
// =============================================================================
// 4B. TEMPORAL MOTION TRACKING & KINEMATIC RECOGNITION ENGINE
// =============================================================================
const MOTION_BUFFER_MAX = 25; // ~0.8s of video history at 30 FPS
const motionBuffer = [];

// Dynamic motion sign persistence & turnaround grace window state
let lastMotionMatch = null;
let lastMotionTimestamp = 0;
const MOTION_PERSISTENCE_MS = 380; // Keep motion sign alive across wave turnarounds

function recordHandMotionSnapshot(handsLandmarks, timestamp) {
  if (!handsLandmarks || handsLandmarks.length === 0) {
    if (motionBuffer.length > 0 && timestamp - motionBuffer[motionBuffer.length - 1].timestamp > 350) {
      motionBuffer.length = 0;
      lastMotionMatch = null;
    }
    return;
  }

  const h1 = handsLandmarks[0];
  const h2 = handsLandmarks.length > 1 ? handsLandmarks[1] : null;

  motionBuffer.push({
    timestamp,
    handsCount: handsLandmarks.length,
    // Hand 1 key kinematic points: wrist, knuckle (MCP 9), thumb, index, middle, pinky
    h1Wrist: { x: h1[0].x, y: h1[0].y, z: h1[0].z },
    h1Mcp: { x: h1[9].x, y: h1[9].y, z: h1[9].z }, // Palm center
    h1ThumbTip: { x: h1[4].x, y: h1[4].y, z: h1[4].z },
    h1IndexTip: { x: h1[8].x, y: h1[8].y, z: h1[8].z },
    h1MiddleTip: { x: h1[12].x, y: h1[12].y, z: h1[12].z },
    h1PinkyTip: { x: h1[20].x, y: h1[20].y, z: h1[20].z },
    // Hand 2 key kinematic points
    h2Wrist: h2 ? { x: h2[0].x, y: h2[0].y, z: h2[0].z } : null,
    h2Mcp: h2 ? { x: h2[9].x, y: h2[9].y, z: h2[9].z } : null,
    h2ThumbTip: h2 ? { x: h2[4].x, y: h2[4].y, z: h2[4].z } : null,
    h2IndexTip: h2 ? { x: h2[8].x, y: h2[8].y, z: h2[8].z } : null,
    h2MiddleTip: h2 ? { x: h2[12].x, y: h2[12].y, z: h2[12].z } : null,
    h2PinkyTip: h2 ? { x: h2[20].x, y: h2[20].y, z: h2[20].z } : null,
  });

  if (motionBuffer.length > MOTION_BUFFER_MAX) {
    motionBuffer.shift();
  }
}

/**
 * Robust kinematic oscillation analyzer.
 * Accumulates movement across consecutive frames until stroke reverses direction by >= minStroke.
 * Properly handles smooth frame-to-frame deltas without dropping under fixed per-frame limits.
 */
function analyzeKinematicOscillation(values, minStroke = 0.02) {
  if (!values || values.length < 3) {
    return { reversals: 0, totalDist: 0, maxSpan: 0 };
  }

  let reversals = 0;
  let totalDist = 0;
  let minVal = values[0];
  let maxVal = values[0];
  let overallMin = values[0];
  let overallMax = values[0];
  let currentDirection = 0; // +1 = moving positive, -1 = moving negative

  for (let i = 1; i < values.length; i++) {
    const v = values[i];
    const prev = values[i - 1];
    totalDist += Math.abs(v - prev);

    if (v < overallMin) overallMin = v;
    if (v > overallMax) overallMax = v;

    if (currentDirection === 0) {
      if (v - minVal > minStroke) {
        currentDirection = 1;
        maxVal = v;
      } else if (maxVal - v > minStroke) {
        currentDirection = -1;
        minVal = v;
      }
    } else if (currentDirection === 1) {
      if (v > maxVal) {
        maxVal = v;
      } else if (maxVal - v > minStroke) {
        reversals++;
        currentDirection = -1;
        minVal = v;
      }
    } else if (currentDirection === -1) {
      if (v < minVal) {
        minVal = v;
      } else if (v - minVal > minStroke) {
        reversals++;
        currentDirection = 1;
        maxVal = v;
      }
    }
  }

  const maxSpan = overallMax - overallMin;
  return { reversals, totalDist, maxSpan };
}

function detectDynamicMotionSign(handsLandmarks, now) {
  if (motionBuffer.length < 5) return null;

  const handsCount = handsLandmarks.length;
  const h1 = getFingerStates(handsLandmarks[0]);
  const h2 = handsCount > 1 ? getFingerStates(handsLandmarks[1]) : null;

  // Extract kinematic series for lateral and vertical hand movement
  const h1WristX = motionBuffer.map((s) => s.h1Wrist.x);
  const h1WristY = motionBuffer.map((s) => s.h1Wrist.y);
  const h1TipX = motionBuffer.map((s) => s.h1MiddleTip.x);
  const h1TipY = motionBuffer.map((s) => s.h1MiddleTip.y);
  const h1KnuckleX = motionBuffer.map((s) => s.h1Mcp.x);

  const oscWristX = analyzeKinematicOscillation(h1WristX, 0.018);
  const oscWristY = analyzeKinematicOscillation(h1WristY, 0.016);
  const oscTipX = analyzeKinematicOscillation(h1TipX, 0.020);
  const oscTipY = analyzeKinematicOscillation(h1TipY, 0.018);
  const oscKnuckleX = analyzeKinematicOscillation(h1KnuckleX, 0.018);

  const lateralReversalsH1 = Math.max(oscWristX.reversals, oscTipX.reversals, oscKnuckleX.reversals);
  const lateralSpanH1 = Math.max(oscWristX.maxSpan, oscTipX.maxSpan, oscKnuckleX.maxSpan);
  const lateralDistH1 = Math.max(oscWristX.totalDist, oscTipX.totalDist, oscKnuckleX.totalDist);

  // 1. DUAL HAND WAVING / FLAPPING -> EMERGENCY (🚨)
  if (handsCount >= 2 && h2) {
    const h2WristX = motionBuffer.map((s) => (s.h2Wrist ? s.h2Wrist.x : s.h1Wrist.x));
    const h2WristY = motionBuffer.map((s) => (s.h2Wrist ? s.h2Wrist.y : s.h1Wrist.y));
    const oscH2X = analyzeKinematicOscillation(h2WristX, 0.018);
    const oscH2Y = analyzeKinematicOscillation(h2WristY, 0.018);

    const allH1Up = h1.indexExt && h1.middleExt && h1.ringExt;
    const allH2Up = h2.indexExt && h2.middleExt && h2.ringExt;

    const dualOsc =
      (lateralReversalsH1 >= 1 || oscWristY.reversals >= 1) &&
      (oscH2X.reversals >= 1 || oscH2Y.reversals >= 1) &&
      (lateralDistH1 > 0.07 || oscH2X.totalDist > 0.07 || oscWristY.totalDist > 0.07);

    if (allH1Up && allH2Up && dualOsc) {
      return buildVocabMatch("EMERGENCY", 0.99, true);
    }
  }

  // 2. SINGLE HAND WAVING -> HELLO (👋)
  const allH1Up = (h1.indexExt && h1.middleExt && h1.ringExt) || (h1.indexExt && h1.middleExt && h1.pinkyExt);
  if (allH1Up) {
    const isWaving =
      (lateralReversalsH1 >= 2 && lateralSpanH1 >= 0.030) ||
      (lateralReversalsH1 >= 1 && lateralSpanH1 >= 0.050) ||
      lateralDistH1 >= 0.11;

    if (isWaving) {
      return buildVocabMatch("HELLO", 0.98, true);
    }
  }

  // 4. VERTICAL TAPPING ON OPPOSITE WRIST -> TIME (⏰) or DOCTOR (🩺)
  // Supports both hand orders: H1 tapping on H2, or H2 tapping on H1!
  if (handsCount >= 2 && h2) {
    // Direction A: H1 fingers tapping H2 wrist
    const tapDistsA_Index = motionBuffer.map((s) => (s.h2Wrist ? dist2D(s.h1IndexTip, s.h2Wrist) : 999));
    const tapDistsA_Middle = motionBuffer.map((s) => (s.h2Wrist ? dist2D(s.h1MiddleTip, s.h2Wrist) : 999));
    const minTapDistA = Math.min(...tapDistsA_Index, ...tapDistsA_Middle);
    const oscTapA = analyzeKinematicOscillation(tapDistsA_Index, 0.012);

    // Direction B: H2 fingers tapping H1 wrist
    const tapDistsB_Index = motionBuffer.map((s) => (s.h2IndexTip && s.h1Wrist ? dist2D(s.h2IndexTip, s.h1Wrist) : 999));
    const tapDistsB_Middle = motionBuffer.map((s) => (s.h2MiddleTip && s.h1Wrist ? dist2D(s.h2MiddleTip, s.h1Wrist) : 999));
    const minTapDistB = Math.min(...tapDistsB_Index, ...tapDistsB_Middle);
    const oscTapB = analyzeKinematicOscillation(tapDistsB_Index, 0.012);

    if (minTapDistA < 0.22 && (oscTapA.reversals >= 2 || (oscTapA.reversals >= 1 && oscTapA.totalDist > 0.035))) {
      if (h1.indexExt && h1.middleExt) {
        return buildVocabMatch("DOCTOR", 0.98, true);
      } else if (h1.indexExt) {
        return buildVocabMatch("TIME", 0.97, true);
      }
    }

    if (minTapDistB < 0.22 && (oscTapB.reversals >= 2 || (oscTapB.reversals >= 1 && oscTapB.totalDist > 0.035))) {
      if (h2.indexExt && h2.middleExt) {
        return buildVocabMatch("DOCTOR", 0.98, true);
      } else if (h2.indexExt) {
        return buildVocabMatch("TIME", 0.97, true);
      }
    }
  }

  // 4b. SINGLE HAND WRIST-WATCH TAPPING -> TIME (⏰)
  if (h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt) {
    const oscIndexY = analyzeKinematicOscillation(motionBuffer.map((s) => s.h1IndexTip.y), 0.015);
    if (oscIndexY.reversals >= 2 && oscIndexY.maxSpan >= 0.028) {
      return buildVocabMatch("TIME", 0.96, true);
    }
  }

  // 5. THUMB RUBBING FINGERTIPS -> MONEY (💵)
  const thumbIndexDists = motionBuffer.map((s) => dist2D(s.h1ThumbTip, s.h1IndexTip));
  const oscThumbDist = analyzeKinematicOscillation(thumbIndexDists, 0.007);
  const oscThumbX = analyzeKinematicOscillation(motionBuffer.map((s) => s.h1ThumbTip.x), 0.008);
  const avgThumbIndexDist = thumbIndexDists.reduce((a, b) => a + b, 0) / thumbIndexDists.length;

  if (avgThumbIndexDist < 0.12 && !h1.ringExt && (oscThumbDist.reversals >= 2 || oscThumbX.reversals >= 2)) {
    return buildVocabMatch("MONEY", 0.97, true);
  }

  // 6. FINGERS FLUTTERING / WIGGLING -> WAIT (⏳)
  if (allH1Up) {
    const oscIndexY = analyzeKinematicOscillation(motionBuffer.map((s) => s.h1IndexTip.y), 0.012);
    const oscPinkyY = analyzeKinematicOscillation(motionBuffer.map((s) => s.h1PinkyTip.y), 0.012);
    const wristStable = oscWristX.reversals <= 1 && oscWristY.reversals <= 1;

    if (wristStable && (oscIndexY.reversals >= 2 || oscPinkyY.reversals >= 2)) {
      return buildVocabMatch("WAIT", 0.97, true);
    }
  }

  // 7. TWO INDEX FINGERS TWISTING / JABBING INWARD -> PAIN (🤕)
  if (handsCount >= 2 && h2) {
    const indexIndexDists = motionBuffer.map((s) => (s.h2IndexTip ? dist2D(s.h1IndexTip, s.h2IndexTip) : 999));
    const minIndexDist = Math.min(...indexIndexDists);
    const oscIndexDist = analyzeKinematicOscillation(indexIndexDists, 0.010);
    const oscIndexY1 = analyzeKinematicOscillation(motionBuffer.map((s) => s.h1IndexTip.y), 0.012);
    const oscIndexY2 = analyzeKinematicOscillation(motionBuffer.map((s) => (s.h2IndexTip ? s.h2IndexTip.y : 0)), 0.012);

    if (
      h1.indexExt &&
      !h1.middleExt &&
      h2.indexExt &&
      !h2.middleExt &&
      minIndexDist < 0.24 &&
      (oscIndexDist.reversals >= 2 || oscIndexY1.reversals >= 2 || oscIndexY2.reversals >= 2)
    ) {
      return buildVocabMatch("PAIN", 0.98, true);
    }
  }

  // 8. MIDDLE FINGER STIRRING IN FLAT PALM -> MEDICINE (💊)
  if (handsCount >= 2 && h2) {
    // Symmetrical check: H1 stirring in H2 palm, OR H2 stirring in H1 palm
    const medDistsA = motionBuffer.map((s) => (s.h2Mcp ? dist2D(s.h1MiddleTip, s.h2Mcp) : 999));
    const minMedA = Math.min(...medDistsA);
    const oscMedA = analyzeKinematicOscillation(medDistsA, 0.010);

    const medDistsB = motionBuffer.map((s) => (s.h2MiddleTip && s.h1Mcp ? dist2D(s.h2MiddleTip, s.h1Mcp) : 999));
    const minMedB = Math.min(...medDistsB);
    const oscMedB = analyzeKinematicOscillation(medDistsB, 0.010);

    const h2PalmFlat = h2.indexExt && h2.middleExt && h2.ringExt;
    const h1PalmFlat = h1.indexExt && h1.middleExt && h1.ringExt;

    if (h2PalmFlat && minMedA < 0.18 && (oscMedA.reversals >= 2 || oscMedA.totalDist > 0.04)) {
      return buildVocabMatch("MEDICINE", 0.97, true);
    }
    if (h1PalmFlat && minMedB < 0.18 && (oscMedB.reversals >= 2 || oscMedB.totalDist > 0.04)) {
      return buildVocabMatch("MEDICINE", 0.97, true);
    }
  }

  // 9. OPEN PALM SHAKE -> WHERE (❓)
  const palmUpward = h1.indexExt && h1.middleExt && h1.indexTip.y > h1.wrist.y - 0.26;
  if (palmUpward && lateralReversalsH1 >= 2 && (lateralSpanH1 >= 0.028 || lateralDistH1 >= 0.045)) {
    return buildVocabMatch("WHERE", 0.97, true);
  }

  // 10. HAND FLIP / LANDING ON PALM -> REPEAT (🔁)
  if (handsCount >= 2 && h2) {
    const startH1Y = motionBuffer[0].h1Wrist.y;
    const endH1Y = motionBuffer[motionBuffer.length - 1].h1Wrist.y;
    const lastSnap = motionBuffer[motionBuffer.length - 1];
    const endDist = lastSnap.h2Wrist ? dist2D(lastSnap.h1Wrist, lastSnap.h2Wrist) : 999;

    const allH2Up = h2.indexExt && h2.middleExt;
    if (allH2Up && endH1Y - startH1Y > 0.04 && endDist < 0.18) {
      return buildVocabMatch("REPEAT", 0.96, true);
    }
  }

  // 11. OUTWARD FLICK / SPREAD -> DONE (✅)
  if (handsCount >= 2 && motionBuffer.length >= 8) {
    const firstSnap = motionBuffer[0];
    const lastSnap = motionBuffer[motionBuffer.length - 1];
    if (firstSnap.h2Wrist && lastSnap.h2Wrist) {
      const startWristDist = dist2D(firstSnap.h1Wrist, firstSnap.h2Wrist);
      const endWristDist = dist2D(lastSnap.h1Wrist, lastSnap.h2Wrist);

      if (endWristDist - startWristDist > 0.11) {
        return buildVocabMatch("DONE", 0.98, true);
      }
    }
  }

  // 12. INDEX SWEEP ACROSS CHEST -> WE (👥)
  if (h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt) {
    if (lateralSpanH1 >= 0.055 || (lateralReversalsH1 >= 1 && lateralDistH1 >= 0.065)) {
      return buildVocabMatch("WE", 0.97, true);
    }
  }

  // 13. PINCH & PULL OUTWARD AT CHEST -> LIKE (✨)
  const thumbMiddleDist = dist2D(h1.thumbTip, h1.middleTip);
  if (thumbMiddleDist < 0.065 && h1.indexExt && motionBuffer.length >= 5) {
    const firstZ = motionBuffer[0].h1Wrist.z || 0;
    const lastZ = motionBuffer[motionBuffer.length - 1].h1Wrist.z || 0;
    const zDelta = Math.abs(lastZ - firstZ);
    const yDelta = Math.abs(motionBuffer[motionBuffer.length - 1].h1Wrist.y - motionBuffer[0].h1Wrist.y);
    if (zDelta > 0.015 || yDelta > 0.02 || oscWristX.totalDist > 0.035) {
      return buildVocabMatch("LIKE", 0.98, true);
    }
  }

  return null;
}

function calculateRealTimeGesture(handsLandmarks) {
  if (!handsLandmarks || handsLandmarks.length === 0) {
    return null;
  }

  const now = performance.now();
  recordHandMotionSnapshot(handsLandmarks, now);

  // 1. Dynamic Motion Sign Detection (Takes highest priority when movement is active)
  const motionMatch = detectDynamicMotionSign(handsLandmarks, now);
  if (motionMatch) {
    lastMotionMatch = motionMatch;
    lastMotionTimestamp = now;
    return motionMatch;
  }

  // 1b. Motion Turnaround Grace Window:
  // If user was actively executing a dynamic sign in the last 380ms, maintain it across momentary turning points
  if (lastMotionMatch && now - lastMotionTimestamp < MOTION_PERSISTENCE_MS) {
    return lastMotionMatch;
  }

  const handsCount = handsLandmarks.length;
  const h1 = getFingerStates(handsLandmarks[0]);

  // ---------------------------------------------------------------------------
  // A. TWO-HAND VOCABULARY (42 Points)
  // ---------------------------------------------------------------------------
  if (handsCount >= 2) {
    const h2 = getFingerStates(handsLandmarks[1]);

    const wristDist = dist2D(h1.wrist, h2.wrist);
    const indexTipsDist = dist2D(h1.indexTip, h2.indexTip);
    const thumbTipsDist = dist2D(h1.thumbTip, h2.thumbTip);
    const middleTipsDist = dist2D(h1.middleTip, h2.middleTip);
    const pinkyTipsDist = dist2D(h1.pinkyTip, h2.pinkyTip);

    const allH1Up = h1.indexExt && h1.middleExt && h1.ringExt;
    const allH2Up = h2.indexExt && h2.middleExt && h2.ringExt;

    // 1. HEART (42 pts)
    if (
      h1.thumbExt &&
      h2.thumbExt &&
      thumbTipsDist < 0.08 &&
      indexTipsDist < 0.08 &&
      !h1.pinkyExt &&
      !h2.pinkyExt &&
      wristDist < 0.34
    ) {
      return buildVocabMatch("HEART", 0.98);
    }

    // 2. STOP (42 pts - Crossed Wrists X: wrists must overlap in X center)
    if (wristDist < 0.13 && indexTipsDist > 0.22 && Math.abs(h1.wrist.x - h2.wrist.x) < 0.08) {
      return buildVocabMatch("STOP", 0.96);
    }

    // 3. THANK YOU / NAMASTE (42 pts - Both palms touching upright prayer)
    if (
      allH1Up &&
      allH2Up &&
      wristDist < 0.28 &&
      indexTipsDist < 0.14 &&
      middleTipsDist < 0.14 &&
      thumbTipsDist < 0.16
    ) {
      return buildVocabMatch("THANK_YOU", 0.98);
    }

    // 4. FRIEND (42 pts - Index fingers hooked interlinked touching)
    if (
      indexTipsDist < 0.13 &&
      h1.indexExt &&
      h2.indexExt &&
      !h1.middleExt &&
      !h2.middleExt &&
      !h1.pinkyExt &&
      !h2.pinkyExt &&
      wristDist < 0.32
    ) {
      return buildVocabMatch("FRIEND", 0.95);
    }

    // 5. WE (42 pts - Dual index fingers pointing forward/together side-by-side)
    if (
      h1.indexExt &&
      h2.indexExt &&
      !h1.middleExt &&
      !h2.middleExt &&
      !h1.pinkyExt &&
      !h2.pinkyExt &&
      indexTipsDist >= 0.13 &&
      indexTipsDist < 0.30 &&
      wristDist < 0.36
    ) {
      return buildVocabMatch("WE", 0.96);
    }

    // 6. HELP (42 pts - One fist resting on flat palm of other hand)
    const h1FlatH2Fist = allH1Up && !h2.indexExt && !h2.middleExt;
    const h2FlatH1Fist = allH2Up && !h1.indexExt && !h1.middleExt;
    if ((h1FlatH2Fist || h2FlatH1Fist) && wristDist < 0.28) {
      return buildVocabMatch("HELP", 0.95);
    }

    // 7. MORE (42 pts - Both hands pinched, tips touching)
    if (
      !h1.indexExt &&
      !h2.indexExt &&
      indexTipsDist < 0.09 &&
      thumbTipsDist < 0.09 &&
      !h1.ringExt &&
      !h2.ringExt &&
      wristDist < 0.25
    ) {
      return buildVocabMatch("MORE", 0.94);
    }

    // 8. TOGETHER (42 pts - Both fists touching)
    if (
      wristDist < 0.25 &&
      !h1.indexExt &&
      !h1.middleExt &&
      !h2.indexExt &&
      !h2.middleExt
    ) {
      return buildVocabMatch("TOGETHER", 0.94);
    }

    // 9. DONE (42 pts - Dual thumbs up)
    if (h1.thumbExt && h1.thumbUp && !h1.indexExt && !h1.middleExt && h2.thumbExt && h2.thumbUp && !h2.indexExt && !h2.middleExt) {
      return buildVocabMatch("DONE", 0.98);
    }

    // 10. FOOD (42 pts - Both open palms held upward side-by-side like plate/book)
    if (
      pinkyTipsDist < 0.16 &&
      wristDist < 0.26 &&
      allH1Up &&
      allH2Up &&
      h1.indexTip.y > h1.wrist.y - 0.2
    ) {
      return buildVocabMatch("FOOD", 0.94);
    }

    // 11. PLEASE (42 pts - Both open palms cupped together gently)
    if (allH1Up && allH2Up && wristDist < 0.25 && indexTipsDist < 0.20) {
      return buildVocabMatch("PLEASE", 0.93);
    }

    // 12. WELCOME (42 pts - Both open palms spread wide welcoming)
    if (allH1Up && allH2Up && h1.pinkyExt && h2.pinkyExt && wristDist > 0.32) {
      return buildVocabMatch("WELCOME", 0.95);
    }

    // 13. FAMILY (42 pts - Both hands forming F-shape touching)
    const h1F = dist2D(h1.thumbTip, h1.indexTip) < 0.08 && h1.middleExt && h1.ringExt;
    const h2F = dist2D(h2.thumbTip, h2.indexTip) < 0.08 && h2.middleExt && h2.ringExt;
    if (h1F && h2F && wristDist < 0.28) {
      return buildVocabMatch("FAMILY", 0.95);
    }

    // 14. PAIN Static fallback (42 pts - Both index fingers pointing inward)
    if (h1.indexExt && !h1.middleExt && h2.indexExt && !h2.middleExt && indexTipsDist < 0.18) {
      return buildVocabMatch("PAIN", 0.94);
    }

    // 15. MEDICINE Static fallback
    if (allH2Up && !h1.indexExt && h1.middleExt && dist2D(h1.middleTip, h2.wrist) < 0.16) {
      return buildVocabMatch("MEDICINE", 0.94);
    }

    // 16. WAIT Static fallback (Both hands open facing chest)
    if (allH1Up && allH2Up && h1.pinkyExt && h2.pinkyExt && wristDist < 0.30) {
      return buildVocabMatch("WAIT", 0.93);
    }
  }

  // ---------------------------------------------------------------------------
  // B. SINGLE-HAND VOCABULARY (21 Points)
  // ---------------------------------------------------------------------------
  if (handsCount === 1) {
    // 17. I / ME (Index finger pointing at chest OR pinky upright 'I' handshape)
    const isIndexPointing = h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt && !h1.thumbExt;
    const isPinkyI = h1.pinkyExt && !h1.indexExt && !h1.middleExt && !h1.ringExt;
    if (isIndexPointing || isPinkyI) {
      return buildVocabMatch("I", 0.97);
    }

    // 18. LIKE (Thumb & Middle finger pinching together in 8-handshape at chest)
    if (dist2D(h1.thumbTip, h1.middleTip) < 0.065 && h1.indexExt && h1.pinkyExt) {
      return buildVocabMatch("LIKE", 0.97);
    }

    // 19. I LOVE YOU (ASL ILY - Thumb + Index + Pinky)
    if (h1.thumbExt && h1.indexExt && !h1.middleExt && !h1.ringExt && h1.pinkyExt) {
      return buildVocabMatch("LOVE", 0.97);
    }

    // 20. GOOD (OK Sign - Thumb and Index circle)
    const okDist = dist2D(h1.thumbTip, h1.indexTip);
    if (okDist < 0.085 && h1.middleExt && h1.ringExt && h1.pinkyExt) {
      return buildVocabMatch("GOOD", 0.96);
    }

    // 21. WATER (W sign - Index + Middle + Ring upright)
    if (h1.indexExt && h1.middleExt && h1.ringExt && !h1.pinkyExt) {
      return buildVocabMatch("WATER", 0.95);
    }

    // 22. PEACE (V sign - Index + Middle upright and spread)
    if (h1.indexExt && h1.middleExt && !h1.ringExt && !h1.pinkyExt && dist2D(h1.indexTip, h1.middleTip) > 0.035) {
      return buildVocabMatch("PEACE", 0.96);
    }

    // 23. HOSPITAL (H-handshape: Index + Middle straight out together)
    if (h1.indexExt && h1.middleExt && !h1.ringExt && !h1.pinkyExt && dist2D(h1.indexTip, h1.middleTip) <= 0.035) {
      return buildVocabMatch("HOSPITAL", 0.94);
    }

    // 24. YES (Thumb Up, above index knuckle)
    if (
      h1.thumbExt &&
      h1.thumbTip.y < h1.indexMcp.y - 0.035 &&
      !h1.indexExt &&
      !h1.middleExt &&
      !h1.ringExt &&
      !h1.pinkyExt
    ) {
      return buildVocabMatch("YES", 0.97);
    }

    // 25. NO (Thumb Down, below thumb MCP)
    if (
      h1.thumbExt &&
      h1.thumbTip.y > h1.thumbMcp.y + 0.04 &&
      !h1.indexExt &&
      !h1.middleExt &&
      !h1.ringExt &&
      !h1.pinkyExt
    ) {
      return buildVocabMatch("NO", 0.95);
    }

    // 26. HELLO Static Pose (Open Palm Upright)
    if (h1.indexExt && h1.middleExt && h1.ringExt && h1.pinkyExt && h1.fingersUp) {
      return buildVocabMatch("HELLO", 0.94);
    }

    // 27. BAD (Hand turned downward)
    if (h1.fingersDown && !h1.thumbExt) {
      return buildVocabMatch("BAD", 0.93);
    }

    // 28. MONEY Static Pinch
    if (dist2D(h1.thumbTip, h1.indexTip) < 0.05 && dist2D(h1.thumbTip, h1.middleTip) < 0.06 && !h1.ringExt) {
      return buildVocabMatch("MONEY", 0.93);
    }

    // 30. WHERE Static Open Palm
    if (h1.indexExt && h1.middleExt && h1.indexTip.y > h1.wrist.y - 0.25) {
      return buildVocabMatch("WHERE", 0.92);
    }

    // 31. SORRY (Closed Fist over chest)
    if (!h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt && !h1.thumbUp) {
      return buildVocabMatch("SORRY", 0.92);
    }
  }

  return null;
}

// =============================================================================
// 5. EXTRACT & FLATTEN DUAL-HAND LANDMARKS (126 FLOATS = 42 POINTS × 3)
// =============================================================================
function extractDualHandCoordinates(handsLandmarks) {
  // Returns 126 floats (42 points total)
  const coords = new Array(126).fill(0.0);

  if (!handsLandmarks || handsLandmarks.length === 0) {
    return coords;
  }

  // Hand 1 (indices 0..62)
  const h1 = handsLandmarks[0];
  for (let i = 0; i < h1.length && i < 21; i++) {
    const pt = h1[i];
    coords[i * 3] = Number(pt.x.toFixed(6));
    coords[i * 3 + 1] = Number(pt.y.toFixed(6));
    coords[i * 3 + 2] = Number(pt.z.toFixed(6));
  }

  // Hand 2 (indices 63..125)
  if (handsLandmarks.length > 1) {
    const h2 = handsLandmarks[1];
    for (let i = 0; i < h2.length && i < 21; i++) {
      const pt = h2[i];
      coords[63 + i * 3] = Number(pt.x.toFixed(6));
      coords[63 + i * 3 + 1] = Number(pt.y.toFixed(6));
      coords[63 + i * 3 + 2] = Number(pt.z.toFixed(6));
    }
  }

  return coords;
}

function updatePayloadMonitor(coords126, handsCount) {
  const pointsCount = handsCount * 21;
  payloadStatus.textContent = `${pointsCount}/42 pts (${coords126.length} floats)`;

  const sample = coords126
    .slice(0, 8)
    .map((v) => (v >= 0 ? ` ${v.toFixed(3)}` : v.toFixed(3)))
    .join(", ");
  payloadPreview.textContent = `[ ${sample}, ... +118 more (Hands: ${handsCount}) ]`;
}

// =============================================================================
// 6. SKELETAL RENDERING WITH DISTINCT DUAL-HAND VISUALS
// =============================================================================
function renderDualHandSkeletons(handsLandmarks) {
  if (!showLandmarks || !drawingUtils || !HandLandmarkerClass || !handsLandmarks) return;

  const w = canvas.width;
  const h = canvas.height;

  handsLandmarks.forEach((hand, idx) => {
    // Hand 1: Warm Tangerine (#F26522) & White (#FFFFFF); Hand 2: Light Gray (#EBEBEB) & Orange (#F26522)
    const isHand1 = idx === 0;
    const connectorColor = isHand1 ? "#F26522" : "#EBEBEB";
    const landmarkColor = isHand1 ? "#FFFFFF" : "#F26522";

    drawingUtils.drawConnectors(hand, HandLandmarkerClass.HAND_CONNECTIONS, {
      color: connectorColor,
      lineWidth: 3.5,
    });
    drawingUtils.drawLandmarks(hand, {
      color: landmarkColor,
      fillColor: "#FFFFFF",
      lineWidth: 1.5,
      radius: 4,
    });

    // Draw Wrist Floating Badge Label directly on canvas
    if (hand.length > 0) {
      const wrist = hand[0];
      const wx = wrist.x * w;
      const wy = wrist.y * h;

      canvasCtx.save();
      canvasCtx.font = "bold 11px 'JetBrains Mono', monospace";
      const labelText = isHand1 ? "HAND 1: TANGERINE (21 pts)" : "HAND 2: WHITE (21 pts)";
      const metrics = canvasCtx.measureText(labelText);
      const padding = 5;

      canvasCtx.fillStyle = "rgba(26, 26, 26, 0.9)";
      canvasCtx.strokeStyle = connectorColor;
      canvasCtx.lineWidth = 1;
      canvasCtx.beginPath();
      canvasCtx.roundRect(wx - 6, wy - 22, metrics.width + padding * 2, 18, 4);
      canvasCtx.fill();
      canvasCtx.stroke();

      canvasCtx.fillStyle = landmarkColor;
      canvasCtx.fillText(labelText, wx - 6 + padding, wy - 9);
      canvasCtx.restore();
    }
  });
}

// =============================================================================
// 7. REAL-TIME LANDMARK & GESTURE PROCESSING LOOP
// =============================================================================
function processLiveVideoFeed() {
  if (!webcamRunning || !handLandmarker) {
    requestAnimationFrame(processLiveVideoFeed);
    return;
  }

  const now = performance.now();

  // Rendering FPS calculation
  frameCount++;
  if (now - lastFpsUpdateTime >= 1000) {
    currentFps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
    fpsCounter.textContent = `${currentFps}`;
    frameCount = 0;
    lastFpsUpdateTime = now;
  }

  // Detect landmarks when new video frame is ready
  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;

    const results = handLandmarker.detectForVideo(video, now);

    // Clear canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = results.landmarks || [];
    const handsCount = hands.length;
    const pointsCount = Math.min(handsCount, 2) * 21;

    // Update Telemetry indicators
    ptsCounter.textContent = `${pointsCount}/42`;

    if (handsCount >= 2) {
      reticleBadge.textContent = "2 Hands Detected (42 pts)";
      reticleBadge.className = "reticle-badge dual";
      hand1Indicator.className = "hand-chip active-cyan";
      hand2Indicator.className = "hand-chip active-amber";
    } else if (handsCount === 1) {
      reticleBadge.textContent = "1 Hand Detected (21 pts)";
      reticleBadge.className = "reticle-badge detected";
      hand1Indicator.className = "hand-chip active-cyan";
      hand2Indicator.className = "hand-chip offline";
    } else {
      reticleBadge.textContent = "Awaiting Hands";
      reticleBadge.className = "reticle-badge searching";
      hand1Indicator.className = "hand-chip offline";
      hand2Indicator.className = "hand-chip offline";
    }

    if (handsCount > 0) {
      // 0. Hands detected - cancel any active silence countdown
      cancelSilenceTimer();

      // 1. Draw dual hand skeleton
      renderDualHandSkeletons(hands);

      // 2. Flatten coordinates into 126 floats (42 points x 3)
      const coords126 = extractDualHandCoordinates(hands);
      updatePayloadMonitor(coords126, handsCount);

      // 3. Real-Time Gesture Calculation
      const detectedGesture = calculateRealTimeGesture(hands);
      handleGestureCalculation(detectedGesture, handsCount, now);

      // 4. Concurrently transmit coordinates to local Express backend for ML telemetry
      if (now - lastPredictTime >= PREDICT_INTERVAL_MS && !isRequestInFlight) {
        lastPredictTime = now;
        sendLandmarksToBackend(coords126);
      }
    } else {
      // No hands present
      resetGestureStaging();
      predictionDisplay.textContent = "—";
      predictionDisplay.classList.add("empty");
      predictionSubtext.textContent = draftSentence
        ? `Draft sentence: "${draftSentence}" (Rest hands 2s to send)`
        : "Show one or both hands (42 points)";
      confidencePercent.textContent = "0.0%";
      confidenceBar.style.width = "0%";
      payloadStatus.textContent = "Idle (0/42 pts)";
      payloadPreview.textContent = "[ awaiting hand detection (0/42 points)... ]";

      // 0 hands detected: Trigger 2.0s continuous silence timer if a draft sentence exists
      handleSilenceDetection();
    }
  }

  requestAnimationFrame(processLiveVideoFeed);
}

// =============================================================================
// 8. REAL-TIME GESTURE CALCULATION & SENTENCE STAGING LOGIC
// =============================================================================
let lastValidGestureTime = 0;

function handleGestureCalculation(gestureObj, handsCount, now) {
  if (!gestureObj) {
    // Tracking tolerance: if we had an active staging gesture and tracking flickered for < 280ms, retain staging
    if (stagingGestureKey && now - lastValidGestureTime < 280) {
      return;
    }
    resetGestureStaging();
    predictionDisplay.textContent = "…";
    predictionDisplay.classList.remove("empty");
    predictionSubtext.textContent = handsCount >= 2 ? "Calculating dual-hand vocabulary..." : "Calculating single-hand vocabulary...";
    return;
  }

  lastValidGestureTime = now;
  activeGestureObj = gestureObj;
  highlightVocabChip(gestureObj.key);

  const isMotion = !!gestureObj.isMotion;
  // Dynamic motion gestures commit at 650ms (since 2-3 motion cycles already span ~400-600ms); static poses commit at 1100ms
  const targetHoldMs = isMotion ? 650 : AUTO_SEND_HOLD_MS;

  // Update Hero Card
  predictionDisplay.textContent = `${gestureObj.emoji} ${gestureObj.label}`;
  predictionDisplay.classList.remove("empty");
  const motionTag = isMotion
    ? (currentLanguage === "ta" ? "⚡ இயங்கு சைகை" : "⚡ Dynamic Motion Sign")
    : (currentLanguage === "ta" ? "நிலையான சைகை" : "Static Sign Pose");
  const ptsUnit = currentLanguage === "ta" ? "புள்ளிகள்" : "pts";
  predictionSubtext.textContent = `${gestureObj.label} • ${motionTag} • ${gestureObj.tip || gestureObj.fullText} (${gestureObj.pointsCount} ${ptsUnit})`;

  const confPercent = Math.round(gestureObj.confidence * 100);
  confidencePercent.textContent = `${confPercent}%`;
  confidenceBar.style.width = `${confPercent}%`;
  confidenceLabel.textContent = isMotion
    ? (currentLanguage === "ta" ? `இயக்க துல்லியம் (${gestureObj.pointsCount} புள்ளிகள்)` : `Kinematic Motion Accuracy (${gestureObj.pointsCount} pts)`)
    : (gestureObj.handsCount === 2
        ? (currentLanguage === "ta" ? "இரு கை துல்லியம் (42 புள்ளிகள்)" : "Dual-Hand Accuracy (42 pts)")
        : (currentLanguage === "ta" ? "ஒரு கை துல்லியம் (21 புள்ளிகள்)" : "Single-Hand Accuracy (21 pts)"));

  // Real-Time Staging & Auto-Send Buffer
  gestureStagingBar.classList.add("active");
  gestureStagingBar.classList.toggle("motion-active", isMotion);
  stagingGestureName.textContent = `${gestureObj.emoji} ${gestureObj.label}`;
  stagingConfBadge.textContent = isMotion ? `${confPercent}% ⚡` : `${confPercent}%`;

  const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;

  if (stagingGestureKey !== gestureObj.key) {
    // New vocabulary word recognized, start hold timer
    stagingGestureKey = gestureObj.key;
    stagingStartTime = now;
    hasCommittedCurrentHold = false;
    stagingProgressFill.style.width = "0%";
    stagingStatusText.textContent = isAutoSendEnabled
      ? (isMotion ? t.holdMotionActive : t.holdSteady)
      : t.holdReady;
  } else {
    // Same vocabulary word held or motion continued
    const elapsed = now - stagingStartTime;
    const progress = Math.min(100, Math.round((elapsed / targetHoldMs) * 100));
    stagingProgressFill.style.width = `${progress}%`;

    const remainingSec = Math.max(0, (targetHoldMs - elapsed) / 1000).toFixed(1);

    if (hasCommittedCurrentHold) {
      // Word already appended for this continuous hold; prevent duplicate additions
      stagingStatusText.textContent = currentLanguage === "ta"
        ? `✓ "${gestureObj.label}" வரைவில் உள்ளது. அனுப்ப கைகளை இறக்கவும்.`
        : `✓ "${gestureObj.label}" in draft. Lower hands to send.`;
      stagingProgressFill.style.width = "100%";
    } else {
      stagingStatusText.textContent = isAutoSendEnabled
        ? (isMotion
            ? (currentLanguage === "ta" ? `⚡ இயக்கம்: ${remainingSec} வினாடியில் சேரும்` : `⚡ Motion Active: ${remainingSec}s to add`)
            : (currentLanguage === "ta" ? `பிடிக்கவும்: ${remainingSec} வினாடி` : `Hold: ${remainingSec}s to add`))
        : (currentLanguage === "ta" ? "அனுப்பு 🚀 அல்லது Enter அழுத்தவும்" : "Press Send 🚀 or Enter");

      // Auto-Append word trigger when held for targetHoldMs
      if (isAutoSendEnabled && elapsed >= targetHoldMs) {
        appendWordToSentence(gestureObj);
        hasCommittedCurrentHold = true;
        lastCommittedGestureKey = gestureObj.key;
        lastCommitTimestamp = now;
        stagingProgressFill.style.width = "100%";
      }
    }
  }
}

function resetGestureStaging() {
  stagingGestureKey = "";
  stagingStartTime = 0;
  hasCommittedCurrentHold = false;
  lastValidGestureTime = 0;
  lastMotionMatch = null;
  lastMotionTimestamp = 0;
  stagingProgressFill.style.width = "0%";
  const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
  stagingStatusText.textContent = draftSentence
    ? (currentLanguage === "ta" ? `வரைவு: "${draftSentence}" (அனுப்ப 2 வினாடி கைகளை எடுக்கவும்)` : `Draft: "${draftSentence}" (Rest hands 2s to send)`)
    : t.holdToCommit;
  gestureStagingBar.classList.remove("active");
  gestureStagingBar.classList.remove("motion-active");
  stagingGestureName.textContent = draftSentence ? `Draft: "${draftSentence}"` : t.waitingForSign;
  stagingConfBadge.textContent = "--%";
  highlightVocabChip(null);
}

// =============================================================================
// 9. CONTINUOUS SENTENCE CONSTRUCTION & CHAT ENGINE
// =============================================================================

/**
 * 1. Sentence Building:
 * Appends the recognized gesture word to the running draftSentence.
 * Displays it live in the text input box.
 * Prevents duplicate words from being added back-to-back (e.g. holding 'Help' for 3 seconds).
 */
function appendWordToSentence(gestureObj) {
  const word = currentLanguage === "ta"
    ? (gestureObj.wordTa || gestureObj.labelTa || gestureObj.label)
    : (gestureObj.label || gestureObj.key);
  if (!word) return;

  // Sync draftSentence with manual input if user typed something first
  if (chatInput.value.trim() && !draftSentence) {
    draftSentence = chatInput.value.trim();
  }

  // Prevent duplicate words from being added back-to-back
  const currentTokens = draftSentence.trim().split(/\s+/).filter(Boolean);
  const lastToken = currentTokens.length > 0 ? currentTokens[currentTokens.length - 1] : "";

  if (lastToken.toLowerCase() === word.toLowerCase() || lastAppendedWord.toLowerCase() === word.toLowerCase()) {
    console.log(`[*] Duplicate back-to-back word prevented: "${word}"`);
    stagingStatusText.textContent = currentLanguage === "ta" ? `"${word}" ஏற்கனவே வாக்கியத்தில் உள்ளது` : `"${word}" already in sentence`;
    return;
  }

  // Cancel any active silence countdown when a new word is actively added
  cancelSilenceTimer();

  // Append word with proper spacing
  if (draftSentence.trim().length > 0) {
    draftSentence = `${draftSentence.trim()} ${word}`;
  } else {
    draftSentence = word;
  }
  lastAppendedWord = word;

  // Display live in the text input box
  chatInput.value = draftSentence;

  stagingStatusText.textContent = currentLanguage === "ta" ? `✓ சேர்க்கப்பட்டது "${word}"` : `✓ Added "${word}"`;
  console.log(`[*] Sentence updated: "${draftSentence}"`);
}

/**
 * 2. The Silence Timer:
 * Starts a 2.0s countdown when zero hands are detected in the frame.
 * If zero hands persist for 2.0 continuous seconds, triggers finalizeSentence().
 */
function handleSilenceDetection() {
  const currentSentence = (chatInput.value || draftSentence || "").trim();
  if (!currentSentence) {
    cancelSilenceTimer();
    return;
  }

  if (!silenceTimerId) {
    silenceStartTime = performance.now();
    reticleBadge.textContent = "Silence: 2.0s to Send";
    reticleBadge.className = "reticle-badge standby";

    silenceTimerId = setTimeout(async () => {
      silenceTimerId = null;
      await finalizeSentence();
    }, SILENCE_TIMEOUT_MS);
  } else {
    const elapsed = performance.now() - silenceStartTime;
    const remaining = Math.max(0, (SILENCE_TIMEOUT_MS - elapsed) / 1000).toFixed(1);
    reticleBadge.textContent = `Silence: ${remaining}s to Send`;
  }
}

function cancelSilenceTimer() {
  if (silenceTimerId) {
    clearTimeout(silenceTimerId);
    silenceTimerId = null;
  }
}

/**
 * 4. Bonus: LLM Smoothing Engine (Placeholder)
 * Asynchronously passes the raw keyword string (e.g. "I need help" or "HELP PLEASE WATER")
 * to a lightweight LLM endpoint (e.g. OpenAI, Anthropic, Gemini, Ollama, or local Express API)
 * to convert it into natural conversational English before it is spoken & displayed.
 *
 * @param {string} rawSentence - Raw concatenated sign language words.
 * @returns {Promise<string>} - Polished, natural conversational sentence.
 */
async function smoothSentenceWithLLM(rawSentence) {
  if (!rawSentence || !rawSentence.trim()) return rawSentence;

  // Toggle this flag to true once your backend LLM route or cloud endpoint is connected
  const ENABLE_LLM_SMOOTHING = false;
  const LLM_API_ENDPOINT = "http://localhost:3000/api/smooth-sentence";

  if (!ENABLE_LLM_SMOOTHING) {
    // Graceful fallback to raw sentence when LLM endpoint is inactive
    return rawSentence;
  }

  try {
    const response = await fetch(LLM_API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: "Convert the following American Sign Language keyword stream into natural, conversational English:",
        sentence: rawSentence,
      }),
    });

    if (!response.ok) {
      throw new Error(`LLM endpoint returned HTTP ${response.status}`);
    }

    const data = await response.json();
    return data.smoothedSentence || data.text || rawSentence;
  } catch (error) {
    console.warn("[!] LLM Smoothing unavailable, falling back to raw sentence:", error.message);
    return rawSentence;
  }
}

/**
 * 3. Execution (finalizeSentence):
 * Takes the complete draftSentence, passes it through LLM smoothing,
 * pushes it to the UI as a single chat bubble, triggers voice readout via speakText(),
 * and clears the buffer for the next sentence.
 */
async function finalizeSentence() {
  cancelSilenceTimer();

  // Sync with any manual typing edits the user might have made in chatInput
  const rawSentence = (chatInput.value || draftSentence || "").trim();
  if (!rawSentence) {
    draftSentence = "";
    lastAppendedWord = "";
    chatInput.value = "";
    return;
  }

  console.log(`[*] Finalizing sentence: "${rawSentence}"`);

  // Reset staging UI & indicators
  resetGestureStaging();
  reticleBadge.textContent = "Sentence Dispatched";
  reticleBadge.className = "reticle-badge detected";

  // Optional LLM Smoothing: convert raw keywords into natural conversational English
  let finalSentence = rawSentence;
  try {
    finalSentence = await smoothSentenceWithLLM(rawSentence);
  } catch (err) {
    console.warn("[!] Error during LLM smoothing, using raw sentence:", err);
    finalSentence = rawSentence;
  }

  const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
  const wordCount = finalSentence.split(/\s+/).filter(Boolean).length;

  const senderLabel = currentLanguage === "ta" ? "நீங்கள் (சைகையாளர்)" : "You (Signer)";
  const sentenceMeta = currentLanguage === "ta"
    ? `வாக்கியம் (${wordCount} வார்த்தைகள்) • ${timestamp}`
    : `Sentence (${wordCount} words) • ${timestamp}`;

  // 1. Push to UI as a single chat bubble
  addChatMessage({
    sender: senderLabel,
    text: finalSentence,
    meta: sentenceMeta,
    isUser: true,
  });

  // 2. Trigger voice readout
  if (isTtsEnabled) {
    speakText(finalSentence);
  }

  // 3. Clear the buffer for the next sentence
  draftSentence = "";
  lastAppendedWord = "";
  hasCommittedCurrentHold = false;
  chatInput.value = "";
  stagingStatusText.textContent = currentLanguage === "ta"
    ? "வாக்கியம் அனுப்பப்பட்டது. சைகைக்காக காத்திருக்கிறது..."
    : "Sentence sent. Waiting for sign...";

  // 4. Automated contextual Assistant Response
  setTimeout(() => {
    const respTime = new Date().toLocaleTimeString("en-US", { hour12: false });
    const replyText = generateSentenceReply(finalSentence);
    const assistantSender = currentLanguage === "ta" ? "🤖 கவுண்டர் #04 மொழிபெயர்ப்பாளர்" : "🤖 Counter #04 Interpreter";
    const assistantMeta = currentLanguage === "ta" ? `பொது சேவை உதவியாளர் • ${respTime}` : `Civic Terminal Assistant • ${respTime}`;

    addChatMessage({
      sender: assistantSender,
      text: replyText,
      meta: assistantMeta,
      isUser: false,
    });

    if (isTtsEnabled) {
      setTimeout(() => speakText(replyText), 300);
    }
  }, 450);
}

function generateSentenceReply(sentence) {
  if (currentLanguage === "ta") {
    const lower = sentence.toLowerCase();
    if (lower.includes("தண்ணீர்") || lower.includes("உணவு")) {
      return "உங்களுக்கு குடிநீர் அல்லது உணவு தேவை என்பதை புரிந்து கொண்டோம். பணியாளர் ஒருவர் உதவிக்கு வருகிறார்.";
    }
    if (lower.includes("உதவி")) {
      return "உதவி எச்சரிக்கை பெறப்பட்டது. கவுண்டர் #04 அதிகாரிக்கு உங்கள் கோரிக்கை தெரிவிக்கப்பட்டுள்ளது.";
    }
    if (lower.includes("நன்றி") || lower.includes("தயவுசெய்து")) {
      return "மிக்க மகிழ்ச்சி! உங்களுக்கு மேலும் ஏதேனும் உதவி தேவைப்பட்டால் தெரிவிக்கவும்.";
    }
    if (lower.includes("வணக்கம்") || lower.includes("நல்வரவு")) {
      return "வணக்கம்! கவுண்டர் #04 குடிமக்கள் சேவை மையத்திற்கு நல்வரவு. உங்களுக்கு எவ்வாறு உதவ முடியும்?";
    }
    if (lower.includes("மன்னிக்கவும்")) {
      return "பரவாயில்லை! பொறுமையாக தொடரலாம், நாங்கள் உங்களுக்கு உதவ தயாராக உள்ளோம்.";
    }
    if (lower.includes("பிடிக்கும்")) {
      return "எங்கள் சேவை உங்களுக்கு பிடித்ததில் மிக்க மகிழ்ச்சி! உங்களுக்கு சேவை செய்வதில் பெருமை கொள்கிறோம்.";
    }
    if (lower.includes("நாம்") || lower.includes("நாங்கள்")) {
      return "உங்கள் குழுவிற்கு மனமார்ந்த நல்வரவு! கவுண்டர் #04-ல் உங்களுக்கு உதவ நாங்கள் தயாராக உள்ளோம்.";
    }
    if (lower.includes("நான்") || lower.includes("என்னை")) {
      return `உங்கள் தனிப்பட்ட கோரிக்கை பெறப்பட்டது: "${sentence}". உடனடியாக கவனிக்கிறோம்.`;
    }
    return `பெறப்பட்டது: "${sentence}". கவுண்டர் #04-ல் உங்கள் கோரிக்கை செயல்படுத்தப்படுகிறது.`;
  }

  const upper = sentence.toUpperCase();
  if (upper.includes("WATER") || upper.includes("FOOD")) {
    return "I understand you need refreshments or assistance with food/water. An attendant is on the way.";
  }
  if (upper.includes("HELP")) {
    return "Help is on the way. An officer at Counter #04 has been notified of your request.";
  }
  if (upper.includes("THANK") || upper.includes("PLEASE")) {
    return "You're very welcome! Please let us know if there is anything else we can assist you with.";
  }
  if (upper.includes("HELLO") || upper.includes("WELCOME")) {
    return "Hello and welcome to Counter #04 Citizen Services! How can I assist you today?";
  }
  if (upper.includes("SORRY")) {
    return "No worries at all! Take your time, we are here to assist you.";
  }
  if (upper.includes("LIKE")) {
    return "We are thrilled to know you like this! We are glad to serve you.";
  }
  if (upper.includes("WE")) {
    return "Greetings to your group! How can we assist your party today?";
  }
  if (upper.includes("I ") || upper.startsWith("I") || upper.includes(" ME")) {
    return `Received your personal request: "${sentence}". Assisting you right now.`;
  }
  return `Received: "${sentence}". Processing your request at Counter #04.`;
}

// Keep commitGestureToChat for backward compatibility if directly invoked
function commitGestureToChat(gestureObj) {
  appendWordToSentence(gestureObj);
}

function addChatMessage({ sender, text, meta, isUser }) {
  const msgEl = document.createElement("div");
  msgEl.className = `chat-msg ${isUser ? "user" : "assistant"}`;

  const speakBtnLabel = currentLanguage === "ta" ? "🔊 பேசு" : "🔊 Speak";
  const copyBtnLabel = currentLanguage === "ta" ? "📋 நகலெடு" : "📋 Copy";

  msgEl.innerHTML = `
    <div class="chat-msg-header">
      <span class="sender-name">${sender}</span>
      <span class="msg-meta">${meta}</span>
    </div>
    <div class="chat-msg-body">
      ${text}
      <div class="chat-bubble-actions">
        <button class="chat-mini-btn btn-speak-bubble" title="Read message aloud">${speakBtnLabel}</button>
        <button class="chat-mini-btn btn-copy-bubble" title="Copy message text">${copyBtnLabel}</button>
      </div>
    </div>
  `;

  // Attach bubble actions
  const speakBtn = msgEl.querySelector(".btn-speak-bubble");
  speakBtn.addEventListener("click", () => speakText(text));

  const copyBtn = msgEl.querySelector(".btn-copy-bubble");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = currentLanguage === "ta" ? "நகலெடுக்கப்பட்டது!" : "Copied!";
      setTimeout(() => (copyBtn.textContent = copyBtnLabel), 1200);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  });

  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function speakText(text) {
  if (!window.speechSynthesis) return;

  // Clean emoji characters from speech readout for natural voice output
  const cleanText = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "").trim();
  if (!cleanText) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  if (currentLanguage === "ta") {
    utterance.lang = "ta-IN";
    const voices = window.speechSynthesis.getVoices();
    const tamilVoice = voices.find(
      (v) =>
        v.lang === "ta-IN" ||
        v.lang === "ta_IN" ||
        (v.lang && v.lang.toLowerCase().startsWith("ta")) ||
        (v.name && v.name.toLowerCase().includes("tamil"))
    );
    if (tamilVoice) {
      utterance.voice = tamilVoice;
    }
  } else {
    utterance.lang = "en-US";
    const voices = window.speechSynthesis.getVoices();
    const enVoice = voices.find((v) => v.lang === "en-US" || v.lang === "en-GB");
    if (enVoice) {
      utterance.voice = enVoice;
    }
  }

  window.speechSynthesis.speak(utterance);
}

// =============================================================================
// 10. BACKEND COMMUNICATION (FETCH API -> localhost:3000/predict)
// =============================================================================
async function sendLandmarksToBackend(coords126) {
  isRequestInFlight = true;
  payloadStatus.textContent = "Transmitting (126 floats)...";
  const startTime = performance.now();

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        landmarks: coords126,
        features: coords126,
      }),
    });

    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    setBackendOnline(true);
  } catch (error) {
    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;
    setBackendOnline(false);
  } finally {
    isRequestInFlight = false;
  }
}

function setBackendOnline(isOnline) {
  if (isOnline !== backendOnline) {
    backendOnline = isOnline;
    const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
    if (isOnline) {
      backendBadge.className = "kiosk-status-badge online";
      backendStatusText.textContent = t.backendConnected;
    } else {
      backendBadge.className = "kiosk-status-badge standby";
      backendStatusText.textContent = t.backendOffline;
    }
  }
}

// =============================================================================
// 11. EVENT LISTENERS & CHATBOX CONTROLS
// =============================================================================
function initKioskEventListeners() {
  // 0. Language Settings Buttons (Header & Chatbox)
  if (btnLangEn) {
    btnLangEn.addEventListener("click", () => setLanguage("en"));
  }
  if (btnLangTa) {
    btnLangTa.addEventListener("click", () => setLanguage("ta"));
  }
  if (btnToggleLang) {
    btnToggleLang.addEventListener("click", () => {
      setLanguage(currentLanguage === "en" ? "ta" : "en");
    });
  }

  // 1. Camera Toggle
  btnToggleCam.addEventListener("click", () => {
    if (webcamRunning) {
      stopWebcam();
    } else {
      loaderOverlay.classList.remove("hidden");
      loaderStatus.textContent = currentLanguage === "ta" ? "கேமரா இயக்கப்படுகிறது..." : "Activating Webcam...";
      startWebcam();
    }
  });

  // 2. Mirror View Toggle
  btnToggleMirror.addEventListener("click", () => {
    isMirrored = !isMirrored;
    video.classList.toggle("mirrored", isMirrored);
    canvas.classList.toggle("mirrored", isMirrored);
    btnToggleMirror.classList.toggle("active", isMirrored);
  });

  // 3. Landmarks Overlay Toggle
  btnToggleLandmarks.addEventListener("click", () => {
    showLandmarks = !showLandmarks;
    btnToggleLandmarks.classList.toggle("active", showLandmarks);
    if (!showLandmarks) {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
  });

  // 3b. Float Picture-in-Picture Toggle
  if (btnTogglePip && viewportPanel) {
    btnTogglePip.addEventListener("click", () => {
      const isPip = viewportPanel.classList.toggle("pip-mode");
      btnTogglePip.classList.toggle("active", isPip);
      const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
      if (pipBtnText) {
        pipBtnText.textContent = isPip ? t.dockPip : t.floatPip;
      }
    });
  }

  // 4. TTS Readout Toggle
  btnToggleTts.addEventListener("click", () => {
    isTtsEnabled = !isTtsEnabled;
    btnToggleTts.classList.toggle("active", isTtsEnabled);
    const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
    if (btnToggleTtsText) {
      btnToggleTtsText.textContent = isTtsEnabled ? t.ttsOn : t.ttsOff;
    }
  });

  // 5. Auto-Send Toggle
  btnToggleAutosend.addEventListener("click", () => {
    isAutoSendEnabled = !isAutoSendEnabled;
    btnToggleAutosend.classList.toggle("active", isAutoSendEnabled);
    const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
    if (btnToggleAutosendText) {
      btnToggleAutosendText.textContent = isAutoSendEnabled ? t.autoSend : (currentLanguage === "ta" ? "⏸️ கைமுறை அனுப்பு" : "⏸️ Manual Send");
    }
  });

  // 6. Clear Chat
  btnClearChat.addEventListener("click", () => {
    cancelSilenceTimer();
    draftSentence = "";
    lastAppendedWord = "";
    hasCommittedCurrentHold = false;
    chatInput.value = "";
    const t = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
    chatMessages.innerHTML = `
      <div class="chat-msg assistant">
        <div class="chat-msg-header">
          <span class="sender-name">${t.senderAssistant}</span>
          <span class="msg-meta">${t.chatCleared}</span>
        </div>
        <div class="chat-msg-body">
          ${t.chatClearedBody}
        </div>
      </div>
    `;
    resetGestureStaging();
  });

  // 7. Export Chat Transcript
  btnExportChat.addEventListener("click", () => {
    const messages = chatMessages.querySelectorAll(".chat-msg");
    if (!messages || messages.length === 0) return;

    let transcript = "=== SIGN LANGUAGE BRIDGE - ACCESSIBILITY TRANSCRIPT ===\n";
    transcript += `Generated: ${new Date().toLocaleString()}\n`;
    transcript += "Counter #04 • Public Hub (42 Points Dual Hand Tracking)\n";
    transcript += `Language: ${currentLanguage === "ta" ? "தமிழ் (Tamil)" : "English"}\n`;
    transcript += "=".repeat(60) + "\n\n";

    messages.forEach((m) => {
      const sender = m.querySelector(".sender-name")?.textContent || "Speaker";
      const meta = m.querySelector(".msg-meta")?.textContent || "";
      const body = m.querySelector(".chat-msg-body")?.childNodes[0]?.textContent?.trim() || "";
      transcript += `[${meta}] ${sender}:\n${body}\n\n`;
    });

    const blob = new Blob([transcript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sign_language_chat_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // 8. Send Button (Immediately dispatches draft sentence)
  btnSendChat.addEventListener("click", () => {
    sendCurrentInputOrActiveGesture();
  });

  // 9. Input Keyboard Enter
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendCurrentInputOrActiveGesture();
    }
  });

  // 9b. Live sync of manual edits in chatInput with draftSentence
  chatInput.addEventListener("input", () => {
    draftSentence = chatInput.value;
    const tokens = draftSentence.trim().split(/\s+/).filter(Boolean);
    lastAppendedWord = tokens.length > 0 ? tokens[tokens.length - 1] : "";
    cancelSilenceTimer();
  });

  // 10. Space button
  btnSpace.addEventListener("click", () => {
    chatInput.value += " ";
    draftSentence = chatInput.value;
    chatInput.focus();
  });

  // 11. Backspace button
  btnBackspace.addEventListener("click", () => {
    chatInput.value = chatInput.value.slice(0, -1);
    draftSentence = chatInput.value;
    const tokens = draftSentence.trim().split(/\s+/).filter(Boolean);
    lastAppendedWord = tokens.length > 0 ? tokens[tokens.length - 1] : "";
    chatInput.focus();
  });

  // 12. Speak latest message
  btnSpeakTranscript.addEventListener("click", () => {
    const lastMsg = chatMessages.querySelector(".chat-msg:last-child .chat-msg-body");
    if (lastMsg) {
      speakText(lastMsg.textContent);
    }
  });
}

function sendCurrentInputOrActiveGesture() {
  const text = (chatInput.value || draftSentence || "").trim();

  if (text) {
    draftSentence = text;
    finalizeSentence();
  } else if (activeGestureObj) {
    appendWordToSentence(activeGestureObj);
  }
}

// =============================================================================
// 12. APPLICATION BOOTSTRAP
// =============================================================================
async function bootstrap() {
  console.log("[*] Initializing Dual-Hand (42 Points) Sign Language Terminal...");
  startKioskClock();
  initKioskEventListeners();
  setLanguage(currentLanguage);

  const visionReady = await initMediaPipeVision();
  if (visionReady) {
    const cameraReady = await startWebcam();
    if (cameraReady) {
      processLiveVideoFeed();
    }
  }
}

window.addEventListener("DOMContentLoaded", bootstrap);
