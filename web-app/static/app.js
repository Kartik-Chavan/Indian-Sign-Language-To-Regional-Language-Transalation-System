// Configuration
const POLL_INTERVAL = 300; // 300ms between detection polls
const TRANSLATION_TIMEOUT = 5000; // 5 second timeout for translations

// DOM Elements
const videoFeed = document.getElementById('video-feed');
const signSequence = document.getElementById('sign-sequence');
const translationResult = document.getElementById('translation-result');
const translateBtn = document.getElementById('translate-btn');
const resetBtn = document.getElementById('reset-btn');
const languageSelect = document.getElementById('language');
const statusDisplay = document.getElementById('status-display');

// Application State
let detectedSigns = [];
let isTranslating = false;
let lastDetectionTime = 0;

// Initialize the application
function initApp() {
    setupEventListeners();
    startDetectionPolling();
    updateStatus("System ready - waiting for signs");
}

// Set up all event listeners
function setupEventListeners() {
    translateBtn.addEventListener('click', handleTranslation);
    resetBtn.addEventListener('click', handleReset);
}

// Start polling for detections
function startDetectionPolling() {
    const poll = async () => {
        try {
            const now = Date.now();
            if (now - lastDetectionTime >= POLL_INTERVAL) {
                await checkForDetections();
                lastDetectionTime = now;
            }
        } catch (error) {
            console.error("Polling error:", error);
            updateStatus("Detection error - check console");
        }
        setTimeout(poll, 100); // Fast poll interval with detection throttling
    };
    poll();
}

// Check for new detections from server
async function checkForDetections() {
    const response = await fetch('/process');
    const data = await response.json();
    
    if (data.status === "success") {
        detectedSigns = data.sequence || [];
        updateSignDisplay();
        
        if (detectedSigns.length > 0) {
            updateStatus(`Detected: ${detectedSigns.join(", ")}`);
        } else {
            updateStatus("Waiting for signs...");
        }
    }
}

// Update the sign display
function updateSignDisplay() {
    signSequence.textContent = detectedSigns.length > 0 
        ? detectedSigns.join(" → ") 
        : "No signs detected";
}

// Handle translation request
async function handleTranslation() {
    if (isTranslating || detectedSigns.length === 0) return;
    
    isTranslating = true;
    translateBtn.disabled = true;
    translationResult.textContent = "Translating...";
    updateStatus("Processing translation...");
    
    // Debug logs
    console.log("Selected language:", languageSelect.value);
    console.log("Sending signs:", detectedSigns);
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TRANSLATION_TIMEOUT);
        
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signs: detectedSigns,
                lang: languageSelect.value
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const data = await response.json();
        
        // Debug log for translation response
        console.log("Translation response:", data);
        
        if (data.status === "success") {
            translationResult.textContent = data.translation;
            updateStatus("Translation complete");
        } else {
            translationResult.textContent = "Translation error";
            updateStatus(`Error: ${data.error || "Unknown error"}`);
        }
    } catch (error) {
        console.error("Translation error:", error);
        translationResult.textContent = error.name === "AbortError" 
            ? "Translation timeout" 
            : "Connection error";
        updateStatus("Translation failed - check console");
    } finally {
        isTranslating = false;
        translateBtn.disabled = false;
    }
}

// Handle reset request
async function handleReset() {
    try {
        resetBtn.disabled = true;
        updateStatus("Resetting...");
        
        const response = await fetch('/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === "reset") {
            detectedSigns = [];
            updateSignDisplay();
            translationResult.textContent = "";
            updateStatus("System reset - ready for new signs");
            
            // Briefly show confirmation
            const originalText = resetBtn.textContent;
            resetBtn.textContent = "✓ Reset Done";
            setTimeout(() => {
                resetBtn.textContent = originalText;
                resetBtn.disabled = false;
            }, 2000);
        }
    } catch (error) {
        console.error("Reset error:", error);
        updateStatus("Reset failed - check console");
        resetBtn.disabled = false;
    }
}

// Update status display
function updateStatus(message) {
    statusDisplay.textContent = message;
    console.log(`Status: ${message}`);
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);