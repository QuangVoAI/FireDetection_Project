// ============================================================
// 🔧 Aura Sentinel — App Logic
// ============================================================
// Apple Liquid Glass PWA for Fire Detection
// ============================================================

// ============================================================
// 🔧 Aura Sentinel — Safe App Logic for Verbatim UI
// ============================================================

// ─── State ───
let selectedFile = null;
let cameraStream = null;
let facingMode = 'environment';
let autoDetectInterval = null;
let isAutoDetecting = false;

// ─── DOM Elements (Safe) ───
const video = document.getElementById('camera-video');
const canvas = document.getElementById('camera-canvas');
const overlay = document.getElementById('camera-overlay');
const liveBadge = document.getElementById('live-badge');
const fpsBadge = document.getElementById('fps-badge');
const cameraAlert = document.getElementById('camera-alert');
const sahiGrid = document.getElementById('sahi-grid');
const scanLine = document.querySelector('.scan-line');
const detectionOverlay = document.getElementById('detection-overlay');

// ============================================================
// 1. Tab Navigation
// ============================================================
const navBtns = document.querySelectorAll('.nav-btn');
if (navBtns.length > 0) {
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.target;
            if (!tabName) return;

            document.querySelectorAll('.tab-content').forEach(s => s.classList.add('hidden'));
            const target = document.getElementById('tab-' + tabName);
            if (target) {
                target.classList.remove('hidden');
                target.classList.add('active');
            }

            const mapBg = document.getElementById('map-background');
            if(mapBg) mapBg.style.display = (tabName === 'sentinel') ? 'none' : 'block';
            
            if (tabName === 'me' && typeof loadRecordings === 'function') loadRecordings();
        });
    });
    const def = document.querySelectorAll('.nav-btn[data-target="sentinel"]');
    if(def.length > 0) def[0].click();
}

// ============================================================
// 2. PWA — Service Worker
// ============================================================
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then((registration) => {
            console.log('✅ Service Worker registered');
        })
        .catch(err => console.log('SW error:', err));
}

// ============================================================
// Health / Offline Checks
// ============================================================
const offlineBanner = document.getElementById('offline-banner');
function updateOnlineStatus() {
    if(!offlineBanner) return;
    if (navigator.onLine) {
        offlineBanner.style.display = 'none';
        offlineBanner.className = 'alert-banner';
    } else {
        offlineBanner.style.display = 'block';
        offlineBanner.className = 'alert-banner fire';
    }
}
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();

async function checkHealth() {
    const statusDot = document.getElementById('status-dot');
    const statusLabel = document.getElementById('status-label');
    if(!statusDot || !statusLabel) return;
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
    } catch (e) { }
}
setInterval(checkHealth, 15000);

// ============================================================
// 3. Camera API
// ============================================================
async function startCamera() {
    if(!video) return;
    try {
        const constraints = { video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false };
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        video.style.display = 'block';
        if(overlay) overlay.classList.add('hidden');
        if(liveBadge) liveBadge.classList.add('active');
        if(fpsBadge) fpsBadge.style.display = 'block';
        if(sahiGrid) sahiGrid.style.opacity = '0.3';
        if(scanLine) scanLine.style.display = 'block';

        // Hide static background image if it exists
        const bgImgs = document.querySelectorAll('img.w-full.h-full.object-cover');
        bgImgs.forEach(img => {
            if(img.alt && img.alt.includes('Dark misty')) {
                img.style.display = 'none';
            }
        });
    } catch (err) {
        console.error('Camera error:', err);
    }
}

// ============================================================
// Capture & Detect
// ============================================================
async function captureAndDetect() {
    if (!cameraStream || !video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    const confidence = 0.35; // Default

    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');

    try {
        const res = await fetch(`/api/detect?confidence=${confidence}`, {
            method: 'POST', body: formData
        });

        if (!res.ok) throw new Error('Server error');
        const data = await res.json();
        drawBoxes(data.detections);
    } catch (err) {
        console.error('Detect error:', err);
    }
}

// ============================================================
// Draw Bounding Boxes matching UI mockup style
// ============================================================
function drawBoxes(detections) {
    if (!detectionOverlay || !video) return;
    detectionOverlay.innerHTML = '';
    
    // Prevent division by zero if video hasn't loaded dimensions
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    detections.forEach(d => {
        // Support both backend shapes:
        // - { xmin, ymin, xmax, ymax, class_name, confidence }
        // - { bbox: [x1,y1,x2,y2], class_name, confidence }
        const class_name = d.class_name;
        const confidence = d.confidence;

        const xmin = (typeof d.xmin === 'number') ? d.xmin : (Array.isArray(d.bbox) ? d.bbox[0] : 0);
        const ymin = (typeof d.ymin === 'number') ? d.ymin : (Array.isArray(d.bbox) ? d.bbox[1] : 0);
        const xmax = (typeof d.xmax === 'number') ? d.xmax : (Array.isArray(d.bbox) ? d.bbox[2] : 0);
        const ymax = (typeof d.ymax === 'number') ? d.ymax : (Array.isArray(d.bbox) ? d.bbox[3] : 0);

        const width = xmax - xmin;
        const height = ymax - ymin;
        
        const topPct = (ymin / vh) * 100;
        const leftPct = (xmin / vw) * 100;
        const widthPct = (width / vw) * 100;
        const heightPct = (height / vh) * 100;

        const isFire = class_name === 'Fire';
        const clrBox = isFire ? 'border-apple-red/50 bg-apple-red/10 animate-pulse' : 'border-apple-blue/30 bg-apple-blue/5';
        const clrDot = isFire ? 'bg-apple-red' : 'bg-apple-blue shadow-[0_0_8px_rgba(0,122,255,0.6)]';
        const cornerColor = isFire ? 'border-neon-amber' : 'border-neon-cyan';
        
        const html = `
        <div class="absolute" style="top:${topPct}%; left:${leftPct}%; width:${widthPct}%; height:${heightPct}%;">
            <!-- Label -->
            <div class="absolute -top-7 left-0 flex items-center space-x-1.5 glass-panel px-2.5 py-1 rounded-lg border-white/20 whitespace-nowrap">
                <div class="w-1.5 h-1.5 rounded-full ${clrDot}"></div>
                <span class="text-[9px] font-bold text-white uppercase tracking-[0.1em]">${class_name}</span>
                <span class="text-[9px] font-medium text-white/50">${(confidence*100).toFixed(0)}%</span>
            </div>
            <!-- Box -->
            <div class="w-full h-full border-[1px] ${clrBox} rounded-xl">
                <div class="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 ${cornerColor}"></div>
                <div class="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 ${cornerColor}"></div>
                <div class="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 ${cornerColor}"></div>
                <div class="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 ${cornerColor}"></div>
            </div>
        </div>`;
        detectionOverlay.insertAdjacentHTML('beforeend', html);
    });
}

// ============================================================
// Panic & Initialization
// ============================================================
const panicBtn = document.getElementById('panic-btn');
if(panicBtn) {
    panicBtn.addEventListener('click', () => {
        if (navigator.vibrate) navigator.vibrate([300, 100, 300, 100, 500]);
        alert('🚨 PANIC! Tín hiệu khẩn cấp đã được gửi.');
    });
}

// Auto-start features on page load
document.addEventListener('DOMContentLoaded', () => {
    // If we are on the Sentinel page (has the video element)
    if (document.getElementById('camera-video')) {
        // Start WebRTC camera and auto-detect
        startCamera().then(() => {
            if(!isAutoDetecting) {
                isAutoDetecting = true;
                captureAndDetect();
                autoDetectInterval = setInterval(captureAndDetect, 2000);
            }
        });
    }

    // Toggle camera button (if there's a floating action button for it)
    const toggleCamBtn = document.getElementById('toggle-cam-btn');
    if (toggleCamBtn) {
        toggleCamBtn.addEventListener('click', switchCamera);
    }
    
    // Auto-init Real Map on Alleyways Page
    if (document.getElementById('alleyways-page')) {
        initLeafletMap();
    }
});

// ============================================================
// Real Map API (Leaflet.js integration)
// ============================================================
function initLeafletMap() {
    // Dynamically inject Leaflet CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    document.head.appendChild(link);

    // Dynamically inject Leaflet JS
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload = () => {
        // Wait a slight moment for layout to settle
        setTimeout(() => {
            // Tôn Đức Thắng University coordinates (hoặc mặc định Saigon)
            const mapContainer = document.getElementById('leaflet-map');
            if(!mapContainer) return;
            
            // 10.732, 106.699 là Toạ độ đại học Tôn Đức Thắng Quận 7
            const map = L.map('leaflet-map').setView([10.73266, 106.69976], 16);

            // Add standard OpenStreetMap tiles
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 19,
            }).addTo(map);

            // Hide UI zoom controls to keep the mobile mockup aesthetic clean
            map.zoomControl.remove();
            
            // Custom Fire Hydrant / Safe Zone Icons
            const safeIcon = L.divIcon({
                html: '<div style="width: 24px; height: 24px; background: rgba(52,199,89,0.8); border: 2px solid white; border-radius: 50%; box-shadow: 0 0 10px rgba(52,199,89,0.5);"></div>',
                className: '',
                iconSize: [24, 24]
            });
            
            const hydrantIcon = L.divIcon({
                html: '<div style="width: 24px; height: 24px; background: rgba(0,122,255,0.8); border: 2px solid white; border-radius: 50%; box-shadow: 0 0 10px rgba(0,122,255,0.5);"></div>',
                className: '',
                iconSize: [24, 24]
            });

            // Add some dummy markers corresponding to the Alleyways mockup UI
            L.marker([10.733, 106.700], {icon: safeIcon}).addTo(map).bindPopup('Alley 4B Community Hall');
            L.marker([10.731, 106.698], {icon: safeIcon}).addTo(map).bindPopup('East Gate Garden');
            L.marker([10.7325, 106.699], {icon: hydrantIcon}).addTo(map).bindPopup('Station Alpha');
            
            console.log('✅ Real map API (Leaflet) initialized');
        }, 100);
    };
    document.body.appendChild(script);
}
