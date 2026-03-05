// ============================================================
// 🔧 Service Worker — Cho phép PWA chạy offline
// ============================================================
// Service Worker là gì?
//   - Là script chạy nền trong browser (không trên tab)
//   - Có thể cache files → app chạy offline
//   - Có thể nhận push notification
//   - Chạy ngay cả khi đóng tab
//
// TẠI SAO CẦN?
//   Khi cài PWA trên iPhone, user mở lên cần load nhanh.
//   Service Worker cache HTML/CSS/JS → mở app tức thì.
//   Chỉ gọi API detect khi có mạng.
// ============================================================

const CACHE_NAME = 'fire-detection-v1';
const STATIC_ASSETS = [
    '/',
    '/static/index.html',
    '/static/manifest.json',
];

// --- Install: cache static assets ---
self.addEventListener('install', (event) => {
    console.log('[SW] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    // Kích hoạt ngay, không chờ SW cũ
    self.skipWaiting();
});

// --- Activate: xóa cache cũ ---
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...');
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            );
        })
    );
    self.clients.claim();
});

// --- Fetch: serve từ cache hoặc network ---
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // API requests → luôn fetch từ network (cần data mới nhất)
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(fetch(event.request));
        return;
    }

    // Static assets → cache first, fallback to network
    event.respondWith(
        caches.match(event.request).then((cached) => {
            return cached || fetch(event.request).then((response) => {
                // Cache response mới
                const responseClone = response.clone();
                caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, responseClone);
                });
                return response;
            });
        })
    );
});
