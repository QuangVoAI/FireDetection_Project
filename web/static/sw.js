// ============================================================
// 🔧 Service Worker — PWA Phát hiện Cháy Sớm
// ============================================================
//
// Service Worker là gì?
//   - Script chạy nền trong browser (không trên tab)
//   - Cache files → app chạy offline
//   - Quản lý network requests
//   - Chạy ngay cả khi đóng tab
//
// CHIẾN LƯỢC CACHING:
//   1. Static assets (HTML, CSS, JS, icons) → Cache First
//      Ưu tiên cache → nhanh. Fallback network → cập nhật cache.
//   2. API requests (/api/*) → Network Only
//      Luôn gọi server (cần data real-time). Nếu offline → trả lỗi.
//   3. Google Fonts → Stale While Revalidate
//      Serve từ cache + fetch cập nhật ẩn.
//   4. Offline fallback → Hiển thị trang "Không có mạng"
// ============================================================

const CACHE_VERSION = 'v3';
const CACHE_STATIC = `fire-detect-static-${CACHE_VERSION}`;
const CACHE_DYNAMIC = `fire-detect-dynamic-${CACHE_VERSION}`;

// Danh sách assets cần pre-cache khi install
const PRECACHE_ASSETS = [
    '/',
    '/static/css/app.css',
    '/static/js/app.js',
    '/static/manifest.json',
    '/static/icons/favicon.png',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/static/icons/apple-touch-icon.png',
];

// ============================================================
// INSTALL — Pre-cache static assets
// ============================================================
self.addEventListener('install', (event) => {
    console.log('[SW] Installing...', CACHE_VERSION);

    event.waitUntil(
        caches.open(CACHE_STATIC)
            .then((cache) => {
                console.log('[SW] Pre-caching static assets');
                return cache.addAll(PRECACHE_ASSETS);
            })
            .then(() => {
                // Kích hoạt ngay, không chờ SW cũ
                return self.skipWaiting();
            })
    );
});

// ============================================================
// ACTIVATE — Xóa cache cũ
// ============================================================
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...', CACHE_VERSION);

    event.waitUntil(
        caches.keys()
            .then((keys) => {
                return Promise.all(
                    keys
                        .filter((key) => key !== CACHE_STATIC && key !== CACHE_DYNAMIC)
                        .map((key) => {
                            console.log('[SW] Deleting old cache:', key);
                            return caches.delete(key);
                        })
                );
            })
            .then(() => {
                // Claim tất cả tabs → SW mới kiểm soát ngay
                return self.clients.claim();
            })
    );
});

// ============================================================
// FETCH — Chiến lược caching cho từng loại request
// ============================================================
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // ─── 1. API requests → Network Only ───
    // Detection API cần data real-time, không cache
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request)
                .catch(() => {
                    // Offline → trả JSON error
                    return new Response(
                        JSON.stringify({
                            error: 'Không có kết nối mạng',
                            offline: true
                        }),
                        {
                            status: 503,
                            headers: { 'Content-Type': 'application/json' }
                        }
                    );
                })
        );
        return;
    }

    // ─── 2. Google Fonts → Stale While Revalidate ───
    if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
        event.respondWith(
            caches.open(CACHE_DYNAMIC).then((cache) => {
                return cache.match(event.request).then((cached) => {
                    const fetchPromise = fetch(event.request).then((response) => {
                        cache.put(event.request, response.clone());
                        return response;
                    }).catch(() => cached); // Offline → dùng cache

                    return cached || fetchPromise;
                });
            })
        );
        return;
    }

    // ─── 3. Static assets → Cache First, Network Fallback ───
    event.respondWith(
        caches.match(event.request)
            .then((cached) => {
                if (cached) {
                    // Có trong cache → trả ngay
                    // Đồng thời fetch mới để cập nhật cache (background)
                    fetch(event.request)
                        .then((response) => {
                            if (response && response.status === 200) {
                                caches.open(CACHE_STATIC).then((cache) => {
                                    cache.put(event.request, response);
                                });
                            }
                        })
                        .catch(() => { }); // Ignore network errors for background update

                    return cached;
                }

                // Không có cache → fetch từ network
                return fetch(event.request)
                    .then((response) => {
                        // Cache response mới cho lần sau
                        if (response && response.status === 200) {
                            const responseClone = response.clone();
                            caches.open(CACHE_DYNAMIC).then((cache) => {
                                cache.put(event.request, responseClone);
                            });
                        }
                        return response;
                    })
                    .catch(() => {
                        // Offline + không có cache
                        // Nếu request HTML → hiển thị offline page
                        if (event.request.headers.get('accept')?.includes('text/html')) {
                            return generateOfflinePage();
                        }
                        return new Response('Offline', { status: 503 });
                    });
            })
    );
});

// ============================================================
// Offline Page — Trang hiển thị khi mất mạng
// ============================================================
function generateOfflinePage() {
    const html = `
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Không có mạng — FireDetect</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0a0a1a;
            color: #e8e8f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px;
            text-align: center;
        }
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse at 30% 50%, rgba(255,68,68,0.05) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 30%, rgba(255,140,0,0.04) 0%, transparent 60%);
            pointer-events: none;
        }
        .container { position: relative; z-index: 1; max-width: 360px; }
        .icon {
            font-size: 72px;
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, #ff4444, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }
        p {
            color: #9090b0;
            font-size: 15px;
            line-height: 1.6;
            margin-bottom: 28px;
        }
        .retry-btn {
            padding: 14px 40px;
            background: linear-gradient(135deg, #ff4444, #ff8c00);
            color: white;
            border: none;
            border-radius: 28px;
            font-size: 16px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 20px rgba(255,68,68,0.3);
        }
        .retry-btn:active { transform: scale(0.96); }
        .status {
            margin-top: 32px;
            padding: 10px 18px;
            background: rgba(255,68,68,0.08);
            border: 1px solid rgba(255,68,68,0.15);
            border-radius: 10px;
            font-size: 13px;
            color: #ff6666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">📡</div>
        <h1>Không có kết nối mạng</h1>
        <p>Ứng dụng cần kết nối internet để phát hiện lửa/khói. Hãy kiểm tra kết nối Wi-Fi hoặc dữ liệu di động của bạn.</p>
        <button class="retry-btn" onclick="location.reload()">🔄 Thử lại</button>
        <div class="status">⚡ Camera và detection cần kết nối đến server</div>
    </div>
</body>
</html>`;

    return new Response(html, {
        status: 503,
        headers: { 'Content-Type': 'text/html; charset=utf-8' }
    });
}

// ============================================================
// Background Sync — (Tùy chọn) Gửi lại detection khi có mạng
// ============================================================
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }

    // Clear cache khi user yêu cầu
    if (event.data && event.data.type === 'CLEAR_CACHE') {
        caches.keys().then((keys) => {
            return Promise.all(keys.map((key) => caches.delete(key)));
        }).then(() => {
            console.log('[SW] All caches cleared');
        });
    }
});
