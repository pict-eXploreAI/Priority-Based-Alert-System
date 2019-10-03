const staticCacheName = 'site-static';
const assets = [
    '/static/',
    '/templates/index.html',
    '/static/assets/js/app.js',
    '/static/assets/js/argon.js',,
    '/static/assets/js/argon.min.js',
    '/static/assets/css/argon.js',
    '/static/assets/css/argon.min.js',
    '/static/assets/css/bootstrap/bootstrap-grid.css',
    '/static/assets/css/bootstrap/bootstrap-grid.min.css',
    '/static/assets/css/bootstrap/bootstrap-reboot.css',
    '/static/assets/css/bootstrap/bootstrap-reboot.min.css',
    '/static/assets/css/bootstrap/bootstrap.css',
    '/static/assets/css/bootstrap/bootstrap.min.css',

];

//Install
self.addEventListener('install',evt => {
    //console.log('service worker has been installed');
    evt.waitUntil(
    caches.open(staticCacheName).then(cache => {
        console.log('Caching');
        cache.addAll(assets);
    })
    );
});

//activation

self.addEventListener('activate',evt=>{
    console.log("service worker has been activated");
});

//fetch
self.addEventListener('fetch',evt => {
    //console.log('fetch event',evt);
    evt.respondWith(
        caches.match(evt.request).then(cacheRes =>{
            return cacheRes || fetch(evt.request);
        })
    )
});

