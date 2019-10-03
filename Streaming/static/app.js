if('serviceWorker' in navigator){
    navigator.serviceWorker.register('/static/sw.js')
    .then((reg)=>console.log('service worker registered',reg))
    .catch((err)=>console.log('service not registered',err))

}