// Applied in <head> before first paint so the stored theme never flashes.
// message.js adopts the value and handles toggling/persistence.
(function () {
  try {
    var t = localStorage.getItem('ui.theme');
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  } catch (e) { /* best effort */ }
})();
