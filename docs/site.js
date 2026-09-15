/* Play only the lightweight highlight videos currently in view. */
(() => {
  const videos = [...document.querySelectorAll('video[data-autoplay]')];
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  if (motion.matches || navigator.connection?.saveData || !('IntersectionObserver' in window)) return;
  const inView = new Set();
  const observer = new IntersectionObserver(entries => {
    for (const {target: video, isIntersecting} of entries) {
      if (isIntersecting) {
        inView.add(video);
        if (!document.hidden) video.play().catch(() => {});
      } else {
        inView.delete(video);
        video.pause();
      }
    }
  }, {threshold: 0.35});
  videos.forEach(video => observer.observe(video));
  document.addEventListener('visibilitychange', () => {
    for (const video of videos) {
      if (document.hidden) video.pause();
      else if (inView.has(video)) video.play().catch(() => {});
    }
  });
})();
