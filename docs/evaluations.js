/* Static archive browser. The manifest contains source IDs, not outcome labels. */
(() => {
  const data = window.CHICGRASP_EPISODES;
  const grid = document.getElementById('episode-grid');
  const count = document.getElementById('episode-count');
  const method = grid.dataset.method || 'Diffusion Policy';
  const assetBase = grid.dataset.assetBase || 'assets/evaluations/';
  if (!data) { count.textContent = 'The episode index could not load.'; return; }
  const dialog = document.getElementById('episode-dialog');
  const player = document.getElementById('episode-player');
  const previous = document.getElementById('previous-episode');
  const next = document.getElementById('next-episode');
  let current = 0;
  function openEpisode(index) {
    current = index;
    const episode = data.episodes[index];
    const label = String(episode.episode).padStart(3, '0');
    document.getElementById('episode-title').textContent = `${method} · EP ${label}`;
    document.getElementById('episode-info').textContent = `Wrist camera (0) · ${episode.source_seconds.toFixed(1)} s recorded · 4× playback · Grasp phase`;
    player.src = assetBase + episode.clip;
    player.poster = assetBase + episode.poster;
    document.getElementById('episode-download').href = player.src;
    previous.disabled = index === 0;
    next.disabled = index === data.episodes.length - 1;
    if (!dialog.open) dialog.showModal();
    player.play().catch(() => {});
  }
  data.episodes.forEach((episode, index) => {
    const label = String(episode.episode).padStart(3, '0');
    const button = document.createElement('button');
    button.className = 'episode-tile'; button.type = 'button'; button.dataset.episode = label;
    button.setAttribute('aria-label', `Play ${method} episode ${label}`);
    const image = document.createElement('img');
    image.src = assetBase + episode.poster; image.alt = ''; image.loading = 'lazy';
    image.width = 640; image.height = 360;
    const caption = document.createElement('span'); caption.textContent = label;
    button.append(image, caption); button.addEventListener('click', () => openEpisode(index));
    grid.append(button);
  });
  document.getElementById('episode-filter').addEventListener('input', event => {
    const query = event.target.value.trim().replace(/^ep\s*/i, ''); let shown = 0;
    grid.querySelectorAll('button').forEach(button => {
      button.hidden = !button.dataset.episode.includes(query);
      if (!button.hidden) shown++;
    });
    count.textContent = `${shown} ${shown === 1 ? 'episode' : 'episodes'}`;
  });
  document.getElementById('close-episode').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { player.pause(); player.removeAttribute('src'); player.load(); });
  previous.addEventListener('click', () => { if (current > 0) openEpisode(current - 1); });
  next.addEventListener('click', () => { if (current < data.episodes.length - 1) openEpisode(current + 1); });
  dialog.addEventListener('keydown', event => {
    if (event.target === player) return;
    if (event.key === 'ArrowLeft' && current > 0) { event.preventDefault(); openEpisode(current - 1); }
    if (event.key === 'ArrowRight' && current < data.episodes.length - 1) { event.preventDefault(); openEpisode(current + 1); }
  });
})();
