# Editing the ChicGrasp portfolio

Live project page: <https://AmirrezaDavar.github.io/ChicGrasp/>

GitHub Pages publishes the `docs/` directory on `main`. Changes to these static files appear after GitHub's Pages build completes. There is no package installation or build step.

## Page files

- `index.html`: project introduction, highlights, overview placeholder, evaluation grid, published results, hardware, contribution, and ongoing work.
- `evaluations.html`, `evaluations.js`: all 142 available DP recordings, with a searchable poster grid and an individual player.
- `evaluations-ibc.html`, `evaluations-lstm-gmm.html`: separate three-episode baseline grids and browsers, using the same `evaluations.js`.
- `policy.html`, `app.js`: recorded numerical action-generation explorer and robot/jaw commands.
- `action-videos.html`: the 23 individual action-overlay examples.
- `evidence.html`: source coverage, model replay, projection assumptions, and distinctions between archive recordings and published outcomes.
- `style.css`, `site.js`: shared styling and viewport-aware highlight playback.
- `assets/`: figures, videos, manifests, and exported numerical data.

## Add the finalized 80-second overview

1. Add the final MP4 under `docs/assets/media/`, for example `chicgrasp_overview_final.mp4`.
2. Add its poster image and captions, if available.
3. In `index.html`, replace only the `div` with `id="overview-slot"` with the final player below. Replace the adjacent sentence about finalization as well.
4. Update the video link in the root README.

```html
<video id="overview-slot" class="film" controls playsinline preload="metadata"
       poster="assets/figures/overview_final.jpg">
  <source src="assets/media/chicgrasp_overview_final.mp4" type="video/mp4">
  <track kind="captions" src="assets/media/overview_final.vtt"
         srclang="en" label="English">
</video>
```

Only include the poster and caption attributes when those files have been added. The placeholder intentionally has no video source; previous 80-second edits remain in the external archive.

## Add recordings or change the grid

`presentation/build_evaluation_grid.py` creates the wrist-camera (camera 0) individual 4× previews, 4K/1080p mosaics, posters, and source manifest from the supplied archive. Its current inventory explicitly expects episodes 000–141. Extend that inventory and its validation together when adding recordings.

```bash
python presentation/build_evaluation_grid.py \
  --data-root /path/to/ChicGrasp-data \
  --output /path/to/generated/evaluation_grid_wrist --camera 0
```

Copy the generated MP4s, JPGs, `episodes/`, `episodes.js`, and `manifest.json` into `docs/assets/evaluations/`. Intermediate row videos in `work/` stay outside the website. See the source manifest for timing and episode placement.

The current recordings stop before lift and scripted rehang. Add full-task recordings and a verified trial mapping before presenting a gallery as all labeled pick-and-lift successes or as the paper's complete evaluation set. Existing published result tables remain sourced to the paper.

## Preview before publishing

```bash
python -m http.server 8000 --directory docs
# Open http://localhost:8000
```

Check desktop and mobile layout, video playback, episode search, dialog navigation, and the action explorer. Preserve the published hardware / ongoing simulation distinction when adding material. Use the Matplotlib scripts in `presentation/` to update research plots, and include corresponding source CSVs.

## Baseline evaluation grids

The IBC and LSTM-GMM sections each show all three available wrist-camera recordings. Generate these in a new output folder:

```bash
python presentation/build_baseline_grids.py \
  --data-root /path/to/ChicGrasp-data \
  --output /path/to/generated/baseline_evaluation_grids
```

Copy the generated `ibc/` and `lstm-gmm/` folders into `docs/assets/evaluations/`. Each includes a full-resolution grid, a web grid, posters, individual episodes, and a manifest with source hashes and timings. The two grids use common starts, 4× speed, and held endings. Add more recordings only after updating the layout and archive-coverage text together.
