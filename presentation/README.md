# Rebuild the ChicGrasp presentation

This folder rebuilds the publication tables, plots, actual offline model traces, CAD view, and 84-second film. Raw source archives remain external. No script here sends robot commands.

## Outputs

- `docs/index.html`: responsive project page and interactive denoising explorer; works over HTTP or directly from disk.
- `docs/evidence.html`: sources, discrepancies, and interpretation.
- `docs/assets/figures/`: SVG, PDF, PNG, CAD still, and film contact sheet.
- `docs/assets/media/chicgrasp_84s.mp4`: 1920×1080, 24 FPS, 84 seconds, no audio.
- `docs/assets/media/diffusion_action_generation.mp4`: eight-second actual denoising replay.
- `docs/assets/media/gripper_cad_orbit.mp4`: eight-second orbit of the composed CAD assembly.
- `docs/assets/data/`: source inventory, CSVs, captured trace, provenance, and validation.

## 1. Recreate tables and training curves

The paper transcription is `published_results.csv`. It includes the per-chicken nominal/disturbed counts, with all three methods represented. The paper's inconsistent disturbed subtotal is documented in the evidence page; it is not used as a replacement for its rows.

```bash
python presentation/analyze_sources.py --data-root /path/to/ChicGrasp-data
```

Dependencies: Python 3.11, NumPy, PyYAML, Zarr 2.x. ZIP files are read directly; no bulk extraction is needed. Training logs are deduplicated by global step before selecting the final loss record in each epoch. Different loss objectives are plotted separately.

## 2. Capture actual diffusion iterates

Use the existing `robodiff` environment compatible with this repository (the completed replay used Python 3.9, torch 1.12.1, diffusers 0.11.1, and an RTX 4080):

```bash
python presentation/replay_denoising.py \
  --data-root /path/to/ChicGrasp-data \
  --episode 14 --times 5 12 20 --steps 100 --device cuda:0
```

The replay reads the roughly 5.9 GB uncompressed checkpoint into memory instead of extracting it. Allow approximately 15 GB of available system memory, plus GPU memory for the policy. It loads EMA state with strict key matching and wraps the scheduler to capture each real denoising result. It never instantiates the robot environment.

The policy inputs use recorded video at elapsed episode time and the corresponding state rows. Exact camera timestamps are unavailable, so alignment is approximate. These are newly sampled plans, not the original deployment outputs. The checkpoint retains eight action values; the display emphasizes XYZ and two jaws.

## 3. Export local CAD

Use a Python environment with `pxr` (OpenUSD), then point to the existing assembly:

```bash
python presentation/export_cad.py /path/to/gripper_customized_1.usd
```

The export reads instance proxies, visible mesh triangles, and composed transforms. It leaves the original CAD unchanged. Camera motion and display colors are handled by the renderer. `cad_provenance.json` records the source and referenced layer hashes.

## 4. Build figures and render the film

```bash
python presentation/build_figures.py
python presentation/render_media.py \
  --data-root /path/to/ChicGrasp-data \
  --cinematic /path/to/chicken_workbench_showcase.mp4 \
  --stills-only
# Inspect docs/assets/figures/film_contact_sheet.jpg, then:
python presentation/render_media.py \
  --data-root /path/to/ChicGrasp-data \
  --cinematic /path/to/chicken_workbench_showcase.mp4
```

Dependencies: Matplotlib, NumPy, Pillow, OpenCV, FFmpeg and ffprobe. The renderer streams RGB frames to FFmpeg, avoiding a large frame sequence on disk. No generated illustration substitutes for hardware evidence. Film timings are exported in `edit_decision_list.json`. Captions and the optional narration script are authored in the repository and should be revised if the edit changes.

## 5. Validate and preview

```bash
python presentation/validate.py
python -m http.server 8000 --directory docs
```

Visit `http://localhost:8000`. `validate.py` checks local asset links, table arithmetic, finite trace values and dimensions, correspondence of final scheduler output to policy output, jaw transitions, video dimensions/durations, and full MP4 decoding. Browser interaction and responsive layout are checked separately.

## Film structure

| Time | Content |
| --- | --- |
| 0–8 s | Real hardware and the published headline result |
| 8–16 s | Camera orbit around the local CAD assembly |
| 16–24 s | Three observation views and the reported dataset |
| 24–42 s | Three actual model-denoising replays |
| 42–52 s | Recorded independent jaw commands; 0.5× playback |
| 52–65 s | Published benchmark and scope |
| 65–75 s | Separate ongoing Isaac Lab cinematic |
| 75–84 s | Technical ownership, repository, and paper |

## Publication notes

The HTML is static and ready to serve from `docs/`. A Pages URL should only replace local README links once the site is actually enabled and verified. The repository contains selected compact media, not full checkpoints or raw datasets. The existing `.gitignore` is narrowly overridden for curated `docs/assets/` exports.

The README corrects the old placeholder citation, removes references to absent download scripts, and documents the actual archive layout. It preserves training/controller source and the existing Onshape and Box resource links.
