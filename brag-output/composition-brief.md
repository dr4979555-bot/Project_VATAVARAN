# Hyperframes Composition Brief: VATAWARAN

## Objective

Create a short cinematic launch-style brag video for VATAWARAN — an AI-driven extreme weather tracking engine built for Smart India Hackathon 2026.

## Output

- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 22 seconds

## Source Material

- Project root: `d:\VATAWARAN`
- Primary files read: `frontend/index.html`, `backend/main.py`, `backend/gov_services.py`, `README.md`
- Product name: VATAWARAN
- Tagline / strongest claim: "AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine"
- Key UI or visual moment to recreate: The newspaper-styled masthead with Playfair Display title, the synoptic map with 3 anomaly trajectories, the diffusion denoising counter, and the government telemetry ribbon with 5 agency pills
- Copy that must appear verbatim:
  - "VATAWARAN"
  - "METEOROLOGICAL INTELLIGENCE BULLETIN"
  - "5 km resolution. 10-day forecast. One engine."
  - "SIH 2026 · Problem Statement #26078"
  - "NCMRWF NEPS-G (12 km)" / "IMDAA Baseline" / "IMD Ground AWS" / "ISRO MOSDAC (INSAT-3DR)" / "NDMA Sachet (CAP v1.2)"
  - "Stage II · Diffusion Downscaling"
  - "BOB-CYCLONE-2026-01" / "NWI-HEATWAVE-2026-03" / "WHR-RAIN-2026-07"
  - "ITU-T X.1303 COMPLIANT"

## Creative Direction

- Tone preset: cinematic
- Creative direction: classified government intelligence brief, cold war weather bureau aesthetic
- Interpretation: Dramatic reveals with confidence. Big serif type. Double-rule borders. Slow builds that pay off. The pacing says "this is important." Authoritative and striking, not playful.
- Angle: A secret intelligence brief about the weather. Government seals. Classified dossiers. The weather is the enemy, and VATAWARAN tracks it.
- Hook: "VATAWARAN" letterpress-stamps in massive Playfair Display, "METEOROLOGICAL INTELLIGENCE BULLETIN" types out below, bureau-red classified stamp appears
- Outro / punchline: "5 km resolution. 10-day forecast. One engine." → ITU-T X.1303 COMPLIANT badge → SIH 2026 stamp
- Avoid:
  - Generic SaaS language ("streamline your workflow")
  - Abstract filler visuals
  - Any visual redesign away from the aged-paper newspaper aesthetic

## Visual Identity

- Background: #faf7f0 (aged paper with fractal noise SVG grain)
- Text: #1a1510 (ink-deep) for headlines, #2c2418 (ink-base) for body
- Accent: #8b1a1a (bureau red), #1a2744 (navy), #1e3d1e (forest green for status), #7a4f00 (amber)
- Rules/borders: #c0a878, #a08858 (double-rule newspaper borders)
- Display font: Playfair Display 900 (Google Fonts)
- Body font: EB Garamond (Google Fonts)
- Mono font: IBM Plex Mono (Google Fonts)
- Visual references from the project: newspaper masthead with ornament emblem, double-rule borders, anomaly dossier cards with EFI score bars, government telemetry ribbon with green-dot status pills, DDIM diffusion progress bar

## Storyboard

Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:

1. "The Masthead" — 3s — VATAWARAN title stamps in, subtitle types, classified stamp appears
2. "The Synoptic Chart" — 5s — India map with 3 anomaly trajectories drawing in, EFI scores
3. "The Diffusion Engine" — 5s — Step counter 00→50, progress bar fills, physics QA passes
4. "Government Feeds Online" — 4s — 5 agency pills light up one by one with status badges
5. "The Verdict" — 5s — "5 km resolution. 10-day forecast. One engine." → CAP badge → SIH stamp

## Audio

- Audio role: cinematic support bed — dramatic, building, institutional
- Audio arc: quiet impact opening → building through map reveals → peak at diffusion completion → confident stride at gov feeds → resolving final beat on verdict
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3
- Music treatment: Fade in from 0s at low volume, build through scenes 2-3, peak energy at scene 3 completion, confident stride in scene 4, resolve to final beat in scene 5, fade out at 22s
- Music cue guidance: Bundled preset at `<skill-dir>/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`. Tempo 109.96 BPM. Strong cues: target 8.74s for map trajectory reveal, 13.11s for diffusion completion, 22.93s for final stamp. Beat grid 6.00–13.00s window for sequential gov-pill reveals.
- Audio-reactive treatment: subtle; use music RMS/bass to gently pulse paper grain opacity and make anomaly track glow breathe — brand-specific, not generic waveform
- Audio-coupled moments:
  - Scene 1 title stamp — impact thud on VATAWARAN appearance
  - Scene 1 subtitle — subtle key ticks on typing animation
  - Scene 2 trajectory reveals — each track draws on a beat
  - Scene 3 counter ticks — mechanical click on step increments
  - Scene 4 gov pills — each pill appears on a beat (5 pills over ~3s)
  - Scene 5 taglines — each line on a strong beat; final stamp thud
- SFX selection guidance: Professional, restrained. Bureau/institutional feel — not cartoon. Use impact sounds for stamps, interface sounds for data reveals, keyboard sounds for typing.
- SFX analysis guidance: `<skill-dir>/assets/sfx/sfx-analysis.md` and `<skill-dir>/assets/sfx/sfx-analysis.json`. Prefer low high-frequency-risk sounds for repeated moments.
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: copy the chosen music and any Hyperframes-selected SFX into `brag-output/composition/assets/`

## Hyperframes Instructions

Load the composition-building Hyperframes domain skills — `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, `hyperframes-cli`. /brag is its own workflow: do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo / launch-video workflow. Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:

- Show at least one real UI, copy, or visual element from the source project.
- Keep all text readable in the final render.
- Keep the video within 15-25 seconds.
- Include the planned music/SFX layer.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet. Choose SFX after the visual animation exists.
- Treat music cue metadata as optional timing hints. Ignore cues that hurt readability, scene pacing, or the product story.
- Major reveals may move toward nearby strong cues within about 0.15s. Use only 1-3 strong cue locks.
- Use local assets for audio. Run `hyperframes check` before render.
- Keep creation and rendering local.
