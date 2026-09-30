# Brag Plan: VATAWARAN

## What is this app?

VATAWARAN is an AI-driven weather intelligence engine that tracks extreme weather anomalies across India using Spherical Graph Neural Networks and diffusion-based downscaling — built for Smart India Hackathon 2026 Problem Statement #26078, connecting directly to NCMRWF, IMD, ISRO MOSDAC, and NDMA government feeds.

## The angle

This isn't a weather app. It's a classified meteorological intelligence bulletin — the dashboard looks like a vintage government dispatch newspaper, with aged paper textures, Playfair Display mastheads, double-rule borders, and bureau-red accent stamps. The video leans into this aesthetic hard: it's a *secret intelligence brief* about the weather. Government seals. Classified dossiers. The weather is the enemy, and VATAWARAN tracks it.

## Hook (first 2–3 seconds)

A dramatic flash — the word **VATAWARAN** slams onto aged paper in giant Playfair Display 900-weight letterpress. Below it: "METEOROLOGICAL INTELLIGENCE BULLETIN" in monospaced kicker text. A bureau-red classified stamp appears with a satisfying thud.

## Key moments (the middle)

- **The synoptic chart**: The MapLibre GL map centered on India with three glowing anomaly trajectories — cyclone, heatwave, flash-flood — drawing themselves across the subcontinent with bounding-box overlays.
- **The diffusion countdown**: The 50-step DDIM denoising progress bar filling segment-by-segment, step counter ticking from 00 to 50, with the substep text "Denoising latent field... Mass divergence: 0.012".
- **The government telemetry ribbon**: Five green "SYNCED / ONLINE / READY" pills lighting up one by one — NCMRWF NEPS-G, IMDAA Baseline, IMD Ground AWS, ISRO MOSDAC, NDMA Sachet.

## Outro / punchline

"5 km resolution. 10-day forecast. One engine." — holds for beat. Then the NDMA CAP XML export badge appears: "ITU-T X.1303 COMPLIANT" with a final bureau seal, and the SIH 2026 · PS #26078 stamp.

## User flow worth showing

1. **Entry**: Dashboard loads — the newspaper masthead appears, "Systems Operational" stamp blinks, anomaly bulletin sidebar populates with 3 dossiers.
2. **Key action**: User clicks a cyclone anomaly card → the map zooms to Bay of Bengal → trajectory track animates across coastal Odisha → "Stage II Diffusion Downscaling" overlay triggers with the 50-step DDIM counter.
3. **Result**: 5 km heatmap blooms across the map, severity assessment appears: "EXTREME — Peak Wind Gust 155 km/h", and the NDMA CAP alert is ready to export.

## Tone

- Preset: cinematic
- Creative direction: classified government intelligence brief, cold war weather bureau aesthetic
- Interpretation: Dramatic reveals with confidence. Big serif type. Double-rule borders. Slow builds that pay off. The pacing says "this is important" — not playful, not corporate, but authoritative and striking.

## Format

Landscape — 1920x1080

## Duration

22 seconds

## Visual identity (from the project)

- Background: #faf7f0 (aged paper) with fractal noise SVG grain texture
- Accent: #8b1a1a (bureau red), #1a2744 (navy), #1e3d1e (forest green for status)
- Text: #1a1510 (ink-deep), #2c2418 (ink-base)
- Display font: Playfair Display (900, letterpress feel)
- Body font: EB Garamond (serif, newspaper broadsheet)
- Mono font: IBM Plex Mono (dossier data, telemetry)
- Strongest visual element: The newspaper masthead with ornament emblem, double-rule borders, and the government telemetry ribbon with five agency pills

## Share copy (draft)

"We built an AI that tracks cyclones, heatwaves, and flash floods across India — 5 km resolution, 10-day lookahead, plugged into 5 government feeds. VATAWARAN. SIH 2026 #26078."

## Audio direction

- Role: cinematic support bed — dramatic, building, institutional
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3 (109.96 BPM — the most dramatic/cinematic of the available tracks)
- Music treatment: Fade in from 0s, low during the hook, build through the middle reveals, reach peak during the diffusion counter, subtle fade for outro hold
- Music cue guidance: Preset available, tempo 109.96 BPM. Target strong cues at 8.74s (map trajectory reveal), 13.11s (diffusion counter peak), 22.93s (final stamp). Beat grid from 6.00–13.00s for the sequential gov-pill reveals.
- Audio-reactive treatment: subtle; use bass RMS to gently pulse the paper grain opacity and make the anomaly track glow breathe
- SFX posture: sparse, professional; motion-matched to key reveals only
- Audio-coupled moments: the government pill one-by-one reveals should land on beats; the diffusion counter tick should have subtle mechanical clicks; the final stamp should have a satisfying thud
- Restraint rule: no cartoon sounds, no whooshes — this is a government intelligence bureau, not a game show

## Storyboard

### Scene 1 — "The Masthead" — 3s

A flash of aged paper texture. The word **VATAWARAN** letterpress-stamps in at 900-weight Playfair Display, massive, centered. Below: "METEOROLOGICAL INTELLIGENCE BULLETIN" types out in IBM Plex Mono. A bureau-red "CLASSIFIED" stamp appears offset.

Sequential/interaction: Title stamps in → subtitle types → stamp appears

Audio intent: Dramatic opening impact

Audio-coupled idea: Letterpress thud on title stamp; subtle key ticks on typing subtitle; seal stamp sound

Music: Low cinematic start, building

Transition mood: dramatic → Scene 2

### Scene 2 — "The Synoptic Chart" — 5s

India map (dark styled, muted tones). Three anomaly trajectories draw themselves one by one across the map: Cyclone track (bureau red) curving up the Bay of Bengal, Heatwave ridge (amber) spreading across Rajasthan, Flash flood zone (navy) in the Western Himalaya. Each gets a brief label: "BOB-CYCLONE-2026-01", "NWI-HEATWAVE-2026-03", "WHR-RAIN-2026-07". EFI scores appear beside each: 0.96, 0.91, 0.88.

Sequential/interaction: Three tracks draw in one by one, each with its ID label and EFI badge

Audio intent: Building tension, each track landing is a moment

Audio-coupled idea: Each trajectory reveal lands on a beat; subtle radar-sweep sound as tracks draw

Music: Building, mid energy

Transition mood: clean → Scene 3

### Scene 3 — "The Diffusion Engine" — 5s

Close-up of the diffusion processing overlay. Step counter ticking: 00 → 10 → 25 → 40 → 50. The segmented progress bar fills. Substep text cycles: "Initializing noise tensor" → "Denoising latent field" → "Physics QA: mass_divergence = 0.012 ✓". The "5 km" resolution badge pulses. A stylized heatmap pattern blooms behind the counter.

Sequential/interaction: Counter ticks up; progress bar fills; substep text cycles

Audio intent: Mechanical precision, building to completion

Audio-coupled idea: Tick sound on counter steps; subtle mechanical whir; satisfying "complete" chime at 50

Music: Peak energy at completion

Transition mood: dramatic → Scene 4

### Scene 4 — "Government Feeds Online" — 4s

The government telemetry ribbon. Five pills appear one by one, each lighting up with a green status dot:

1. "NCMRWF NEPS-G (12 km)" → SYNCED
2. "IMDAA Baseline" → ONLINE
3. "IMD Ground AWS" → ONLINE
4. "ISRO MOSDAC (INSAT-3DR)" → ONLINE
5. "NDMA Sachet (CAP v1.2)" → READY

Below: "14 Ground Stations · 5 Government Agencies · 1 Engine"

Sequential/interaction: Five pills appear one by one, each on a beat

Audio intent: Confident accumulation — each agency connecting reinforces authority

Audio-coupled idea: Each pill arrival on a beat; subtle data-sync blip

Music: Strong rhythm, confident stride

Transition mood: clean → Scene 5

### Scene 5 — "The Verdict" — 5s

Full-bleed aged paper background. Large centered text in Playfair Display:

"5 km resolution."
"10-day forecast."
"One engine."

Each line appears one by one with a hold. Then the NDMA CAP export badge: "ITU-T X.1303 COMPLIANT" with official seal styling. Bottom: "SIH 2026 · Problem Statement #26078" in IBM Plex Mono. Final: "VATAWARAN" wordmark with ornament rule.

Sequential/interaction: Three taglines appear one by one → CAP badge → SIH stamp → wordmark

Audio intent: Landing. Confidence. Authority.

Audio-coupled idea: Each tagline on a strong beat; final stamp thud

Music: Peak → resolve to final beat → fade

Transition mood: final

**Music mood for this video:** cinematic, institutional, building

**Audio summary:** Starts quiet with impact stamp, builds through map reveals and diffusion sequence, peaks at government feed activation, resolves with a confident final beat on the verdict taglines.
