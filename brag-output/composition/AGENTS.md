# HyperFrames Composition Guide

## Overview

This project is a video composition built with HyperFrames.

### Workflows

- **Product launch video**: Creates short, polished launch videos and showcases from project source code.
- **Motion graphics**: Short design-led animation sequences.
- **Audio sync**: Beat-synced timeline animation locked to audio cues.

### Key Rules

> Always check the timeline with `npx hyperframes check` before rendering.
> Ensure all fonts and assets are locally cached or injected deterministically.
> Maintain high visual contrast (WCAG AA minimum 4.5:1 for readable text).

### Useful Commands

```bash
# Validate the composition layout, contrast, and timing
npx hyperframes check

# Preview in browser studio
npx hyperframes preview

# Render final MP4 output
npx hyperframes render --output ../brag.mp4
```

### Composition Structure

```text
composition/
  ├── index.html       # Main HTML, CSS, and GSAP timeline definition
  ├── hyperframes.json # Project metadata and resolution
  ├── meta.json        # Composition properties
  └── assets/          # Local audio and media files
```
