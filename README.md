# Whiteboarder

## Preamble
This paper implementation is the result of a one hour session with Claude Code. 

I attempted to create this once by hand as a project precursor. Eventually I let it go for other projects.

The strategy here was simple: point Claude to the paper and let it create a first implementation with a details logger, test and return the logs, let Claude tune the detector, repeat from step 2. Works *scarily* well

Model used is Opus 5 on high effort. CC logs said I used ~569k tokens, the majority of which was likely through reading images and detections

Original Paper: [Whiteboard Scanning and Image Enhancement](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Digital-Signal-Processing.pdf) 

## Whiteboarder
Live detection and rectification of a whiteboard — or any flat rectangular
object — from a webcam, implementing Zhang & He, *Whiteboard Scanning and Image
Enhancement* (MSR-TR-2003-39).

Two windows: the camera feed with the fitted quadrangle drawn on it, and the
rectified fronto-parallel view of whatever was found.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python whiteboard.py            # --camera 0 --width 640 --height 480
.venv/bin/python test_whiteboard.py       # self-check, no camera needed
```

| key | |
|---|---|
| `q` / `Esc` | quit |
| `e` | toggle enhancement (§3.4) |
| `d` | toggle the edge and Hough views |
| `s` | save every view plus a JSON record under `logs/` |
| `r` | record a continuous segment of raw frames under `logs/segments/` |

`--scale` (default 0.5) sets the resolution detection runs at; rectification
always uses the full frame. `--record-seconds` (default 20) bounds a recording.

## How it works

| stage | what it does |
|---|---|
| §3.1 edges | Sobel, thresholded at `GRADIENT_THRESHOLD`, each edge keeping its gradient orientation |
| §3.1 Hough | oriented accumulator over the full circle, so the two sides of a border stay distinct |
| §3.1 quadrangles | four-line combinations passing the paper's five criteria |
| §3.1 verification | `LineSupport` prefix-sums each line's real edge coverage, so every candidate is scored, not just the largest |
| selection | quality leads; near-ties are settled by `interior_spread`, since a target is *one surface* and a spurious quadrangle straddles several |
| §3.2 | aspect ratio and focal length from the single view |
| §3.3 | warp to a rectangle of the estimated aspect |
| §3.4 | white balance against the estimated board colour, then an S-curve |

Stability is handled separately: `Tracker` holds the quadrangle successive
frames agree on, and the tracked quadrangle is fed back into selection so a
contender matching it wins near-ties.

## Where it stands

Measured on logged frames, aspect within 0.15 of the true 1.5 for whiteboards
and IoU ≥ 0.5 against hand-labelled targets otherwise:

| | whiteboard stills | book + page | painting | detection |
|---|---|---|---|---|
| | 19/25 | 3/4 | 2/3 | ~54 ms/frame at 320×240 |

The detection improvements are all single-frame; only `Tracker` and the
tracking prior need a sequence, and both are no-ops without one.

Known failures: a page on a white desk, whose boundary is genuinely absent
from the image at any threshold; and fast camera motion, where blur costs most
of the edges.

## Logs

`s` writes `logs/{frame,camera_detection,edges,hough,rectified_whiteboard}/<stamp>.png`
and `logs/data/<stamp>.json`. `r` writes `logs/segments/<stamp>/` — raw frames
plus `records.jsonl`, one line per frame carrying that frame's own detection
and the tracker's held quadrangle side by side. Only raw frames are recorded;
every other view is recomputable from them.

## Notes and ideas

- **Candidate generation is the remaining bottleneck.** On the frames still
  missed, no candidate has the right geometry, so no amount of better ranking
  reaches them. Worth checking whether those cluster at particular poses.
- **`MAX_PAIRS = 60` keeps only the widest-separated line pairs**, the same
  "biggest wins" bias removed from scoring. Raising it surfaces small targets
  but costs time in the Python loop in `form_quadrangles` — vectorising that
  loop would make the cap unnecessary.
- **Corner precision, not candidate choice, caps the aspect ratio.** Quadrangles
  come out with the right area and imprecise corners. Sub-pixel line fitting,
  or averaging corners over frames the tracker already agrees on, is the lever.
- **Calibrate the focal length once.** Equation (21) swings 242–990 px for a
  fixed lens; the live loop takes a running median, but a single frame cannot.
  Pass a known focal to `estimate_aspect_ratio` for one-shot use.
- **Two evaluations disagree** on `MIN_LINE_VOTES`: +6 frames on independent
  stills, neutral on continuous pans. Consecutive pan frames are not
  independent samples. Varied segments — different rooms, poses, objects —
  would settle it.
- **Ranking is a trade, not a solution.** Brightness, interior flatness and
  boundary contrast each win one class of scene by giving up another. Returning
  the top few candidates instead of forcing one answer may serve callers better.
- **`Tracker.agree` is not rotation-invariant.** A quadrangle turning through
  45° can flip which corner comes first and read as a disagreement. Compare
  over all four rotations if that shows up as flapping.
- **Low-contrast targets want a different cue** than gradient magnitude —
  colour, or the shadow an object casts on the surface behind it.
