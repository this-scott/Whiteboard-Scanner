# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `README.md` for what the project is, how to run it, and the current
numbers. This file covers what the README does not: how to measure a change
here, and the mistakes already made.

## Commands

```bash
.venv/bin/python whiteboard.py          # live, needs a camera
.venv/bin/python test_whiteboard.py     # the whole self-check, ~5s, no camera
```

There is no test framework and no lint config. `test_whiteboard.py` is a
single `demo()` of asserts over a synthetic scene, plus `check_tracker()` and
`check_recording()`. To run one part, call it directly:

```bash
.venv/bin/python -c "import test_whiteboard as t; t.check_tracker()"
```

Offline replay against real frames is how everything is actually evaluated —
write a throwaway script that imports `whiteboard` and loops over `logs/`.
Never add such scripts to the repo.

## Architecture

`whiteboard.py` is the whole program, in the paper's section order. Two things
are worth knowing before changing anything:

**Selection, not detection, is the hard part.** The correct quadrangle is
usually *among* the candidates; picking it is what fails. `quality` (fraction
of perimeter backed by real edges) leaves near-ties, and `interior_spread`
settles them on the principle that a target is one surface. Brightness,
interior flatness and boundary contrast were each tried; each wins one class
of scene by giving up another. Treat a new ranking idea as a trade until
measured on all three scene classes.

**`LineSupport` is why every candidate can be scored.** It prefix-sums each
Hough line's real edge coverage once per frame, so a side's supported length
is two array lookups and the whole batch scores in one vectorised pass. Before
it, only the largest few hundred candidates could be afforded, and the board is
never the largest thing in a room. Keep scoring batch-wide.

Stability is separate from accuracy and layered on top: `Tracker` holds the
quadrangle successive frames agree on, and `detect_whiteboard(frame, previous)`
prefers a contender matching it. Both are no-ops when there is no history, so
single-frame behaviour is unaffected.

The constants at the top of the file are the tuning surface, and each carries
the measurement that set it. `MIN_LINE_VOTES` and `GRADIENT_THRESHOLD` are the
two with the largest effect and the sharpest trade-offs.

## Evaluating a change

Data lives in `logs/` (gitignored, ~300 MB):

- `logs/frame/*.png` — independent stills, varied poses, with `logs/data/*.json`
- `logs/segments/<stamp>/` — continuous runs at 5–12 fps, raw frames plus
  `records.jsonl` carrying each frame's own detection and the tracker's held
  quadrangle

Scene classes and their ground truth, all of which must be checked together:

| class | frames | metric |
|---|---|---|
| whiteboard | 25 stills, plus segments `232728` / `232754` | aspect within 0.15 of 1.5 |
| book + page on a desk | 4 stills | IoU ≥ 0.5 against hand labels |
| dog painting | 3 stills | IoU ≥ 0.5 against hand labels |
| stability | near-identical consecutive pairs (mean pixel diff < 2) | fraction of corner jumps > 50 px |

The hand labels were read off a coordinate grid by eye and are good to about
±10 px, so a difference of one or two frames is noise. Say so rather than
claiming an improvement.

## Mistakes already made here

- **Benchmarked against the wrong baseline.** A comparison file built from
  `HEAD` before the first commit was reused for several rounds and reported a
  slowdown as a speedup. Always `git show <commit>:whiteboard.py > /tmp/old.py`
  for the commit actually being compared against.
- **Optimised an intermediate metric.** Lowering `GRADIENT_THRESHOLD` to 15
  raised how much of a target's perimeter registered as edges, exactly as
  intended, and made the aspect ratio worse — weak edges drag the line fits off
  the boundary. Measure the output, not the stage.
- **A median hid a frozen tracker.** "0.0 px median movement" read as perfect
  stability; the tracker was stuck on its first detection and scoring 1/4 where
  the detector alone scored 3/4. Prefer counting bad events over averaging.
- **Reconstructed frames by inpainting the overlay away.** The green polygon is
  drawn *on* the object's border, so removing it destroys the edges under test.
  This is why raw frames are saved; use `logs/frame/` and `logs/segments/`.
- **Reported a ceiling that was an artifact.** "14/25 is the ceiling for any
  ranking, so candidate generation is the bottleneck" was measured on a
  candidate set built mostly from noise lines. Clearing them gave 19/25.
  A ceiling is only as real as the pipeline that produced it.
- **Correlated frames are not independent samples.** 270 frames of one slow pan
  carry far less information than 25 varied stills, and the two disagree about
  `MIN_LINE_VOTES`. Weight accordingly.

## Conventions

Comments explain why a value or approach was chosen, usually with the
measurement that decided it; keep that when editing rather than trimming to a
description of what the line does. Commit messages carry the before/after
numbers. Deliberate shortcuts are marked `ponytail:` with their ceiling and
upgrade path.
