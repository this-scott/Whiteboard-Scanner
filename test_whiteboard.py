"""Self-check for the detector: a synthetic room with a whiteboard in it.

The scene reproduces the failure the live logs showed -- a door frame, a
picture frame, a cornice and a skirting board span a rectangle that is real,
fully edge-supported and *larger* than the whiteboard, so ranking candidates
by size picks the wall instead of the board.

Run: python test_whiteboard.py
"""

import cv2
import numpy as np

import whiteboard as wb

BOARD = np.array([[45, 40], [250, 28], [255, 190], [50, 200]], np.float32)


def scene(clutter, fill=(232, 232, 230)):
    """Beige wall, a rectangular target, and rectangles that compete with it."""
    image = np.full((240, 320, 3), 185, np.uint8)
    image[:, 298:303] = 90                                  # door frame
    image[:, 6:9] = 100                                     # picture frame
    cv2.fillConvexPoly(image, BOARD.astype(np.int32), fill)
    cv2.polylines(image, [BOARD.astype(np.int32)], True, (70, 70, 70), 2)
    ink = (120, 90, 60) if sum(fill) / 3 > 128 else (230, 230, 230)
    cv2.putText(image, "abc", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1)
    if clutter:
        cv2.rectangle(image, (10, 150), (40, 235), (150, 150, 150), 1)
        cv2.rectangle(image, (14, 160), (36, 225), (140, 140, 140), 1)
        cv2.rectangle(image, (270, 20), (295, 120), (120, 120, 120), 1)
        cv2.line(image, (0, 218), (320, 226), (120, 120, 120), 2)   # skirting
        cv2.line(image, (0, 12), (320, 6), (150, 150, 150), 2)      # cornice
        cv2.rectangle(image, (120, 205), (190, 214), (60, 60, 60), -1)
        cv2.line(image, (262, 60), (300, 150), (110, 110, 110), 2)
    return cv2.GaussianBlur(image, (3, 3), 0)


def corner_error(corners):
    return float(np.abs(wb.order_corners(corners)
                        - wb.order_corners(BOARD)).max())


def demo():
    for clutter in (False, True):
        corners, _, _ = wb.detect_whiteboard(scene(clutter))
        assert corners is not None, "no board found (clutter=%s)" % clutter
        error = corner_error(corners)
        # The border is drawn 2 px thick, so the fit lands within a few pixels
        # of its centre line; the wall rectangle it must not pick is 50 px out.
        assert error < 8.0, "clutter=%s corner error %.1f px" % (clutter, error)
        print("clutter=%-5s corner error %.1f px" % (clutter, error))

    # The target is picked out by being one surface, not by being a bright
    # one, so a dark book on a pale desk has to work as well as a whiteboard.
    for name, fill in (("black", (30, 30, 28)), ("book brown", (60, 110, 150)),
                       ("mid grey", (140, 140, 138))):
        corners, _, _ = wb.detect_whiteboard(scene(True, fill))
        assert corners is not None, "no target found (%s)" % name
        error = corner_error(corners)
        assert error < 8.0, "%s target: corner error %.1f px" % (name, error)
        print("%-10s among competing rectangles, corner error %.1f px" % (name, error))

    # Rectifying a known rectangle must return its aspect ratio.
    corners, _, _ = wb.detect_whiteboard(scene(True))
    corners = wb.order_corners(corners)
    aspect, focal, solved = wb.estimate_aspect_ratio(corners, 320, 240)
    truth = (np.linalg.norm(BOARD[1] - BOARD[0])
             / np.linalg.norm(BOARD[3] - BOARD[0]))
    assert abs(aspect - truth) < 0.25, "aspect %.2f, expected %.2f" % (aspect, truth)
    assert wb.rectify(scene(True), corners, aspect) is not None
    print("aspect %.2f (drawn %.2f)  focal %.0f px" % (aspect, truth, focal))

    # A caller-supplied focal length is used as given, and the value this view
    # would have solved for comes back regardless.
    fixed, used, again = wb.estimate_aspect_ratio(corners, 320, 240, 500.0)
    assert used == 500.0 and again == solved

    check_tracker()
    print("ok")


def check_tracker():
    """The tracker keeps what frames agree on and ignores what they don't."""
    here = np.array([[100., 100.], [300., 100.], [300., 260.], [100., 260.]])
    rival = here + 90.0                       # a different quadrangle entirely
    jitter = lambda q, n: q + np.array([[n, -n], [-n, n], [n, n], [-n, -n]])

    # The first detection is taken at once: nothing is held yet to protect.
    t = wb.Tracker()
    assert np.allclose(t.update(here), here)

    # One frame of a rival cannot take the view away, and the target returning
    # finds the tracker still on it.
    for _ in range(4):
        t.update(jitter(here, 2.0))
    assert t.agree(t.update(rival), here), "a single rival frame stole the view"
    assert t.agree(t.update(here), here)

    # A rival that keeps showing up is the camera moving, and does take over.
    for _ in range(wb.TRACK_AGREEMENT):
        held = t.update(rival)
    assert t.agree(held, rival), "a persistent rival never took over"

    # Jitter is averaged down rather than followed frame for frame.
    t = wb.Tracker()
    t.update(here)
    noisy = [jitter(here, 6.0 if i % 2 else -6.0) for i in range(8)]
    held = [t.update(q) for q in noisy]
    moved = max(float(np.abs(held[i] - held[i - 1]).max()) for i in range(1, len(held)))
    fed = max(float(np.abs(noisy[i] - noisy[i - 1]).max()) for i in range(1, len(noisy)))
    assert moved < fed, "smoothing did not reduce jitter (%.1f vs %.1f)" % (moved, fed)

    # A held quadrangle survives a gap in detection, then is dropped.
    t = wb.Tracker()
    t.update(here)
    for _ in range(wb.TRACK_PATIENCE):
        assert t.update(None) is not None, "dropped the target too eagerly"
    assert t.update(None) is None, "held a target through a long blackout"
    print("tracker   flicker rejected, move adopted, jitter %.1f -> %.1f px"
          % (fed, moved))


if __name__ == "__main__":
    demo()
