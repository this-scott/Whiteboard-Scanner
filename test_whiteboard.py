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


def scene(clutter):
    """Beige wall, a whiteboard, and rectangles that compete with it."""
    image = np.full((240, 320, 3), 185, np.uint8)
    image[:, 298:303] = 90                                  # door frame
    image[:, 6:9] = 100                                     # picture frame
    cv2.fillConvexPoly(image, BOARD.astype(np.int32), (232, 232, 230))
    cv2.polylines(image, [BOARD.astype(np.int32)], True, (70, 70, 70), 2)
    cv2.putText(image, "abc", (90, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (120, 90, 60), 1)                      # writing
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

    print("ok")


if __name__ == "__main__":
    demo()
