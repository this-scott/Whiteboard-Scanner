"""Live whiteboard detection and rectification.

Implements the algorithm from Zhang & He, "Whiteboard Scanning and Image
Enhancement" (MSR-TR-2003-39):

  3.1  Automatic whiteboard detection
       edge detection -> oriented Hough transform -> quadrangle formation
       -> quadrangle verification -> quadrangle refining
  3.2  Physical aspect ratio and focal length from a single view
  3.3  Rectification to a rectangle of the estimated aspect ratio
  3.4  White balancing and image enhancement

Two live windows: the camera feed with the fitted quadrangle overlaid, and
the rectified fronto-parallel view of the detected board.

Keys:  q/ESC quit   e toggle enhancement   d toggle debug (edge+Hough) views
       s save every view plus a JSON record under logs/
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# --- Section 3.1 parameters (values from the paper) -------------------------
GRADIENT_THRESHOLD = 15      # T_G, on G = |Gx| + |Gy|. The paper's 40 suits a
                             # dark-framed board on a plain wall; a page on a
                             # white desk or a book against dark carpet has
                             # boundaries far weaker than that, and at 40 most
                             # of the target's own perimeter is simply unseen.
RHO_BIN = 5.0                # rho cell size, pixels
THETA_BIN = np.deg2rad(2.0)  # theta cell size, 2 degrees
PEAK_FRACTION = 0.05         # keep local maxima above 5% of the max vote
SUPPRESSION_THETA = 3        # non-maximum suppression half-window, theta cells
SUPPRESSION_RHO = 2          # non-maximum suppression half-window, rho cells
OPPOSITE_TOLERANCE = np.deg2rad(30.0)   # opposite lines within 30 deg of 180
PERPENDICULAR_TOLERANCE = np.deg2rad(30.0)  # neighbours within 30 deg of 90
SUPPORT_DISTANCE = 3.0       # an edge within 3 px of a side supports it
REFINE_NEIGHBOURHOOD = 10.0  # line refitting neighbourhood, pixels
QUALITY_THRESHOLD = 0.5      # a side must be mostly covered by real edges
QUALITY_TIE = 0.05           # quadrangles within this of the best are ranked
                             # on interior spread. A low-contrast target never
                             # tops the support ranking outright, so the band
                             # has to be wide enough to keep it in contention.
MAX_EDGE_DENSITY = 0.8       # above this the frame is texture, not a scene

# --- Section 3.4 parameters -------------------------------------------------
CELL_SIZE = 15               # whiteboard-color cells, 15x15 pixels
TOP_PERCENTILE = 25          # average the top 25% luminance pixels per cell
S_CURVE_P = 0.75             # steepness of the S-shaped curve

MAX_LINES = 60               # cap on Hough peaks, keeps quadrangle search sane
MAX_PAIRS = 60               # opposing line pairs kept, widest separation first
MAX_QUADRANGLES = 4000       # cap on quadrangles built per frame


# ---------------------------------------------------------------------------
# 3.1  Edge detection
# ---------------------------------------------------------------------------

def detect_edges(gray):
    """Sobel gradient, thresholded at T_G, with per-edge orientation.

    The paper approximates the gradient magnitude as G = |Gx| + |Gy| and calls
    a pixel an edge when G > T_G. Each edge carries an orientation
    theta = atan2(Gy, Gx) in [-180, 180) degrees, which is what makes the
    Hough transform below an *oriented* one.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.abs(gx) + np.abs(gy)
    mask = magnitude > GRADIENT_THRESHOLD

    ys, xs = np.nonzero(mask)
    theta = np.arctan2(gy[ys, xs], gx[ys, xs])
    return xs.astype(np.float32), ys.astype(np.float32), theta, mask


# ---------------------------------------------------------------------------
# 3.1  Oriented Hough transform
# ---------------------------------------------------------------------------

class HoughSpace:
    """Accumulator over (rho, theta) with theta spanning the full circle.

    Unlike the textbook transform, theta ranges over [-pi, pi) so that two
    lines lying on opposite sides of a border -- which have gradients pointing
    in opposite directions -- land in different cells instead of being merged.
    """

    def __init__(self, width, height):
        diagonal = float(np.hypot(width, height))
        self.rho_min = -diagonal
        self.n_rho = int(np.ceil(2 * diagonal / RHO_BIN)) + 1
        self.n_theta = int(np.ceil(2 * np.pi / THETA_BIN)) + 1
        self.accumulator = np.zeros((self.n_theta, self.n_rho), np.float32)

    def vote(self, xs, ys, theta):
        """Each edge votes once, for the line through it with its orientation."""
        rho = xs * np.cos(theta) + ys * np.sin(theta)
        ti = np.clip(((theta + np.pi) / THETA_BIN).astype(np.int32),
                     0, self.n_theta - 1)
        ri = np.clip(((rho - self.rho_min) / RHO_BIN).astype(np.int32),
                     0, self.n_rho - 1)
        np.add.at(self.accumulator, (ti, ri), 1.0)
        return self.accumulator

    def peaks(self):
        """Local maxima with more than PEAK_FRACTION of the top vote count.

        Suppression happens on the raw accumulator. Smoothing along theta is
        tempting -- the gradient orientations along one edge scatter by a few
        degrees -- but with 2 degree cells even a modest blur bleeds a slanted
        line's votes into the strong horizontal and vertical columns that a
        room full of straight edges always produces, and the peak returned for
        each board side snaps to the nearest axis. The votes for a real edge
        already concentrate well enough to stand on their own, so the
        accumulator is left alone and only the winner within a small
        neighbourhood is kept.
        """
        acc = self.accumulator
        peak_value = acc.max()
        if peak_value <= 0:
            return []
        threshold = PEAK_FRACTION * peak_value

        # Circular padding along theta, so the suppression window wraps.
        pad = SUPPRESSION_THETA
        wrapped = np.vstack([acc[-pad:], acc, acc[:pad]])
        window = np.ones((2 * pad + 1, 2 * SUPPRESSION_RHO + 1), np.uint8)
        dilated = cv2.dilate(wrapped, window)[pad:-pad]
        ti, ri = np.nonzero((acc >= dilated) & (acc > threshold))

        lines = [(ri[i] * RHO_BIN + self.rho_min,
                  ti[i] * THETA_BIN - np.pi,
                  float(acc[ti[i], ri[i]])) for i in range(len(ti))]
        lines.sort(key=lambda line: -line[2])
        return lines[:MAX_LINES]


def line_intersection(a, b):
    """Intersect two lines given in normal form x cos(t) + y sin(t) = rho."""
    rho_a, theta_a = a[0], a[1]
    rho_b, theta_b = b[0], b[1]
    determinant = np.sin(theta_b - theta_a)
    if abs(determinant) < 1e-6:
        return None
    x = (rho_a * np.sin(theta_b) - rho_b * np.sin(theta_a)) / determinant
    y = (rho_b * np.cos(theta_a) - rho_a * np.cos(theta_b)) / determinant
    return np.array([x, y], np.float64)


def angle_difference(a, b):
    """Signed difference a - b wrapped into [-pi, pi)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def similar_orientation(a, b, tolerance=OPPOSITE_TOLERANCE):
    """True where two gradient orientations agree up to a 180 degree flip.

    A side's supporting edges all lie along its normal, but whether the
    gradient points into or out of the board depends on which side is
    brighter, so orientations are compared modulo pi.
    """
    difference = np.abs(angle_difference(a, b))
    return np.minimum(difference, np.pi - difference) < tolerance


# ---------------------------------------------------------------------------
# 3.1  Quadrangle formation
# ---------------------------------------------------------------------------

def form_quadrangles(lines, width, height):
    """Enumerate four-line combinations that pass the paper's five criteria.

    Lines are taken as two opposing pairs (top/bottom, left/right). The
    criteria prune the combinatorial explosion down to plausible boards:

      1. opposite lines have near-opposite orientations (180 within 30 deg)
      2. opposite lines are far apart (|drho| > one fifth of width or height)
      3. neighbouring lines meet at close to +/-90 deg (within 30 deg)
      4. orientations run consistently clockwise or counter-clockwise
      5. the quadrangle is big enough (circumference > (W + H) / 4)
    """
    min_separation = min(width, height) / 5.0
    min_circumference = (width + height) / 4.0
    n = len(lines)

    # Criteria 1 and 2 depend only on a pair, so pair up once and reuse.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            opposite = abs(abs(angle_difference(lines[i][1], lines[j][1])) - np.pi)
            if opposite > OPPOSITE_TOLERANCE:
                continue
            separation = abs(abs(lines[i][0]) - abs(lines[j][0]))
            if separation < min_separation:
                continue
            pairs.append((separation, i, j))

    # Widely separated pairs first, so the largest quadrangles -- the ones the
    # caller scores first and the ones a framed whiteboard produces -- are
    # built before the budget runs out. A cluttered room can otherwise form
    # tens of thousands of candidates, and merely building them costs more
    # than the whole frame's time allowance.
    pairs.sort(reverse=True)
    pairs = [(i, j) for _, i, j in pairs[:MAX_PAIRS]]

    quadrangles = []
    for p in range(len(pairs)):
        if len(quadrangles) >= MAX_QUADRANGLES:
            break
        for q in range(p + 1, len(pairs)):
            a, c = pairs[p]
            b, d = pairs[q]
            if len({a, b, c, d}) != 4:
                continue

            # Criterion 3: the two pairs must be roughly perpendicular.
            cross = abs(angle_difference(lines[a][1], lines[b][1]))
            if abs(cross - np.pi / 2) > PERPENDICULAR_TOLERANCE:
                continue

            for second in ((b, d), (d, b)):
                sides = [a, second[0], c, second[1]]
                corners = []
                for k in range(4):
                    point = line_intersection(lines[sides[k]],
                                              lines[sides[(k + 1) % 4]])
                    if point is None:
                        break
                    corners.append(point)
                if len(corners) != 4:
                    continue
                corners = np.array(corners)

                # Criterion 4: consistent turning direction (a convex,
                # non self-intersecting quadrangle).
                edges = np.roll(corners, -1, axis=0) - corners
                following = np.roll(edges, -1, axis=0)
                turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
                if not (np.all(turns > 0) or np.all(turns < 0)):
                    continue

                # Criterion 5: big enough.
                circumference = float(np.sum(np.linalg.norm(edges, axis=1)))
                if circumference < min_circumference:
                    continue

                # corners[k] is sides[k] ^ sides[k+1], so the segment leaving
                # corners[k] runs along sides[k+1]. Rotate the sides so that
                # sides[k] owns the segment corners[k] -> corners[k+1], which
                # is the pairing the verification and refining steps assume.
                quadrangles.append((corners, sides[1:] + sides[:1], circumference))

    if not quadrangles:
        return None, None, None
    # Stacked, because verification scores the whole batch in one pass.
    return (np.stack([q[0] for q in quadrangles]),
            np.array([q[1] for q in quadrangles]),
            np.array([q[2] for q in quadrangles]))


# ---------------------------------------------------------------------------
# 3.1  Quadrangle verification
# ---------------------------------------------------------------------------

class LineSupport:
    """Where along each Hough line the real edges are, as prefix sums.

    Verification asks the same question for every candidate: how much of this
    segment is backed by edges. Every segment lies on one of the few dozen
    Hough lines, so the answer is precomputed per line instead of per
    candidate -- bin the edges that support the line into one-pixel cells
    along it, then cumulatively sum, and a segment's covered length becomes
    the difference of two lookups.

    That difference is what lets `quality` score *every* candidate. Scoring
    used to cost a scan per side, so only the largest few hundred quadrangles
    could be afforded, and since the board is never the largest thing a
    cluttered room produces -- a door frame and a picture frame make a much
    bigger quadrangle -- the true board was routinely dropped before it was
    ever scored.
    """

    def __init__(self, lines, xs, ys, theta, width, height):
        # One bin per pixel of projected distance along the line. Corners can
        # sit outside the image, so the axis spans the diagonal either way.
        self.offset = float(np.hypot(width, height))
        self.bins = int(2 * self.offset) + 2
        self.angles = np.array([line[1] for line in lines])
        self.prefix = np.zeros((len(lines), self.bins), np.int32)
        for i, (rho, angle, _) in enumerate(lines):
            # An edge supports a line when it lies within 3 px of it and its
            # gradient points along the line's normal, modulo pi.
            distance = np.abs(xs * np.cos(angle) + ys * np.sin(angle) - rho)
            near = np.nonzero((distance < SUPPORT_DISTANCE)
                              & similar_orientation(theta, angle))[0]
            if len(near) == 0:
                continue
            along = -xs[near] * np.sin(angle) + ys[near] * np.cos(angle)
            cells = np.clip((along + self.offset).astype(np.int32), 0, self.bins - 1)
            # Set rather than add: edges are two or three pixels thick and
            # bunch up around writing, so a count of pixels lets a short side
            # crossing a scribble out-score a long genuine border. Coverage of
            # the side's length is what the ratio is meant to express.
            self.prefix[i, cells] = 1
        np.cumsum(self.prefix, axis=1, out=self.prefix)

    def covered(self, corners, sides):
        """Covered length of each side of each candidate. Shapes (M,4,2), (M,4)."""
        angles = self.angles[sides]
        following = np.roll(corners, -1, axis=1)
        starts = -corners[..., 0] * np.sin(angles) + corners[..., 1] * np.cos(angles)
        ends = -following[..., 0] * np.sin(angles) + following[..., 1] * np.cos(angles)
        low = np.clip(np.minimum(starts, ends) + self.offset, 0, self.bins - 1)
        high = np.clip(np.maximum(starts, ends) + self.offset, 0, self.bins - 1)
        return (self.prefix[sides, high.astype(np.int32)]
                - self.prefix[sides, low.astype(np.int32)])


def interior_spread(gray, corners):
    """Luminance spread inside each quadrangle, sampled on a coarse grid.

    Edge support alone cannot tell the target from any other well-formed
    rectangle in the scene: a door frame with a cornice, or two desk edges
    with the edge of a magazine, span quadrangles that are real, straight and
    fully supported, and are often *larger* than the target, so the paper's
    "prefer the biggest" rule hands them the win. What separates them is that
    the target is one object and they are not -- a board, a page or a book
    cover is one surface under one light, while a spurious quadrangle
    straddles desk, carpet and clutter, and its interior says so.

    This measure has to stay indifferent to what the object *is*: brightness
    would work for a whiteboard and then reject a dark book on a pale desk.
    Spread is a property of a surface, not of its colour.

    The grid is inset from the border so that the object's own frame or
    shadow is not sampled, and nearest-neighbour is enough at this resolution.
    """
    u, v = np.meshgrid(np.linspace(0.15, 0.85, 6), np.linspace(0.15, 0.85, 6))
    u, v = u.ravel(), v.ravel()
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], 1)
    points = np.einsum("pk,mkc->mpc", weights, corners)
    x = np.clip(points[..., 0], 0, gray.shape[1] - 1).astype(np.int32)
    y = np.clip(points[..., 1], 0, gray.shape[0] - 1).astype(np.int32)
    return gray[y, x].std(axis=1)


def quality(corners, sides, circumference, support):
    """Fraction of each circumference actually backed by oriented edges.

    Hough lines are infinite and say nothing about where their support lies,
    so a quadrangle of four well-placed lines can still be spurious (Fig. 2).
    The quality measure is the ratio of supported perimeter to the whole
    perimeter, scored for every candidate at once.
    """
    lengths = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    # Cap coverage at each side's own length: a bin only exists because that
    # stretch of the side is covered, so more bins than the side is long means
    # rounding, not extra evidence. Left uncapped the measure exceeds 1 and
    # stops being a fraction, which lets a small crisp rectangle beat a large
    # one and forces the size comparison to arbitrate what quality should have
    # settled.
    covered = np.minimum(support.covered(corners, sides), lengths)
    return covered.sum(axis=1) / circumference


# ---------------------------------------------------------------------------
# 3.1  Quadrangle refining
# ---------------------------------------------------------------------------

def refine_side(side, p0, p1, xs, ys, theta):
    """Refit one side to its supporting edges, rejecting outliers.

    The Hough lines are only accurate to the discretisation of the parameter
    space. The paper collects the edges within 10 px of the side that have a
    similar orientation, discards outliers with least-median-squares, and
    least-squares fits the remainder.
    """
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return side
    normal = np.array([-direction[1], direction[0]]) / length

    offsets = np.column_stack([xs - p0[0], ys - p0[1]])
    distance = np.abs(offsets @ normal)
    along = (offsets @ direction) / length
    oriented = similar_orientation(theta, side[1])
    selected = np.nonzero((distance < REFINE_NEIGHBOURHOOD) &
                          (along >= 0) & (along <= length) & oriented)[0]
    if len(selected) < 10:
        return side

    px, py = xs[selected], ys[selected]

    # Least-median-squares: the current line is the initial estimate, so
    # residuals against it give a robust scale from which to cut outliers.
    residual = np.abs(px * np.cos(side[1]) + py * np.sin(side[1]) - side[0])
    median = np.median(residual)
    scale = 1.4826 * max(median, 1e-3)
    inliers = residual < 2.5 * scale
    if np.count_nonzero(inliers) < 10:
        return side
    px, py = px[inliers], py[inliers]

    # Total least squares: the fitted direction is the eigenvector of the
    # scatter matrix with the smaller eigenvalue.
    mean_x, mean_y = px.mean(), py.mean()
    centred = np.column_stack([px - mean_x, py - mean_y])
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    nx, ny = vt[1]
    new_theta = np.arctan2(ny, nx)
    new_rho = mean_x * np.cos(new_theta) + mean_y * np.sin(new_theta)

    # Keep the normal pointing the same way as the original, so that the
    # orientation bookkeeping used elsewhere stays valid.
    if abs(angle_difference(new_theta, side[1])) > np.pi / 2:
        new_theta = (new_theta + np.pi + np.pi) % (2 * np.pi) - np.pi
        new_rho = -new_rho
    return (new_rho, new_theta, side[2])


def refine_line(line, xs, ys, theta):
    """Refit a Hough peak to the edges supporting it, anywhere in the image.

    A peak is only accurate to the size of a Hough cell -- 5 pixels by 2
    degrees -- and 2 degrees of angular error is enough to badly skew the
    aspect ratio recovered in section 3.2. Worse, the criteria in
    `form_quadrangles` compare these angles against each other, so quantised
    peaks make genuine boards look non-rectangular and get them rejected
    before refining ever runs. This is the same least-median-squares then
    least-squares fit the paper applies to the sides of the winning
    quadrangle, hoisted earlier and applied to bare lines, which have no
    endpoints to bound the search.
    """
    rho, angle = line[0], line[1]
    distance = np.abs(xs * np.cos(angle) + ys * np.sin(angle) - rho)
    selected = np.nonzero((distance < RHO_BIN) & similar_orientation(theta, angle))[0]
    if len(selected) < 10:
        return line

    px, py = xs[selected], ys[selected]
    residual = distance[selected]
    scale = 1.4826 * max(np.median(residual), 1e-3)
    inliers = residual < 2.5 * scale
    if np.count_nonzero(inliers) < 10:
        return line
    px, py = px[inliers], py[inliers]

    mean_x, mean_y = px.mean(), py.mean()
    centred = np.column_stack([px - mean_x, py - mean_y])
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    nx, ny = vt[1]
    new_theta = np.arctan2(ny, nx)
    new_rho = mean_x * np.cos(new_theta) + mean_y * np.sin(new_theta)
    if abs(angle_difference(new_theta, angle)) > np.pi / 2:
        new_theta = angle_difference(new_theta + np.pi, 0.0)
        new_rho = -new_rho
    return (new_rho, new_theta, line[2])


def deduplicate(lines):
    """Drop lines that refining has collapsed onto one another."""
    kept = []
    for line in lines:
        duplicate = False
        for other in kept:
            if (abs(angle_difference(line[1], other[1])) < np.deg2rad(3.0)
                    and abs(line[0] - other[0]) < RHO_BIN):
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def detect_whiteboard(frame):
    """Full section 3.1 pipeline. Returns (corners, edge mask, hough image)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    xs, ys, theta, mask = detect_edges(gray)
    if len(xs) < 100:
        return None, mask, None

    # In a heavily textured frame every candidate side finds support just by
    # chance, so the quality measure stops discriminating and the detector
    # reports a confident board in what is really noise.
    if len(xs) > MAX_EDGE_DENSITY * gray.size:
        return None, mask, None

    hough = HoughSpace(width, height)
    accumulator = hough.vote(xs, ys, theta)
    lines = deduplicate([refine_line(line, xs, ys, theta) for line in hough.peaks()])
    if len(lines) < 4:
        return None, mask, accumulator

    corners, sides, circumference = form_quadrangles(lines, width, height)
    if corners is None:
        return None, mask, accumulator

    support = LineSupport(lines, xs, ys, theta, width, height)
    scores = quality(corners, sides, circumference, support)

    # Quality leads, and only decides among quadrangles whose support is
    # genuinely comparable to the best one; weighting quality against the
    # other measures lets a marginally bigger quadrangle with visibly worse
    # edge support win by a hair, which under noise is the wrong call.
    contenders = np.nonzero(scores >= max(scores.max() - QUALITY_TIE,
                                          QUALITY_THRESHOLD))[0]
    if len(contenders) == 0:
        return None, mask, accumulator

    # Then the flattest interior wins. Size is not the tie-break the paper
    # assumes: in a cluttered scene the biggest supported quadrangle is
    # generally the one spanning the most junk, and quality on its own prefers
    # small crisp rectangles, since a short perimeter is easier to cover
    # completely than a long one. Both failures share a cause -- the winner is
    # not one surface -- which is what the spread measures.
    best = contenders[np.argmin(interior_spread(gray, corners[contenders]))]
    corners, sides = corners[best], sides[best]

    refined = [refine_side(lines[sides[k]], corners[k], corners[(k + 1) % 4],
                           xs, ys, theta)
               for k in range(4)]
    points = []
    for k in range(4):
        point = line_intersection(refined[k], refined[(k + 1) % 4])
        if point is None:
            return corners, mask, accumulator
        points.append(point)
    return np.array(points), mask, accumulator


# ---------------------------------------------------------------------------
# 3.2  Aspect ratio and focal length from a single view
# ---------------------------------------------------------------------------

def estimate_aspect_ratio(corners, width, height, focal=None):
    """Estimate (aspect ratio, focal length used, focal length solved).

    Implements equations (11), (12), (14), (16), (17), (20) and (21). The
    corners must arrive in the paper's ordering: m1 = (0,0), m2 = (w,0),
    m3 = (0,h), m4 = (w,h) on the board.

    Assumes square pixels (s = 1) and the principal point at the image
    centre, which are the assumptions the paper makes to reduce the two
    constraints to a solvable system.

    A caller with a better focal length than one frame can offer -- the camera
    keeps the same lens from frame to frame, while equation (21) is sensitive
    enough to corner noise to swing by a factor of four between consecutive
    frames of a stationary board -- passes it in, and gets back the value this
    view alone would have solved for, or None where it solves for nothing.
    """
    m1 = np.array([corners[0][0], corners[0][1], 1.0])
    m2 = np.array([corners[1][0], corners[1][1], 1.0])
    m3 = np.array([corners[3][0], corners[3][1], 1.0])
    m4 = np.array([corners[2][0], corners[2][1], 1.0])

    u0, v0, s = width / 2.0, height / 2.0, 1.0

    # Equations (11) and (12).
    denominator_2 = np.dot(np.cross(m2, m4), m3)
    denominator_3 = np.dot(np.cross(m3, m4), m2)
    if abs(denominator_2) < 1e-9 or abs(denominator_3) < 1e-9:
        return None, focal, None
    k2 = np.dot(np.cross(m1, m4), m3) / denominator_2
    k3 = np.dot(np.cross(m1, m4), m2) / denominator_3

    # Equations (14) and (16).
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    n21, n22, n23 = n2
    n31, n32, n33 = n3

    # No solution when k2 = 1 or k3 = 1: the corresponding vanishing point
    # is at infinity, i.e. that pair of board edges is already parallel in
    # the image, and the view carries no perspective to solve for f.
    if abs(n23) < 1e-9 or abs(n33) < 1e-9:
        # Fall back to the bounding-box-free planar case: without a focal
        # length the best available estimate is the projective-free ratio.
        f_squared = None
    else:
        # Equation (21).
        f_squared = -(1.0 / (n23 * n33 * s * s)) * (
            (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) * s * s
            + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))

    # A negative f^2 means the quadrangle is not a plausible projection of a
    # rectangle under these assumptions, and this view solves for nothing.
    solved = None if f_squared is None or f_squared <= 0 else float(np.sqrt(f_squared))

    # Whatever the caller knows beats this view, then this view, then a
    # nominal focal length so that rectification still produces something.
    used = focal or solved or float(max(width, height))

    A = np.array([[used, 0.0, u0],
                  [0.0, s * used, v0],
                  [0.0, 0.0, 1.0]])
    A_inv = np.linalg.inv(A)
    B = A_inv.T @ A_inv  # A^-T A^-1

    # Equation (20).
    numerator = n2 @ B @ n2
    denominator = n3 @ B @ n3
    if denominator <= 1e-12 or numerator <= 0:
        return None, used, solved
    return float(np.sqrt(numerator / denominator)), used, solved


# ---------------------------------------------------------------------------
# 3.3  Rectification
# ---------------------------------------------------------------------------

def order_corners(corners):
    """Order corners clockwise from the top-left, as the rest of the code expects."""
    centre = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    ordered = corners[np.argsort(angles)]
    # Rotate so that the corner closest to the top-left of the image is first.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def rectify(frame, corners, aspect_ratio, max_side=1200):
    """Warp the quadrangle to a rectangle of the estimated aspect ratio.

    The output size follows section 3.3: with W1, W2 the top and bottom side
    lengths and H1, H2 the left and right ones, let W_hat = max(W1, W2),
    H_hat = max(H1, H2) and r_hat = W_hat / H_hat. If r_hat >= r the height is
    scaled down (W = W_hat, H = W / r), otherwise the width is (H = H_hat,
    W = r H). That guarantees every original pixel maps to at least one
    rectified pixel.
    """
    w1 = np.linalg.norm(corners[1] - corners[0])
    w2 = np.linalg.norm(corners[2] - corners[3])
    h1 = np.linalg.norm(corners[3] - corners[0])
    h2 = np.linalg.norm(corners[2] - corners[1])
    w_hat, h_hat = max(w1, w2), max(h1, h2)
    if h_hat < 1e-6:
        return None
    r_hat = w_hat / h_hat

    if r_hat >= aspect_ratio:
        out_w, out_h = w_hat, w_hat / aspect_ratio
    else:
        out_h, out_w = h_hat, aspect_ratio * h_hat

    scale = min(1.0, max_side / max(out_w, out_h))
    out_w = max(int(round(out_w * scale)), 2)
    out_h = max(int(round(out_h * scale)), 2)

    destination = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(frame, H, (out_w, out_h), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# 3.4  White balancing and image enhancement
# ---------------------------------------------------------------------------

def estimate_whiteboard_color(image):
    """The blank-whiteboard image, i.e. the incident light C_light per pixel.

    Computed on a coarse grid because the blank board colour varies smoothly:
    split into 15x15 cells, and within each cell average the colours of the
    pixels in the top 25th percentile of luminance (ink absorbs light, so the
    board itself is the bright end). Cells entirely covered by strokes give a
    wrong colour and are rejected as outliers against a locally fitted plane,
    then replaced by interpolation from their neighbours.
    """
    height, width = image.shape[:2]
    rows = max(height // CELL_SIZE, 1)
    cols = max(width // CELL_SIZE, 1)

    # Crop to a whole number of cells so the image reshapes into a
    # (rows, cols, pixels-per-cell) block and every cell's percentile is taken
    # in one vectorised pass; a Python loop over the ~1000 cells of a rectified
    # board costs more than the rest of the enhancement together. The few
    # leftover pixels at the right and bottom edges are covered by the resize
    # back to full size at the end.
    cell_h, cell_w = height // rows, width // cols
    used_h, used_w = rows * cell_h, cols * cell_w
    cropped = image[:used_h, :used_w]
    luminance = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    blocks = (cropped.reshape(rows, cell_h, cols, cell_w, 3)
              .transpose(0, 2, 1, 3, 4).reshape(rows, cols, cell_h * cell_w, 3))
    block_luminance = (luminance.reshape(rows, cell_h, cols, cell_w)
                       .transpose(0, 2, 1, 3).reshape(rows, cols, cell_h * cell_w))

    # Average the colours of the pixels in the top percentile of luminance.
    cutoff = np.percentile(block_luminance, 100 - TOP_PERCENTILE,
                           axis=2, keepdims=True)
    bright = (block_luminance >= cutoff)[..., None]
    counts = np.maximum(bright.sum(axis=2), 1)
    cells = (np.where(bright, blocks, 0).sum(axis=2) / counts).astype(np.float32)

    # Reject cells that sit far below a locally fitted plane in RGB space.
    # A large-kernel median is a robust stand-in for the local plane fit and
    # keeps this fast enough to run per frame.
    kernel = 5 if min(rows, cols) >= 5 else 3
    if min(rows, cols) >= 3:
        smoothed = cv2.medianBlur(cells, kernel)
        deviation = cells.mean(axis=2) - smoothed.mean(axis=2)
        outliers = deviation < -max(10.0, 2.0 * np.std(deviation))
        if np.any(outliers):
            cells[outliers] = smoothed[outliers]

    cells = cv2.GaussianBlur(cells, (3, 3), 0) if min(rows, cols) >= 3 else cells
    return cv2.resize(cells, (width, height), interpolation=cv2.INTER_LINEAR)


_S_CURVE = np.array(
    [255.0 * (0.5 - 0.5 * np.cos(((i / 255.0) ** S_CURVE_P) * np.pi))
     for i in range(256)], np.float32).clip(0, 255).astype(np.uint8)


def enhance(image):
    """Make the background uniformly white and saturate the pen strokes.

    Step 1: C_out = min(1, C_input / C_light), scaling every pixel by its
    cell's whiteboard colour.
    Step 2: remap each channel through the S-shaped curve
    0.5 - 0.5 cos(C_out^p * pi) with p = 0.75, which suppresses sensor noise
    near white and deepens the strokes.
    """
    light = estimate_whiteboard_color(image)
    scaled = np.divide(image.astype(np.float32), np.maximum(light, 1.0))
    out = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(out, _S_CURVE)


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------

def draw_overlay(frame, corners, aspect_ratio, focal, quality_text):
    """Camera view with the fitted quadrangle and its corners drawn on."""
    view = frame.copy()
    if corners is not None:
        polygon = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in corners.astype(np.int32):
            cv2.circle(view, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"aspect {aspect_ratio:.3f}  f {focal:.0f}px" if aspect_ratio else "aspect n/a"
        cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(view, "no whiteboard detected", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(view, quality_text, (10, view.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return view


def hough_image(accumulator):
    """The Hough space as a displayable image, rho horizontal and theta vertical."""
    if accumulator is None:
        return None
    normalised = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(normalised.astype(np.uint8), (480, 240))


def save_screens(root, overlay, rectified, hough, edges, frame, small,
                 corners, aspect, focal, detected_this_frame, elapsed,
                 enhancement, scale):
    """Write every view plus a JSON record of the detection under `root`.

    One timestamp names all the files of a capture, so a record and the images
    it describes line up by filename across the folders. A view that does not
    exist for this frame -- no board found, so no rectified image -- is simply
    not written, and the JSON says so rather than a missing file being the
    only clue.

    The untouched frame is saved alongside the annotated one. Everything else
    here is a view of a decision already taken, so a saved run can be looked
    at but not re-run; the raw frame is what lets a parameter change be
    replayed against the exact images that misbehaved.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    views = {
        "frame": frame,
        "camera_detection": overlay,
        "rectified_whiteboard": rectified,
        "hough": hough,
        "edges": edges,
    }

    written = []
    for folder, image in views.items():
        if image is None:
            continue
        directory = os.path.join(root, folder)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{stamp}.png")
        if cv2.imwrite(path, image):
            written.append(folder)

    # Everything needed to reproduce or audit the detection: the geometry the
    # paper's sections produce, and the settings that produced it.
    record = {
        "timestamp": stamp,
        "captured_at": datetime.now().astimezone().isoformat(),
        "detected": corners is not None,
        "detected_this_frame": bool(detected_this_frame),
        "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "detection": {
            "scale": float(scale),
            "width": int(small.shape[1]),
            "height": int(small.shape[0]),
            "seconds": float(elapsed),
            "fps": float(1.0 / elapsed) if elapsed > 0 else None,
        },
        "enhancement": bool(enhancement),
        "views": {name: (f"{name}/{stamp}.png" if name in written else None)
                  for name in views},
        "parameters": {
            "gradient_threshold": GRADIENT_THRESHOLD,
            "rho_bin": RHO_BIN,
            "theta_bin_degrees": round(float(np.rad2deg(THETA_BIN)), 6),
            "peak_fraction": PEAK_FRACTION,
            "opposite_tolerance_degrees": round(float(np.rad2deg(OPPOSITE_TOLERANCE)), 6),
            "perpendicular_tolerance_degrees": round(
                float(np.rad2deg(PERPENDICULAR_TOLERANCE)), 6),
            "support_distance": SUPPORT_DISTANCE,
            "refine_neighbourhood": REFINE_NEIGHBOURHOOD,
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_tie": QUALITY_TIE,
            "cell_size": CELL_SIZE,
            "top_percentile": TOP_PERCENTILE,
            "s_curve_p": S_CURVE_P,
        },
    }

    if corners is not None:
        # Corners are in full-resolution frame coordinates, clockwise from the
        # top-left, matching m1..m4 in the paper's section 3.2 after ordering.
        ordered = [[float(x), float(y)] for x, y in corners]
        sides = [float(np.linalg.norm(corners[(k + 1) % 4] - corners[k]))
                 for k in range(4)]
        record["quadrangle"] = {
            "corners": ordered,
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "side_lengths": {"top": sides[0], "right": sides[1],
                             "bottom": sides[2], "left": sides[3]},
            "circumference": float(sum(sides)),
        }
        record["estimates"] = {
            "aspect_ratio": float(aspect) if aspect is not None else None,
            "focal_length_pixels": float(focal) if focal is not None else None,
            "principal_point": [float(frame.shape[1]) / 2.0,
                                float(frame.shape[0]) / 2.0],
        }
        if rectified is not None:
            record["rectified"] = {"width": int(rectified.shape[1]),
                                   "height": int(rectified.shape[0])}

    directory = os.path.join(root, "data")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{stamp}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)

    return f"{stamp} -> {', '.join(written) if written else 'data only'} (+data)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="downscale factor for detection; rectification "
                             "always uses the full-resolution frame")
    parser.add_argument("--no-enhance", action="store_true",
                        help="start with section 3.4 enhancement off")
    parser.add_argument("--debug", action="store_true",
                        help="start with edge and Hough views shown")
    parser.add_argument("--logs", default="logs",
                        help="directory the 's' key saves screens and data to")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    enhancement = not args.no_enhance
    debug = args.debug
    # The board is static relative to the camera over short intervals, so the
    # last good detection carries the rectified view through dropped frames.
    last_corners, last_aspect, last_focal = None, None, None
    solved_focals = deque(maxlen=30)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if args.scale != 1.0:
            small = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        else:
            small = frame

        tick = cv2.getTickCount()
        corners, mask, accumulator = detect_whiteboard(small)
        elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency()

        if corners is not None:
            corners = order_corners(corners / args.scale)
            # The lens does not change, so the median of what past views
            # solved for is a better focal length than this view's own.
            steady = float(np.median(solved_focals)) if solved_focals else None
            aspect, focal, solved = estimate_aspect_ratio(
                corners, frame.shape[1], frame.shape[0], steady)
            if solved is not None:
                solved_focals.append(solved)
            if aspect is not None:
                last_corners, last_aspect, last_focal = corners, aspect, focal

        status = f"{1.0 / max(elapsed, 1e-6):.1f} fps detection | " \
                 f"enhance {'on' if enhancement else 'off'} (e) | " \
                 f"save (s) | debug (d) | quit (q)"
        overlay = draw_overlay(frame, last_corners, last_aspect, last_focal, status)
        cv2.imshow("camera + detection", overlay)

        rectified = None
        if last_corners is not None and last_aspect is not None:
            rectified = rectify(frame, last_corners, last_aspect)
            if rectified is not None:
                if enhancement:
                    rectified = enhance(rectified)
                cv2.imshow("rectified whiteboard", rectified)

        # Built every frame, not only under --debug, so that pressing 's' saves
        # the same four views whether or not the debug windows are open.
        edges = (mask * 255).astype(np.uint8) if mask is not None else None
        hough = hough_image(accumulator)

        if debug:
            if edges is not None:
                cv2.imshow("edges", edges)
            if hough is not None:
                cv2.imshow("hough (rho x, theta y)", hough)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("e"):
            enhancement = not enhancement
        if key == ord("s"):
            saved = save_screens(args.logs, overlay, rectified, hough, edges,
                                 frame, small, last_corners, last_aspect,
                                 last_focal, corners is not None, elapsed,
                                 enhancement, args.scale)
            print(f"saved {saved}")
        if key == ord("d"):
            debug = not debug
            if not debug:
                cv2.destroyWindow("edges")
                cv2.destroyWindow("hough (rho x, theta y)")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
