"""Junction geometry (DECISIONS §12.1): lane-level paths through the box.

Traffic-engineering idea: a turning car follows a circular arc whose radius is set by
the kerb (corner) radius, so the box is the road width plus a corner radius on each side,
and every path through it has a true length and a curvature speed limit
v_cap = sqrt(a_lat * r).  Through traffic keeps its lane; a near-side turn leaves the kerb
lane and a far-side turn the median lane, and either may end in the exit lane its next
turn needs (a short straight lead-in or lead-out joins the arc to that lane).  Two paths
conflict where their corridors (widest vehicle + margin) touch; the zones are identical
at every junction and drive the clearance gate and the near-miss monitor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rot(h, degrees: float):
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    return (round(c * h[0] - s * h[1], 12), round(s * h[0] + c * h[1], 12))


@dataclass(frozen=True)
class BoxPath:
    """A lane-level path from the entry lane's box edge (p0, heading h0) to the exit lane's
    box edge (p1, heading h1): straight (turn_sign 0) or lead-in + quarter arc + lead-out."""
    approach: str
    movement: str
    in_lane: int
    out_lane: int
    p0: tuple
    h0: tuple
    p1: tuple
    h1: tuple
    radius: float          # math.inf for a straight path
    centre: tuple | None
    turn_sign: int         # +1 counter-clockwise, -1 clockwise, 0 straight
    lead_in: float         # straight part before the arc [m]
    lead_out: float        # straight part after the arc [m]
    length: float
    v_cap: float           # curvature speed limit [m/s]

    @property
    def key(self):
        return (self.approach, self.movement, self.in_lane, self.out_lane)

    @property
    def arc_start(self) -> float:
        return self.lead_in

    def pose(self, s: float):
        """(x, y, heading_rad) of the point at distance s along the path (extrapolated
        straight beyond either end, so a body that sticks out of the box stays on a line)."""
        h0 = math.atan2(self.h0[1], self.h0[0])
        if self.turn_sign == 0 or s <= self.lead_in:
            return self.p0[0] + self.h0[0] * s, self.p0[1] + self.h0[1] * s, h0
        arc_len = self.radius * math.pi / 2.0
        if s >= self.lead_in + arc_len:
            d = s - self.length
            return self.p1[0] + self.h1[0] * d, self.p1[1] + self.h1[1] * d, math.atan2(self.h1[1], self.h1[0])
        u = (s - self.lead_in) / self.radius
        cx, cy = self.centre
        sx, sy = self.p0[0] + self.h0[0] * self.lead_in, self.p0[1] + self.h0[1] * self.lead_in
        a0 = math.atan2(sy - cy, sx - cx)
        a = a0 + self.turn_sign * u
        return cx + self.radius * math.cos(a), cy + self.radius * math.sin(a), h0 + self.turn_sign * u

    def polyline(self, step: float = 0.25) -> np.ndarray:
        n = max(2, int(math.ceil(self.length / step)) + 1)
        return np.array([self.pose(s)[:2] for s in np.linspace(0.0, self.length, n)])


def make_path(approach, movement, in_lane, out_lane, p0, h0, p1, h1, a_lat: float, v_limit: float) -> BoxPath:
    """Straight line, or the largest quarter arc tangent to both lane centre lines joined to
    the farther one by a straight piece (lead-in if the entry line is longer, else lead-out)."""
    if abs(h0[0] - h1[0]) < 1e-9 and abs(h0[1] - h1[1]) < 1e-9:
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        return BoxPath(approach, movement, in_lane, out_lane, tuple(p0), tuple(h0), tuple(p1), tuple(h1),
                       math.inf, None, 0, 0.0, 0.0, length, v_limit)
    A = np.array([[h0[0], h1[0]], [h0[1], h1[1]]], dtype=float)
    t, u = np.linalg.solve(A, np.array(p1, dtype=float) - np.array(p0, dtype=float))
    if t <= 0 or u <= 0:
        raise ValueError(f"no turning path for {approach}{movement} lane {in_lane}->{out_lane}: t={t:.3f} u={u:.3f}")
    r = float(min(t, u))
    lead_in, lead_out = float(t - r), float(u - r)
    cross = h0[0] * h1[1] - h0[1] * h1[0]
    sign = 1 if cross > 0 else -1
    n = rot(h0, 90.0 * sign)
    sx, sy = p0[0] + h0[0] * lead_in, p0[1] + h0[1] * lead_in
    centre = (sx + n[0] * r, sy + n[1] * r)
    length = lead_in + r * math.pi / 2.0 + lead_out
    v_cap = min(v_limit, math.sqrt(a_lat * r))
    return BoxPath(approach, movement, in_lane, out_lane, tuple(p0), tuple(h0), tuple(p1), tuple(h1),
                   r, centre, sign, lead_in, lead_out, length, v_cap)


def _min_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """For every point of P, its distance to the nearest point of Q."""
    return np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1]).min(axis=1)


def zone_interval(a: BoxPath, b: BoxPath, clearance: float, step: float = 0.1):
    """(s_in, s_out) along `a` where its centre line is within `clearance` of `b`'s, or None."""
    S = np.arange(0.0, a.length + 1e-9, step)
    P = np.array([a.pose(s)[:2] for s in S])
    Q = b.polyline(step)
    near = _min_dist(P, Q) < clearance
    if not near.any():
        return None
    idx = np.nonzero(near)[0]
    return (max(0.0, float(S[idx[0]]) - step / 2), min(a.length, float(S[idx[-1]]) + step / 2))


def divergence_length(a: BoxPath, b: BoxPath, clearance: float, step: float = 0.1) -> float:
    """Distance along `a` after which `a` stays at least `clearance` away from `b` (paths that
    leave the same entry lane): a vehicle on `b` stops being a leader of one on `a` once its
    rear bumper has passed this point on its own path."""
    S = np.arange(0.0, a.length + 1e-9, step)
    P = np.array([a.pose(s)[:2] for s in S])
    near = _min_dist(P, b.polyline(step)) < clearance
    if not near.any():
        return 0.0
    return min(a.length, float(S[np.nonzero(near)[0][-1]]) + step)
