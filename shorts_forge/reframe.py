"""Reframe 16:9 footage to 9:16 by following the speaker's face.

1. Sample frames inside the clip (default 4 per second) with OpenCV.
2. Detect faces: YuNet (a 230 KB ONNX model bundled with the package, OpenCV >= 4.8),
   or a Haar cascade when YuNet is unavailable. No detector / no OpenCV -> center crop.
3. Follow one subject (largest confident face, preferring the one nearest the last
   position), fill short gaps, smooth with a moving median + exponential filter, and
   apply a dead zone so the virtual camera holds still until the subject really moves.
4. Compress the path to a few keyframes and emit a piecewise-linear ffmpeg ``crop``
   expression, so the actual cropping is done by ffmpeg in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    cx: float  # center x, pixels in the original frame
    cy: float
    w: float
    h: float
    score: float


@dataclass
class CropPlan:
    """Where the 9:16 window sits over time (times relative to the clip start)."""

    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    keyframes: list[tuple[float, int]] = field(default_factory=list)  # (t, x) or (t, y)
    axis: str = "x"  # "x" pans horizontally (wide source), "y" vertically (tall source)
    mode: str = "center"  # face | center
    detector: str = "none"
    face_coverage: float = 0.0  # share of sampled frames with a face

    def offset_at(self, t: float) -> float:
        k = self.keyframes
        if not k:
            return 0.0
        if t <= k[0][0]:
            return float(k[0][1])
        for (t0, x0), (t1, x1) in zip(k, k[1:]):
            if t <= t1:
                return x0 + (x1 - x0) * (t - t0) / max(1e-9, t1 - t0)
        return float(k[-1][1])

    def ffmpeg_filter(self) -> str:
        """``crop=...`` filter string; the offset is a piecewise-linear function of ``t``."""
        expr = piecewise_expr(self.keyframes)
        if self.axis == "x":
            return f"crop=w={self.crop_w}:h={self.crop_h}:x='{expr}':y=0"
        return f"crop=w={self.crop_w}:h={self.crop_h}:x=0:y='{expr}'"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "detector": self.detector, "axis": self.axis,
                "crop": [self.crop_w, self.crop_h], "source": [self.src_w, self.src_h],
                "face_coverage": round(self.face_coverage, 3),
                "keyframes": [[round(t, 3), int(v)] for t, v in self.keyframes]}


# --------------------------------------------------------------------------- geometry

def crop_size(src_w: int, src_h: int, aspect: float = 9 / 16) -> tuple[int, int, str]:
    """Largest window of the target aspect that fits the source (even sizes for x264)."""
    if src_w / src_h > aspect:
        h = src_h - src_h % 2
        w = int(round(h * aspect))
        return w - w % 2, h, "x"
    w = src_w - src_w % 2
    h = int(round(w / aspect))
    return w, min(h - h % 2, src_h - src_h % 2), "y"


def piecewise_expr(keyframes: list[tuple[float, int]]) -> str:
    """Nested ffmpeg ``if(lt(t,..))`` expression interpolating linearly between keyframes."""
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return str(int(keyframes[0][1]))
    expr = str(int(keyframes[-1][1]))
    for (t0, x0), (t1, x1) in reversed(list(zip(keyframes, keyframes[1:]))):
        if x0 == x1:
            seg = str(int(x0))
        else:
            slope = (x1 - x0) / max(1e-6, t1 - t0)
            seg = f"{int(x0)}+({slope:.4f})*(t-{t0:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


# --------------------------------------------------------------------------- tracking

def fill_gaps(track: list[float | None], max_gap: int = 8) -> list[float | None]:
    """Linearly bridge gaps up to ``max_gap`` samples; hold the edge values at the ends."""
    out = list(track)
    known = [i for i, v in enumerate(out) if v is not None]
    if not known:
        return out
    for i in range(known[0]):
        out[i] = out[known[0]]
    for i in range(known[-1] + 1, len(out)):
        out[i] = out[known[-1]]
    for a, b in zip(known, known[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        va, vb = out[a], out[b]
        for k in range(1, gap + 1):
            out[a + k] = va + (vb - va) * k / (gap + 1) if gap <= max_gap else (va if k <= gap / 2 else vb)
    return out


def smooth_path(values: list[float], window: int = 5, alpha: float = 0.35,
                deadzone: float = 0.0) -> list[float]:
    """Median filter (kills detector jitter) -> EMA (eases motion) -> dead zone (holds still)."""
    if not values:
        return []
    half = window // 2
    med = []
    for i in range(len(values)):
        win = sorted(values[max(0, i - half): i + half + 1])
        med.append(win[len(win) // 2])
    ema = [med[0]]
    for v in med[1:]:
        ema.append(ema[-1] + alpha * (v - ema[-1]))
    if deadzone <= 0:
        return ema
    out = [ema[0]]
    anchor = ema[0]
    moving = False
    for v in ema[1:]:
        if not moving and abs(v - anchor) > deadzone:
            moving = True
        if moving:
            anchor = v
            # settle once the subject slows down
            if len(out) >= 1 and abs(v - out[-1]) < deadzone * 0.08:
                moving = False
        out.append(anchor)
    return out


def simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker on (t, x) so the ffmpeg expression stays short."""
    if len(points) < 3:
        return list(points)
    (t0, x0), (t1, x1) = points[0], points[-1]
    worst, idx = -1.0, 0
    for i in range(1, len(points) - 1):
        t, x = points[i]
        interp = x0 + (x1 - x0) * (t - t0) / max(1e-9, t1 - t0)
        d = abs(x - interp)
        if d > worst:
            worst, idx = d, i
    if worst <= tolerance:
        return [points[0], points[-1]]
    left = simplify(points[: idx + 1], tolerance)
    return left[:-1] + simplify(points[idx:], tolerance)


def choose_face(faces: list[Face], prev_cx: float | None, frame_w: float) -> Face | None:
    """Biggest confident face, with a bonus for staying near the previous subject."""
    if not faces:
        return None

    def rank(f: Face) -> float:
        r = f.w * f.h * (0.5 + f.score)
        if prev_cx is not None:
            r *= 1.0 + 1.5 * max(0.0, 1 - abs(f.cx - prev_cx) / (0.25 * frame_w))
        return r

    return max(faces, key=rank)


def plan_from_track(track: list[tuple[float, float | None]], src_w: int, src_h: int,
                    detector: str = "none", fps_samples: float = 4.0,
                    deadzone_frac: float = 0.06, headroom: float = 0.0) -> CropPlan:
    """Turn sampled face centers (None = no face) into a smoothed crop plan."""
    cw, ch, axis = crop_size(src_w, src_h)
    span = (src_w - cw) if axis == "x" else (src_h - ch)
    size = cw if axis == "x" else ch
    found = [c for _, c in track if c is not None]
    coverage = len(found) / len(track) if track else 0.0
    if span <= 0 or not found or coverage < 0.15:
        mid = max(0, span // 2)
        return CropPlan(src_w, src_h, cw, ch, [(0.0, mid)], axis, "center", detector, coverage)
    filled = fill_gaps([c for _, c in track], max_gap=int(2 * fps_samples))
    centers = smooth_path([float(v) for v in filled], window=5, alpha=0.35,
                          deadzone=deadzone_frac * (src_w if axis == "x" else src_h))
    pts = []
    for (t, _), c in zip(track, centers):
        off = c - size / 2 - (headroom * size if axis == "y" else 0)
        pts.append((t, float(min(max(0.0, off), span))))
    keys = simplify(pts, tolerance=max(2.0, 0.01 * src_w))
    return CropPlan(src_w, src_h, cw, ch, [(round(t, 3), int(round(x))) for t, x in keys],
                    axis, "face", detector, coverage)


# --------------------------------------------------------------------------- detectors

class FaceDetector:
    name = "none"

    def detect(self, frame) -> list[Face]:  # pragma: no cover - interface
        return []


def _quiet_opencv(cv2) -> None:
    """OpenCV 5 logs a harmless backend warning when YuNet loads; keep the CLI output clean."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:  # pragma: no cover - older builds without the logging module
        pass


class YuNetDetector(FaceDetector):
    name = "yunet"

    def __init__(self, model: Path = YUNET_MODEL, score: float = 0.6, max_side: int = 640):
        import cv2

        self.cv2 = cv2
        self.max_side = max_side
        _quiet_opencv(cv2)
        self.net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score, 0.3, 50)

    def detect(self, frame) -> list[Face]:
        h, w = frame.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = self.cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, res = self.net.detect(small)
        if res is None:
            return []
        return [Face((r[0] + r[2] / 2) / s, (r[1] + r[3] / 2) / s, r[2] / s, r[3] / s, float(r[-1]))
                for r in res]


class HaarDetector(FaceDetector):
    name = "haar"

    def __init__(self, cascade: Path, max_side: int = 640):
        import cv2

        self.cv2 = cv2
        self.max_side = max_side
        self.clf = cv2.CascadeClassifier(str(cascade))
        if self.clf.empty():
            raise RuntimeError(f"cannot load Haar cascade {cascade}")

    def detect(self, frame) -> list[Face]:
        h, w = frame.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = self.cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        gray = self.cv2.cvtColor(small, self.cv2.COLOR_BGR2GRAY)
        boxes = self.clf.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        return [Face((x + bw / 2) / s, (y + bh / 2) / s, bw / s, bh / s, 0.8)
                for x, y, bw, bh in boxes]


def _haar_path() -> Path | None:
    try:
        import cv2
    except ImportError:
        return None
    base = Path(getattr(getattr(cv2, "data", None), "haarcascades", "") or ".")
    p = base / "haarcascade_frontalface_default.xml"
    return p if p.is_file() else None


def make_detector(kind: str = "auto") -> FaceDetector | None:
    """``auto`` | ``yunet`` | ``haar`` | ``none``. Returns None when nothing is usable."""
    if kind == "none":
        return None
    try:
        import cv2
    except ImportError:
        if kind != "auto":
            raise RuntimeError("face tracking needs OpenCV: pip install opencv-python-headless")
        return None
    if kind in ("auto", "yunet") and hasattr(cv2, "FaceDetectorYN") and YUNET_MODEL.is_file():
        try:
            return YuNetDetector()
        except Exception:  # pragma: no cover - old/odd OpenCV builds
            if kind == "yunet":
                raise
    if kind in ("auto", "haar"):
        p = _haar_path()
        if p:
            return HaarDetector(p)
    if kind != "auto":
        raise RuntimeError(f"face detector {kind!r} is not available in this OpenCV build")
    return None


def sample_track(video: Path, start: float, end: float, detector: FaceDetector,
                 fps_samples: float = 4.0) -> tuple[list[tuple[float, float | None]], int, int]:
    """Detect the subject at ``fps_samples`` per second between start and end."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_samples)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    track: list[tuple[float, float | None]] = []
    prev: float | None = None
    idx = 0
    horizontal = w / h > 9 / 16
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end + 1e-3:
            break
        if idx % step == 0 and t >= start - 0.5:
            ok, frame = cap.retrieve()
            if ok:
                face = choose_face(detector.detect(frame), prev, w)
                c = None
                if face:
                    c = face.cx if horizontal else face.cy
                    prev = face.cx
                track.append((max(0.0, t - start), c))
        idx += 1
    cap.release()
    return track, w, h


def plan_crop(video: Path, start: float, end: float, src_size: tuple[int, int],
              layout: str = "auto", detector: FaceDetector | None = None,
              fps_samples: float = 4.0) -> CropPlan:
    """Build the crop plan for one clip. ``layout``: auto | face | center."""
    w, h = src_size
    if layout == "center" or detector is None:
        return plan_from_track([], w, h, "none")
    track, w2, h2 = sample_track(video, start, end, detector, fps_samples)
    plan = plan_from_track(track, w2 or w, h2 or h, detector.name, fps_samples)
    return plan


def describe(plan: CropPlan) -> str:
    if plan.mode == "center":
        return f"center crop ({plan.detector}, faces in {plan.face_coverage:.0%} of samples)"
    xs = [v for _, v in plan.keyframes]
    return (f"face track via {plan.detector}: {len(plan.keyframes)} keyframes, "
            f"{plan.axis} range {min(xs)}-{max(xs)}px, faces in {plan.face_coverage:.0%} of samples")


__all__ = ["CropPlan", "Face", "crop_size", "piecewise_expr", "fill_gaps", "smooth_path",
           "simplify", "choose_face", "plan_from_track", "make_detector", "plan_crop",
           "describe"]
