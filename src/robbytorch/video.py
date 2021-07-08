"""Utils for handling video files and streams."""
import pathlib
import time
import warnings
from typing import cast, List, Optional, Tuple

import numpy as np
import cv2

from .utils import PathLike


class VideoCapture:
    """A thin wrapper simplifying the use of `cv2.VideoCapture`.

    Actually I would recommend using pyav instead, since it seems to have much better error
    handling (and is more pythonic overall).
    """

    def __init__(self, path: Optional[PathLike] = None, rgb: bool = False):
        self.path: Optional[PathLike] = path
        self.rgb: bool = rgb  # Whether to convert BGR colors to RGB.
        self.width: int = 0
        self.height: int = 0
        # Note the actual number of retriavable frames may be different, and there's no way to tell.
        # It seems there's either `n_frames` or `n_frames - 1` retrievable frames?
        self.n_frames: int = 0
        self.frame_rate: float = 0.0
        # Approximate video length in float seconds (equals `n_frames / self.frame_rate`).
        self.length: float = 0.0
        # 4-character code describing video codec.
        # See `https://en.wikipedia.org/wiki/FourCC`.
        self.fourcc: bytes = b""

        self.cur_frame: int = 0  # Current position in frames (e.g. 1 after getting first frame).
        self.cur_time: float = 0.0  # Current position in float seconds.

        self.capture = cv2.VideoCapture()
        self.capture.setExceptionMode(True)
        self.prev_frame_time: float = 0.0

        if path:
            self.open(path)

    def open(self, path: PathLike):
        self.path = pathlib.Path(path).expanduser()
        if not self.path.is_file():
            raise RuntimeError(f"Video file does not exists: {path}")
        r = self.capture.open(str(self.path))
        assert r  # We setExceptionMode(True), so this should never be False.

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        self.length = self.n_frames / self.frame_rate if self.frame_rate else 0
        self.fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder="little")

        if not self.n_frames:
            raise RuntimeError("Failed to load {path}.")

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Return next frame (shape HWC, uint8 BGR) or None if stream ended.

        This grabs and retrieves (decodes) the next frame, updating `cur_frame` and `cur_time`.

        In opencv, there's apparently no way to tell the difference between errors and actual
        end-of-stream, even for simple mp4 files. We assume everything's fine if we retrieved
        `n_frames` or `n_frames - 1` frames (or if `n_frames` is zero), otherwise we warn.

        The returned ndarray has shape HWC, dtype uint8, BGR format (unless self.rgb is True).
        """
        try:
            success, frame = self.capture.read()
        except cv2.error:
            # There's apparently no way to tell the difference between errors and end-of-stream.
            # One missing frame seems to be normal, warn if more.
            if self.n_frames and self.cur_frame < self.n_frames - 1:
                warnings.warn(
                    f"Failed to read after {self.cur_frame} frames of {self.n_frames}.",
                    stacklevel=2,
                )
                raise
            return None
        assert success is True and isinstance(frame, np.ndarray)
        assert frame.shape == (self.height, self.width, 3)
        # Update `cur_time` and `cur_frame`.
        self.cur_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        if self.n_frames and self.cur_frame > self.n_frames:
            raise RuntimeError(f"Read too many: {self.cur_frame} frames out of {self.n_frames}.")
        self.cur_time = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def goto_frame(self, frame_id: int):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    def goto_time(self, seconds: float):
        self.capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)

    def release(self):
        """Closes video file or capturing device, deallocating memory.

        This is automatically called when `open()`ing another video and on destruction."""
        self.capture.release()
        self.path = None

    def __repr__(self) -> str:
        max_len = 30
        path = str(self.path)
        if len(path) > max_len:
            path = "..." + path[-(max_len - 3):]
        return (
            f"<VideoCapture {path} {self.width}x{self.height} {self.fourcc!r} "
            f"{self.length}s {self.frame_rate} fps {self.n_frames} frames>"
        )

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, frame_id: int) -> np.ndarray:
        if frame_id < 0:
            frame_id += self.n_frames
        if self.cur_frame != frame_id:
            self.goto_frame(frame_id)
        frame = self.get_next_frame()
        if frame is None:
            raise KeyError(f"Invalid frame index: {frame_id} out of {self.n_frames}.")
        return frame

    def check(self) -> None:
        """Check exact number and shape of all frames, raise AssertionError if invalid."""
        real_n_frames = 0
        while True:
            frame = self.get_next_frame()
            if frame is None:
                break
            real_n_frames += 1
            assert frame.dtype == np.uint8
            assert frame.shape == (self.height, self.width, 3)
        assert self.n_frames == real_n_frames
        self.goto_frame(0)


class Rect:
    """A simple structure for axis-parallel rectangles.

    Rect always has satisfies:
        - `left <= right and top <= bottom`
        - `left + width == right`
        - `top + height == bottom`
    """

    def __init__(
        self,
        left: Optional[int] = None,
        top: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        right: Optional[int] = None,
        bottom: Optional[int] = None,
    ):
        def complete(
            lo: Optional[int], d: Optional[int], hi: Optional[int]
        ) -> Tuple[int, int, int]:
            if [hi is not None, d is not None, lo is not None].count(True) != 2:
                raise ValueError("Exactly 2 of left/width/right and top/height/bottom expected.")
            if hi is None:
                hi = cast(int, lo) + cast(int, d)
            if lo is None:
                lo = cast(int, hi) - cast(int, d)
            if d is None:
                d = cast(int, hi) - cast(int, lo)
            return (lo, d, hi)

        (left, width, right) = complete(left, width, right)
        (top, height, bottom) = complete(top, height, bottom)
        self.left: int = left
        self.width: int = width
        self.right: int = right
        self.top: int = top
        self.height: int = height
        self.bottom: int = bottom
        if self.width < 0:
            raise ValueError("`left` must be smaller than `right`.")
        if self.height < 0:
            raise ValueError("`top` must be smaller than `bottom`.")

    @property
    def left_top(self) -> Tuple[int, int]:
        return (self.left, self.top)

    @property
    def right_bottom(self) -> Tuple[int, int]:
        return (self.right, self.bottom)

    @property
    def left_bottom(self) -> Tuple[int, int]:
        return (self.left, self.bottom)

    @property
    def top_bottom(self) -> slice:
        return slice(self.top, self.bottom)

    @property
    def left_right(self) -> slice:
        return slice(self.left, self.right)

    def contains(self, other: "Rect") -> bool:
        return (
            other.left >= self.left
            and other.right <= self.right
            and other.top >= self.top
            and other.bottom <= self.bottom
        )

    def relative_to(self, other: "Rect") -> "Rect":
        """Return `self` with coordinates relative to `other`, assuming it contains `self`."""
        if not other.contains(self):
            raise ValueError("self.relative_to(other): self should be contained in other.")
        return Rect(
            width=self.width,
            height=self.height,
            left=self.left - other.left,
            top=self.top - other.top,
        )

    def view(self, array: np.ndarray) -> np.ndarray:
        """Return a view of `array` cropped to `self`."""
        if len(array.shape) == 3:
            return array[self.top_bottom, self.left_right, :]
        elif len(array.shape) == 2:
            return array[self.top_bottom, self.left_right]
        elif len(array.shape) == 4:
            return array[:, self.top_bottom, self.left_right, :]
        else:
            raise ValueError("Expected 2, 3, or 4 dimensional array.")

    def draw(self, img: np.ndarray, color, thickness: float = 1) -> None:
        """Draw a border around `self` in `array`."""
        cv2.rectangle(img, self.left_top, self.right_bottom, color, thickness)

    def fill(self, img: np.ndarray, color) -> None:
        """Fill `self` rectangle in `array`."""
        self.view(img)[...] = color
        # cv2.rectangle(img, self.left_top, self.right_bottom, color, cv2.FILLED)


def put_text(
    img: np.ndarray,
    text: str,
    left_bottom: Tuple[int, int],
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.0,
    line_type=1,
) -> None:
    """Thin wrapper around cv2.putText with some sane defaults."""
    cv2.putText(img, text, left_bottom, font, scale, color, line_type)


class RateMeasurer:
    """A simple class for measuring the rate (ticks per second) of anything.

    This is estimated as 1 over exponentially decaying average of time between calls to `tick()`.
    """

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.prev_tick: float = 0.0
        self.avg_time: float = float("nan")  # Average time between ticks.

    def tick(self) -> float:
        """Return ticks-per-second estimate, or zero on first tick."""
        now = time.monotonic()
        if not self.prev_tick:
            self.prev_tick = now
            return 0.0
        if not self.avg_time or np.isnan(self.avg_time) or not self.momentum:
            self.avg_time = now - self.prev_tick
        else:
            self.avg_time = self.momentum * self.avg_time + (1 - self.momentum) * (
                now - self.prev_tick
            )
        self.prev_tick = now
        return 1.0 / self.avg_time


def get_ffmpeg_encode_args(
    format: str = "H264", crf: Optional[float] = None
) -> Tuple[List[str], str]:
    """Return a reasonable ffmpeg argument list and extension for a given video format.

    Use "H264" for good defaults or "FFV1" for actually lossless.
    "H265" seems bad for small height x width.

    Args:
        - format: one of "VP9", "H264", "H265" ,"FFV1" (or veriations thereof).
        - crf: "Constant Rate Factor", a quality param available for many codecs, lower is better.
            The default is some choice that is closer to visually lossless than typically.
            0 is interpreted as (actually) lossless when possible, though you might need to avoid
            RGB to YUV color space conversion as well to make it actually lossless.
            Available ranges, recommended ranges, default recommended values, and values we use are:
            - VP9: 0-63, 15-35, 31, 30.
            - H264: 0-51, 17-?, 23, 18.
            - H265: 0-51, ?-?, 28, 18.
            - FFV1: always lossless.

    For more details on e.g. recommended crf values and other available options:
        https://trac.ffmpeg.org/wiki/Encode/H.264
        https://trac.ffmpeg.org/wiki/Encode/H.265
        https://trac.ffmpeg.org/wiki/Encode/VP9
        https://developers.google.com/media/vp9/settings/vod/
        https://trac.ffmpeg.org/wiki/Encode/FFV1

    Returns `args, ext`, where:
        - args: a list of arguments describing the encoding for ffmpeg.
        - ext: a file extension appropriate for this format, like "mp4", "webm", "mkv".
            Give ffmpeg an output path with this extension (so that it uses a compatible container).
    """
    format = format.upper()
    if format in ["VP9", "VP90"]:
        ext = "webm"
        args = "-c:v vp9".split()
        if crf is None:
            crf = 30
        if not crf:
            args += "-lossless 1".split()
        else:
            args += f"-b:v 0 -crf {crf}".split()
    elif format in ["H264", "H.264", "X264"]:
        ext = "mp4"
        args = "-c:v libx264 -preset slower -tune fastdecode".split()
        if crf is None:
            crf = 18
        args += f"-crf {crf}".split()
    elif format in ["H265", "H.265", "X265", "HEVC"]:
        ext = "mp4"
        args = "-c:v libx265 -preset slow -tune fastdecode".split()
        if crf is None:
            crf = 18
        params = "log-level=error"
        # I can't find a way to prevent the encoder from using all availables cores,
        # none of these options work.
        # ":frame-threads=1:pools=none:wpp=0:pme=0:pmode=0:lookahead-slices=0:lookahead-threads=0"
        if not crf:
            args += f"-x265-params lossless=1:{params}".split()
        else:
            args += f"-crf {crf}".split()
            args += f"-x265-params {params}".split()
    elif format == "FFV1":
        ext = "mkv"
        args = "-c:v ffv1 -level 3".split()
        if crf:
            raise ValueError(f"Format FFV1 is always lossless, crf={crf} unsupported.")
    else:
        raise ValueError(f"Unknown format: {format}")
    return args, ext
