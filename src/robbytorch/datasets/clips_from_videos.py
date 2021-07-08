from typing import cast, Any, Callable, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms

from ..utils import PathLike
from ..video import VideoCapture


class ClipsFromVideos(torch.utils.data.Dataset):
    """A dataset consisting of short clips or frames from a given list of long videos.

    Given paths to videos and their classes, this dataset provides clips from these videos
    (as tensors consisting of `clip_length` frames), or individual frames. By default all
    possible clips of `clip_length` consecutive frames are returned.

    Args:
    - df: a `pd.DataFrame` whose rows describe videos;, with columns `path`, `class`, and
            optionally `n_frames`.
    - classes: List of class names to be used. This also becomes the index-to-label map
        (clips will be labelled with the index of their class in this list).

    - clip_length: Length of every clip in frames. Ignored if `return_images` is true.
    - clip_stride: Distance between first frames of consecutive clips. In other words,
        only clips whose first frame's index is divisible by `clip_stride` are used.
    - clip_offset: Changes the behaviour of `clip_stride` so that only clips whose first
        frame's index is congruent to `clip_offset` mod `min(clip_stride, len(video))` are used.
        If None, randomly select an offset once at `__init__`, independently for each video.
    - clip_dilation: Distance between consecutive frames taken into a clip.
    - random_clip_per_video: If true, return only one random clip per video per epoch.
    - return_images: If true, yield individual frames, without the time dimension.
        Overrides `clip_length=1`.

    - transform: If given, takes a single ndarray frame and returns a transformed torch tensor.
        Default: `torchvision.transforms.ToTensor`, returns 3xHxW tensors in range 0.0..1.0.
        (Frames before `transform` are ndarrays of shape HxWx3, dtype=uint8, 0..255 RGB.)
    """
    def __init__(self,
                 videos_df: pd.DataFrame,
                 classes: List[str],
                 clip_length: int = 8,
                 clip_stride: int = 1,
                 clip_offset: Optional[int] = 0,
                 clip_dilation: int = 1,
                 random_clip_per_video: bool = False,
                 return_images: bool = False,
                 transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None):
        self.videos_df = videos_df.reset_index()
        self.classes = list(classes)

        self.clip_length = 1 if return_images else clip_length
        self.clip_stride = clip_stride
        self.clip_offset = clip_offset
        self.clip_dilation = clip_dilation
        self.return_images = return_images
        self.random_clip_per_video = random_clip_per_video

        self.transform = transform or torchvision.transforms.ToTensor()

        self.class_to_idx: Dict[str, int] = {cls: i for i, cls in enumerate(self.classes)}

        # # A list mapping a video's index to a list of frame filenames.
        # # This is used when videos are ImageFolders, for faster access.
        # self.image_folders: List[List[str]] = []

        # A list mapping a video's index to a list of indices of frames that can be used as
        # first frames of clips (as specified by `clip_stride` and `clip_offset`).
        self.first_frame_indices: List[List[int]] = []
        # A list of all clips, as (video_id, first frame index) tuples.
        # If `random_clip_per_video` is True, this only lists (video_id, -1) pairs.
        self.clips: List[Tuple[int, int]] = []

        self.loaded_video = VideoCapture(rgb=True)
        self.loaded_path: PathLike = ""

        # Check arguments.
        if "class" not in videos_df.columns or "path" not in videos_df.columns:
            raise ValueError("videos_df should have 'class' and 'path' columns.")
        if "n_frames" not in videos_df.columns:
            videos_df["n_frames"] = None
        if min(self.clip_length, self.clip_stride, self.clip_dilation) < 1:
            raise ValueError("clip length, stride, duration should be >=1.")

        # Compute self.image_folders, self.first_frame_indices, and self.clips.
        for i, row in self.videos_df.iterrows():
            n_frames = cast(Optional[int], row["n_frames"])
            if n_frames is None:
                n_frames = self._get_n_frames(row["path"])
            self.add_video(i, n_frames)

    def _get_n_frames(self, path: PathLike) -> int:
        return VideoCapture(path).n_frames

    def add_video(self, video_id: int, n_frames: int):
        # Compute `first_frame_indices`.
        clip_span = (self.clip_length - 1) * self.clip_dilation + 1
        maximum = n_frames - clip_span  # last possible first_frame_index
        assert maximum >= 0, f"No clips got selected from a video of length {n_frames}."
        stride = min(self.clip_stride, maximum + 1)
        if self.clip_offset is not None:
            minimum = self.clip_offset % stride
        else:
            minimum = self._random_int(0, stride)
        lst = list(range(minimum, maximum + 1, stride))
        assert lst, f"No clips got selected from a video of length {n_frames}."
        self.first_frame_indices.append(lst)

        # Compute `clips`.
        if self.random_clip_per_video:
            self.clips.append((video_id, -1))
        else:
            for first_frame_index in lst:
                self.clips.append((video_id, first_frame_index))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a `(clip, class_id)` dataitem for a given integer key.

        `class_id` is the class index in `self.classes`.
        `clip` is a torch tensor of shape `clip_length` x 3 x H x W, float RGB,
            or just 3xHxW if `self.return_images` is True (assuming `transform` calls `ToTensor`).
        """
        video_id, first_frame_index = self.clips[index]
        row = self.videos_df.iloc[video_id]
        class_id = self.class_to_idx[row["class"]]
        path = row["path"]
        if first_frame_index == -1:
            first_frame_index = self._random_from(self.first_frame_indices[video_id])
        if self.loaded_path != path:
            self.loaded_video.open(path)
            self.loaded_path = path

        clip: List[torch.Tensor] = []
        for i in range(self.clip_length):
            frame_id = first_frame_index + i * self.clip_dilation
            frame = self.loaded_video[frame_id]
            clip.append(self.transform(frame))

        if self.return_images:
            return clip[0], class_id
        else:
            return torch.stack(clip), class_id

    def __len__(self) -> int:
        return len(self.clips)

    def _random_int(self, low, high) -> int:
        """Select an random int between `low` and `high-1` using `torch` randomness."""
        result = torch.randint(low, high, (1,))
        return int(result.item())

    def _random_from(self, lst: List[Any]):
        """Select a random element from a list using `torch` randomness."""
        return lst[self._random_int(0, len(lst))]
