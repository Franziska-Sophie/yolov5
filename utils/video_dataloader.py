import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import ThreadPool
from pathlib import Path
import psutil
from tqdm import tqdm

from utils.torch_utils import torch_distributed_zero_first
from utils.general import LOGGER, NUM_THREADS, TQDM_BAR_FORMAT
from utils.dataloaders import SmartDistributedSampler

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class VideoYOLODataset(Dataset):
    """
    DDP-friendly dataset that loads frames from video files and optionally caches them into RAM.
    Index entries are (video_path: Path, frame_id: int, label_path: Path).
    """

    def __init__(
        self,
        video_root,
        label_root,
        img_size=640,
        frame_skip=1,
        sample_frames=None,
        transform=None,
        cache_images=False,  # "ram" to cache
        rank: int = LOCAL_RANK,
        world_size: int = WORLD_SIZE,
        prefix: str = "",
    ):
        self.video_root = Path(video_root)
        self.label_root = Path(label_root)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.sample_frames = sample_frames
        self.transform = transform
        self.cache_images = cache_images
        self.rank = rank
        self.world_size = world_size if world_size >= 1 else 1
        self.prefix = prefix

        # build full index then partition per rank (so each rank handles a disjoint subset)
        full_index = self._build_index()
        if sample_frames:
            full_index = random.sample(full_index, min(sample_frames, len(full_index)))

        # partition index per rank to avoid every rank caching everything
        if (self.rank is not None) and (self.rank >= 0) and (self.world_size > 1):
            self.index = [full_index[i] for i in range(len(full_index)) if (i % self.world_size) == self.rank]
            LOGGER.info(f"{prefix}[rank{self.rank}] Using {len(self.index)} frames (partitioned of {len(full_index)})")
        else:
            self.index = full_index
            LOGGER.info(f"{prefix}Using {len(self.index)} frames (single-process)")

        self.n = len(self.index)

        # load labels into memory (small)
        self.labels = []
        for _, _, label_file in self.index:
            if label_file and label_file.exists():
                with open(label_file, "r") as f:
                    lines = [list(map(float, ln.strip().split())) for ln in f.readlines() if ln.strip()]
                if len(lines) > 0:
                    self.labels.append(np.array(lines, dtype=np.float32))
                else:
                    self.labels.append(np.zeros((0, 5), dtype=np.float32))
            else:
                self.labels.append(np.zeros((0, 5), dtype=np.float32))

        # placeholder shapes (we'll use resized shapes)
        self.shapes = [(self.img_size, self.img_size)] * self.n

        # prepare frame cache container (list aligned with self.index)
        self.ims = [None] * self.n

        # Optionally cache frames into RAM (per-rank subset)
        if cache_images == "ram":
            if self.check_cache_ram(prefix=prefix):
                try:
                    self._cache_frames_to_ram(prefix=prefix)
                except Exception as e:
                    LOGGER.error(f"{prefix}Caching failed with error: {e}. Falling back to on-demand reads.")
                    self.ims = [None] * self.n
            else:
                LOGGER.warning(f"{prefix}Not enough RAM to cache frames, using on-demand reads.")
        # else: no caching, use on-demand

        LOGGER.info(f"{prefix}✅ Dataset ready: {len(self.index)} items (rank={self.rank})")

    def _build_index(self):
        """Build (video_path, frame_id, label_path) triplets by scanning label folders."""
        idx = []
        # find candidate videos
        video_files = sorted(
            [f for f in self.video_root.glob("*.*") if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
        )
        for video_path in video_files:
            video_name = video_path.stem
            label_folder = self.label_root / "labels" / video_name
            if not label_folder.exists():
                LOGGER.warning(f"no label folder for video '{video_name}' at {label_folder}")
                continue
            # label filenames expected like: video_name_frame_000123.txt
            frame_labels = sorted(label_folder.glob(f"{video_name}_frame_*.txt"))
            for label_file in frame_labels:
                frame_str = label_file.stem.split("_frame_")[-1]
                if not frame_str.isdigit():
                    continue
                frame_id = int(frame_str)
                if frame_id % self.frame_skip != 0:
                    continue
                idx.append((video_path, frame_id, label_file))
        return idx

    def check_cache_ram(self, safety_margin=0.1, sample_n=5, prefix=""):
        """Estimate whether there's enough available RAM to cache this rank's subset."""
        gb = 1 << 30
        if self.n == 0:
            return False
        # sample up to sample_n frames to estimate size
        sample_n = min(sample_n, self.n)
        sample_idxs = random.sample(range(self.n), sample_n)
        sample_bytes = 0
        for si in sample_idxs:
            video_path, frame_id, _ = self.index[si]
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                continue
            # estimate bytes after resizing
            h0, w0 = frame.shape[:2]
            r = self.img_size / max(h0, w0)
            h_ = int(round(h0 * r))
            w_ = int(round(w0 * r))
            sample_bytes += h_ * w_ * 3 * np.dtype(np.uint8).itemsize
        if sample_bytes == 0:
            return False
        avg = sample_bytes / sample_n
        total_bytes = avg * self.n
        mem = psutil.virtual_memory()
        need_gb = total_bytes / gb
        available_gb = mem.available / gb
        LOGGER.info(f"{prefix}Estimate caching rank {self.rank}: need {need_gb:.2f}GB, available {available_gb:.2f}GB")
        return total_bytes * (1 + safety_margin) < mem.available

    def _cache_frames_to_ram(self, prefix=""):
        """Cache all frames in this rank's index into RAM. Raises on error."""
        LOGGER.info(f"{prefix}[rank{self.rank}] Caching {self.n} frames into RAM with {NUM_THREADS} threads...")
        total_bytes = 0
        gb = 1 << 30

        def load_frame_pair(pair):
            video_path, frame_id, _ = pair
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # convert to RGB to be consistent with later pipeline
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        # use ThreadPool to parallelize reading (IO-bound)
        with ThreadPool(NUM_THREADS) as pool:
            # imap returns results in order, we need index i to store into self.ims
            it = pool.imap(load_frame_pair, self.index)
            for i, frame in enumerate(tqdm(it, total=self.n, bar_format=TQDM_BAR_FORMAT, disable=(self.rank > 0))):
                self.ims[i] = frame
                total_bytes += frame.nbytes
                if (i + 1) % 200 == 0:
                    LOGGER.info(f"{prefix}[rank{self.rank}] Cached {i + 1}/{self.n} frames ({total_bytes / gb:.2f} GB)")
        LOGGER.info(f"{prefix}[rank{self.rank}] ✅ Finished caching {self.n} frames ({total_bytes / gb:.2f} GB)")
        return

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # idx refers to the rank-local index (0..n-1)
        video_path, frame_id, label_path = self.index[idx]

        frame = self.ims[idx]
        if frame is None:
            # on-demand read
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # to tensor: (C,H,W), float32 [0..1]
        img = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        targets = (
            torch.tensor(self.labels[idx], dtype=torch.float32)
            if self.labels[idx].size > 0
            else torch.zeros((0, 5), dtype=torch.float32)
        )

        if self.transform:
            img, targets = self.transform(img, targets)

        return img, targets


def yolo_video_collate(batch):
    """
    Collate function for VideoYOLODataset compatible with YOLOv5 train.py.
    Returns imgs, targets, paths, shapes.
    """
    imgs, targets, paths, shapes = [], [], [], []

    for i, (img, target) in enumerate(batch):
        imgs.append(img)
        paths.append(f"video_frame_{i}")  # arbitrary frame identifier
        shapes.append(torch.tensor(img.shape[1:], dtype=torch.float32))  # H,W

        if target.numel() > 0:
            b = torch.full((target.shape[0], 1), i)  # batch index
            targets.append(torch.cat((b, target), dim=1))

    imgs = torch.stack(imgs, 0)
    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 6))

    shapes = torch.stack(shapes, 0)  # [B,2]

    return imgs, targets, paths, shapes


def create_video_yolo_dataloader(
    video_root,
    label_root,
    imgsz=640,
    batch_size=16,
    workers=4,
    frame_skip=1,
    sample_frames=None,
    shuffle=True,
    rank=-1,
    world_size=WORLD_SIZE,
    cache_images=False,
):
    with torch_distributed_zero_first(rank):
        dataset = VideoYOLODataset(
            video_root=video_root,
            label_root=label_root,
            img_size=imgsz,
            frame_skip=frame_skip,
            sample_frames=sample_frames,
            transform=None,
            cache_images=cache_images,
            rank=rank,
        )

    sampler = None
    if (rank is not None) and (rank >= 0) and (world_size > 1):
        sampler = SmartDistributedSampler(dataset, shuffle=shuffle)

    # IMPORTANT: if cache_images=="ram" prefer num_workers=0 to avoid worker processes duplicating the cached RAM.
    num_workers = workers if not cache_images == "ram" else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=yolo_video_collate,
    )
