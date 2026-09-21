"""Scene clustering by perceptual hash.

The 2 378 frames come from one capture day and a handful of fixed cameras, so
the pool is dominated by near-duplicates: 73% of frames sit in clusters of ten or
more, and the three largest scenes alone are 55% of the data. A random train/test
split would put near-identical frames on both sides and inflate every metric --
which is how this project previously arrived at a meaningless mAP of 0.958.

Every split downstream groups on the `scene_id` produced here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

#: Hamming distance (out of 240 bits) below which two frames are the same scene.
DEFAULT_THRESHOLD = 24
#: Difference-hash side length; the hash is SIDE x SIDE bits.
HASH_SIDE = 16

_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def dhash(path: Path, side: int = HASH_SIDE) -> np.ndarray:
    """Difference hash: is each pixel brighter than the one to its right?"""
    with Image.open(path) as im:
        grey = im.convert("L").resize((side + 1, side), Image.BILINEAR)
    a = np.asarray(grey, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hash_directory(directory: Path) -> tuple[list[str], np.ndarray]:
    """Hash every image in `directory`, returning (filenames, packed hashes)."""
    names, rows = [], []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            rows.append(dhash(path))
            names.append(path.name)
        except OSError:
            continue  # unreadable frame; excluded rather than guessed at
    return names, np.packbits(np.array(rows), axis=1)


def cluster(packed: np.ndarray, threshold: int = DEFAULT_THRESHOLD,
            chunk: int = 256) -> np.ndarray:
    """Union-find over pairwise Hamming distance. Returns a scene id per row.

    Chunked so the full N x N distance matrix is never materialised.
    """
    n = len(packed)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for start in range(0, n, chunk):
        block = packed[start:start + chunk]
        dist = _POPCOUNT[np.bitwise_xor(block[:, None, :], packed[None, :, :])].sum(-1)
        for row in range(dist.shape[0]):
            i = start + row
            for j in np.nonzero(dist[row] <= threshold)[0]:
                if j > i:
                    union(i, int(j))

    roots = {}
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        root = find(i)
        out[i] = roots.setdefault(root, len(roots))
    return out


# ---------------------------------------------------------------------------
# Camera identity
# ---------------------------------------------------------------------------

def camera_hash(path: Path, side: int = 8, blur: float = 6.0) -> np.ndarray:
    """Signature of the room, not the moment.

    `dhash` above measures pixel difference, which is the wrong quantity for
    grouping a split: one fixed camera produces frames 30-70 bits apart simply
    because people walked through it and the light changed between shifts. Using
    it as a camera proxy let 82% of "unseen scene" test frames come from cameras
    that were also in training.

    Heavy blur plus an 8x8 downscale removes the people and keeps the room --
    wall layout, fixtures, viewing angle -- which is what identifies a camera.
    """
    with Image.open(path) as im:
        grey = (im.convert("L")
                  .filter(ImageFilter.GaussianBlur(blur))
                  .resize((side + 1, side), Image.BILINEAR))
    a = np.asarray(grey, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hash_cameras(directory: Path) -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            rows.append(camera_hash(path))
            names.append(path.name)
        except OSError:
            continue
    return names, np.packbits(np.array(rows), axis=1)
