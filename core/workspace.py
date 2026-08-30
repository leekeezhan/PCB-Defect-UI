"""
workspace.py
============
Where this application looks for *data* on disk.

Module 4 lives in its own repository and reaches Modules 1, 2 and 3 over HTTP,
so it no longer sits inside the shared project tree and can no longer assume
that ``Preprocessed_Dataset/`` is one directory above it. Nothing in the
inspection path needs those folders — every algorithmic stage arrives over the
network. They are still worth locating when they happen to be present, because
the sample pickers, the checkpoint picker and the project-video picker become
much friendlier when there is real data to point at.

The workspace is therefore *optional* and resolved in this order:

1. the folder configured in the sidebar, if the operator set one;
2. the ``PCB_WORKSPACE`` environment variable;
3. a sibling checkout of the shared repository, if one is next to this one;
4. this repository's own root.

None of the four has to contain anything. An empty result is an ordinary state
and every caller is written to cope with it.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Root of this repository — the directory holding ``app.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Environment variable that overrides the workspace location.
ENV_WORKSPACE = "PCB_WORKSPACE"

#: Names of sibling checkouts of the shared project, tried in order.
_SIBLING_NAMES = ("Image-Processing", "Image Processing", "PCB-Defect-Inspection")

#: Dataset folders the upstream modules are known to write, in display order.
_DATASET_CANDIDATES = {
    "Preprocessed_Dataset (Module 1 output)": "Preprocessed_Dataset",
    "Clean_Dataset (Module 1 resized)": "Clean_Dataset",
    "Calibrated_Dataset (Module 2 output)": "Calibrated_Dataset",
    "PCB_DATASET (raw capture)": "PCB_DATASET",
    "samples (local)": "samples",
}


def repo_root() -> Path:
    """The root of this repository. Always exists."""
    return REPO_ROOT


def sibling_checkout() -> Path | None:
    """
    Locate a checkout of the shared project beside this one.

    A common local layout is::

        C:\\Users\\you\\Image-Processing\\      <- teammates' modules and datasets
        C:\\Users\\you\\PCB-Defect-UI\\         <- this repository

    Returns:
        The sibling directory, or ``None`` when there is no such folder.
    """
    parent = REPO_ROOT.parent
    for name in _SIBLING_NAMES:
        candidate = parent / name
        if candidate.is_dir() and candidate != REPO_ROOT:
            return candidate
    return None


def workspace_root(configured: str | os.PathLike | None = None) -> Path | None:
    """
    Resolve the data workspace.

    Args:
        configured: the folder set in the sidebar. Wins over everything else
            when it names a directory that exists.

    Returns:
        A directory, or ``None`` if even this repository's root is unreadable
        (which should not happen, but the callers tolerate it).
    """
    if configured:
        candidate = Path(str(configured).strip().strip('"'))
        if candidate.is_dir():
            return candidate

    from_env = os.environ.get(ENV_WORKSPACE, "").strip().strip('"')
    if from_env and Path(from_env).is_dir():
        return Path(from_env)

    sibling = sibling_checkout()
    if sibling is not None:
        return sibling

    return REPO_ROOT if REPO_ROOT.is_dir() else None


def dataset_folders(root: str | os.PathLike | None) -> dict[str, Path]:
    """
    Collect the dataset folders present under ``root``.

    Args:
        root: the workspace, as returned by :func:`workspace_root`.

    Returns:
        Mapping of label to existing directory, in display order. Folders that
        are not there are simply omitted — on a fresh clone the mapping is
        empty and the pickers hide themselves.
    """
    if root is None:
        return {}
    base = Path(root)
    if not base.is_dir():
        return {}
    found = {
        label: base / name
        for label, name in _DATASET_CANDIDATES.items()
        if (base / name).is_dir()
    }
    return found
