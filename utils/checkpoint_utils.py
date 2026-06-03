"""
Checkpoint and resume helpers for staged RL training.

Ported from obstacle_avoidance_mission/scripts/train.py (L81-140).
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> str | None:
    """Return the path of the checkpoint with the highest step count, or None.

    Scans *checkpoint_dir* for files matching the pattern::

        {prefix}_{steps}_steps.zip

    where *steps* is a non-negative integer.  Prefixes that contain
    underscores (e.g. ``ppo_drone_stage1``) are handled correctly because
    step extraction strips exactly ``prefix + "_"`` from the left and
    ``"_steps.zip"`` from the right before parsing the remainder as an int.

    Args:
        checkpoint_dir: Directory to scan.
        prefix: Filename prefix, e.g. ``"ppo_drone"`` or ``"ppo_drone_stage1"``.

    Returns:
        Absolute path to the checkpoint with the highest step count, or
        ``None`` if the directory does not exist or contains no matching files.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    candidates: list[tuple[int, str]] = []

    for fname in os.listdir(checkpoint_dir):
        if fname.startswith(prefix) and fname.endswith(".zip") and "_steps" in fname:
            try:
                step_part = fname.replace(prefix + "_", "").replace("_steps.zip", "")
                steps = int(step_part)
                candidates.append((steps, os.path.join(checkpoint_dir, fname)))
            except ValueError:
                pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _stage_start_step_metadata_path(checkpoint_dir: str, stage: int) -> str:
    """Return the path of the JSON metadata file for *stage*.

    Args:
        checkpoint_dir: Root checkpoint directory.
        stage: Training stage index.

    Returns:
        Absolute path ``{checkpoint_dir}/stage{stage}_start_step.json``.
    """
    return os.path.join(checkpoint_dir, f"stage{stage}_start_step.json")


def load_stage_start_step(checkpoint_dir: str, stage: int) -> int:
    """Load the recorded start step for *stage* from disk.

    Reads the JSON file written by :func:`save_stage_start_step` and returns
    ``data["step"]`` as an integer.

    Args:
        checkpoint_dir: Root checkpoint directory.
        stage: Training stage index.

    Returns:
        The saved step value, or ``0`` if the file is missing or malformed.
    """
    path = _stage_start_step_metadata_path(checkpoint_dir, stage)
    if not os.path.exists(path):
        logger.debug(
            "[STAGE START STEP] metadata not found stage=%d path=%s, returning 0",
            stage,
            path,
        )
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        value = int(data["step"])
        logger.info(
            "[STAGE START STEP] loaded stage=%d path=%s value=%d",
            stage,
            path,
            value,
        )
        return value
    except Exception as exc:
        logger.warning(
            "[STAGE START STEP] failed to load stage=%d path=%s: %s",
            stage,
            path,
            exc,
        )
        return 0


def save_stage_start_step(
    checkpoint_dir: str,
    stage: int,
    step: int,
    source_path: str,
) -> None:
    """Persist the start step for *stage* to disk atomically.

    Writes ``{"step": step, "wall_time": <unix timestamp>, "source": source_path}``
    to a temporary file in *checkpoint_dir* then renames it into place via
    :func:`os.replace`, guaranteeing that readers never see a partial write.

    Args:
        checkpoint_dir: Root checkpoint directory (created if absent).
        stage: Training stage index.
        step: Step value to record.
        source_path: Path to the source checkpoint that triggered this save
            (recorded for audit purposes).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = _stage_start_step_metadata_path(checkpoint_dir, stage)
    tmp_path = path + ".tmp"

    payload = {
        "step": int(step),
        "wall_time": float(time.time()),
        "source": str(source_path),
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    os.replace(tmp_path, path)

    logger.info(
        "[STAGE START STEP] saved stage=%d path=%s value=%d source=%s",
        stage,
        path,
        int(step),
        source_path,
    )
