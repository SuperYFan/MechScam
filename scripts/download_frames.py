#!/usr/bin/env python3
"""Rebuild the MechScam video frames from the original YouTube videos.

Frames are not distributed with this repository (see "Data access and copyright" in README.md). This
script downloads each video by its YouTube ID with yt-dlp and extracts the same number of uniformly
spaced frames listed in its sample.json, writing them to

    data/mechscam/samples/<collection>/<video_id>/frames/000001.jpg, 000002.jpg, ...

Frame k (0-based) of a video with T decoded frames and N target frames is frame
int(k * (T - 1) / (N - 1)), i.e. np.linspace(0, T - 1, N, dtype=int), saved as JPEG (ffmpeg -q:v 2).

Some videos have been removed from YouTube since collection; they are reported as unavailable. Re-encoded
downloads may differ slightly from the frames used in the paper (the sha256 values in sample.json will then
not match); contact the authors for the original frames.

Requirements: yt-dlp and ffmpeg/ffprobe on PATH (pip install yt-dlp).

    python scripts/download_frames.py --limit 5             # try a few videos first
    python scripts/download_frames.py --workers 4           # all videos, resumable
    python scripts/download_frames.py --ids-file my_ids.txt # only some sample_ids (one per line)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def uniform_indices(total: int, target: int) -> list[int]:
    if target == 1:
        return [0]
    return [int(i * (total - 1) / (target - 1)) for i in range(target)]


def count_frames(ffprobe: str, video: Path) -> int:
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def extract(ffmpeg: str, ffprobe: str, video: Path, out_dir: Path, names: list[str]) -> None:
    total = count_frames(ffprobe, video)
    target = len(names)
    if total < target:
        raise RuntimeError(f"video has {total} frames, {target} needed")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        tmp = Path(tmp)
        for trim in range(5):  # container frame counts are sometimes one or two too high
            for f in tmp.glob("*.jpg"):
                f.unlink()
            select = "+".join(f"eq(n\\,{i})" for i in uniform_indices(total - trim, target))
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                            "-vf", f"select={select}", "-vsync", "0", "-q:v", "2", str(tmp / "%06d.jpg")],
                           check=True)
            made = sorted(tmp.glob("*.jpg"))
            if len(made) == target:
                out_dir.mkdir(parents=True, exist_ok=True)
                for src, name in zip(made, names):
                    shutil.move(str(src), out_dir / Path(name).name)
                return
        raise RuntimeError(f"ffmpeg produced {len(made)} of {target} frames")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(row: dict, args) -> tuple[str, str]:
    sample_dir = args.dataset_dir / Path(row["sample_path"]).parent
    sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
    frames = sample.get("frames", [])
    names = [f["file"] for f in frames]
    out_dir = sample_dir / "frames"
    if names and all((sample_dir / n).exists() for n in names):
        return row["sample_id"], "exists"
    url = f"https://www.youtube.com/watch?v={row['video_id']}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run([args.yt_dlp, "-q", "--no-warnings", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                            "-o", str(Path(tmp) / "video.%(ext)s"), url], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            lines = (exc.stderr or "").strip().splitlines()
            return row["sample_id"], "unavailable: " + (lines[-1] if lines else "yt-dlp failed")
        video = next(Path(tmp).glob("video.*"))
        try:
            extract(args.ffmpeg, args.ffprobe, video, out_dir, names)
        except Exception as exc:  # noqa: BLE001
            return row["sample_id"], f"extract_failed: {exc}"
    same = sum(sha256(sample_dir / f["file"]) == f.get("sha256") for f in frames)
    return row["sample_id"], f"ok ({same}/{len(frames)} frames identical to the released hashes)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-dir", type=Path, default=ROOT / "data/mechscam")
    p.add_argument("--ids-file", type=Path, help="Only these sample_ids (one per line)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--yt-dlp", dest="yt_dlp", default="yt-dlp")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--log", type=Path, default=ROOT / "work/download_frames.jsonl")
    args = p.parse_args()
    for tool in (args.yt_dlp, args.ffmpeg, args.ffprobe):
        if shutil.which(tool) is None:
            print(f"{tool} not found on PATH", file=sys.stderr)
            return 2
    rows = [json.loads(l) for l in (args.dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if args.ids_file:
        keep = {l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()}
        rows = [r for r in rows if r["sample_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(args.workers) as pool, args.log.open("a", encoding="utf-8") as log:
        futures = [pool.submit(process, r, args) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, status = fut.result()
            key = status.split(":")[0].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            log.write(json.dumps({"sample_id": sid, "status": status}) + "\n")
            if i % 50 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
