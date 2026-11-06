#!/usr/bin/env python3
"""Create a saved soccer tracking video. Run: python generate_video.py"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=project / "OSUshortclip.mp4")
    parser.add_argument("--output-dir", type=Path, help="Default: a new folder in output_videos")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--device", help="cpu, mps (Apple GPU), or 0 (CUDA)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--ball-conf", type=float, default=0.10, help="Minimum ball confidence; lower values admit more detections")
    parser.add_argument("--ball-start", nargs=2, type=float, metavar=("X", "Y"), help="Initial ball center in original-video pixels")
    parser.add_argument("--all-ball-candidates", action="store_true", help="Debug: show every ball candidate")
    parser.add_argument("--max-frames", type=int, default=0, help="0 exports the full clip")
    args = parser.parse_args()
    source = args.video.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Video not found: {source}")
    if args.imgsz <= 0 or args.max_frames < 0:
        parser.error("--imgsz must be positive and --max-frames cannot be negative")
    if not 0 < args.ball_conf <= 1:
        parser.error("--ball-conf must be greater than 0 and at most 1")
    destination = (
        args.output_dir.expanduser().resolve() if args.output_dir else
        project / "output_videos" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    # Use the same Python environment and the existing tracking implementation.
    model_path = Path(args.model).expanduser()
    if model_path.is_file():
        model = str(model_path.resolve())
    elif (project / model_path).is_file():
        model = str((project / model_path).resolve())
    else:
        model = str(model_path)
    command = [
        sys.executable, str(project / "track_soccer.py"), str(source),
        "--output-dir", str(destination), "--model", model,
        "--imgsz", str(args.imgsz), "--max-frames", str(args.max_frames),
        "--ball-conf", str(args.ball_conf),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.ball_start is not None:
        command.extend(["--ball-start", *map(str, args.ball_start)])
    if args.all_ball_candidates:
        command.append("--all-ball-candidates")
    print("Generating annotated video. Progress will appear below.", flush=True)
    try:
        subprocess.run(command, cwd=project, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt:
        raise SystemExit("Export interrupted; any video in the output folder may be incomplete.")
    video = destination / f"{source.stem}_tracked.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise SystemExit("Tracking finished without producing a video.")
    print(f"\nYour output video is ready:\n{video}", flush=True)


if __name__ == "__main__":
    main()
