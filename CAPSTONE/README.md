# Soccer video tracking

To generate the full annotated video with one command:

```bash
source .venv/bin/activate
python generate_video.py
```

This uses `OSUshortclip.mp4` beside the script and creates a new folder under
`output_videos/` on each run. The final message gives the saved MP4 path.
You can also supply another clip: `python generate_video.py path/to/clip.mp4`.
It uses `track_soccer.py`, which must stay beside it. The MP4 is silent.
Keep `ball_tracker.py` in the same folder as well.

Only one plausible match ball is displayed by default. To help exclude shoes
and the penalty spot in this clip, initialize the ball at its first-frame center:

```bash
python generate_video.py --ball-conf 0.05 --imgsz 1920 --ball-start 1306 614
```

The coordinates are original-video pixels and must be changed for another clip.
Ball selection combines confidence, motion continuity, and a penalty for boxes
inside the bottom 28% of person detections. Camera motion updates the search
position. It does not reject a selected ball simply because it stops. These are
heuristics: feet can still be confused with a ball, and real balls inside player
boxes can be missed. Use `--all-ball-candidates` to inspect unfiltered candidates.
If a manually initialized ball leaves the expected search area, it stays hidden
rather than switching to a distant candidate. Restart with a different start
position when necessary. A soccer-trained detector with labeled balls and
unlabeled shoes/field markings in the training frames is the stronger long-term
fix: [custom training](https://docs.ultralytics.com/modes/train/).

Use the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python track_soccer.py OSUshortclip.mp4
```

The first run downloads YOLO26 Small weights (`yolo26s.pt`). Outputs are saved in
`runs/soccer/`: an annotated MP4, `tracks.csv` with detection boxes and IDs, and
`field.csv` with the estimated visible grass polygon for each frame. CSV frame
numbers start at zero. Player anchors are the bottom center of their boxes;
ball anchors are the box centers. All positions are image pixels.

For an initial preview, process 60 frames in a separate directory:

```bash
python track_soccer.py OSUshortclip.mp4 --max-frames 60 --show --output-dir runs/preview
```

On Apple Silicon, try `--device mps`. For faster inference, try
`--model yolo26n.pt --imgsz 960`. For smaller balls, try `--imgsz 1920`
(slower). `--ball-conf 0.08` admits weaker ball detections and more false positives.
Use a new `--output-dir` for each run; existing results are never overwritten.
The output MP4 is silent and keeps the input frame rate and dimensions.

## What the output means

- Person detections include players, goalkeepers and referees; there is no team
  classification. People outside the estimated grass are hidden by default;
  use `--include-off-field` to disable that filter.
- ByteTrack associates person detections between frames. Ball detections are
  captured before ByteTrack can filter them, then associated by center position
  and velocity without a box-overlap requirement or a confirmation delay.
  Ball and person IDs use separate namespaces; identify tracks by kind and ID.
  A short history can reconnect gaps, but absent detections are not drawn or
  exported as measured locations. A separate selector shows at most one
  candidate using confidence, motion and player-foot proximity. False positives
  can still be selected, and nearby candidates can exchange IDs.
  IDs can change after
  occlusions or camera cuts. Trails are in screen coordinates, so camera motion
  affects them. Missing detections are not reported as measured locations.
- The field outline estimates the largest green grass region. Optical-flow
  feature matches estimate camera pan, rotation and zoom, then align the previous
  estimate before smoothing it with the new grass mask. Unreliable matches or
  large segmentation changes reset the estimate. Newly exposed image areas use
  fresh segmentation. Screen edges are not drawn as observed field boundaries.
  The polygon in `field.csv` still closes along the image border where clipped.
  The outline is approximate and may include grass outside the touchlines;
  it does not locate pitch landmarks or convert pixels into meters.
- General pretrained weights can miss small, blurred or occluded soccer balls.
  This is a baseline, not a guarantee of continuous ball tracking. A custom
  soccer detector can be supplied with `--model path/to/best.pt`; it must have
  player/person and ball classes using names recognized by the script.

YOLO API references: [tracking](https://docs.ultralytics.com/modes/track/) and
[YOLO26 models](https://docs.ultralytics.com/models/yolo26/).
