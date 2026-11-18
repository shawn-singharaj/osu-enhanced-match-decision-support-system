# OSU Men's Soccer Decision Support System

Senior Capstone Project | Oregon State University

## Overview

This project is developing a video analysis and decision support system for Oregon State University's men's soccer program. The goal is to process match footage, track players and the ball, map their locations onto a 2D soccer field, and identify useful events for coaches to review.

The initial design uses **one camera** and processes recorded match video. The architecture also leaves room for future features such as requesting and analyzing a recent clip during a match.

> **Project status:** In development. The folders and commands below describe the planned repository organization; not every module is implemented yet.

## System Architecture

The current top-level system diagram contains these components:

1. **Replay Controller PCB and Power PCB** — Provide physical replay controls and power. The controller sends USB keyboard commands to the GUI.
2. **Graphical User Interface (GUI)** — Accepts user controls, selects match footage, and displays results.
3. **Video Chunking & Merging** — Divides long videos into manageable segments and later combines timestamped results.
4. **Player Detection & Tracking** — Detects players, assigns tracking IDs, and identifies teams.
5. **Ball Detection & Tracking** — Detects and tracks the ball.
6. **Field Mapping** — Converts player and ball pixel coordinates into field coordinates using video/field calibration.
7. **Possession** — Estimates which team has possession.
8. **Special Event Detection** — Identifies selected events using player/ball positions and possession data.
9. **Phase Labeling & Timestamps** — Organizes field, possession, and event information along the match timeline.
10. **GUI / CSV Export** — Presents reviewable results and exports data.

**Main software flow:**

```text
Match Video
    |
    v
GUI --> Video Chunking
              |
        +-----+-----+
        |           |
        v           v
 Player Tracking  Ball Tracking
        |           |
        +-----+-----+
              |
              v
        Field Mapping <--- Video Chunks (for calibration)
              |
        +-----+------------------+
        |                        |
        v                        v
    Possession            Special Event Detection
        |                        ^
        +------------------------|  (possession informs events)
        |                        |
        +------------+-----------+
                     |
                     v
          Phase Labeling & Timestamps
                     |
                     v
             Merge Results
                     |
                     v
               GUI / CSV
```

The physical replay controller and power PCB are separate hardware components linked to the GUI.

## Repository Structure

```text
osu-soccer-capstone/
├── README.md
├── requirements.txt               # Python dependencies (when established)
├── .gitignore
├── .env.example
├── hardware/
│   ├── replay_controller/
│   │   ├── firmware/
│   │   ├── pcb_design/
│   │   └── README.md
│   └── power_pcb/
│       ├── pcb_design/
│       └── README.md
├── software/
│   ├── gui/
│   ├── video_processing/
│   │   ├── chunk_video.py
│   │   └── merge_results.py
│   ├── player_tracking/
│   │   ├── detect_players.py
│   │   ├── track_players.py
│   │   └── classify_teams.py
│   ├── ball_tracking/
│   │   ├── detect_ball.py
│   │   └── track_ball.py
│   ├── field_mapping/
│   │   ├── detect_field.py
│   │   └── map_coordinates.py
│   ├── possession/
│   │   └── detect_possession.py
│   ├── event_detection/
│   │   └── detect_events.py
│   ├── phase_labeling/
│   │   ├── label_phases.py
│   │   └── generate_timestamps.py
│   ├── live_stream/                # Future feature
│   └── pipeline/                   # Future integration/orchestration
├── shared/
│   ├── schemas/
│   ├── config/
│   └── utils/
├── tests/
│   └── fixtures/
├── data/
│   ├── sample_videos/
│   ├── annotations/
│   └── sample_outputs/
├── models/
├── outputs/
│   ├── csv/
│   └── videos/
└── docs/
    ├── architecture/
    ├── interfaces/
    └── verification/
```

Folder contents above are **planned examples**, not a claim that every file already exists.

## Software Blocks

| Block | Folder | Responsibilities | Expected output |
| --- | --- | --- | --- |
| GUI | `software/gui/` | Video selection, playback, replay controls, results display | Selected video/clip and review UI |
| Video processing | `software/video_processing/` | Split videos and merge ordered results | Video chunks with start times; merged results |
| Player tracking | `software/player_tracking/` | YOLO detection, ByteTrack, team classification | Player boxes, IDs, team labels, pixel locations, timestamps |
| Ball tracking | `software/ball_tracking/` | Ball detection and tracking | Ball pixel locations and timestamps |
| Field mapping | `software/field_mapping/` | Field calibration, pixel-to-field conversion | Player/ball field coordinates and ball zone |
| Possession | `software/possession/` | Team-possession estimation | Timestamped possession labels |
| Event detection | `software/event_detection/` | Detect selected game events | Timestamped event records |
| Phase labeling | `software/phase_labeling/` | Assign phase labels and timestamps | Timeline data for export |

Keep each block's internal implementation separate. Modules should exchange data through documented interfaces rather than import each other's internal functions where that would prevent independent verification.

## Shared Data Interfaces

The team should agree on input and output formats **before integrating blocks**. Document final schemas in `docs/interfaces/` and, when useful, validate them with code in `shared/schemas/`.

Example **proposed** player detection record:

```json
{
  "frame_index": 150,
  "timestamp_seconds": 305.0,
  "track_id": 12,
  "team": "OSU",
  "bbox_xyxy": [450, 200, 500, 350],
  "foot_point_px": [475, 350]
}
```

Example **proposed** mapped position record:

```json
{
  "frame_index": 150,
  "timestamp_seconds": 305.0,
  "track_id": 12,
  "team": "OSU",
  "field_x_m": 42.5,
  "field_y_m": 24.1
}
```

These are examples only. Before using them across the team, decide on coordinate origin, field dimensions, class labels, missing-data handling, and timestamp conventions.

**Important for chunking:** Every output needs its position in the *original match timeline*. If tracking restarts on each chunk, a `track_id` is only unique within that chunk until an explicit cross-chunk matching method is added. Include a chunk identifier if needed.

## Getting Started

### Prerequisites

- Git
- Python 3 (exact supported version to be agreed upon)
- FFmpeg for video processing
- A suitable environment for any model inference modules (CPU or supported GPU)

### Clone the repository

```bash
git clone <REPOSITORY_URL>
cd osu-soccer-capstone
```

Replace `<REPOSITORY_URL>` with the team's actual GitHub URL.

### Set up a Python environment

```bash
python -m venv .venv
```

Activate it:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

When the team has finalized `requirements.txt`, install dependencies:

```bash
pip install -r requirements.txt
```

> There is not yet a verified one-command application entry point. Each module's README should document its working command once implemented.

## Testing and Verification

Each block should be testable independently using sample video clips, manually labeled frames, or synthetic CSV/JSON fixtures. A teammate should not need another unfinished block to verify their assigned block.

For each block, provide:

- A README describing what it does and how to run it.
- Documented inputs and outputs.
- At least one reproducible sample input.
- Tests for expected results and relevant error cases.
- Verification steps and numerical acceptance criteria based on the approved engineering requirements.

When a test suite is in place, the team can standardize on a command such as:

```bash
python -m pytest tests/
```

This is a **planned convention**, not a currently verified test command.

## GitHub Workflow

Use `main` as the stable integration branch. Create a feature branch for work on each block or task.

```bash
git checkout main
git pull origin main
git checkout -b feature/player-tracking
```

After making changes:

```bash
git add .
git commit -m "Add initial player detection module"
git push -u origin feature/player-tracking
```

Open a pull request and ask another team member to review it before merging into `main`.

Suggested GitHub Issues include specific, independently verifiable tasks: detecting player bounding boxes, tracking IDs through short occlusions, converting pixels to field coordinates, exporting timestamps, or testing replay controller inputs.

## Handling Long Videos

The initial pipeline is intended for recorded match footage. Instead of requiring a full 90-minute analysis in one job, the video-processing block can split the match into shorter segments and process each one in order.

A reasonable **starting point to test** is 5-minute chunks. Record each chunk's start time so results can be mapped back to the complete match. Preserve tracking state across sequential chunks when practical, or explicitly handle player ID changes at boundaries.

For future in-game review, a recording/buffer component could keep recent short video segments available so the coach can request a specific time range. The system could then show cached results or prioritize that clip for new analysis. This feature is **not part of the current implemented baseline**.

## Project Documentation

- `docs/architecture/` — Top-level system diagrams and design decisions.
- `docs/interfaces/` — Input/output schemas and interface definitions.
- `docs/verification/` — Block verification procedures and test evidence.

## Data and Model Files

Avoid committing full match recordings, large trained model weights, generated video outputs, secrets, or private team data to Git. Keep small, approved sample fixtures for repeatable tests. Use `.gitignore` and a documented storage location for larger assets.

## Team Coordination

Before changing a shared interface, create an Issue or pull request describing the change and its effect on downstream blocks. Keep each block's README updated as the implementation evolves.

## License

License and distribution permissions have not yet been determined. Do not assume the match footage, model weights, or project code can be publicly redistributed.
