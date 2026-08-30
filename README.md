# Module 4 — User Interface Module

**PCB Defect Inspection System** · Mode B, Innovative Solution Development

The operator-facing application for the inspection system. It integrates the
three upstream modules into one end-to-end workflow, presents every intermediate
stage, applies the acceptance criteria, persists the results, and exports the
findings.

**This repository holds Module 4 alone.** Modules 1, 2 and 3 live in their own
repositories and are reached over HTTP, so each module can be developed, fixed
and deployed by its owner without anyone waiting on anyone else. Two contracts
define those boundaries, and both ship with a runnable reference implementation:

| Boundary | Contract | Default port |
|---|---|---|
| Modules 1 & 2 — pre-processing and alignment | [`docs/API_CONTRACT_PIPELINE.md`](docs/API_CONTRACT_PIPELINE.md) | 8100 |
| Module 3 — defect detection | [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) | 8000 |

| Attribute | Details |
|---|---|
| Module | Student 4 — User Interface |
| Framework | Streamlit |
| Inputs | Single image · multi-image upload · folder on disk · recorded video · live camera |
| Pre-processing & alignment | A remote service over HTTP, or a local `image_pipeline.py` |
| Detection | A remote inference service over HTTP, or a local checkpoint |
| Persistence | SQLite (default) or Supabase |
| Reporting | ReportLab (PDF), pandas (CSV) |

---

## Running the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the interface at <http://localhost:8501>. It starts and is
usable with **no** services running: every stage reports itself unavailable and
inspection continues on the raw image, which is by design — see
[Degradation behaviour](#degradation-behaviour).

Point it at the teammates' services in the sidebar:

* **Processing pipeline → Remote API** — base URL of the Modules 1 & 2 service.
* **Detection model → Remote API** — base URL of the Module 3 service.

Both also offer a local option, for working offline or demonstrating without the
teammates' services: *Local module* imports `image_pipeline.py` from a checkout
of the shared repository, and *Local checkpoint* loads a `.pt`/`.pth` file into
this process.

### Sample data (optional)

Nothing in the inspection path reads from disk. If a checkout of the shared
repository is on the same machine, set its folder under **Sample data folder**
in the sidebar — or export `PCB_WORKSPACE` before starting — and the pickers
will offer boards from `PCB_DATASET/`, `Preprocessed_Dataset/` and the rest,
plus any trained checkpoints found there. A sibling checkout named
`Image-Processing` is picked up automatically.

### Running without a teammate's service yet

`local_service/serve.py` answers both contracts using code already on this
machine — the shared repository's real `image_pipeline.py` for Modules 1 & 2,
and this repository's own `core.detector.DefectDetector` for Module 3 — so the
interface can be developed and demoed end to end before a teammate's service
exists. It is a development convenience, not part of this module's own
deliverable; delete it once a real service is running.

```bash
pip install -r local_service/requirements.txt
uvicorn local_service.serve:app --host 127.0.0.1 --port 8000
```

Then put the same URL in **both** sidebar fields — one process answers
`/process` and `/predict`. If Module 3 has not yet delivered a trained
checkpoint, this loads whatever stock, pre-trained-on-COCO `.pt` file it finds
(`yolov10n.pt`, `rtdetr-l.pt`) and says so on the System status page — real
inference, real pre-processing and alignment, wrong class names until the real
checkpoint exists. See the module's own docstring for configuration.

---

## System architecture

The interface performs **no image processing of its own**. Every algorithmic
step is delegated through an adapter to the module that owns it, so the
presentation layer stays independent of how any given module is implemented —
and so either upstream boundary can be swapped between a service and a local
implementation without a single change in the pages. That indifference is what
makes the split into separate repositories work: the pages see one adapter
interface, and the adapter decides whether the work happens over a socket or in
this process.

```
                        ┌─────────────────────────────┐
   Operator ───────────▶│         app.py              │  Module 4
   image / folder /     │  Streamlit pages + routing  │  (this repository)
   video / camera       └──────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  core/pipeline_remote.py      core/remote.py                  core/analysis.py
        │      ─or─                 │    ─or─                          │
  core/pipeline_bridge.py      core/detector.py                        ▼
        │                          │                            verdict + statistics
        ▼                          ▼                                   │
 ┌──────────────────┐      ┌──────────────────┐                        ▼
 │  POST /process   │      │  POST /predict   │             core/report.py  → PDF
 │  Modules 1 & 2   │      │    Module 3      │             core/storage.py → history
 │  service :8100   │      │  service :8000   │             core/video.py   → recorded
 └──────────────────┘      └──────────────────┘             core/live.py    → camera
  API_CONTRACT_          API_CONTRACT.md
  PIPELINE.md            (teammates' repositories)
```

Each upstream box has a local alternative for offline work: `pipeline_bridge.py`
imports `image_pipeline.py` from a checkout of the shared repository, and
`detector.py` loads a checkpoint through Ultralytics (YOLO, RT-DETR) or
torchvision (Faster R-CNN).

### Files

```
PCB-Defect-UI/
├── app.py                  Entry point: page layout, routing, orchestration
├── core/
│   ├── pipeline_remote.py  Adapter for Modules 1 and 2 — HTTP service
│   ├── pipeline_bridge.py  StageResult, the local adapter, and the factory
│   ├── workspace.py        Resolves the optional folder of sample data
│   ├── remote.py           Adapter for Module 3 — HTTP inference service
│   ├── detector.py         Adapter for Module 3 — local checkpoint
│   ├── analysis.py         Acceptance criteria, quality score, statistics
│   ├── viz.py              Bounding-box overlays, colours, encoding
│   ├── report.py           PDF inspection reports (single board and batch)
│   ├── video.py            Frame-by-frame execution over a recorded video
│   ├── live.py             Real-time capture and inspection from a camera
│   └── storage.py          Inspection history — SQLite or Supabase
├── ui/
│   ├── theme.py            Stylesheet and page header
│   ├── sidebar.py          The single configuration panel
│   └── components.py       Verdict banner, stage chips, legend, status rows
├── docs/
│   ├── API_CONTRACT_PIPELINE.md  Wire format for the Modules 1 & 2 service
│   ├── API_CONTRACT.md           Wire format for the Module 3 service
│   └── supabase_schema.sql       Table, indexes and row-level-security policies
├── tests/test_core.py      Backend verification suite (177 checks, no server needed)
└── requirements.txt
```

---

## Pages

### 1. Single board inspection

Upload a board, or pick one from the datasets the upstream modules produced. The
result shows the verdict, the headline figures, and five views of the board:
the detection overlay, the Module 2 aligned image, the Module 1 pre-processed
image, the operator's input, and a before/after pair. Findings are tabulated
per defect with confidence, position and size, and exported as a PDF inspection
certificate, an annotated PNG, or a CSV.

### 2. Batch inspection

Bulk ingestion of a whole folder (recursively, so the per-class dataset layout
works as-is) or a multi-file upload. Produces the production yield, the verdict
split, the defect distribution across the run, a per-board results table, and a
gallery of the boards that were failed or flagged. Exports a PDF batch report
with an annotated evidence appendix, plus two CSVs.

### 3. Video stream inspection

Ingests a recorded MP4 and runs the pipeline across individual frames. Detection
is sampled every *n*-th frame (configurable) while every frame is written to the
output, so the annotated stream plays at the source frame rate. Produces the
annotated video, the worst frame, a defect timeline chart, and a PDF report.

### 4. Live inspection

Two capture modes:

* **Snapshot** — one photograph taken through the browser (`st.camera_input`),
  then inspected exactly like an uploaded board. Needs no camera on the server,
  so it works when the interface is deployed remotely.
* **Continuous stream** — repeatedly captures from a camera attached to the
  machine running the interface, inspecting each frame and reporting a running
  pass rate, effective frame rate and defect timeline.

Continuous mode runs its capture loop in **short chunks** rather than one long
loop. Streamlit only redraws when the page script finishes, so a loop that never
returns would leave the Stop button unclickable; each chunk captures for about
two seconds, saves its totals in the session, and asks for a re-run. The camera
handle is held across chunks rather than reopened.

### 5. History

The yield dashboard. Every result the system has recorded — across all four
inspection modes — with filters by mode and verdict, a quality-over-time chart,
the verdict split, the full record table, CSV and PDF export, and a guarded
clear-history action.

### 6. System status

Diagnostics: which modules resolved, which checkpoint or service loaded and
under which backend, whether a camera is attached, which datasets were
discovered, the installed library versions, and the configuration in force.

---

## Detector: remote or local

Selected in the sidebar under **Detector source**.

**Remote API** (default) forwards each image to Module 3's inference service
over HTTP.

**Local checkpoint** loads the weights into this process instead. The sidebar
discovers every checkpoint in the repository automatically and lists
purpose-trained weights before the stock pre-trained downloads (`yolov10n.pt`,
`rtdetr-l.pt`), which are flagged with a warning because they predict generic
COCO classes rather than defect types. The backend is chosen from the
checkpoint: `.pt` files load through Ultralytics (YOLO and RT-DETR), `.pth` files
through torchvision. The Faster R-CNN loader rebuilds the 8–128 px anchor
generator that Module 3 substitutes for torchvision's 32–512 px default, since
the rescaled defects are only 10–25 px across.

Running detection as a service keeps the machine running the interface free of a
multi-gigabyte PyTorch install, which is what makes deploying the interface to a
small host practical, and lets the detection model be updated or moved to a GPU
box independently of this repository. The wire format is specified in
[`docs/API_CONTRACT.md`](docs/API_CONTRACT.md), which includes a complete
runnable FastAPI reference implementation. If the service is unreachable or
slow, the failure is reported exactly as a local model failure would be and the
rest of the interface keeps working.

### Two models, chosen per page

The measured trade-off between the two trained architectures
(`Student3-Defect Detection/results/comparison_summary.md`, CPU):

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS |
|---|---|---|---|---|---|
| YOLOv10n | 0.566 | 0.249 | 0.565 | 0.527 | 19.2 |
| RT-DETR-L | **0.669** | **0.293** | **0.701** | **0.637** | 2.1 |

RT-DETR-L is more accurate on both axes — and recall matters most in defect
inspection, where a missed short circuit costs far more than a false alarm — but
it is roughly nine times slower. Neither is the right answer everywhere, so the
sidebar accepts **two** checkpoints: the primary model, used for single-board and
batch inspection where half a second per board is imperceptible, and an optional
**fast model** used by the video and live pages, where throughput is what makes
the page usable at all.

---

## Acceptance criteria

The verdict is not simply "any detection fails the board". Three rules are
applied in order, all configurable from the sidebar:

1. **Critical defects fail outright.** `open_circuit` and `short` break board
   continuity, so a single confident instance fails the board whatever the
   allowance.
2. **The allowance is then checked.** More confident detections than
   *Defects allowed per board* fails the board.
3. **Uncertain findings go to review, not to failure.** A board whose only
   findings score below the *Manual-review confidence* is marked `REVIEW`, so a
   marginal detection prompts a human look rather than scrapping the board.

A **quality score** out of 100 accompanies the verdict. Each detection subtracts
`25 × severity × confidence`, where severity is highest for the defects that
break the board electrically. The score ranks boards by how badly they are
affected; the verdict, not the score, decides acceptance.

---

## Inspection history

Persisting each verdict is what turns the interface from a viewer into the
*data analysis dashboard* the assignment's shared requirements call for.

| Store | When to use |
|---|---|
| **SQLite** (default) | Nothing to install, nothing to configure, no network. The right choice for a live demonstration. Writes `inspection_history.db` in the sample-data folder, or beside the application — add it to `.gitignore`, it is run data rather than source. |
| **Supabase** | Hosted Postgres, so the history is shared across every machine running the system. Run [`docs/supabase_schema.sql`](docs/supabase_schema.sql) in the Supabase SQL editor first, then paste the project URL and the **anon** key into the sidebar. |
| **Off** | No history is recorded; the History page explains that and everything else still works. |

Use the anon key together with the row-level-security policies in the schema
file. The service-role key bypasses row-level security entirely and does not
belong inside a desktop application.

Live capture writes its frames in one call per chunk rather than one call per
frame, so a long session does not make hundreds of round trips.

---

## Degradation behaviour

The interface is built so that a missing upstream artefact never produces a
stack trace on screen. Each of the following is a reported state, not a crash:

| Situation | Behaviour |
|---|---|
| The Modules 1 & 2 service is down, slow, or returns unexpected JSON | Detection runs on the raw image; the banner and System status page give the status code and reason, and a batch continues |
| `image_pipeline.py` not found, when the pipeline is set to run locally | Detection runs on the raw image; the banner and System status page explain why |
| Module 1 returns nothing | The original image is carried forward, and a note says so |
| Module 2 finds no four-corner board boundary | Detection continues on the pre-processed image; the stage chip reads "not applied" |
| No Module 3 checkpoint available | The pre-processing and alignment stages stay fully usable; no defects are reported |
| The inference service is down, slow, or returns unexpected JSON | Reported per image with the status code and reason; a batch continues |
| Supabase unreachable or its library missing | The failure is shown once; inspections continue unaffected |
| No camera attached | Continuous live mode says so; snapshot mode still works through the browser |
| `reportlab` missing or PDF generation fails | The other exports still work; the failure is shown as a warning |
| A file in a batch cannot be decoded | It is skipped and counted, and the run continues |

This matters more here than it did when all four modules shared one repository:
a fresh clone of this repository contains no image data, no checkpoints and no
running services, and the interface must still start, explain what it cannot
reach, and inspect an uploaded board with whatever it can.

---

## Execution modes

Independently of *where* Modules 1 and 2 run, the sidebar asks *how* they should
execute:

* **Shared module functions** (default) — the plain functions the two modules
  export. Fast and robust; the right choice for a live demonstration.
* **Teammates' notebook functions** — the function-definition cells of the
  original notebooks, loaded and executed. Slower, but it demonstrates that the
  system drives the teammates' own code rather than a re-implementation of it.

The choice is forwarded to the processing service as the `mode` form field. A
service that does not implement notebook mode says so in its `/health` response
and the System status page reports it as unavailable, which is accurate rather
than a fault. Running locally, notebook mode calls
`image_pipeline.student2_from_student1_array` and falls back to the shared
functions if the notebook cannot be loaded.

---

## Verification

```bash
python tests/test_core.py
```

177 checks over the acceptance rules, the statistics, the annotation and encoding
helpers, the module bridge (including its degradation paths), both detector
adapters, both PDF builders, the video pipeline, the live-capture loop and both
history stores. Both remote adapters are tested against real stub HTTP services
bound to a socket, so the multipart encoding and JSON parsing are genuinely
exercised rather than mocked. The suite needs no Streamlit server, no trained
weights, no camera and no network.

---

## Known notes

* Streamlit deprecated `use_container_width` in favour of `width="stretch"`.
  This code keeps `use_container_width` because `width` requires Streamlit 1.49
  or later, and compatibility with the environment the teammates' notebooks run
  in matters more than the console warning it prints.
* Continuous live capture reads the camera on the machine running Streamlit, not
  the machine viewing the page. Deployed remotely, use snapshot mode.
