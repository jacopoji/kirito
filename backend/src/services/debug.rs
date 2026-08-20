use std::{
    fs,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex},
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use include_dir::{Dir, include_dir};
use opencv::{
    core::{Mat, MatTraitConst, ModifyInplace, Point, Rect, Vector},
    highgui::{MouseEventTypes, destroy_all_windows, set_mouse_callback, wait_key},
    imgcodecs::{IMREAD_COLOR, imdecode},
    imgproc::{COLOR_BGR2BGRA, cvt_color_def},
    videoio::{VideoCapture, VideoCaptureTrait, VideoWriter, VideoWriterTrait},
};
use platforms::Window;
use rand::distr::SampleString;
use rand_distr::Alphanumeric;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{
        broadcast::{self, Receiver, Sender},
        mpsc::{self},
    },
    task::spawn_blocking,
};

use crate::{
    DebugState, TransparentShapeDifficulty,
    bridge::{Input, MouseKind},
    debug::{debug_annotate_transparent_shape, debug_shape_tracks_combined},
    detect::{DefaultDetector, Detector},
    ecs::Resources,
    mat::OwnedMat,
    models::Localization,
    run::FPS,
    solvers::{RuneSolver, TransparentShapeSolver, ViolettaSolver},
    utils::DatasetDir,
};

/// Embedded transparent shape test fixtures - video plus its ground-truth
/// sidecar (see [`GroundTruthPoint`]) - hoisted to module scope (rather than
/// each accessor method declaring its own `include_bytes!`/`include_str!`
/// statics) so `test_transparent_shape`, `annotate_transparent_shape` and
/// `evaluate_transparent_shape` all share a single embedded copy instead of
/// tripling the binary's fixture payload.
static TRANSPARENT_SHAPE_NORMAL_VIDEO: &[u8] =
    include_bytes!(env!("TRANSPARENT_SHAPE_TEST_NORMAL_VIDEO"));
static TRANSPARENT_SHAPE_HARD_VIDEO: &[u8] =
    include_bytes!(env!("TRANSPARENT_SHAPE_TEST_HARD_VIDEO"));
static TRANSPARENT_SHAPE_FAILURE_VIDEO: &[u8] =
    include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_VIDEO"));
static TRANSPARENT_SHAPE_FAILURE_2_VIDEO: &[u8] =
    include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_2_VIDEO"));
static TRANSPARENT_SHAPE_FAILURE_3_VIDEO: &[u8] =
    include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_3_VIDEO"));

static TRANSPARENT_SHAPE_NORMAL_GROUND_TRUTH: &str =
    include_str!(env!("TRANSPARENT_SHAPE_GROUND_TRUTH_NORMAL"));
static TRANSPARENT_SHAPE_HARD_GROUND_TRUTH: &str =
    include_str!(env!("TRANSPARENT_SHAPE_GROUND_TRUTH_HARD"));
static TRANSPARENT_SHAPE_FAILURE_GROUND_TRUTH: &str =
    include_str!(env!("TRANSPARENT_SHAPE_GROUND_TRUTH_FAILURE"));
static TRANSPARENT_SHAPE_FAILURE_2_GROUND_TRUTH: &str =
    include_str!(env!("TRANSPARENT_SHAPE_GROUND_TRUTH_FAILURE_2"));
static TRANSPARENT_SHAPE_FAILURE_3_GROUND_TRUTH: &str =
    include_str!(env!("TRANSPARENT_SHAPE_GROUND_TRUTH_FAILURE_3"));

/// `(base fixture name, embedded video bytes, embedded ground-truth JSON)`
/// for a difficulty - the base name (no extension) doubles as both the
/// `.mp4`'s and the `.ground_truth.json`'s filename stem, so a ground truth
/// file annotated into the dataset folder can be copied into
/// `backend/resources/` under the exact name this function will look for.
fn transparent_shape_fixture(
    difficulty: TransparentShapeDifficulty,
) -> (&'static str, &'static [u8], &'static str) {
    match difficulty {
        TransparentShapeDifficulty::Normal => (
            "transparent_shape_test_normal",
            TRANSPARENT_SHAPE_NORMAL_VIDEO,
            TRANSPARENT_SHAPE_NORMAL_GROUND_TRUTH,
        ),
        TransparentShapeDifficulty::Hard => (
            "transparent_shape_test_hard",
            TRANSPARENT_SHAPE_HARD_VIDEO,
            TRANSPARENT_SHAPE_HARD_GROUND_TRUTH,
        ),
        TransparentShapeDifficulty::Failure => (
            "transparent_shape_test_failure",
            TRANSPARENT_SHAPE_FAILURE_VIDEO,
            TRANSPARENT_SHAPE_FAILURE_GROUND_TRUTH,
        ),
        TransparentShapeDifficulty::Failure2 => (
            "transparent_shape_test_failure_2",
            TRANSPARENT_SHAPE_FAILURE_2_VIDEO,
            TRANSPARENT_SHAPE_FAILURE_2_GROUND_TRUTH,
        ),
        TransparentShapeDifficulty::Failure3 => (
            "transparent_shape_test_failure_3",
            TRANSPARENT_SHAPE_FAILURE_3_VIDEO,
            TRANSPARENT_SHAPE_FAILURE_3_GROUND_TRUTH,
        ),
    }
}

/// A single human-annotated ground-truth keyframe: the real target's
/// region-relative pixel center at `frame` (matching `TransparentShapeSolver::solve`'s
/// own `frame_index`/coordinate space). Sparse by design - see
/// `interpolate_ground_truth` for how the gaps are filled in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct GroundTruthPoint {
    frame: u64,
    x: i32,
    y: i32,
}

/// Loads a fixture's ground truth, preferring a dataset-folder copy (written
/// by `annotate_transparent_shape`, possibly mid-annotation/corrected) over
/// the embedded resource copy (what's actually committed), falling back to
/// an empty set if neither parses. Always returned sorted by frame - callers
/// (`interpolate_ground_truth`) rely on that ordering.
fn load_ground_truth(name: &str, embedded: &str) -> Vec<GroundTruthPoint> {
    let dataset_file = DatasetDir::GroundTruth
        .to_folder()
        .join(format!("{name}.ground_truth.json"));
    let contents = fs::read_to_string(&dataset_file).unwrap_or_else(|_| embedded.to_string());
    let mut points = serde_json::from_str::<Vec<GroundTruthPoint>>(&contents).unwrap_or_default();
    points.sort_by_key(|point| point.frame);
    points
}

/// Writes a fixture's annotated ground truth to the dataset folder - not
/// `backend/resources` directly, so an in-progress/experimental annotation
/// pass never silently changes what's committed. Copy the resulting file
/// into `backend/resources/<name>.ground_truth.json` by hand once satisfied,
/// mirroring how new video fixtures are already added to this repo.
fn save_ground_truth(name: &str, mut points: Vec<GroundTruthPoint>) -> PathBuf {
    points.sort_by_key(|point| point.frame);
    points.dedup_by_key(|point| point.frame);

    let file = DatasetDir::GroundTruth
        .to_folder()
        .join(format!("{name}.ground_truth.json"));
    let json = serde_json::to_string_pretty(&points).unwrap();
    fs::write(&file, json).unwrap();
    file
}

/// Opens a fixture video for direct, synchronous frame-by-frame reading -
/// used by `annotate_transparent_shape` and `evaluate_transparent_shape`
/// instead of `frame_receiver_from_video`'s realtime-paced background
/// producer, whose bounded channel silently drops frames once its 3-frame
/// buffer fills (fine for real-time playback, but would desync a paused
/// annotation session's or a slower-than-realtime evaluation pass's frame
/// numbering from the actual video - unacceptable for ground truth, which
/// must match `TransparentShapeSolver::solve`'s own `frame_index` exactly).
fn open_video_capture(file: &PathBuf) -> VideoCapture {
    VideoCapture::from_file_def(file.to_str().expect("invalid UTF-8 path"))
        .expect("failed to open video")
}

/// Reads and BGRA-converts the next frame from `capture`, or `None` once the
/// video is exhausted. See `open_video_capture`'s doc comment for why this
/// (rather than the channel-based `frame_receiver_from_video`) backs
/// annotation/evaluation.
fn read_next_frame(capture: &mut VideoCapture) -> Option<Mat> {
    let mut frame = Mat::default();
    if !capture.read(&mut frame).unwrap_or(false) {
        return None;
    }

    convert_bgr_to_bgra(&mut frame);
    Some(frame)
}

/// Linearly interpolates the ground-truth position at `frame` between its
/// surrounding keyframes, clamping to the first/last keyframe outside their
/// range. `points` must be sorted by frame (see `load_ground_truth`).
fn interpolate_ground_truth(points: &[GroundTruthPoint], frame: u64) -> Option<Point> {
    let first = points.first()?;
    if frame <= first.frame {
        return Some(Point::new(first.x, first.y));
    }
    let last = points.last().expect("non-empty, checked above");
    if frame >= last.frame {
        return Some(Point::new(last.x, last.y));
    }

    points.windows(2).find_map(|pair| {
        let [a, b] = pair else {
            unreachable!("windows(2) always yields 2 elements")
        };
        if frame < a.frame || frame > b.frame {
            return None;
        }
        if a.frame == b.frame {
            return Some(Point::new(a.x, a.y));
        }

        let t = (frame - a.frame) as f64 / (b.frame - a.frame) as f64;
        let x = a.x as f64 + (b.x - a.x) as f64 * t;
        let y = a.y as f64 + (b.y - a.y) as f64 * t;
        Some(Point::new(x.round() as i32, y.round() as i32))
    })
}

#[derive(Debug)]
pub struct DebugService {
    state: Sender<DebugState>,
    writer: Option<VideoWriter>,
}

impl Default for DebugService {
    fn default() -> Self {
        Self {
            state: broadcast::channel(1).0,
            writer: None,
        }
    }
}

impl DebugService {
    pub fn poll(&mut self, resources: &mut Resources) {
        if let Some(writer) = self.writer.as_mut()
            && let Some(detector) = resources.detector.as_ref()
        {
            writer.write(&detector.mat()).unwrap();
        }

        if self.state.is_empty() {
            let _ = self.state.send(DebugState {
                is_recording: self.writer.is_some(),
                is_rune_auto_saving: resources.debug.auto_save_rune,
                is_lie_detector_auto_recording: resources.debug.auto_record_lie_detector,
            });
        }
    }

    pub fn subscribe_state(&self) -> Receiver<DebugState> {
        self.state.subscribe()
    }

    pub fn record_video(&mut self, resources: &mut Resources, start: bool) {
        if !start {
            self.writer = None;
            return;
        }

        if resources.detector.is_none() {
            return;
        }

        let detector = resources.detector();
        let frame_size = detector.mat().size().unwrap();

        let id = Alphanumeric.sample_string(&mut rand::rng(), 8);
        let file = DatasetDir::Recordings.to_folder().join(format!("{id}.mp4"));
        let fourcc = VideoWriter::fourcc('H', 'V', 'C', '1').unwrap();

        let mut writer =
            VideoWriter::new(file.to_str().unwrap(), fourcc, FPS as f64, frame_size, true).unwrap();
        writer.write(&detector.mat()).unwrap();

        self.writer = Some(writer);
    }

    pub fn test_spin_rune(&self) {
        static SPIN_TEST_DIR: Dir<'static> = include_dir!("$SPIN_TEST_DIR");
        static SPIN_TEST_IMAGES: LazyLock<Vec<Mat>> = LazyLock::new(|| {
            let mut files = SPIN_TEST_DIR.files().collect::<Vec<_>>();
            files.sort_by_key(|file| file.path().to_str().unwrap());
            files
                .into_iter()
                .map(|file| {
                    let vec = Vector::from_slice(file.contents());
                    let mut mat = imdecode(&vec, IMREAD_COLOR).unwrap();
                    convert_bgr_to_bgra(&mut mat);
                    mat
                })
                .collect()
        });

        spawn_blocking(move || {
            let mut solver = RuneSolver::debug();
            for detector in SPIN_TEST_IMAGES
                .clone()
                .into_iter()
                .map(OwnedMat::from)
                .map(|mat| DefaultDetector::new(mat, Arc::new(Localization::default())))
            {
                solver.solve(&detector);
            }
            destroy_all_windows().unwrap();
        });
    }

    pub fn test_transparent_shape(
        &self,
        mut input: Box<dyn Input>,
        difficulty: TransparentShapeDifficulty,
    ) {
        spawn_blocking(move || {
            let (name, video, _ground_truth) = transparent_shape_fixture(difficulty);
            let file_name = format!("{name}.mp4");
            log::debug!(target: "backend/player", "starting transparent shape test: {file_name}");

            let file = DatasetDir::Root.to_folder().join(&file_name);
            if !file.exists() {
                let _ = fs::write(&file, video);
            }

            let mut frame_rx = frame_receiver_from_video(file);
            let mut solver = TransparentShapeSolver::debug();
            let mut candidate_solver = TransparentShapeSolver::debug_candidate();
            let localization = Arc::new(Localization::default());
            log::debug!(
                target: "backend/player",
                "shipped_run_id={} candidate_run_id={}",
                solver.run_id(),
                candidate_solver.run_id(),
            );

            input.set_window(Window::new("Main HighGUI"));

            // Starts playing; space toggles pause, n single-steps while
            // paused, q/Esc stops the test early. A single combined window
            // (both models overlaid, distinct colors) replaces two
            // independently playing windows, which turned out to be
            // unreliable to compare by eye in real time.
            let mut paused = false;

            loop {
                if frame_rx.is_closed() {
                    return;
                }

                if let Ok(frame) = frame_rx.try_recv() {
                    let region = Rect::new(0, 0, frame.cols(), frame.rows());
                    let detector =
                        DefaultDetector::new(OwnedMat::from(frame), localization.clone());
                    // Same frame through both models so the combined overlay
                    // is directly comparable frame-for-frame.
                    let cursor = solver.solve(&detector, region);
                    candidate_solver.solve(&detector, region);

                    if let Some(cursor) = cursor {
                        input.send_mouse(cursor.x, cursor.y, MouseKind::Move);
                    }

                    let (shipped_tracks, shipped_cursor, bg_direction) = solver.last_debug_state();
                    let (candidate_tracks, candidate_cursor, _) =
                        candidate_solver.last_debug_state();
                    debug_shape_tracks_combined(
                        "Shape Tracks (shipped vs candidate)",
                        Point::new(50, 50),
                        &detector.mat().roi(region).unwrap(),
                        bg_direction,
                        shipped_tracks,
                        shipped_cursor,
                        candidate_tracks,
                        candidate_cursor,
                        paused,
                    );

                    loop {
                        let wait_ms = if paused { 0 } else { 33 };
                        match wait_key(wait_ms).unwrap() {
                            32 => {
                                // space: toggle play/pause
                                paused = !paused;
                                if !paused {
                                    break;
                                }
                            }
                            113 | 27 => return,     // q/Esc: stop the test early
                            110 if paused => break, // n: single-step while paused
                            _ if !paused => break,
                            _ => {}
                        }
                    }
                }
            }
        });
    }

    /// Ground-truth authoring tool for a `TransparentShapeDifficulty` fixture -
    /// reuses `test_transparent_shape`'s playback controls (space=play/pause,
    /// n=step) but runs no solver at all, since a ground truth is a human
    /// judgment call (which shape actually rotates, per
    /// `project_lie_detector_shape_solver_tracking`'s domain notes), not
    /// something to infer from tracking output. Click the real target while
    /// paused, then Enter/'s' to save that click as the current frame's
    /// keyframe and advance; 'd' clears an existing keyframe for the current
    /// frame; q/Esc writes everything annotated so far to the dataset
    /// folder and exits. See `save_ground_truth`'s doc comment for why this
    /// writes to the dataset folder rather than `backend/resources` directly.
    pub fn annotate_transparent_shape(&self, difficulty: TransparentShapeDifficulty) {
        spawn_blocking(move || {
            let (name, video, ground_truth_json) = transparent_shape_fixture(difficulty);
            let file_name = format!("{name}.mp4");
            log::debug!(target: "backend/player", "starting transparent shape annotation: {file_name}");

            let file = DatasetDir::Root.to_folder().join(&file_name);
            if !file.exists() {
                let _ = fs::write(&file, video);
            }

            let mut points = load_ground_truth(name, ground_truth_json);
            let mut capture = open_video_capture(&file);

            let window_name = format!("Annotate Ground Truth - {name}");
            let clicked: Arc<Mutex<Option<Point>>> = Arc::new(Mutex::new(None));
            let mut callback_registered = false;
            // Starts paused so the very first frame can be annotated
            // immediately instead of racing past it.
            let mut paused = true;
            let mut frame_index = 0u64;

            loop {
                let Some(frame) = read_next_frame(&mut capture) else {
                    break;
                };
                frame_index += 1;
                let mut pending: Option<Point> = None;

                loop {
                    let saved = points
                        .iter()
                        .find(|point| point.frame == frame_index)
                        .map(|point| Point::new(point.x, point.y));
                    debug_annotate_transparent_shape(
                        &window_name,
                        &frame,
                        frame_index,
                        saved,
                        pending,
                        points.len(),
                        paused,
                    );

                    if !callback_registered {
                        let clicked = clicked.clone();
                        let _ = set_mouse_callback(
                            &window_name,
                            Some(Box::new(move |event, x, y, _flags| {
                                if event == MouseEventTypes::EVENT_LBUTTONDOWN as i32 {
                                    *clicked.lock().unwrap() = Some(Point::new(x, y));
                                }
                            })),
                        );
                        callback_registered = true;
                    }

                    if let Some(point) = clicked.lock().unwrap().take() {
                        pending = Some(point);
                    }

                    let wait_ms = if paused { 30 } else { 33 };
                    match wait_key(wait_ms).unwrap() {
                        32 => {
                            // space: toggle play/pause
                            paused = !paused;
                            if !paused {
                                break;
                            }
                        }
                        113 | 27 => {
                            // q/Esc: save what's annotated so far and stop
                            let file = save_ground_truth(name, points.clone());
                            log::debug!(target: "backend/player", "saved {} ground truth point(s) to {file:?}", points.len());
                            destroy_all_windows().unwrap();
                            return;
                        }
                        13 | 115 if pending.is_some() => {
                            // Enter/'s': confirm the pending click for this
                            // frame as its ground truth, then advance
                            let point = pending.take().unwrap();
                            points.retain(|existing| existing.frame != frame_index);
                            points.push(GroundTruthPoint {
                                frame: frame_index,
                                x: point.x,
                                y: point.y,
                            });
                            break;
                        }
                        100 => {
                            // 'd': clear this frame's existing ground truth point
                            points.retain(|existing| existing.frame != frame_index);
                        }
                        110 if paused => break, // n: single-step while paused
                        _ if !paused => break,
                        _ => {}
                    }
                }
            }

            let file = save_ground_truth(name, points.clone());
            log::debug!(target: "backend/player", "saved {} ground truth point(s) to {file:?}", points.len());
            destroy_all_windows().unwrap();
        });
    }

    /// Scores the shipped and candidate solvers for a `TransparentShapeDifficulty`
    /// fixture against its human-annotated ground truth (see
    /// `annotate_transparent_shape`), printing one summary line per solver via
    /// `debug!(target: "backend/player", ...)` - grep for `frame_evaluation` -
    /// plus a full per-frame report written to the dataset folder for later
    /// archiving/diffing. Headless (no HighGUI window, no mouse movement):
    /// this is purely a measurement pass, not something to eyeball.
    pub fn evaluate_transparent_shape(&self, difficulty: TransparentShapeDifficulty) {
        spawn_blocking(move || {
            let (name, video, ground_truth_json) = transparent_shape_fixture(difficulty);
            let file_name = format!("{name}.mp4");
            log::debug!(target: "backend/player", "starting transparent shape evaluation: {file_name}");

            let file = DatasetDir::Root.to_folder().join(&file_name);
            if !file.exists() {
                let _ = fs::write(&file, video);
            }

            let points = load_ground_truth(name, ground_truth_json);
            if points.is_empty() {
                log::debug!(target: "backend/player", "frame_evaluation fixture={name} skipped: no ground truth annotated yet");
                return;
            }

            let mut capture = open_video_capture(&file);
            let localization = Arc::new(Localization::default());
            let mut solver = TransparentShapeSolver::debug();
            let mut candidate_solver = TransparentShapeSolver::debug_candidate();
            log::debug!(
                target: "backend/player",
                "evaluating fixture={name} shipped_run_id={} candidate_run_id={}",
                solver.run_id(),
                candidate_solver.run_id(),
            );

            let mut shipped_errors: Vec<(u64, Point, Point, f64)> = Vec::new();
            let mut candidate_errors: Vec<(u64, Point, Point, f64)> = Vec::new();
            let mut frame_index = 0u64;

            while let Some(frame) = read_next_frame(&mut capture) {
                frame_index += 1;
                let region = Rect::new(0, 0, frame.cols(), frame.rows());
                let detector = DefaultDetector::new(OwnedMat::from(frame), localization.clone());

                // Same frame through both models so the two reports line up
                // frame-for-frame, matching test_transparent_shape's setup.
                let shipped_cursor = solver.solve(&detector, region);
                let candidate_cursor = candidate_solver.solve(&detector, region);

                let Some(ground_truth) = interpolate_ground_truth(&points, frame_index) else {
                    continue;
                };
                if let Some(cursor) = shipped_cursor {
                    shipped_errors.push((
                        frame_index,
                        ground_truth,
                        cursor,
                        pixel_distance(ground_truth, cursor),
                    ));
                }
                if let Some(cursor) = candidate_cursor {
                    candidate_errors.push((
                        frame_index,
                        ground_truth,
                        cursor,
                        pixel_distance(ground_truth, cursor),
                    ));
                }
            }

            let (switches, dead_reckoning) = solver.debug_counters();
            summarize_transparent_shape_evaluation(
                "shipped",
                solver.run_id(),
                name,
                &shipped_errors,
                switches,
                dead_reckoning,
                points.len(),
            );
            let (switches, dead_reckoning) = candidate_solver.debug_counters();
            summarize_transparent_shape_evaluation(
                "candidate",
                candidate_solver.run_id(),
                name,
                &candidate_errors,
                switches,
                dead_reckoning,
                points.len(),
            );
        });
    }

    pub fn diagnose_cuda(&self) -> String {
        crate::detect::diagnose_cuda()
    }

    pub fn test_violetta(&self, mut input: Box<dyn Input>) {
        static VIDEO: &[u8] = include_bytes!(env!("VIOLETTA_TEST_VIDEO"));

        spawn_blocking(move || {
            let file = DatasetDir::Root.to_folder().join("violetta_test.mp4");
            if !file.exists() {
                let _ = fs::write(&file, VIDEO);
            }

            let mut frame_rx = frame_receiver_from_video(file);
            let mut solver = ViolettaSolver::debug();
            let localization = Arc::new(Localization::default());

            input.set_window(Window::new("Main HighGUI"));

            loop {
                if frame_rx.is_closed() {
                    return;
                }

                if let Ok(frame) = frame_rx.try_recv() {
                    let region = Rect::new(0, 0, frame.cols(), frame.rows());
                    let detector =
                        DefaultDetector::new(OwnedMat::from(frame), localization.clone());
                    if let Some(cursor) = solver.solve(&detector, region) {
                        input.send_mouse(cursor.x, cursor.y, MouseKind::Move);
                    }
                }
            }
        });
    }
}

fn frame_receiver_from_video(file: PathBuf) -> mpsc::Receiver<Mat> {
    fn read_and_send_frame(capture: &mut VideoCapture, tx: &mpsc::Sender<Mat>) -> bool {
        let mut frame = Mat::default();
        if !capture.read(&mut frame).unwrap_or(false) {
            return false;
        }

        convert_bgr_to_bgra(&mut frame);
        let _ = tx.try_send(frame);

        true
    }

    let (tx, rx) = mpsc::channel(3);
    let mut capture = VideoCapture::from_file_def(file.to_str().expect("invalid UTF-8 path"))
        .expect("failed to open video");

    spawn_blocking(move || {
        loop_with_fps(FPS, || read_and_send_frame(&mut capture, &tx));
    });

    rx
}

fn convert_bgr_to_bgra(frame: &mut Mat) {
    unsafe {
        frame.modify_inplace(|src, dst| {
            cvt_color_def(src, dst, COLOR_BGR2BGRA).expect("color conversion failed");
        });
    }
}

fn loop_with_fps(fps: u32, mut on_tick: impl FnMut() -> bool) {
    let nanos_per_frame = (1_000_000_000 / fps) as u128;
    loop {
        let start = Instant::now();

        if !on_tick() {
            return;
        }

        let now = Instant::now();
        let elapsed_duration = now.duration_since(start);
        let elapsed_nanos = elapsed_duration.as_nanos();
        if elapsed_nanos <= nanos_per_frame {
            sleep(Duration::new(0, (nanos_per_frame - elapsed_nanos) as u32));
        }
    }
}

fn pixel_distance(a: Point, b: Point) -> f64 {
    (((a.x - b.x).pow(2) + (a.y - b.y).pow(2)) as f64).sqrt()
}

/// A solver's cursor is considered a correct lock (not just "closer than the
/// alternative") within this many pixels of the annotated ground truth -
/// roughly half a typical detected shape's width, i.e. still landing on the
/// real target rather than a neighboring decoy.
const GROUND_TRUTH_LOCK_RADIUS_PX: f64 = 60.0;

#[derive(Serialize)]
struct FrameEvaluation {
    frame: u64,
    ground_truth_x: i32,
    ground_truth_y: i32,
    cursor_x: i32,
    cursor_y: i32,
    error: f64,
    locked: bool,
}

/// Logs one `frame_evaluation` summary line for `solver_name` against
/// `fixture_name`'s ground truth - grep-able the same way
/// `test_transparent_shape`'s existing per-frame log already is, so a tuning
/// attempt's before/after can be diffed without eyeballing the replay - and
/// writes the full per-frame breakdown to
/// `<dataset>/ground_truth/evaluations/` for later archiving.
fn summarize_transparent_shape_evaluation(
    solver_name: &str,
    run_id: u64,
    fixture_name: &str,
    frames: &[(u64, Point, Point, f64)],
    switches: u32,
    dead_reckoning_frames: u32,
    annotated_count: usize,
) {
    if frames.is_empty() {
        log::debug!(
            target: "backend/player",
            "frame_evaluation solver={solver_name} run_id={run_id} fixture={fixture_name} skipped: solver returned no cursor for any frame"
        );
        return;
    }

    let mut errors = frames
        .iter()
        .map(|&(_, _, _, error)| error)
        .collect::<Vec<_>>();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let median = errors[errors.len() / 2];
    let p95 = errors[(((errors.len() - 1) as f64) * 0.95).round() as usize];
    let max = *errors.last().unwrap();
    let locked_count = errors
        .iter()
        .filter(|&&error| error <= GROUND_TRUTH_LOCK_RADIUS_PX)
        .count();
    let locked_pct = locked_count as f64 / errors.len() as f64 * 100.0;

    log::debug!(
        target: "backend/player",
        "frame_evaluation solver={solver_name} run_id={run_id} fixture={fixture_name} frames={} annotated={annotated_count} mean_err={mean:.1} median_err={median:.1} p95_err={p95:.1} max_err={max:.1} locked_pct={locked_pct:.1} switches={switches} dead_reckoning_frames={dead_reckoning_frames}",
        errors.len(),
    );

    let report = frames
        .iter()
        .map(|&(frame, ground_truth, cursor, error)| FrameEvaluation {
            frame,
            ground_truth_x: ground_truth.x,
            ground_truth_y: ground_truth.y,
            cursor_x: cursor.x,
            cursor_y: cursor.y,
            error,
            locked: error <= GROUND_TRUTH_LOCK_RADIUS_PX,
        })
        .collect::<Vec<_>>();
    let Ok(json) = serde_json::to_string_pretty(&report) else {
        return;
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let dir = DatasetDir::GroundTruth.to_folder().join("evaluations");
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(
        dir.join(format!("{fixture_name}_{solver_name}_{timestamp}.json")),
        json,
    );
}
