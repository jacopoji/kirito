use std::{
    fs,
    path::PathBuf,
    sync::{Arc, LazyLock},
    thread::sleep,
    time::{Duration, Instant},
};

use include_dir::{Dir, include_dir};
use opencv::{
    core::{Mat, MatTraitConst, ModifyInplace, Point, Rect, Vector},
    highgui::{destroy_all_windows, wait_key},
    imgcodecs::{IMREAD_COLOR, imdecode},
    imgproc::{COLOR_BGR2BGRA, cvt_color_def},
    videoio::{VideoCapture, VideoCaptureTrait, VideoWriter, VideoWriterTrait},
};
use platforms::Window;
use rand::distr::SampleString;
use rand_distr::Alphanumeric;
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
    debug::debug_shape_tracks_combined,
    detect::{DefaultDetector, Detector},
    ecs::Resources,
    mat::OwnedMat,
    models::Localization,
    run::FPS,
    solvers::{RuneSolver, TransparentShapeSolver, ViolettaSolver},
    utils::DatasetDir,
};

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
        static NORMAL_VIDEO: &[u8] = include_bytes!(env!("TRANSPARENT_SHAPE_TEST_NORMAL_VIDEO"));
        static HARD_VIDEO: &[u8] = include_bytes!(env!("TRANSPARENT_SHAPE_TEST_HARD_VIDEO"));
        static FAILURE_VIDEO: &[u8] = include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_VIDEO"));
        static FAILURE_2_VIDEO: &[u8] =
            include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_2_VIDEO"));
        static FAILURE_3_VIDEO: &[u8] =
            include_bytes!(env!("TRANSPARENT_SHAPE_TEST_FAILURE_3_VIDEO"));

        spawn_blocking(move || {
            let (name, video) = match difficulty {
                TransparentShapeDifficulty::Normal => {
                    ("transparent_shape_test_normal.mp4", NORMAL_VIDEO)
                }
                TransparentShapeDifficulty::Hard => ("transparent_shape_test_hard.mp4", HARD_VIDEO),
                TransparentShapeDifficulty::Failure => {
                    ("transparent_shape_test_failure.mp4", FAILURE_VIDEO)
                }
                TransparentShapeDifficulty::Failure2 => {
                    ("transparent_shape_test_failure_2.mp4", FAILURE_2_VIDEO)
                }
                TransparentShapeDifficulty::Failure3 => {
                    ("transparent_shape_test_failure_3.mp4", FAILURE_3_VIDEO)
                }
            };
            log::debug!(target: "backend/player", "starting transparent shape test: {name}");

            let file = DatasetDir::Root.to_folder().join(name);
            if !file.exists() {
                let _ = fs::write(&file, video);
            }

            let mut frame_rx = frame_receiver_from_video(file);
            let mut solver = TransparentShapeSolver::debug();
            let mut candidate_solver = TransparentShapeSolver::debug_candidate();
            let localization = Arc::new(Localization::default());

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
