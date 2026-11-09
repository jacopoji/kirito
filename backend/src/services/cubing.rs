use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use log::info;
use opencv::core::Rect;
use tokio::sync::watch;

use crate::{
    bridge::{InputKeyOptions, KeyKind},
    cubing::{
        RollHistory, evaluate, format_totals, potential_lines_roi, select_lines, stat_totals,
    },
    detect::Detector,
    ecs::Resources,
    models::{CubingSettings, CubingState, CubingStatus, CubingStopReason},
    notification::NotificationKind,
    task::{Task, Update, update_detection_task},
};

/// How long the user has to focus the game before the first roll.
const COUNTDOWN: Duration = Duration::from_secs(5);
/// Presses per roll: one on Reset, the rest through its confirmation prompts.
const PRESSES_PER_ROLL: u32 = 6;
const FIRST_PRESS_GAP: Duration = Duration::from_millis(100);
const PRESS_GAP: Duration = Duration::from_millis(30);
/// How long a roll takes to replace the lines on screen.
///
/// Reading any sooner sees the old lines, and rolling again would throw away an unseen result.
const ROLL_SETTLE: Duration = Duration::from_millis(600);
const ROLL_SETTLE_JITTER_MS: u64 = 150;
const KEY_HOLD_MS: std::ops::RangeInclusive<u64> = 20..=35;
/// How long to wait for a new frame after a roll has settled.
const FRAME_TIMEOUT: Duration = Duration::from_secs(5);
/// Rereads before giving up on a window showing no text, which also happens mid-animation.
const MAX_EMPTY_READS: u32 = 5;
const EMPTY_READ_DELAY: Duration = Duration::from_millis(300);
const MAX_LOG_LEN: usize = 100;

#[derive(Debug)]
enum Phase {
    Idle,
    Locating,
    Reading,
    Countdown {
        until: Instant,
    },
    Pressing {
        remaining: u32,
        next_at: Instant,
    },
    /// Waiting for a roll to show, then for a frame captured after `last_frame`.
    Waiting {
        until: Instant,
        timeout: Instant,
        last_frame: Arc<dyn Detector>,
    },
}

/// Rolls the open cube window until the potential satisfies the configured rules.
///
/// Runs from the tick loop while the bot is halted, so the two never send keys over each other.
#[derive(Debug)]
pub struct CubingService {
    state_tx: watch::Sender<CubingState>,
    phase: Phase,
    settings: CubingSettings,
    roi: Rect,
    /// Whether the next read is the check of the potential the item already has.
    is_initial_read: bool,
    empty_reads: u32,
    history: RollHistory,
    locate_task: Option<Task<Result<Rect>>>,
    read_task: Option<Task<Result<Vec<String>>>>,
}

impl Default for CubingService {
    fn default() -> Self {
        Self {
            state_tx: watch::channel(CubingState::default()).0,
            phase: Phase::Idle,
            settings: CubingSettings::default(),
            roi: Rect::default(),
            is_initial_read: true,
            empty_reads: 0,
            history: RollHistory::default(),
            locate_task: None,
            read_task: None,
        }
    }
}

impl CubingService {
    pub fn subscribe_state(&self) -> watch::Receiver<CubingState> {
        self.state_tx.subscribe()
    }

    pub fn is_running(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }

    pub fn start(&mut self, settings: CubingSettings, resources: &Resources) {
        if self.is_running() {
            return;
        }

        let refused = if !resources.operation.halting() {
            Some(CubingStopReason::BotRunning)
        } else if !settings.has_usable_rule() {
            Some(CubingStopReason::NoUsableRule)
        } else {
            None
        };
        if let Some(reason) = refused {
            self.state_tx.send_replace(CubingState {
                status: CubingStatus::Stopped(reason),
                ..CubingState::default()
            });
            return;
        }

        self.settings = settings;
        self.is_initial_read = true;
        self.empty_reads = 0;
        self.history = RollHistory::default();
        self.locate_task = None;
        self.read_task = None;
        self.phase = Phase::Locating;
        self.state_tx.send_replace(CubingState {
            status: CubingStatus::Starting,
            ..CubingState::default()
        });
    }

    pub fn stop(&mut self) {
        if self.is_running() {
            self.finish(CubingStopReason::User);
        }
    }

    pub fn poll(&mut self, resources: &mut Resources) {
        self.poll_at(resources, Instant::now());
    }

    fn poll_at(&mut self, resources: &mut Resources, now: Instant) {
        if !self.is_running() {
            return;
        }
        if !resources.operation.halting() {
            self.finish(CubingStopReason::BotStarted);
            return;
        }
        if resources.detector.is_none() {
            self.finish(CubingStopReason::CaptureStalled);
            return;
        }

        match &mut self.phase {
            Phase::Idle => (),
            Phase::Locating => self.update_locating(resources),
            Phase::Reading => self.update_reading(resources, now),
            Phase::Countdown { until } => {
                let until = *until;
                if now >= until {
                    self.set_status(CubingStatus::Rolling);
                    self.phase = Phase::Reading;
                } else {
                    let secs = (until - now).as_secs_f32().ceil() as u32;
                    self.set_status(CubingStatus::Countdown(secs));
                }
            }
            Phase::Pressing { remaining, next_at } => {
                if now < *next_at || !resources.input.is_key_cleared(KeyKind::Space) {
                    return;
                }

                let hold_ms = resources.rng.random_range(KEY_HOLD_MS);
                resources.input.send_key_with_options(
                    KeyKind::Space,
                    InputKeyOptions::default().down_ms(hold_ms),
                );
                *remaining -= 1;

                if *remaining > 0 {
                    let gap = if *remaining == PRESSES_PER_ROLL - 1 {
                        FIRST_PRESS_GAP
                    } else {
                        PRESS_GAP
                    };
                    *next_at = now + gap;
                } else {
                    let jitter = resources.rng.random_range(0..=ROLL_SETTLE_JITTER_MS);
                    let settle = ROLL_SETTLE + Duration::from_millis(jitter);
                    self.wait_for_new_frame(resources, now, settle);
                }
            }
            Phase::Waiting {
                until,
                timeout,
                last_frame,
            } => {
                if now < *until {
                    return;
                }

                let current = resources.detector_cloned();
                if !is_same_frame(&current, last_frame) {
                    self.phase = Phase::Reading;
                } else if now >= *timeout {
                    self.finish(CubingStopReason::CaptureStalled);
                }
            }
        }
    }

    fn wait_for_new_frame(&mut self, resources: &Resources, now: Instant, wait: Duration) {
        self.phase = Phase::Waiting {
            until: now + wait,
            timeout: now + wait + FRAME_TIMEOUT,
            last_frame: resources.detector_cloned(),
        };
    }

    fn update_locating(&mut self, resources: &Resources) {
        let kind = self.settings.cube_kind;
        match update_detection_task(resources, 0, &mut self.locate_task, move |detector| {
            detector.detect_cube_reset_button(kind)
        }) {
            Update::Ok(reset_button) => {
                self.locate_task = None;
                self.roi = potential_lines_roi(kind, reset_button);
                self.phase = Phase::Reading;
            }
            Update::Err(_) => {
                self.locate_task = None;
                self.finish(CubingStopReason::ResetButtonNotFound);
            }
            Update::Pending => (),
        }
    }

    fn update_reading(&mut self, resources: &mut Resources, now: Instant) {
        let roi = self.roi;
        match update_detection_task(resources, 0, &mut self.read_task, move |detector| {
            Ok(detector.detect_cube_potential_rows(roi))
        }) {
            Update::Ok(rows) => {
                self.read_task = None;
                let lines = select_lines(&rows);
                if lines.is_empty() {
                    self.on_nothing_read(resources, now);
                } else {
                    self.empty_reads = 0;
                    self.on_lines_read(resources, now, lines);
                }
            }
            Update::Err(_) => {
                self.read_task = None;
                self.on_nothing_read(resources, now);
            }
            Update::Pending => (),
        }
    }

    fn on_nothing_read(&mut self, resources: &Resources, now: Instant) {
        self.empty_reads += 1;
        if self.empty_reads >= MAX_EMPTY_READS {
            self.finish(CubingStopReason::NoLinesRead);
        } else {
            self.wait_for_new_frame(resources, now, EMPTY_READ_DELAY);
        }
    }

    fn on_lines_read(&mut self, resources: &Resources, now: Instant, lines: Vec<String>) {
        let totals = format_totals(&stat_totals(&lines));
        let text = lines.join(" | ");

        // The first read is only of what the item already had, so it is not evidence of being
        // unable to roll.
        if !self.is_initial_read && self.history.push_is_stuck(&lines) {
            self.push_log(format!("{text}    STOP (same result 5 times)"));
            self.finish(CubingStopReason::SameResult);
            return;
        }

        if let Some(reason) = evaluate(&lines, &self.settings) {
            self.push_log(format!("{text}    PASS ({reason}; {totals})"));
            resources
                .notification
                .schedule_notification(NotificationKind::CubingRequirementsMet);
            self.finish(CubingStopReason::RequirementsMet);
            return;
        }

        self.push_log(format!("{text}    REJECT ({totals})"));
        if self.is_initial_read {
            self.is_initial_read = false;
            self.set_status(CubingStatus::Countdown(COUNTDOWN.as_secs() as u32));
            self.phase = Phase::Countdown {
                until: now + COUNTDOWN,
            };
        } else {
            self.state_tx.send_modify(|state| state.rolls += 1);
            self.phase = Phase::Pressing {
                remaining: PRESSES_PER_ROLL,
                next_at: now,
            };
        }
    }

    fn finish(&mut self, reason: CubingStopReason) {
        if let Some(mut task) = self.locate_task.take() {
            task.abort();
        }
        if let Some(mut task) = self.read_task.take() {
            task.abort();
        }
        self.phase = Phase::Idle;
        info!(target: "backend/cubing", "cubing stopped: {reason}");
        self.set_status(CubingStatus::Stopped(reason));
    }

    fn set_status(&self, status: CubingStatus) {
        self.state_tx.send_if_modified(|state| {
            let changed = state.status != status;
            state.status = status;
            changed
        });
    }

    fn push_log(&self, line: String) {
        self.state_tx.send_modify(|state| {
            state.log.insert(0, line);
            state.log.truncate(MAX_LOG_LEN);
        });
    }
}

#[inline]
fn is_same_frame(first: &Arc<dyn Detector>, second: &Arc<dyn Detector>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(first), Arc::as_ptr(second))
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use anyhow::anyhow;
    use mockall::predicate::eq;
    use tokio::task::yield_now;

    use super::*;
    use crate::{
        bridge::MockInput,
        detect::MockDetector,
        models::{CubeKind, CubingStat},
        operation::OperationState,
    };

    const FAILING: [&str; 3] = ["STR: +9%", "Skill MP Cost -30%", "DEX: +6%"];
    const PASSING: [&str; 3] = ["LUK: +12%", "LUK: +9%", "DEX: +6%"];

    fn settings() -> CubingSettings {
        CubingSettings {
            cube_kind: CubeKind::Bright,
            threshold_enabled: true,
            threshold: 21,
            threshold_stats: vec![CubingStat::Luk],
            ..CubingSettings::default()
        }
    }

    fn rows(rows: [&str; 3]) -> Vec<String> {
        rows.iter().map(|row| row.to_string()).collect()
    }

    fn detector(lines: [&str; 3]) -> Arc<dyn Detector> {
        let rows = rows(lines);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_cube_reset_button()
            .returning(|_| Ok(Rect::new(558, 643, 138, 33)));
        detector
            .expect_detect_cube_potential_rows()
            .returning(move |_| rows.clone());
        Arc::new(detector)
    }

    fn input(expected_presses: usize) -> MockInput {
        let mut input = MockInput::default();
        input.expect_is_key_cleared().returning(|_| true);
        input
            .expect_send_key_with_options()
            .with(eq(KeyKind::Space), mockall::predicate::always())
            .times(expected_presses)
            .return_const(());
        input
    }

    fn resources(input: MockInput, detector: Arc<dyn Detector>) -> Resources {
        let mut resources = Resources::new(Some(input), None);
        resources.detector = Some(detector);
        resources.operation.state = OperationState::Halting;
        resources
    }

    /// Polls until the phase changes away from an in-flight detection task.
    async fn poll_task(service: &mut CubingService, resources: &mut Resources, now: Instant) {
        for _ in 0..500 {
            service.poll_at(resources, now);
            if !matches!(service.phase, Phase::Locating | Phase::Reading) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
            yield_now().await;
        }
        panic!("detection task did not finish");
    }

    fn status(service: &CubingService) -> CubingStatus {
        service.subscribe_state().borrow().status
    }

    /// Runs one locate and initial read, leaving `service` in the countdown.
    async fn run_to_countdown(
        service: &mut CubingService,
        resources: &mut Resources,
        now: Instant,
    ) {
        service.start(settings(), resources);
        poll_task(service, resources, now).await;
        assert_matches!(service.phase, Phase::Countdown { .. });
    }

    /// Runs a roll's key presses and waits, leaving `service` about to read the result.
    fn press_and_settle(
        service: &mut CubingService,
        resources: &mut Resources,
        now: &mut Instant,
        next_frame: Arc<dyn Detector>,
    ) {
        for _ in 0..PRESSES_PER_ROLL {
            service.poll_at(resources, *now);
            *now += FIRST_PRESS_GAP;
        }
        assert_matches!(service.phase, Phase::Waiting { .. });
        resources.detector = Some(next_frame);
        *now += ROLL_SETTLE + Duration::from_millis(ROLL_SETTLE_JITTER_MS);
        service.poll_at(resources, *now);
        assert_matches!(service.phase, Phase::Reading);
    }

    #[tokio::test]
    async fn refuses_to_start_while_the_bot_runs() {
        let mut service = CubingService::default();
        let mut resources = resources(input(0), detector(FAILING));
        resources.operation.state = OperationState::Running;

        service.start(settings(), &resources);

        assert!(!service.is_running());
        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::BotRunning)
        );
    }

    #[tokio::test]
    async fn refuses_to_start_without_a_usable_rule() {
        let mut service = CubingService::default();
        let resources = resources(input(0), detector(FAILING));

        service.start(CubingSettings::default(), &resources);

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::NoUsableRule)
        );
    }

    #[tokio::test]
    async fn stops_without_pressing_when_the_window_is_not_open() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_cube_reset_button()
            .returning(|_| Err(anyhow!("not found")));
        let mut service = CubingService::default();
        let mut resources = resources(input(0), Arc::new(detector));

        service.start(settings(), &resources);
        poll_task(&mut service, &mut resources, Instant::now()).await;

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::ResetButtonNotFound)
        );
    }

    #[tokio::test]
    async fn stops_without_pressing_when_the_item_already_passes() {
        let mut service = CubingService::default();
        let mut resources = resources(input(0), detector(PASSING));
        let now = Instant::now();

        service.start(settings(), &resources);
        poll_task(&mut service, &mut resources, now).await;

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::RequirementsMet)
        );
    }

    #[tokio::test]
    async fn rolls_until_a_result_passes_and_does_not_press_after() {
        let mut service = CubingService::default();
        let mut resources = resources(input(PRESSES_PER_ROLL as usize), detector(FAILING));
        let mut now = Instant::now();

        run_to_countdown(&mut service, &mut resources, now).await;
        now += COUNTDOWN;
        service.poll_at(&mut resources, now);
        assert_eq!(status(&service), CubingStatus::Rolling);
        poll_task(&mut service, &mut resources, now).await;

        press_and_settle(&mut service, &mut resources, &mut now, detector(PASSING));
        poll_task(&mut service, &mut resources, now).await;

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::RequirementsMet)
        );
        assert_eq!(service.subscribe_state().borrow().rolls, 1);
    }

    #[tokio::test]
    async fn stops_when_rolling_stops_changing_the_result() {
        let mut service = CubingService::default();
        // The fifth identical read stops before pressing for a fifth time.
        let mut resources = resources(input(PRESSES_PER_ROLL as usize * 4), detector(FAILING));
        let mut now = Instant::now();

        run_to_countdown(&mut service, &mut resources, now).await;
        now += COUNTDOWN;
        service.poll_at(&mut resources, now);
        poll_task(&mut service, &mut resources, now).await;
        for _ in 0..4 {
            press_and_settle(&mut service, &mut resources, &mut now, detector(FAILING));
            poll_task(&mut service, &mut resources, now).await;
        }

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::SameResult)
        );
    }

    #[tokio::test]
    async fn stops_when_the_bot_starts_mid_run() {
        let mut service = CubingService::default();
        let mut resources = resources(input(0), detector(FAILING));
        let now = Instant::now();
        run_to_countdown(&mut service, &mut resources, now).await;

        resources.operation.state = OperationState::Running;
        service.poll_at(&mut resources, now);

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::BotStarted)
        );
    }

    #[tokio::test]
    async fn stops_when_no_new_frame_arrives_after_a_roll() {
        let mut service = CubingService::default();
        let frame = detector(FAILING);
        let mut resources = resources(input(PRESSES_PER_ROLL as usize), frame.clone());
        let mut now = Instant::now();
        run_to_countdown(&mut service, &mut resources, now).await;
        now += COUNTDOWN;
        service.poll_at(&mut resources, now);
        poll_task(&mut service, &mut resources, now).await;

        for _ in 0..PRESSES_PER_ROLL {
            service.poll_at(&mut resources, now);
            now += FIRST_PRESS_GAP;
        }
        now += ROLL_SETTLE + Duration::from_millis(ROLL_SETTLE_JITTER_MS);
        service.poll_at(&mut resources, now);
        assert_matches!(service.phase, Phase::Waiting { .. });
        now += FRAME_TIMEOUT;
        service.poll_at(&mut resources, now);

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::CaptureStalled)
        );
    }

    #[tokio::test]
    async fn stops_after_repeatedly_reading_nothing() {
        let mut detector_mock = MockDetector::default();
        detector_mock
            .expect_detect_cube_reset_button()
            .returning(|_| Ok(Rect::new(558, 643, 138, 33)));
        detector_mock
            .expect_detect_cube_potential_rows()
            .returning(|_| vec![]);
        let mut service = CubingService::default();
        let mut resources = resources(input(0), Arc::new(detector_mock));
        let mut now = Instant::now();

        service.start(settings(), &resources);
        poll_task(&mut service, &mut resources, now).await;
        for _ in 1..MAX_EMPTY_READS {
            let mut next = MockDetector::default();
            next.expect_detect_cube_potential_rows()
                .returning(|_| vec![]);
            resources.detector = Some(Arc::new(next));
            now += EMPTY_READ_DELAY;
            service.poll_at(&mut resources, now);
            poll_task(&mut service, &mut resources, now).await;
        }

        assert_eq!(
            status(&service),
            CubingStatus::Stopped(CubingStopReason::NoLinesRead)
        );
    }
}
