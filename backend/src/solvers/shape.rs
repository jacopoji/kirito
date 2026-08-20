use std::{
    collections::HashMap,
    ops::Div,
    sync::atomic::{AtomicU64, Ordering},
};

use log::debug;
use opencv::core::{Point, Point_, Point2d, Rect};

use crate::{
    detect::Detector,
    run::FPS,
    tracker::{ByteTracker, Detection, IouGating, STrack},
};

/// Source for `TransparentShapeSolver::run_id` - a per-instance id included
/// in every one of this solver's `debug!` lines so they can be told apart
/// when multiple solver instances log concurrently (e.g. two
/// `evaluate_transparent_shape` calls, or that call's own shipped/candidate
/// pair, running on different `spawn_blocking` threads at once). Without
/// this, per-frame log lines from separate runs interleave indistinguishably.
static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct TransparentShapeSolver {
    tracker: ByteTracker,
    current_track_id: Option<u64>,
    candidate_track_id: Option<u64>,
    candidate_track_count: u32,
    last_cursor: Option<Point>,
    last_velocity: Option<Point2d>,
    bg_direction: Point2d,
    /// Consecutive frames spent dead-reckoning because the current track
    /// wasn't found (see `solve`'s `None` branch) - reset back to zero the
    /// moment a real track match resumes. Also gates `update_and_find_best_track`'s
    /// switch to spatial re-acquisition and how aggressively `lost_relaxation`
    /// loosens matching thresholds - unlike dead reckoning itself (see
    /// `dead_reckoning_velocity`), this never stops incrementing on its own.
    lost_frame_count: u32,
    /// Whether `update_and_find_best_track` is still allowed to grant the
    /// hysteresis bonus/angle relaxation to the current track. Starts `true`
    /// and latches permanently `false` the first time the lock is ever lost
    /// or switched (see `solve`'s `None` branch and the switch check in its
    /// `Some` branch) - never re-enabled afterward, even once a new track
    /// stabilizes. Domain fact motivating this: during the puzzle's opening
    /// countdown the real target is static and centered, so the very first
    /// lock (`update_initial_track_if_needed`) is trustworthy with certainty
    /// no later re-lock can match - once that original anchor is lost, every
    /// subsequent pick is just a guess like any other, and protecting a
    /// guess with the same confidence as the known-correct start is what let
    /// a bad reacquisition entrench itself on the Failure 4 replay. A
    /// re-earned-trust version (grant hysteresis back after N stable frames)
    /// was tried and measured worse overall - see git history around this
    /// comment - net-improved Failure 4 but regressed Failure 2's mean error
    /// (72.9 to 99.3) by letting a wrong mid-round lock re-entrench too.
    hysteresis_active: bool,
    /// The velocity currently driving indefinite dead reckoning while lost -
    /// seeded from `last_velocity` on the first lost frame, then decayed and
    /// bounced off `region`'s edges every subsequent lost frame (see `solve`'s
    /// `None` branch). Kept separate from `last_velocity` (the last *real*
    /// detected velocity) so dead reckoning can fade its own confidence over
    /// time without corrupting the seed a future dead-reckoning run would
    /// start from. Reset to `None` the moment a real track match resumes.
    dead_reckoning_velocity: Option<Point2d>,
    /// EMA-smoothed `track_background_score` per track id - see the KNOWN
    /// ISSUE comment above `track_background_score`. The flat hysteresis
    /// bonus/angle relaxation there guards against a challenger edging out
    /// the current track by a small margin, but does nothing when two
    /// tracks are both genuinely, strongly diverging from the background at
    /// once (real observed case: two tracks both >=45 degrees off, raw
    /// score swinging 0.3-0.5 frame to frame as `bg_direction` itself keeps
    /// drifting) - the bonus is dwarfed and the winner ping-pongs every few
    /// frames. Smoothing damps single-frame swings so only a sustained lead
    /// changes the winner, without touching the existing threshold/bonus
    /// logic (smoothing wraps the already-biased score).
    track_scores: HashMap<u64, f64>,
    #[cfg(debug_assertions)]
    is_debugging: bool,
    /// When set, `solve` runs detection through the candidate (in-training)
    /// model instead of the shipped one - lets the debug test harness run
    /// two solver instances over the same video and compare tracking
    /// behavior side by side. Debug-only.
    #[cfg(debug_assertions)]
    use_candidate_model: bool,
    /// This frame's tracks/cursor, captured for the debug harness to read
    /// after `solve` returns instead of drawing them immediately - lets two
    /// solver instances (shipped vs. candidate) be combined into a single
    /// overlay by the caller rather than each popping its own window.
    /// Debug-only.
    #[cfg(debug_assertions)]
    last_debug_tracks: Vec<STrack>,
    #[cfg(debug_assertions)]
    last_debug_cursor: Point,
    frame_index: u64,
    /// Count of "shape id switches" events (see `solve`'s `Some` branch) -
    /// exposed via `debug_counters` for the ground-truth evaluation harness
    /// to report alongside pixel-error metrics. Debug-only.
    #[cfg(debug_assertions)]
    debug_switch_count: u32,
    /// Count of frames that entered dead reckoning (see `solve`'s `None`
    /// branch) - exposed via `debug_counters`. Debug-only.
    #[cfg(debug_assertions)]
    debug_dead_reckoning_count: u32,
    /// See `NEXT_RUN_ID`'s doc comment.
    run_id: u64,
}

impl Default for TransparentShapeSolver {
    fn default() -> Self {
        Self {
            tracker: ByteTracker::new(FPS as u64, 0.25, 0.1, 0.25, IouGating::None),
            current_track_id: None,
            candidate_track_id: None,
            candidate_track_count: 0,
            last_cursor: None,
            last_velocity: None,
            bg_direction: Point2d::default(),
            lost_frame_count: 0,
            hysteresis_active: true,
            dead_reckoning_velocity: None,
            track_scores: HashMap::new(),
            #[cfg(debug_assertions)]
            is_debugging: false,
            #[cfg(debug_assertions)]
            use_candidate_model: false,
            #[cfg(debug_assertions)]
            last_debug_tracks: Vec::new(),
            #[cfg(debug_assertions)]
            last_debug_cursor: Point::default(),
            frame_index: 0,
            #[cfg(debug_assertions)]
            debug_switch_count: 0,
            #[cfg(debug_assertions)]
            debug_dead_reckoning_count: 0,
            run_id: NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl TransparentShapeSolver {
    #[cfg(debug_assertions)]
    pub fn debug() -> Self {
        let mut default = Self::default();
        default.is_debugging = true;
        default
    }

    /// Same as [`Self::debug`] but runs the candidate (in-training) model -
    /// pair with a normal `debug()` instance fed the same frames to compare
    /// the two side by side.
    #[cfg(debug_assertions)]
    pub fn debug_candidate() -> Self {
        let mut default = Self::debug();
        default.use_candidate_model = true;
        default
    }

    #[cfg(debug_assertions)]
    fn detect_shapes(&self, detector: &dyn Detector, region: Rect) -> Vec<(Rect, f32)> {
        if self.use_candidate_model {
            detector.detect_transparent_shapes_candidate(region)
        } else {
            detector.detect_transparent_shapes(region)
        }
    }

    #[cfg(not(debug_assertions))]
    fn detect_shapes(&self, detector: &dyn Detector, region: Rect) -> Vec<(Rect, f32)> {
        detector.detect_transparent_shapes(region)
    }

    /// This frame's tracks, predicted cursor (region-relative) and
    /// background direction, as last captured by `solve` - for the debug
    /// harness to combine multiple solver instances into a single overlay.
    #[cfg(debug_assertions)]
    pub fn last_debug_state(&self) -> (&[STrack], Point, Point2d) {
        (
            &self.last_debug_tracks,
            self.last_debug_cursor,
            self.bg_direction,
        )
    }

    /// This instance's unique id - see `NEXT_RUN_ID`'s doc comment.
    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    /// `(shape id switches, frames spent dead reckoning)` accumulated so far
    /// - for the ground-truth evaluation harness to report alongside pixel-
    /// error metrics. Debug-only.
    #[cfg(debug_assertions)]
    pub fn debug_counters(&self) -> (u32, u32) {
        (self.debug_switch_count, self.debug_dead_reckoning_count)
    }

    pub fn solve(&mut self, detector: &dyn Detector, region: Rect) -> Option<Point> {
        self.frame_index += 1;

        let shapes = self.detect_shapes(detector, region);
        let tracks = self.tracker.update(
            shapes
                .into_iter()
                .map(|(bbox, score)| Detection::new(bbox, score))
                .collect(),
        );

        self.update_initial_track_if_needed(region, &tracks);
        self.update_background_direction(&tracks);

        debug!(
            target: "backend/player",
            "run_id={} frame={} bg_direction=({:.3},{:.3}) current_track={:?} tracks={:?}",
            self.run_id,
            self.frame_index,
            self.bg_direction.x,
            self.bg_direction.y,
            self.current_track_id,
            tracks
                .iter()
                .map(|t| (t.track_id(), t.rect(), t.kalman_velocity()))
                .collect::<Vec<_>>(),
        );

        let best_track = self.update_and_find_best_track(&tracks, region);

        match best_track {
            Some(track) => {
                self.lost_frame_count = 0;
                self.dead_reckoning_velocity = None;

                let next_cursor = predicted_center(track);
                if self.current_track_id != Some(track.track_id()) {
                    debug!(target: "backend/player", "run_id={} frame={} shape id switches from {:?} to {}", self.run_id, self.frame_index, self.current_track_id, track.track_id());
                    #[cfg(debug_assertions)]
                    {
                        self.debug_switch_count += 1;
                    }
                    // Only the very first (countdown-anchored) lock is
                    // trustworthy enough to protect with hysteresis - see
                    // `hysteresis_active`'s doc comment. `current_track_id`
                    // starts `None`, so this branch also fires for that
                    // first-ever lock; guard against latching it off then.
                    if self.current_track_id.is_some() {
                        self.hysteresis_active = false;
                    }
                }
                self.current_track_id = Some(track.track_id());
                self.last_cursor = Some(next_cursor);
                self.last_velocity = Some(track.kalman_velocity());

                #[cfg(debug_assertions)]
                if self.is_debugging {
                    self.last_debug_tracks = tracks.clone();
                    self.last_debug_cursor = next_cursor;
                }

                Some(region.tl() + next_cursor)
            }
            None => {
                self.lost_frame_count += 1;
                if self.lost_frame_count == 1 {
                    debug!(target: "backend/player", "run_id={} frame={} lost track, entering dead reckoning", self.run_id, self.frame_index);
                }
                // Losing the target at all - not just switching to a
                // different one - is itself reason enough to stop trusting
                // hysteresis for the rest of this round; see
                // `hysteresis_active`'s doc comment.
                self.hysteresis_active = false;
                #[cfg(debug_assertions)]
                {
                    self.debug_dead_reckoning_count += 1;
                }

                let last_cursor = self.last_cursor?;
                let seed_velocity = self.last_velocity.expect("set if last_cursor set") * 1.5;
                let velocity = self.dead_reckoning_velocity.get_or_insert(seed_velocity);
                *velocity *= DEAD_RECKONING_DECAY;

                let mut next_cursor =
                    last_cursor + Point::new(velocity.x.round() as i32, velocity.y.round() as i32);

                // Bounce off the puzzle region's edges instead of freezing or
                // walking off-screen: reflect both the position and the
                // stored velocity's sign on whichever axis overshot, so the
                // guess keeps moving somewhere plausible inside the puzzle
                // indefinitely - the game almost certainly reads a
                // completely stationary cursor as "not tracking" and fails
                // the puzzle outright, which is worse than a merely
                // imperfect guess.
                if next_cursor.x < 0 {
                    next_cursor.x = -next_cursor.x;
                    velocity.x = -velocity.x;
                } else if next_cursor.x >= region.width {
                    next_cursor.x = 2 * (region.width - 1) - next_cursor.x;
                    velocity.x = -velocity.x;
                }
                if next_cursor.y < 0 {
                    next_cursor.y = -next_cursor.y;
                    velocity.y = -velocity.y;
                } else if next_cursor.y >= region.height {
                    next_cursor.y = 2 * (region.height - 1) - next_cursor.y;
                    velocity.y = -velocity.y;
                }
                // Defensive clamp in case a single reflection wasn't enough
                // to land back in-bounds (e.g. a large stale velocity in a
                // narrow region).
                next_cursor.x = next_cursor.x.clamp(0, region.width - 1);
                next_cursor.y = next_cursor.y.clamp(0, region.height - 1);

                self.last_cursor = Some(next_cursor);

                #[cfg(debug_assertions)]
                if self.is_debugging {
                    self.last_debug_tracks = tracks.clone();
                    self.last_debug_cursor = next_cursor;
                }

                Some(region.tl() + next_cursor)
            }
        }
    }

    fn update_initial_track_if_needed(&mut self, region: Rect, tracks: &[STrack]) {
        if self.current_track_id.is_none() {
            let region_mid = mid_point(Rect::new(0, 0, region.width, region.height));
            if let Some(track) = find_track_closest_to(region_mid, tracks) {
                self.current_track_id = Some(track.track_id());
                self.last_cursor = Some(mid_point(track.rect()));
                self.last_velocity = Some(track.kalman_velocity());
            }
        }
    }

    fn update_background_direction(&mut self, tracks: &[STrack]) {
        if let Some(direction) = estimate_background_direction(self.last_cursor, tracks)
            .and_then(|direction| unit(self.bg_direction * 0.5 + direction * 0.5))
        {
            self.bg_direction = direction;
        }
    }

    fn update_and_find_best_track<'a>(
        &mut self,
        tracks: &'a [STrack],
        region: Rect,
    ) -> Option<&'a STrack> {
        let current_track_id = self.current_track_id?;
        let last_cursor = self.last_cursor?;
        let bg_direction = self.bg_direction;
        let lost_frame_count = self.lost_frame_count;
        let frame_index = self.frame_index;
        let run_id = self.run_id;
        let track_scores = &mut self.track_scores;
        // See `hysteresis_active`'s doc comment: only the still-unbroken
        // original lock gets hysteresis, not merely whichever track happens
        // to be `current_track_id` right now.
        let hysteresis_eligible = self.hysteresis_active;

        let best_scored = tracks
            .iter()
            .filter(|track| track.track_id() == current_track_id || track.tracklet_len() >= 1)
            .filter_map(|track| {
                let is_current = track.track_id() == current_track_id && hysteresis_eligible;
                let angle = track_background_degree(track, bg_direction);
                let score = track_background_score(
                    track,
                    last_cursor,
                    bg_direction,
                    region,
                    lost_frame_count,
                    is_current,
                );
                debug!(
                    target: "backend/player",
                    "run_id={} frame={} candidate track={} angle={:?} score={:?} lost_frame_count={}",
                    run_id,
                    frame_index,
                    track.track_id(),
                    angle,
                    score,
                    lost_frame_count,
                );
                let score = score?;
                // Hysteresis bonus - see ROLLBACK NOTE above
                // `CURRENT_TRACK_HYSTERESIS_BONUS` below. Before this fix
                // this whole block was just `Some((track, score?))`, no
                // `is_current` variable existed, and every track
                // (current included) competed on raw `score` alone.
                let score = if is_current {
                    score + CURRENT_TRACK_HYSTERESIS_BONUS
                } else {
                    score
                };
                // EMA-smooth against this same track id's own score last
                // frame - see `track_scores`'s doc comment.
                let smoothed = match track_scores.get(&track.track_id()) {
                    Some(&prev) => {
                        SCORE_SMOOTHING_ALPHA * score + (1.0 - SCORE_SMOOTHING_ALPHA) * prev
                    }
                    None => score,
                };
                track_scores.insert(track.track_id(), smoothed);
                Some((track, smoothed))
            })
            .max_by(|(_, a_score), (_, b_score)| a_score.partial_cmp(b_score).unwrap())
            .map(|(track, _)| track);

        // Dead-reckoning in `solve` is still running by this point (it never
        // stops on its own - see `DEAD_RECKONING_DECAY`), but its guess is
        // now too decayed/bounced to trust as a match target here. A
        // detection dropout this long usually means the target drifted far
        // enough while undetected that `ByteTracker`'s own IOU-based
        // re-association against the frozen `lost` track's last position
        // fails once detection resumes - it reappears as a brand new track.
        // Background-divergence scoring above is unreliable once genuinely
        // lost like this: plenty of unrelated decoys keep scoring
        // "diverging enough" every frame regardless (that's the whole
        // reason lock-on gets stolen in the first place), so a fresh but
        // unrelated decoy would otherwise keep winning over the reborn
        // target's own still-thin history. Once this deep into being lost,
        // spatial proximity to the last known position is trusted first,
        // falling back to the score-based pick only if nothing was even
        // detected nearby.
        let match_track = if lost_frame_count > MAX_LOST_FRAMES_FOR_EXTRAPOLATION {
            find_reacquisition_candidate(tracks, last_cursor, lost_frame_count)
                .inspect(|track| {
                    debug!(
                        target: "backend/player",
                        "run_id={} frame={} reacquisition candidate near last known position: track={}",
                        run_id,
                        frame_index,
                        track.track_id(),
                    );
                })
                .or(best_scored)
        } else {
            best_scored
        };

        if let Some(track) = match_track {
            if track.track_id() == current_track_id {
                self.candidate_track_id = None;
                self.candidate_track_count = 0;
            }

            if self.candidate_track_id == Some(track.track_id()) {
                self.candidate_track_count += 1;
            } else {
                self.candidate_track_id = Some(track.track_id());
                self.candidate_track_count = 0;
            }

            if self.candidate_track_count >= 1 {
                self.candidate_track_id = None;
                self.candidate_track_count = 0;
                return Some(track);
            }
        }

        tracks
            .iter()
            .find(|track| track.track_id() == current_track_id)
    }
}

/// The point past which `update_and_find_best_track` switches from waiting
/// for the original track to falling back to spatial re-acquisition, and how
/// long `lost_relaxation` takes to start loosening matching thresholds.
/// Doesn't gate `solve`'s own dead reckoning - see `DEAD_RECKONING_DECAY` for
/// how that stays bounded without an explicit frame cutoff.
const MAX_LOST_FRAMES_FOR_EXTRAPOLATION: u32 = 5;

/// Per-frame decay applied to `dead_reckoning_velocity` while lost - keeps
/// the cursor perpetually moving (never frozen - see `solve`'s `None`
/// branch) without staying confidently wrong for long. At 30fps this halves
/// speed roughly every 14 frames (~0.5s), so a stale pre-loss velocity fades
/// into a near-standstill drift within a couple of seconds instead of
/// confidently sailing away from the target's real (by-then likely
/// different) position the way the uncapped pre-#35 version did.
const DEAD_RECKONING_DECAY: f64 = 0.95;

/// Base search radius for spatial re-acquisition (roughly 1.5x a typical
/// detected shape's own size) - how far from the last known position a
/// detection can be and still be trusted as the reappeared target even
/// right as dead-reckoning gives up, before any extra drift-driven slack
/// (`REACQUISITION_MAX_EXTRA_RADIUS`) kicks in.
const REACQUISITION_BASE_RADIUS: f64 = 200.0;

/// Additional search radius granted at full `lost_relaxation` - the longer
/// the target's been undetected, the further it could plausibly have
/// drifted, mirroring the same growing-tolerance idea `lost_relaxation` was
/// already built for.
const REACQUISITION_MAX_EXTRA_RADIUS: f64 = 200.0;

fn find_reacquisition_candidate(
    tracks: &[STrack],
    last_cursor: Point,
    lost_frame_count: u32,
) -> Option<&STrack> {
    let max_distance = REACQUISITION_BASE_RADIUS
        + lost_relaxation(lost_frame_count) * REACQUISITION_MAX_EXTRA_RADIUS;

    tracks
        .iter()
        .map(|track| (track, (mid_point(track.rect()) - last_cursor).norm()))
        .filter(|&(_, distance)| distance <= max_distance)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(track, _)| track)
}

impl Drop for TransparentShapeSolver {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if self.is_debugging {
            use opencv::highgui::destroy_all_windows;

            use crate::debug::reset_shape_track_windows;

            let _ = destroy_all_windows();
            reset_shape_track_windows();
        }
    }
}

fn find_track_closest_to(point: Point, tracks: &[STrack]) -> Option<&STrack> {
    tracks.iter().min_by_key(|track| {
        let track_region = track.rect();
        let track_mid =
            track_region.tl() + Point::new(track_region.width / 2, track_region.height / 2);

        (point - track_mid).norm() as i32
    })
}

fn mid_point(rect: Rect) -> Point {
    rect.tl() + Point::new(rect.width / 2, rect.height / 2)
}

fn predicted_center(track: &STrack) -> Point {
    let v = track.kalman_velocity();
    let point = mid_point(track.kalman_rect());

    Point::new(
        (point.x as f64 + v.x).round() as i32,
        (point.y as f64 + v.y).round() as i32,
    )
}

/// Below this many consecutive lost frames, the strict background-divergence
/// requirement below is kept as-is - a fast, confident re-match is
/// preferable when one is available. Past `RELAX_FULL_LOST_FRAMES`, the
/// shape could plausibly be moving with the background for a moment and
/// still be the real target, so the angle gate (and the distance window
/// around the last known cursor) are loosened to avoid getting stuck
/// waiting for divergent motion that may not come for a long time.
const RELAX_START_LOST_FRAMES: u32 = 10;
const RELAX_FULL_LOST_FRAMES: u32 = 60;
const BASE_ANGLE_THRESHOLD: f64 = 45.0;
const MIN_ANGLE_THRESHOLD: f64 = 10.0;

// ROLLBACK NOTE: before the hysteresis fix below, `track_background_score`
// took no `is_current` parameter and applied no threshold relaxation - every
// track (including the currently-locked one) computed `angle_threshold` the
// same way, from `lost_relaxation` alone. `scored_candidate` (in
// `update_and_find_best_track`) applied no score bonus either - that
// filter_map body was just:
//
//     let angle = track_background_degree(track, bg_direction);
//     let score = track_background_score(track, last_cursor, bg_direction, region, lost_frame_count);
//     debug!(...);
//     Some((track, score?))
//
// i.e. every track, current included, competed purely on raw
// `track_background_score` via `max_by` - no `is_current` variable existed
// anywhere in this function. To roll back after further testing: drop the
// `is_current: bool` parameter from `track_background_score` below and the
// `if is_current { angle_threshold = ... }` block inside it; drop the
// `is_current` computation and the post-`score?` bonus block in
// `scored_candidate`; delete `CURRENT_TRACK_HYSTERESIS_BONUS` and
// `CURRENT_TRACK_ANGLE_HYSTERESIS`.
//
/// Flat bonus added to the currently-locked track's own `track_background_score`
/// before comparing against challengers - see the hysteresis fix described in
/// the KNOWN ISSUE comment above `track_background_score`. Scores are
/// `angle / 180.0 * distance_penalty`, so a "strongly diverging" track
/// (`angle >= 60.0`, `distance_penalty == 1.0`) scores >= 0.33; 0.15 absorbs
/// a single-frame marginal edge-out between two tracks that both clear the
/// angle threshold, without being so large the current track can never
/// legitimately lose to a genuinely more-diverging challenger.
const CURRENT_TRACK_HYSTERESIS_BONUS: f64 = 0.15;

/// EMA weight given to this frame's (already bonus-adjusted) score when
/// blending with a track id's own smoothed score from last frame - see
/// `track_scores`'s doc comment on `TransparentShapeSolver`. Lower is
/// heavier smoothing/slower to react; 0.4 was picked to still let a
/// genuinely sustained divergence (several consecutive frames) win within
/// 2-3 frames - comparable to the existing 2-frame `candidate_track_count`
/// confirmation debounce - while damping the single-frame 0.3-0.5 raw score
/// swings observed between two simultaneously-diverging real tracks (see
/// `track_background_score`'s KNOWN ISSUE comment). Replay-verified across
/// all 5 fixtures: fewer switches on Normal/Hard, Failure 1 unchanged (its
/// remaining hijacks are a genuine wide divergence gap/detection dropout,
/// not marginal noise - smoothing was never expected to fix that case).
const SCORE_SMOOTHING_ALPHA: f64 = 0.4;

/// Degrees subtracted from the angle threshold when scoring the
/// currently-locked track - replaying the Failure 1 fixture showed the
/// dominant real-world case isn't two tracks both scoring and a challenger
/// edging ahead (what `CURRENT_TRACK_HYSTERESIS_BONUS` guards against): it's
/// the current track's own divergence angle briefly dipping a few degrees
/// under `BASE_ANGLE_THRESHOLD`/the current `lost_relaxation`-adjusted
/// threshold, which disqualifies it from scoring at all (`track_background_score`
/// returns `None`) and hands the frame to whichever decoy still clears the
/// threshold - a bonus added only *after* the threshold check can't help
/// with that. 25.0 keeps genuinely non-diverging tracks (angle under ~20)
/// from ever re-qualifying, while covering the momentary dips (angle in the
/// 20s-30s) observed to trigger unwanted switches on that replay.
const CURRENT_TRACK_ANGLE_HYSTERESIS: f64 = 25.0;

fn lost_relaxation(lost_frame_count: u32) -> f64 {
    if lost_frame_count <= RELAX_START_LOST_FRAMES {
        return 0.0;
    }

    ((lost_frame_count - RELAX_START_LOST_FRAMES) as f64
        / (RELAX_FULL_LOST_FRAMES - RELAX_START_LOST_FRAMES) as f64)
        .min(1.0)
}

// KNOWN ISSUE (hysteresis fix landed below, tested across Normal/Hard/Failure
// 1 replays - see git history for prior tuning attempts if this needs
// revisiting - tracked here since this repo has issues disabled): the
// current track can get abandoned for a different one purely because it
// scored marginally higher this frame, even while the current track is
// still present, stable, and being tracked fine - not a detection dropout at
// all. Originally reproduced via the "Test transparent shape failure"
// (Failure 1) debug replay: at frame 233, track 12 was still smoothly
// tracked (center ~(351, 195), unbroken since well before frame 220) when
// the solver switched to track 42 (center ~(427, 191), ~75px away, a
// different stable shape) purely because it out-scored track 12 that frame.
//
// Root cause: every track - including the current one - is scored fresh
// each frame by `update_and_find_best_track`'s `scored_candidate` and the
// winner taken via `max_by`. There's no bias toward the track already being
// followed, so a stable decoy that happens to score even marginally higher
// wins outright. The 2-consecutive-frame confirmation debounce
// (`candidate_track_id`/`candidate_track_count`) doesn't meaningfully guard
// against this: scores don't fluctuate wildly frame to frame for a stable,
// Kalman-smoothed track, so if a decoy edges out the current track once, it
// typically keeps doing so for the next frame too, clearing the debounce
// easily.
//
// Already tried and reverted: a bounding-box size-oscillation ("wobble")
// bonus meant to favor the true (visibly rotating) target over decoys -
// abandoned because several unrelated decoy shapes wobbled just as much or
// more than the real target (see PR #40's description), making switching
// worse, not better.
//
// Current fix - hysteresis favoring the *currently tracked* shape, in two
// parts (`CURRENT_TRACK_ANGLE_HYSTERESIS`, `CURRENT_TRACK_HYSTERESIS_BONUS`
// above - see the ROLLBACK NOTE above them for exactly what to remove if
// this needs backing out). First attempt only added a flat bonus to the
// current track's score post-threshold, which turned out to barely move
// switch counts on a fresh Failure 1 replay: diagnosing why showed most real
// switches there come from the current track's angle dipping a few degrees
// under `BASE_ANGLE_THRESHOLD` for a single frame, disqualifying it from
// scoring entirely (`track_background_score` returns `None`) before any
// bonus could apply - not two tracks both scoring with a challenger edging
// ahead, the narrower case the bonus alone covers.
// `CURRENT_TRACK_ANGLE_HYSTERESIS` relaxes the angle threshold itself for
// the current track to catch that dominant case; the score bonus stays for
// the narrower one. Manually diagnosed switch-by-switch (not just aggregate
// switch counts) across Normal, Hard, and two separate Failure 1 replays
// after landing this - every remaining switch traced back to either a
// genuine detection dropout or a real, wide (often 2x+) divergence-angle gap
// between the current track and the challenger, not a marginal steal.
fn track_background_score(
    track: &STrack,
    last_cursor: Point,
    bg_direction: Point2d,
    region: Rect,
    lost_frame_count: u32,
    is_current: bool,
) -> Option<f64> {
    let relax = lost_relaxation(lost_frame_count);
    let mut angle_threshold =
        BASE_ANGLE_THRESHOLD - relax * (BASE_ANGLE_THRESHOLD - MIN_ANGLE_THRESHOLD);
    if is_current {
        angle_threshold = (angle_threshold - CURRENT_TRACK_ANGLE_HYSTERESIS).max(0.0);
    }
    let angle =
        track_background_degree(track, bg_direction).filter(|&angle| angle > angle_threshold)?;
    let score = angle / 180.0;
    let strongly_diverging = angle >= 60.0;

    let distance_penalty = if strongly_diverging {
        1.0
    } else {
        let cursor_dir = mid_point(track.rect()) - last_cursor;
        let cursor_squared = (cursor_dir.x.pow(2) + cursor_dir.y.pow(2)) as f64;
        // Widen the plausible-distance window as lost duration grows - the
        // longer the shape hasn't been confirmed, the further it could have
        // legitimately moved from where it was last seen.
        let sigma = (0.25 + 0.5 * relax) * diag(region);
        (-cursor_squared / (2.0 * sigma.powi(2))).exp()
    };
    if distance_penalty <= 0.3 {
        return None;
    }

    Some(score * distance_penalty)
}

fn track_background_degree(track: &STrack, bg_direction: Point2d) -> Option<f64> {
    // Undefined while the background direction hasn't been estimated yet
    // (still the zero vector - see `estimate_background_direction`'s
    // `tracklet_len >= 5` warm-up requirement). Computing an angle against a
    // zero vector doesn't error, but `atan2`'s sign-of-zero handling turns it
    // into a coin flip between 0.0 and 180.0 driven purely by the sign of the
    // track's velocity components, not any real divergence - which let
    // unrelated tracks look "maximally diverging" (angle=180, score=1.0)
    // during the first few frames and caused an immediate, essentially
    // random, lock onto the wrong shape.
    if bg_direction.norm() < 1e-3 {
        return None;
    }

    let dir = unit(track.kalman_velocity())?;
    let dot = dir.dot(bg_direction);
    let det = dir.cross(bg_direction);
    Some(det.atan2(dot).to_degrees().abs())
}

fn estimate_background_direction(last_cursor: Option<Point>, tracks: &[STrack]) -> Option<Point2d> {
    let mut last_rect_contains_cursor = None;
    let filtered = tracks
        .iter()
        .filter(|track| {
            if track.tracklet_len() < 5 {
                return false;
            }

            if last_rect_contains_cursor.is_some_and(|rect: Rect| (rect & track.rect()).area() > 0)
            {
                return false;
            }

            let Some(last_cursor) = last_cursor else {
                return true;
            };

            let rect = track.rect();
            if rect.contains(last_cursor) {
                if last_rect_contains_cursor.is_none() {
                    last_rect_contains_cursor = Some(rect);
                }

                return false;
            }

            let norm = (mid_point(track.rect()) - last_cursor).norm();
            norm >= diag(track.rect())
        })
        .map(STrack::kalman_velocity)
        .collect::<Vec<Point2d>>();
    if filtered.len() < 3 {
        return None;
    }

    let velocity_sum = filtered
        .into_iter()
        .fold(Point2d::default(), |acc, v| acc + v);

    unit(velocity_sum)
}

fn diag(rect: Rect) -> f64 {
    ((rect.width.pow(2) + rect.height.pow(2)) as f64).sqrt()
}

fn unit<T>(point: Point_<T>) -> Option<Point_<T>>
where
    T: Copy,
    Point_<T>: Div<f64, Output = Point_<T>>,
    f64: From<T>,
{
    let norm = point.norm();
    if norm < 1e-3 {
        return None;
    }

    Some(point / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::{ByteTracker, Detection, IouGating};

    // Drives a real `ByteTracker` (rather than hand-rolling an `STrack`,
    // whose Kalman state is only mutable from within the `tracker` module)
    // so the returned track's `kalman_velocity` reflects genuine filter
    // output, not a stubbed value. Per-frame drift is kept small enough to
    // stay well under `associate_high_detections`'s IOU cost<=0.5 match
    // threshold (a 10px jump on a 50px box misses it - IOU drops to ~0.47,
    // cost 0.53 - and gets treated as a new, unconfirmed track instead of
    // continuing the existing one). Real gameplay frame-to-frame deltas are
    // a few px on a much bigger box, comfortably within this.
    fn moving_up_left_track(tracker: &mut ByteTracker) -> STrack {
        tracker.update(vec![Detection::new(Rect::new(100, 100, 50, 50), 1.0)]);
        tracker.update(vec![Detection::new(Rect::new(97, 97, 50, 50), 1.0)]);
        let tracks = tracker.update(vec![Detection::new(Rect::new(94, 94, 50, 50), 1.0)]);

        tracks
            .into_iter()
            .next()
            .expect("detection should have matched a track")
    }

    // Regression test for a track being scored as "maximally diverging from
    // background" (angle=180, score=1.0) purely from atan2's sign-of-zero
    // behavior when `bg_direction` is still the unestablished zero vector -
    // see `track_background_degree`'s doc comment.
    #[test]
    fn track_background_degree_is_none_when_background_unestablished() {
        let mut tracker = ByteTracker::new(30, 0.25, 0.1, 0.25, IouGating::None);

        // Moving up-left gives negative x and y velocity, the exact sign
        // combination that previously produced a spurious angle=180.0
        // against a zero bg_direction.
        let track = moving_up_left_track(&mut tracker);
        let velocity = track.kalman_velocity();
        assert!(
            velocity.x < 0.0 && velocity.y < 0.0,
            "velocity={velocity:?}"
        );

        assert_eq!(
            track_background_degree(&track, Point2d::new(0.0, 0.0)),
            None,
        );
    }

    #[test]
    fn track_background_degree_is_some_once_background_established() {
        let mut tracker = ByteTracker::new(30, 0.25, 0.1, 0.25, IouGating::None);
        let track = moving_up_left_track(&mut tracker);

        assert!(
            track_background_degree(&track, Point2d::new(1.0, 0.0)).is_some(),
            "a real (non-zero) bg_direction should still produce an angle",
        );
    }

    // Full-pipeline regression test for the bug behind `ld_qd1I60pB.mp4` (now
    // committed as `transparent_shape_test_failure_3.mp4`): 7 shapes laid out
    // like the real failing puzzle, 6 of which drift up-left in lockstep -
    // the exact velocity sign that previously made every one of them read as
    // "maximally diverging" (angle=180, score=1.0) while `bg_direction` was
    // still the unestablished zero vector, hijacking lock-on within the first
    // few frames via `max_by`'s last-element-wins tie-break. Unlike the two
    // tests above, which exercise `track_background_degree` directly, this
    // drives the real `solve()` through a `MockDetector` so a regression in
    // how the fix composes with the rest of the pipeline - not just the
    // function in isolation - would also be caught.
    #[test]
    fn solve_does_not_switch_track_while_background_unestablished() {
        use crate::detect::MockDetector;

        let mut detector = MockDetector::new();
        let mut frame = 0i32;
        detector
            .expect_detect_transparent_shapes()
            .returning(move |_| {
                frame += 1;
                let dx = -3 * (frame - 1);
                vec![
                    (Rect::new(310, 186, 128, 136), 1.0), // real target: stays put
                    (Rect::new(606 + dx, 280 + dx, 139, 135), 1.0),
                    (Rect::new(613 + dx, 53 + dx, 133, 131), 1.0),
                    (Rect::new(61 + dx, 202 + dx, 136, 131), 1.0),
                    (Rect::new(369 + dx, 317 + dx, 137, 117), 1.0),
                    (Rect::new(510 + dx, 160 + dx, 132, 137), 1.0),
                    (Rect::new(96 + dx, 303 + dx, 141, 125), 1.0),
                ]
            });

        let region = Rect::new(0, 0, 755, 505);
        let mut solver = TransparentShapeSolver::default();

        solver.solve(&detector, region);
        let initial_track = solver.current_track_id;
        assert!(initial_track.is_some());

        for _ in 0..7 {
            solver.solve(&detector, region);
            assert!(
                solver.bg_direction.norm() >= 1e-3 || solver.current_track_id == initial_track,
                "track switched to {:?} while bg_direction was still unestablished",
                solver.current_track_id,
            );
        }
    }
}
