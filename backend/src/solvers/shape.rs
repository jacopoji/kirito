use std::{
    collections::HashMap,
    ops::Div,
    sync::atomic::{AtomicU64, Ordering},
};

use log::debug;
use opencv::{
    calib3d::{RANSAC, estimate_affine_partial_2d},
    core::{
        DMatch, KeyPoint, KeyPointTraitConst, Mat, MatTraitConst, NORM_HAMMING, Point, Point_,
        Point2d, Point2f, Rect, Vector, count_non_zero, no_array,
    },
    features2d::{BFMatcher, DescriptorMatcherTraitConst, Feature2DTrait, ORB},
};

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
    /// Grayscale, padded crop of each live track's own region on the
    /// previous frame - kept so `update_rotation_scores` can diff it against
    /// this frame's crop of the same track id. See `rotation_bonus`'s doc
    /// comment for why an actually-spinning shape's crop content keeps
    /// producing a nonzero rotation estimate frame to frame while a merely
    /// translating decoy's doesn't. Replaced wholesale (not incrementally
    /// patched) each frame in `update_rotation_scores` so a track id that
    /// drops out and later reappears (a new ByteTracker identity) starts
    /// diffing from scratch instead of against a stale, no-longer-adjacent
    /// crop.
    rotation_crops: HashMap<u64, Mat>,
    /// EMA-smoothed per-track rotation magnitude (absolute degrees/frame,
    /// from `estimate_rotation_degrees`) - see `rotation_bonus`'s doc
    /// comment. Entries are dropped alongside `rotation_crops` for any track
    /// id no longer live, for the same re-acquisition reason.
    rotation_scores: HashMap<u64, f64>,
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
            rotation_crops: HashMap::new(),
            rotation_scores: HashMap::new(),
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

    /// `(shape id switches, frames spent dead reckoning)` accumulated so far,
    /// for the ground-truth evaluation harness to report alongside pixel-error
    /// metrics. Debug-only.
    #[cfg(debug_assertions)]
    pub fn debug_counters(&self) -> (u32, u32) {
        (self.debug_switch_count, self.debug_dead_reckoning_count)
    }

    /// This frame's smoothed rotation magnitude for `track_id`, if
    /// `update_rotation_scores` has one yet - lets the ground-truth
    /// evaluation harness compare the real target's rotation signal against
    /// decoys' directly on real footage, the same way the standalone
    /// prototype script did on the ground-truth fixtures before this was
    /// wired into scoring. Debug-only.
    #[cfg(debug_assertions)]
    pub fn debug_rotation_score(&self, track_id: u64) -> Option<f64> {
        self.rotation_scores.get(&track_id).copied()
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
        self.update_rotation_scores(detector, region, &tracks);

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
                    // `from_last_score`/`from_rotation` are the losing
                    // track's own EMA-smoothed values as of whenever it was
                    // last actually scored - stale (not necessarily *this*
                    // frame's) if it's missing from this frame's candidate
                    // dump entirely, which is itself diagnostic: cross-
                    // reference `candidate track={from_id} is_current_track_id=true`
                    // at this same frame - no such line means the old track
                    // wasn't even detected/tracked this frame (a detection
                    // dropout, which no scoring change could fix); a line
                    // with `score=None` means it was excluded by the angle
                    // gate outright (rotation's bonus, applied only after
                    // that gate passes, couldn't have saved it either); a
                    // line with `score=Some(..)` that still lost in `max_by`
                    // is the one genuine case rotation's bonus can influence.
                    debug!(
                        target: "backend/player",
                        "run_id={} frame={} shape id switches from {:?} (last_score={:?} last_rotation={:?}) to {} (score={:?} rotation={:?})",
                        self.run_id,
                        self.frame_index,
                        self.current_track_id,
                        self.current_track_id.and_then(|id| self.track_scores.get(&id)),
                        self.current_track_id.and_then(|id| self.rotation_scores.get(&id)),
                        track.track_id(),
                        self.track_scores.get(&track.track_id()),
                        self.rotation_scores.get(&track.track_id()),
                    );
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

    /// Diffs each live track's crop against its own crop from last frame -
    /// see `rotation_bonus`'s doc comment for why this separates the
    /// genuinely-spinning target from a merely translating decoy. Failures
    /// (too small a crop, too few matched features, no track history yet)
    /// silently skip that track for this frame rather than erroring - a
    /// missed rotation estimate just leaves `rotation_bonus` at its prior
    /// smoothed value (or 0.0 if none yet), same as a missed detection
    /// leaves `track_background_score` alone.
    fn update_rotation_scores(&mut self, detector: &dyn Detector, region: Rect, tracks: &[STrack]) {
        let grayscale = detector.grayscale();
        let mut next_crops = HashMap::with_capacity(tracks.len());

        for track in tracks {
            let track_id = track.track_id();
            let Some(crop) = rotation_crop(grayscale, region, track.rect()) else {
                continue;
            };

            if let Some(prev_crop) = self.rotation_crops.get(&track_id)
                && let Some(angle) = estimate_rotation_degrees(prev_crop, &crop)
            {
                let magnitude = angle.abs();
                let smoothed = match self.rotation_scores.get(&track_id) {
                    Some(&prev) => {
                        ROTATION_SMOOTHING_ALPHA * magnitude
                            + (1.0 - ROTATION_SMOOTHING_ALPHA) * prev
                    }
                    None => magnitude,
                };
                self.rotation_scores.insert(track_id, smoothed);
            }

            next_crops.insert(track_id, crop);
        }

        // Drop scores for any track id no longer live before swapping in
        // this frame's crops - see `rotation_crops`'s doc comment on why a
        // re-acquired id must start fresh.
        let live_ids: std::collections::HashSet<u64> = next_crops.keys().copied().collect();
        self.rotation_scores.retain(|id, _| live_ids.contains(id));
        self.rotation_crops = next_crops;
    }

    fn update_and_find_best_track<'a>(
        &mut self,
        tracks: &'a [STrack],
        region: Rect,
    ) -> Option<&'a STrack> {
        let current_track_id = self.current_track_id?;
        let last_cursor = self.last_cursor?;
        let last_velocity = self.last_velocity.unwrap_or_default();
        let bg_direction = self.bg_direction;
        let lost_frame_count = self.lost_frame_count;
        let frame_index = self.frame_index;
        let run_id = self.run_id;
        let track_scores = &mut self.track_scores;
        let rotation_scores = &self.rotation_scores;
        // See `hysteresis_active`'s doc comment: only the still-unbroken
        // original lock gets hysteresis, not merely whichever track happens
        // to be `current_track_id` right now.
        let hysteresis_eligible = self.hysteresis_active;

        // Whether the current track is *itself* confirmed this frame - scores
        // on its own merits (with its own hysteresis relaxation) rather than
        // merely being labeled `current_track_id`. Gates the teleport-distance
        // rejection below: a first attempt at this (rejected far challengers
        // whenever `hysteresis_eligible`, full stop) regressed Hard badly
        // (locked_pct 86.6% -> 46.6%) because it also protected an *unconfirmed*
        // current track - one that had itself stopped scoring - walling off
        // every alternative including the real target trying to correct back.
        // Only block far challengers while the current track is still actively
        // proving itself right now; the moment it isn't, alternatives need to
        // be free to compete same as before this change existed.
        let current_track_confirmed = hysteresis_eligible
            && tracks
                .iter()
                .find(|track| track.track_id() == current_track_id)
                .is_some_and(|track| {
                    track_background_score(
                        track,
                        last_cursor,
                        bg_direction,
                        region,
                        lost_frame_count,
                        true,
                    )
                    .is_some()
                });

        let best_scored = tracks
            .iter()
            .filter(|track| track.track_id() == current_track_id || track.tracklet_len() >= 1)
            .filter_map(|track| {
                let is_current_track_id = track.track_id() == current_track_id;
                let is_current = is_current_track_id && hysteresis_eligible;
                let rotation = rotation_scores.get(&track.track_id()).copied();

                // While the current track is actively confirming itself, a
                // challenger has to be within physically plausible reach of
                // it - the real target only ever drifts, it doesn't
                // teleport, so a decoy clear across the puzzle can't be it no
                // matter how strongly its own motion diverges from the
                // background (the `strongly_diverging` branch below normally
                // waives the distance check entirely). Reject it outright
                // rather than let it outscore a lock that's still proving
                // itself correct.
                if current_track_confirmed && !is_current {
                    let drift = mid_point(track.rect()) - last_cursor;
                    let distance = ((drift.x.pow(2) + drift.y.pow(2)) as f64).sqrt();
                    let plausible_radius = MAX_PLAUSIBLE_DRIFT_MARGIN_PX
                        + last_velocity.norm() * MAX_PLAUSIBLE_DRIFT_MULTIPLIER;
                    if distance > plausible_radius {
                        return None;
                    }
                }

                let angle = track_background_degree(track, bg_direction);
                let score = track_background_score(
                    track,
                    last_cursor,
                    bg_direction,
                    region,
                    lost_frame_count,
                    is_current,
                );
                // `is_current_track_id` (not just `is_current`, which also
                // requires `hysteresis_eligible`) is logged so a "shape id
                // switches" event's own frame can be cross-referenced here
                // to tell apart *why* the previously-locked track lost the
                // lock: `score=None` means it was excluded by the angle gate
                // outright (rotation's bonus, applied only after `score?`
                // succeeds, can't help that case - see the reverted gate-
                // relaxation attempt noted above `track_background_score`),
                // whereas a `Some` score here that still lost in `max_by`
                // means it was genuinely outscored by a challenger - the
                // case rotation's bonus actually influences.
                debug!(
                    target: "backend/player",
                    "run_id={} frame={} candidate track={} is_current_track_id={is_current_track_id} angle={:?} score={:?} rotation={:?} lost_frame_count={}",
                    run_id,
                    frame_index,
                    track.track_id(),
                    angle,
                    score,
                    rotation,
                    lost_frame_count,
                );
                let score = score?;
                // Rotation bonus - see `rotation_bonus`'s doc comment. Added
                // to every candidate that already clears the background-
                // divergence gate above (a non-diverging decoy still can't
                // score at all, rotating or not - this only re-ranks among
                // candidates the existing heuristic already considers
                // plausible), before the hysteresis bonus so both apply on
                // top of each other rather than one overriding the other.
                let score = score + rotation_bonus(rotation);
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
        let match_track = if lost_frame_count
            > MAX_LOST_FRAMES_FOR_EXTRAPOLATION + REACQUISITION_GRACE_FRAMES
        {
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

/// Extra pixels of context grabbed around a track's own bounding box before
/// diffing it against last frame's crop - a few edge/background pixels give
/// ORB more to match on than the bare silhouette, mirroring the prototype
/// script this was validated against (see
/// `project_lie_detector_shape_solver_tracking` memory notes).
const ROTATION_CROP_PAD: i32 = 4;

/// Below this size (in either dimension) a crop has too little content for
/// `estimate_rotation_degrees` to find enough keypoints on - not worth
/// attempting ORB detection at all.
const ROTATION_MIN_CROP_SIZE: i32 = 8;

/// Clamps `rect` to `mat`'s bounds, mirroring `detect.rs`'s
/// `clamp_rect_to_mat` (not reused directly - that one is private to
/// `detect.rs` - but the same reasoning applies: a track's region-relative
/// rect plus `ROTATION_CROP_PAD` can end up partially outside the captured
/// frame, and `Mat::roi` panics rather than errors on an out-of-bounds rect).
fn clamp_rect_to_mat(mat: &Mat, rect: Rect) -> Option<Rect> {
    let bounds = Rect::new(0, 0, mat.cols(), mat.rows());
    let clamped = rect & bounds;
    (clamped.width > 0 && clamped.height > 0).then_some(clamped)
}

/// Grabs an owned, padded grayscale crop of `track_rect` (region-relative,
/// as returned by `STrack::rect`) from `grayscale` (the full captured
/// frame) - owned rather than a borrowed `roi` view since
/// `update_rotation_scores` keeps it around past this frame to diff against
/// next frame's crop of the same track id.
fn rotation_crop(grayscale: &Mat, region: Rect, track_rect: Rect) -> Option<Mat> {
    let absolute = Rect::new(
        region.x + track_rect.x - ROTATION_CROP_PAD,
        region.y + track_rect.y - ROTATION_CROP_PAD,
        track_rect.width + 2 * ROTATION_CROP_PAD,
        track_rect.height + 2 * ROTATION_CROP_PAD,
    );
    let clamped = clamp_rect_to_mat(grayscale, absolute)?;
    if clamped.width < ROTATION_MIN_CROP_SIZE || clamped.height < ROTATION_MIN_CROP_SIZE {
        return None;
    }

    grayscale.roi(clamped).ok()?.try_clone().ok()
}

/// Below this many ORB keypoints found in a crop, there isn't enough texture
/// to trust a match against the other frame's crop at all.
const ROTATION_MIN_KEYPOINTS: usize = 6;

/// Below this many Lowe's-ratio-filtered matches, `estimate_affine_partial_2d`
/// is fitting noise rather than a real transform.
const ROTATION_MIN_GOOD_MATCHES: usize = 6;

/// Below this many RANSAC inliers on the fitted transform, it's not trusted
/// even if `estimate_affine_partial_2d` returned something.
const ROTATION_MIN_INLIERS: i32 = 4;

/// Lowe's ratio test threshold for filtering ambiguous ORB matches (a match
/// is kept only if the best candidate is convincingly closer than the
/// second-best) - 0.8 is Lowe's own commonly-used value for binary
/// descriptors like ORB's, looser than the ~0.7 typical for SIFT/SURF since
/// ORB's Hamming-distance matches are naturally noisier.
const ROTATION_LOWE_RATIO: f32 = 0.8;

/// Estimates the rotation (in degrees) between `prev` and `curr` - two
/// grayscale crops of the *same* track's own region on consecutive frames
/// (see `rotation_crop`) - by matching ORB keypoints between them and
/// fitting a similarity transform (`estimate_affine_partial_2d`: rotation +
/// uniform scale + translation, no shear/perspective). Because both crops
/// are already re-centered on the track's own position each frame, any
/// translation component in the fitted transform reflects sub-pixel
/// tracking jitter, not the shape's on-screen movement - what's left in the
/// rotation component is the crop *content* rotating relative to itself,
/// i.e. the shape spinning in place. Returns `None` if either crop is too
/// small/low-texture to match confidently, mirroring how
/// `track_background_score` returns `None` rather than a low-confidence
/// guess when its own gating fails.
fn estimate_rotation_degrees(prev: &Mat, curr: &Mat) -> Option<f64> {
    if prev.rows() < ROTATION_MIN_CROP_SIZE
        || prev.cols() < ROTATION_MIN_CROP_SIZE
        || curr.rows() < ROTATION_MIN_CROP_SIZE
        || curr.cols() < ROTATION_MIN_CROP_SIZE
    {
        return None;
    }

    let mut orb = ORB::create_def().ok()?;
    let mut prev_keypoints = Vector::<KeyPoint>::new();
    let mut prev_descriptors = Mat::default();
    orb.detect_and_compute_def(
        prev,
        &no_array(),
        &mut prev_keypoints,
        &mut prev_descriptors,
    )
    .ok()?;
    let mut curr_keypoints = Vector::<KeyPoint>::new();
    let mut curr_descriptors = Mat::default();
    orb.detect_and_compute_def(
        curr,
        &no_array(),
        &mut curr_keypoints,
        &mut curr_descriptors,
    )
    .ok()?;

    if prev_keypoints.len() < ROTATION_MIN_KEYPOINTS
        || curr_keypoints.len() < ROTATION_MIN_KEYPOINTS
        || prev_descriptors.rows() == 0
        || curr_descriptors.rows() == 0
    {
        return None;
    }

    let matcher = BFMatcher::create(NORM_HAMMING, false).ok()?;
    let mut knn_matches = Vector::<Vector<DMatch>>::new();
    matcher
        .knn_train_match_def(&prev_descriptors, &curr_descriptors, &mut knn_matches, 2)
        .ok()?;

    let mut src = Vector::<Point2f>::new();
    let mut dst = Vector::<Point2f>::new();
    for pair in knn_matches.iter() {
        if pair.len() != 2 {
            continue;
        }
        let best = pair.get(0).ok()?;
        let second = pair.get(1).ok()?;
        if best.distance < ROTATION_LOWE_RATIO * second.distance {
            src.push(prev_keypoints.get(best.query_idx as usize).ok()?.pt());
            dst.push(curr_keypoints.get(best.train_idx as usize).ok()?.pt());
        }
    }

    if src.len() < ROTATION_MIN_GOOD_MATCHES {
        return None;
    }

    let mut inliers = Mat::default();
    let affine =
        estimate_affine_partial_2d(&src, &dst, &mut inliers, RANSAC, 3.0, 2000, 0.99, 10).ok()?;
    if affine.rows() == 0 || count_non_zero(&inliers).ok()? < ROTATION_MIN_INLIERS {
        return None;
    }

    let cos_component = *affine.at_2d::<f64>(0, 0).ok()?;
    let sin_component = *affine.at_2d::<f64>(1, 0).ok()?;
    Some(sin_component.atan2(cos_component).to_degrees())
}

/// Degrees of smoothed rotation magnitude past which `rotation_bonus`
/// reaches roughly half of `ROTATION_BONUS_MAX` - i.e. the score curve's
/// half-saturation point, not a hard cutoff. Picked from the prototype
/// script's measurements across all 6 ground-truth fixtures: decoys'
/// smoothed rotation magnitude centered well under 1.5 degrees/frame while
/// the real target's centered in the 2-7 degrees/frame range (see
/// `project_lie_detector_shape_solver_tracking` memory notes) - 2.0 sits
/// just above the decoy band so ordinary decoy noise earns a small bonus at
/// most, while the target's typical range earns most of the max bonus.
const ROTATION_HALF_SCALE_DEG: f64 = 2.0;

/// Cap on the score bonus `rotation_bonus` can contribute - deliberately
/// smaller than the background-divergence score's own range (`angle / 180.0`,
/// up to 1.0) so a strongly rotating decoy can nudge a close call but can
/// never on its own outweigh a genuinely stronger divergence signal from a
/// competing track.
///
/// 0.2 (the original value here) turned out too small to matter in
/// practice: the `rotation_diagnostic` line added to
/// `evaluate_transparent_shape_sync` measured the real target's smoothed
/// rotation magnitude on real footage at roughly 3-13x decoys' (e.g.
/// Failure 2: target 6.25 vs decoy 0.48 degrees/frame) - a strong,
/// genuinely separating signal - but run through `rotation_bonus`'s curve
/// at the old cap, that only produced a ~0.02-0.11 score differential
/// between a typical target and decoy, swamped by both
/// `CURRENT_TRACK_HYSTERESIS_BONUS` (0.15) and the raw divergence score's
/// own frame-to-frame noise (0.3-0.5, per the KNOWN ISSUE comment above
/// `track_background_score`) - measurably, `eval_shape`'s `mean_err`/
/// `locked_pct`/`switches` were unchanged (or moved by noise-level amounts)
/// across all 6 fixtures with the old cap. 0.6 brings the typical
/// differential up to roughly 0.2-0.3 - comparable to or exceeding the
/// hysteresis bonus, so it can actually swing marginal contests - while
/// still well under the divergence score's own 1.0 ceiling, so it still
/// can't single-handedly override a genuinely much stronger divergence
/// signal from a wrong decoy.
const ROTATION_BONUS_MAX: f64 = 0.6;

/// Converts a track's smoothed rotation magnitude (degrees/frame, from
/// `update_rotation_scores`) into an additive score bonus - a saturating
/// curve (`x / (x + half_scale)`) rather than a hard threshold so a track
/// with no rotation estimate yet (`None` - not enough frames of history, or
/// too little texture to match) gets exactly zero bonus instead of being
/// penalized, and so the bonus grows smoothly rather than jumping the
/// moment some arbitrary cutoff is crossed.
fn rotation_bonus(smoothed_degrees: Option<f64>) -> f64 {
    let magnitude = smoothed_degrees.unwrap_or(0.0).max(0.0);
    ROTATION_BONUS_MAX * (magnitude / (magnitude + ROTATION_HALF_SCALE_DEG))
}

/// EMA weight given to this frame's rotation magnitude when blending with a
/// track id's own smoothed rotation from last frame - see `rotation_scores`'s
/// doc comment. 0.15 approximates a ~12-frame rolling window
/// (`2/alpha - 1`), matching the window size the prototype script measured
/// as a good tradeoff: long enough to damp the single-frame noise inherent
/// in per-frame ORB matching, short enough to still reflect a target that's
/// been visibly spinning for under a second.
const ROTATION_SMOOTHING_ALPHA: f64 = 0.15;

// KNOWN ISSUE (upstream of everything in this file - not fixable by tuning
// scoring/tracking logic, tracked here since this repo has issues disabled):
// a meaningful fraction of "shape id switches" events, even on the easy
// `transparent_shape_test_normal.mp4` fixture, trace back to the *detector*
// (`detect_transparent_shapes`, the ONNX model) simply not emitting a box
// for the real target for one or more consecutive frames - not a scoring or
// tracking mistake at all, so no amount of solver-side tuning (background-
// divergence thresholds, rotation bonus, teleport-distance rejection, etc.)
// can fix it; the fix has to be a better-trained model.
//
// Reproduced by replaying the Normal fixture through the debug UI and
// cross-referencing "shape id switches" events against the per-frame
// `tracks=[...]` dump at the same frame (both already logged via `debug!`):
// of Normal's 4 switches in a representative run, 3 were detection dropouts,
// only 1 a genuine scoring contest between two simultaneously-visible
// tracks:
//   - frame 76: track 147 (current) and track 392 (challenger) both present
//     and ~40px apart the whole time - a real, close-range scoring contest,
//     not a dropout.
//   - frame 241: track 392 (current) vanishes from that frame's `tracks=`
//     list entirely (present at 240, absent at 241); a brand-new track 686
//     appears one frame earlier at nearly the same position (~57px away)
//     and inherits the lock - most likely the same physical shape
//     rediscovered under a fresh ByteTracker id after a one-frame miss, not
//     a genuine mis-track.
//   - frames 278-286: track 686 (current, stable for 200+ prior frames)
//     vanishes entirely starting at frame 279 and never reappears; the lock
//     falls to track 148 (~120px away, a different, clearly separate shape)
//     for 8 frames until a new track 746 appears near 686's last known
//     position and reclaims the lock at frame 287 - the most visually
//     obvious kind of "hijack", and purely a multi-frame detection gap.
//
// See `tools/shape_model_training/find_overlapping_shapes.py` and its
// README for the training-side investigation into detection recall gaps
// (shape-on-shape overlap was the leading hypothesis there); the same
// underlying weakness - real-footage frames where the true target's own box
// just isn't produced - is very likely what's showing up here too, just
// without needing two shapes to be overlapping at all. Worth re-checking
// whether a future model iteration measurably reduces these specific
// dropout-driven switches on this fixture, not just aggregate mean_err/
// locked_pct (see `rotation_diagnostic`/`frame_evaluation` in
// `evaluate_transparent_shape_sync` for existing instrumentation).
//
/// The point past which `update_and_find_best_track` switches from waiting
/// for the original track to falling back to spatial re-acquisition, and how
/// long `lost_relaxation` takes to start loosening matching thresholds.
/// Doesn't gate `solve`'s own dead reckoning - see `DEAD_RECKONING_DECAY` for
/// how that stays bounded without an explicit frame cutoff.
const MAX_LOST_FRAMES_FOR_EXTRAPOLATION: u32 = 5;

/// Extra frames past `MAX_LOST_FRAMES_FOR_EXTRAPOLATION` during which
/// `update_and_find_best_track` keeps waiting on score-based matching alone
/// (no `find_reacquisition_candidate` fallback yet) before trusting pure
/// spatial proximity. Grabbing the nearest detection is a last resort, not
/// a first instinct the instant the frame counter crosses a threshold -
/// concretely reproduced on the Normal fixture (see the KNOWN ISSUE above):
/// frames 278-286, where the real target vanished, reacquisition grabbed
/// the nearest *other* shape (~120px away) the moment it became eligible,
/// and the real target reappeared under a fresh track id only a few frames
/// later - close enough in time that a few extra frames of patience would
/// have given `best_scored` a chance to find it on its own once it
/// registered a strong enough divergence angle again, without ever needing
/// to gamble on "nearest wins". Mirrors `CURRENT_TRACK_HYSTERESIS_BONUS`'s
/// reasoning (favor waiting over grabbing the first plausible-looking
/// alternative) but applied to the reacquisition trigger point instead of
/// the score itself, since a raised angle threshold was tried first (see
/// the reverted `RECENTLY_LOST_ANGLE_PENALTY` note above `lost_relaxation`)
/// and didn't reach this code path at all - `find_reacquisition_candidate`
/// never calls `track_background_score`.
///
/// Replay-verified at 5: Failure 1 improved substantially (locked_pct
/// 49.9%->60.8%, mean_err 105.0->96.5), Hard/Failure 3/Failure 4 unaffected,
/// Normal's own frame-278 hijack shrank from 8 frames of wrong lock to 3 (the
/// real target still wasn't detected in time to avoid a wrong grab
/// entirely, but the grab now happens later, closer to when the real target
/// resurfaces) - no regressions anywhere. Doubling to 8 was tried next and
/// was *not* a further improvement: it nearly erased Failure 1's gain
/// (locked_pct back down to 51.9%, mean_err up to 108.4 - worse than the
/// unmodified baseline) in exchange for marginal further gains on Normal
/// and Failure 2 - the same non-monotonic pattern seen tuning
/// `ROTATION_ANGLE_RELAXATION_MAX_DEG` and the NMS IoU export threshold
/// earlier: pushing one lever further doesn't uniformly improve, it just
/// moves which fixture wins and which loses. 5 was kept as the better
/// trade-off (a real win with zero regressions beats a wash). Only two
/// values were tried - a value between 5 and 8 might do better still if
/// this needs revisiting, but see the non-monotonic pattern noted above
/// before assuming a bisection search will converge cleanly.
const REACQUISITION_GRACE_FRAMES: u32 = 5;

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

/// Flat slack (px) added to a velocity-scaled plausible-drift radius, used
/// to reject a challenger outright while the current track is confirmed -
/// see `current_track_confirmed` in `update_and_find_best_track`. Covers
/// Kalman jitter and the target's own size/shape (comparable to
/// `GROUND_TRUTH_LOCK_RADIUS_PX`'s "roughly half a typical shape's width"
/// reasoning) so a legitimately-still-in-place target isn't rejected for
/// merely having a noisy box center.
const MAX_PLAUSIBLE_DRIFT_MARGIN_PX: f64 = 60.0;

/// Multiplier on the current track's own last known speed (px/frame) added
/// to `MAX_PLAUSIBLE_DRIFT_MARGIN_PX` for the same plausible-drift radius.
/// 6x is deliberately generous - `last_velocity` lags by one frame and the
/// real target can accelerate frame to frame (observed up to ~76px/frame in
/// a single burst on the Failure 4 replay, well above its own ~4-6px/frame
/// average) - this only needs to catch a decoy on the opposite side of the
/// puzzle, not shave px off a target that's merely picking up speed.
const MAX_PLAUSIBLE_DRIFT_MULTIPLIER: f64 = 6.0;

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

// TRIED AND REVERTED (recently-lost angle penalty): tried raising
// `angle_threshold` for non-current candidates by 25 degrees (mirroring
// `CURRENT_TRACK_ANGLE_HYSTERESIS`, opposite direction) for
// `1..=MAX_LOST_FRAMES_FOR_EXTRAPOLATION` frames after the current track
// disappears, on the theory that a marginal decoy shouldn't be able to
// steal the lock as easily during that brief window as it could challenge
// a still-visible current track. Replay-verified: zero measurable change
// across all 6 fixtures - `eval_shape` output was bit-identical, and
// tracing the exact switch events showed why: every challenger that won
// during a dropout on these fixtures was diverging at 150-179 degrees (way
// past even the toughened threshold), not marginally. The switches this
// was meant to catch (a decoy narrowly clearing the gate right after a
// loss) don't show up in these fixtures at all - the real culprits are
// either overwhelming-divergence recoveries (this fix structurally can't
// touch) or `find_reacquisition_candidate`'s pure-proximity fallback (which
// never calls `track_background_score`/checks this threshold at all, so a
// gate change here can't reach it either). See the grace-period style fix
// on `find_reacquisition_candidate` instead, which targets that second
// path directly.
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
//
// TRIED AND REVERTED (rotation-based gate relaxation): the same "bonus
// alone isn't enough, the gate itself needs relaxing" lesson looked like it
// should generalize beyond just the current track. Wiring a rotation-
// magnitude score (see `TransparentShapeSolver::rotation_scores`) in as only
// a post-gate `rotation_bonus` measurably separated the real target from
// decoys (`rotation_diagnostic` on real footage: target averaging 3-13x
// decoys' magnitude across all 6 fixtures) but barely moved `eval_shape`'s
// mean_err/locked_pct/switches even at 3x the bonus cap - diagnosing why by
// cross-referencing "shape id switches" events against the per-candidate
// `is_current_track_id` dumps at the same frame showed roughly half of
// those switches lost the real target to angle-gating alone (`score=None`),
// with a real rotation estimate already available that a post-gate-only
// bonus never got a chance to use.
//
// So `track_background_score` took a `rotation: Option<f64>` parameter and
// relaxed `angle_threshold` by up to `ROTATION_ANGLE_RELAXATION_MAX_DEG` for
// *any* candidate with strong rotation, not just the current one (mirroring
// `CURRENT_TRACK_ANGLE_HYSTERESIS`). Replay-verified across all 6 fixtures
// and reverted: at 25.0 degrees (matching `CURRENT_TRACK_ANGLE_HYSTERESIS`),
// it regressed Hard - a previously rock-solid, not-a-documented-problem
// fixture - from locked_pct 86.6/mean_err 36.2/switches 7 to
// 82.7/41.6/9, because unlike the current-track case (bounded to a single
// track), relaxing the gate for *any* rotating candidate let meaningfully
// more marginal candidates into scoring across the board. Halving it to 12.0
// to find a gentler middle ground made things *worse*, not better, on both
// Hard (41.6->48.9 mean_err) and Failure 4 (97.3->130.1 mean_err, worse than
// even the unmodified baseline's 112.3) while mostly erasing the Failure
// 1/2 gains back toward baseline - not a smooth trade-off dial, evidence
// that a small gate change flips *which* candidate wins lock at a handful
// of critical frames, and that winner then holds `current_track_id` (and
// its own hysteresis/dead-reckoning state) for many subsequent frames,
// so small input changes cascade into disproportionate, non-monotonic
// output changes rather than converging. `rotation_bonus` (post-gate only,
// see its doc comment) was kept - it's the variant that measured zero
// regressions on Normal/Hard/Failure 1/2/3 across every value tried, with a
// small real improvement on Failure 4.
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
        // `solve` now also calls `Detector::grayscale` every frame (see
        // `update_rotation_scores`) - an empty Mat makes every track's
        // `rotation_crop` clamp to nothing and skip cleanly, which is fine
        // here since this test only asserts on `bg_direction`/
        // `current_track_id`, not rotation scoring.
        detector.expect_grayscale().return_const(Mat::default());
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

    /// A deliberately asymmetric, high-contrast pattern (not a checkerboard,
    /// to avoid the 90-degree rotational symmetry that could let a wrong
    /// match still look plausible) - gives ORB plenty of distinct corners to
    /// find and match regardless of small rotation/translation.
    /// Large enough to clear the 31px border ORB ignores by default.
    const PATTERN_SIZE: i32 = 300;

    fn textured_pattern() -> Mat {
        use opencv::{
            core::{CV_8UC1, Scalar},
            imgproc::{LINE_8, rectangle},
        };

        let mut mat = Mat::new_rows_cols_with_default(
            PATTERN_SIZE,
            PATTERN_SIZE,
            CV_8UC1,
            Scalar::all(255.0),
        )
        .expect("create pattern mat");
        rectangle(
            &mut mat,
            Rect::new(5, 5, 22, 16),
            Scalar::all(0.0),
            -1,
            LINE_8,
            0,
        )
        .unwrap();
        rectangle(
            &mut mat,
            Rect::new(45, 12, 18, 28),
            Scalar::all(30.0),
            -1,
            LINE_8,
            0,
        )
        .unwrap();
        rectangle(
            &mut mat,
            Rect::new(12, 55, 28, 20),
            Scalar::all(90.0),
            -1,
            LINE_8,
            0,
        )
        .unwrap();
        rectangle(
            &mut mat,
            Rect::new(60, 60, 20, 25),
            Scalar::all(150.0),
            -1,
            LINE_8,
            0,
        )
        .unwrap();
        rectangle(
            &mut mat,
            Rect::new(35, 35, 14, 14),
            Scalar::all(190.0),
            -1,
            LINE_8,
            0,
        )
        .unwrap();

        // Unique, deterministic texture: plain rectangles have too few distinct corners for ORB.
        let mut seed = 7u32;
        let mut next = |bound: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) % bound) as i32
        };
        for _ in 0..400 {
            let rect = Rect::new(
                next(PATTERN_SIZE as u32 - 20),
                next(PATTERN_SIZE as u32 - 20),
                4 + next(14),
                4 + next(14),
            );
            let color = Scalar::all(next(256) as f64);
            rectangle(&mut mat, rect, color, -1, LINE_8, 0).unwrap();
        }
        mat
    }

    fn rotate_pattern(src: &Mat, angle_degrees: f64) -> Mat {
        use opencv::{
            core::Size,
            imgproc::{get_rotation_matrix_2d, warp_affine_def},
        };

        let center = Point2f::new(PATTERN_SIZE as f32 / 2.0, PATTERN_SIZE as f32 / 2.0);
        let rotation_matrix = get_rotation_matrix_2d(center, angle_degrees, 1.0).unwrap();
        let mut dst = Mat::default();
        warp_affine_def(
            src,
            &mut dst,
            &rotation_matrix,
            Size::new(PATTERN_SIZE, PATTERN_SIZE),
        )
        .unwrap();
        dst
    }

    fn translate_pattern(src: &Mat, dx: f64, dy: f64) -> Mat {
        use opencv::{
            core::{MatTrait, Size},
            imgproc::{get_rotation_matrix_2d, warp_affine_def},
        };

        // No dedicated pure-translation constructor is needed - a rotation
        // matrix built with angle=0 is just the identity, so nudging its
        // translation column afterward gives an ordinary translation matrix.
        let mut translation = get_rotation_matrix_2d(
            Point2f::new(PATTERN_SIZE as f32 / 2.0, PATTERN_SIZE as f32 / 2.0),
            0.0,
            1.0,
        )
        .unwrap();
        *translation.at_2d_mut::<f64>(0, 2).unwrap() += dx;
        *translation.at_2d_mut::<f64>(1, 2).unwrap() += dy;

        let mut dst = Mat::default();
        warp_affine_def(
            src,
            &mut dst,
            &translation,
            Size::new(PATTERN_SIZE, PATTERN_SIZE),
        )
        .unwrap();
        dst
    }

    // Regression test for `estimate_rotation_degrees` (the core primitive
    // `rotation_bonus`/`update_rotation_scores` builds on): a crop that's
    // genuinely rotated between frames should report a rotation magnitude
    // close to the applied angle - sign is intentionally not asserted here
    // (depends on `get_rotation_matrix_2d`/`estimate_affine_partial_2d`'s
    // respective conventions lining up, which isn't the thing this test is
    // meant to catch a regression in), only that the *magnitude* is
    // recovered.
    #[test]
    fn estimate_rotation_degrees_detects_applied_rotation() {
        let base = textured_pattern();
        let rotated = rotate_pattern(&base, 15.0);

        let angle = estimate_rotation_degrees(&base, &rotated)
            .expect("expected a rotation estimate for a genuinely rotated crop");
        assert!(
            (angle.abs() - 15.0).abs() < 5.0,
            "expected ~15 degrees of rotation magnitude, got {angle}",
        );
    }

    // A track that's merely translating (the common decoy case - drifting
    // with the rest of the background, not spinning) should not be reported
    // as rotating - this is the whole premise `rotation_bonus` relies on to
    // tell the real (spinning) target apart from decoys.
    #[test]
    fn estimate_rotation_degrees_is_near_zero_for_pure_translation() {
        let base = textured_pattern();
        let translated = translate_pattern(&base, 6.0, -4.0);

        match estimate_rotation_degrees(&base, &translated) {
            None => {}
            Some(angle) => assert!(
                angle.abs() < 3.0,
                "expected near-zero rotation for a pure translation, got {angle}",
            ),
        }
    }
}
