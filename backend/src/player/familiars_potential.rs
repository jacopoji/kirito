use std::{
    fmt::Display,
    sync::Arc,
    time::{Duration, Instant},
};

use log::debug;
use opencv::core::{Mat, Point, Rect};

use super::{
    Player,
    timeout::{Lifecycle, Timeout, next_timeout_lifecycle},
};
use crate::{
    array::Array,
    bridge::{KeyKind, MouseKind},
    detect::FamiliarRank,
    ecs::Resources,
    models::FamiliarRarity,
    player::{PlayerEntity, next_action},
};

/// Number of collection-list cards processed per run before giving up, matching the fixed-size
/// [`Array`] backing [`FamiliarsRevealingPotential::cards`].
///
/// Only the currently-visible page of the collection list is scanned - this run does not scroll
/// for more. A familiar sitting further down the list than this simply waits for a later run,
/// same as any other run whose card happens to still be visible then. Scrolling to guarantee
/// covering the *entire* collection in one pass is a natural follow-up once this simpler version
/// is confirmed working live.
const MAX_CARDS: usize = 64;

/// Internal state machine representing the current state of revealing familiar potentials.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Opening the familiar menu.
    OpenMenu(Timeout),
    /// Clicking the Info tab - reached unconditionally after opening the menu (whether freshly
    /// opened, which defaults to the Setup tab, or found already open on some other tab), since
    /// Info is never the tab a fresh menu-open lands on.
    SwitchToInfoTab(Timeout),
    /// Finding the collection-list cards matching [`FamiliarsRevealingPotential::reveal_rarities`]
    /// on the currently-visible page - see [`MAX_CARDS`]'s docs on why this does not scroll for
    /// more.
    FindCards(Timeout),
    /// Clicking the card at `index` to select it into the main preview panel, then checking
    /// whether its potential is still locked (the `?` box) - a locked box is clicked immediately
    /// (within the same state, on the same tick its bbox is found) to open the unlock-cost popup,
    /// since the box's bbox is only ever available while still locked and is needed again
    /// afterwards to know where to read the revealed result from (see [`State::ReadPotential`]'s
    /// docs). An already-revealed card (no locked box found - either revealed in an earlier run,
    /// or manually by the player) is left alone entirely: skipped without reading or protecting
    /// it, since there is no reliable way to re-locate the result box's bounds without the locked
    /// bbox to anchor on. See this module's own top-level docs for the consequence of this.
    SelectCard(Timeout, u32),
    /// Clicking the unlock-cost popup's `Confirm` button. `Rect` is the potential box's bbox
    /// (captured while still locked, by [`State::SelectCard`]), carried through to
    /// [`State::ReadPotential`] since it is the same screen region either way. `bool` tracks
    /// whether `Confirm` has already been clicked once, so it isn't clicked again on a later tick
    /// - mirrors [`super::familiars_fusion::update_confirm_fuse`]'s own reasoning.
    ConfirmUnlock(Timeout, u32, Rect, bool),
    /// Waiting for the reveal to render, then checking `region` (the potential box) for any good
    /// potential line - see [`crate::detect::Detector::detect_familiar_potential_good_line`]'s
    /// docs. A match captures this card's small collection-list icon into
    /// [`FamiliarsRevealingPotential::protected_icons`].
    ReadPotential(Timeout, u32, Rect),
    /// Advancing to the next card, or completing once every card on the page has been processed.
    NextCard(u32),
    Completing(Timeout, bool),
}

/// Struct for storing familiar potential-reveal data.
#[derive(Debug, Clone)]
pub struct FamiliarsRevealingPotential {
    /// Current state of the potential-reveal state machine.
    state: State,
    /// Only collection-list familiars of these rarities are scanned - mirrors
    /// [`crate::models::FamiliarsFusion::fusable_rarities`]'s reasoning: no point spending Mesos
    /// revealing a rarity that was never going to be a fusion target or fodder candidate anyway.
    reveal_rarities: Array<FamiliarRarity, 2>,
    /// When non-empty, restricts this run to a *targeted* re-check: only Epic-rarity
    /// collection-list cards are considered (ignoring [`Self::reveal_rarities`] entirely), instead
    /// of a full scan of the currently-visible page filtered by [`Self::reveal_rarities`].
    ///
    /// The `Vec<Arc<Mat>>` payload itself - icons of the familiars a chained fusion run just
    /// ranked Rare->Epic, see [`super::familiars_fusion::FamiliarsFusing::newly_epic_icons`]'s
    /// docs - is only ever consulted for emptiness here, not matched against anything. An earlier
    /// version tried to match each icon against a collection-list card's small bbox via
    /// [`Detector::detect_region_template`], but the icon is captured at equipped-slot portrait
    /// scale (54x50, see `familiar_portrait_region`) while a list card's bbox is only 24x32 -
    /// template larger than the search region - which live logs confirmed scores ~0.05-0.10
    /// across every card, nowhere near any usable threshold. Restricting to Epic rarity instead
    /// sidesteps the scale mismatch entirely and is still cheap: an already-revealed Epic card is
    /// skipped in one failed detection by [`State::SelectCard`] anyway.
    ///
    /// Existing purely to avoid re-walking the entire collection page (and re-checking every
    /// already-revealed card) on every single fusion/swap cycle once the one-time full scan is
    /// done - see [`crate::player::PlayerContext::familiars_potential_due`]'s docs for the full
    /// reasoning.
    target_icons: Vec<Arc<Mat>>,
    /// Collection-list cards found by [`State::FindCards`] on the currently-visible page.
    cards: Array<Rect, MAX_CARDS>,
    /// Icons of familiars found to have a good potential line this run - handed off to
    /// [`crate::player::PlayerContext::protected_familiar_icons`] on completion, which accumulates
    /// across every run rather than being replaced by it (see that field's own docs).
    protected_icons: Vec<Arc<Mat>>,
    /// Mouse rest point for other operations.
    mouse_rest: Point,
    /// Whether revealing completed without an unexpected detection failure.
    success: bool,
    /// Next [`Instant`] at which a pending lie detector prompt is (re-)checked for - see the
    /// matching field's fuller docs in `super::familiars_fusion::FamiliarsFusing`.
    lie_detector_check_next_at: Option<Instant>,
}

impl Display for FamiliarsRevealingPotential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.state {
            State::OpenMenu(_) => write!(f, "OpenMenu"),
            State::SwitchToInfoTab(_) => write!(f, "SwitchToInfoTab"),
            State::FindCards(_) => write!(f, "FindCards"),
            State::SelectCard(_, _) => write!(f, "SelectCard"),
            State::ConfirmUnlock(_, _, _, _) => write!(f, "ConfirmUnlock"),
            State::ReadPotential(_, _, _) => write!(f, "ReadPotential"),
            State::NextCard(_) => write!(f, "NextCard"),
            State::Completing(_, _) => write!(f, "Completing"),
        }
    }
}

impl FamiliarsRevealingPotential {
    /// `target_icons` empty means a full scan (filtered by `reveal_rarities`); non-empty means a
    /// targeted re-check of only those icons - see [`Self::target_icons`]'s docs.
    pub fn new(reveal_rarities: Array<FamiliarRarity, 2>, target_icons: Vec<Arc<Mat>>) -> Self {
        Self {
            state: State::OpenMenu(Timeout::default()),
            reveal_rarities,
            target_icons,
            cards: Array::new(),
            protected_icons: Vec::new(),
            mouse_rest: Point::new(50, 50),
            success: true,
            lie_detector_check_next_at: None,
        }
    }

    /// Whether this run is a full scan (as opposed to a targeted re-check) - see
    /// [`Self::target_icons`]'s docs.
    fn is_full_scan(&self) -> bool {
        self.target_icons.is_empty()
    }
}

/// Updates [`Player::FamiliarsRevealingPotential`] contextual state.
///
/// Note: This state does not use any [`Task`], so all detections are blocking. But this should be
/// acceptable for this state.
pub fn update_familiars_revealing_potential_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
) {
    let Player::FamiliarsRevealingPotential(revealing) = &player.state else {
        panic!("state is not familiars revealing potential")
    };
    let mut revealing = revealing.clone();

    let now = Instant::now();
    let should_check_lie_detector = revealing
        .lie_detector_check_next_at
        .is_none_or(|at| now >= at);
    if should_check_lie_detector {
        revealing.lie_detector_check_next_at = Some(now + Duration::from_millis(3000));
    }
    if should_check_lie_detector
        && (resources.detector().detect_lie_detector_shape().is_ok()
            || resources.detector().detect_lie_detector_violetta().is_ok())
    {
        // A lie detector prompt takes priority over revealing potentials - see the matching check
        // (and its fuller rationale) in `super::familiars_fusion::update_familiars_fusing_state`.
        // None of this state's own states involve typing into a text field, so this needs no
        // `is_safe_for_side_key_press` equivalent gate - just the same throttling.
        debug!(target: "backend/player", "aborting familiars revealing potential: lie detector pending");
        player.context.clear_action_completed();
        player.state = Player::Idle;
        return;
    }

    let familiar_key = match player.context.config.familiar_key {
        Some(val) => val,
        None => {
            debug!(target:"backend/player","aborted familiars revealing potential because familiar menu key is not set");
            player.context.clear_action_completed();
            player.state = Player::Idle;
            return;
        }
    };

    match revealing.state {
        State::OpenMenu(_) => update_open_menu(resources, &mut revealing, familiar_key),
        State::SwitchToInfoTab(_) => update_switch_to_info_tab(resources, &mut revealing),
        State::FindCards(_) => update_find_cards(resources, &mut revealing),
        State::SelectCard(_, _) => update_select_card(resources, &mut revealing),
        State::ConfirmUnlock(_, _, _, _) => update_confirm_unlock(resources, &mut revealing),
        State::ReadPotential(_, _, _) => update_read_potential(resources, &mut revealing),
        State::NextCard(index) => update_next_card(&mut revealing, index),
        State::Completing(timeout, completed) => {
            update_completing(resources, &mut revealing, timeout, completed)
        }
    }

    let is_completing = matches!(revealing.state, State::Completing(_, true));
    let success = revealing.success;
    let is_full_scan = revealing.is_full_scan();
    // Only actually taken on the completing tick - see
    // `super::familiars_fusion::update_familiars_fusing_state`'s own docs on why taking
    // unconditionally silently wipes it before the run finishes: `revealing` gets reassigned
    // back onto `player.state` on every *non*-completing tick too, so a good line found on any
    // card but the very last one processed was being cleared long before this state ever reached
    // its own completion and tried to hand the list off - defeating the entire point of this
    // feature.
    let protected_icons = if is_completing {
        std::mem::take(&mut revealing.protected_icons)
    } else {
        Vec::new()
    };

    let next = if is_completing {
        Player::Idle
    } else {
        Player::FamiliarsRevealingPotential(revealing)
    };

    match next_action(&player.context) {
        Some(_) => {
            if is_completing {
                player.context.clear_action_completed();
                player.context.mark_familiars_potential_run();
                player
                    .context
                    .protected_familiar_icons
                    .extend(protected_icons);
                if success {
                    player.context.clear_familiars_potential_fail_count();
                    if is_full_scan {
                        player.context.mark_familiars_potential_full_scan_done();
                    }
                } else {
                    player.context.track_familiars_potential_fail_count();
                }
            }

            player.state = next;
        }
        None => player.state = Player::Idle, // Force cancel if not from action
    }
}

fn update_open_menu(
    resources: &mut Resources,
    revealing: &mut FamiliarsRevealingPotential,
    key: KeyKind,
) {
    let State::OpenMenu(timeout) = revealing.state else {
        panic!("familiars revealing potential state is not opening menu");
    };

    match next_timeout_lifecycle(timeout, 60) {
        Lifecycle::Started(timeout) => {
            resources.input.send_mouse(
                revealing.mouse_rest.x,
                revealing.mouse_rest.y,
                MouseKind::Move,
            );
            if !resources.detector().detect_familiar_menu_opened() {
                resources.input.send_key(key);
            }
            revealing.state = State::OpenMenu(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_menu_opened() {
                revealing.state = State::SwitchToInfoTab(Timeout::default());
                return;
            }

            revealing.success = false;
            revealing.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => {
            if resources.detector().detect_familiar_menu_opened() {
                revealing.state = State::SwitchToInfoTab(Timeout::default());
                return;
            }

            revealing.state = State::OpenMenu(timeout);
        }
    }
}

/// Clicks the Info tab button. A single click is trusted (not retried every tick) since this is
/// an ordinary tab button, not a hover-animated toggle - mirrors
/// [`super::familiars_swap::update_switch_to_setup_tab`].
fn update_switch_to_info_tab(
    resources: &mut Resources,
    revealing: &mut FamiliarsRevealingPotential,
) {
    let State::SwitchToInfoTab(timeout) = revealing.state else {
        panic!("familiars revealing potential state is not switching to info tab");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            let bbox = resources
                .detector()
                .detect_familiar_info_tab_inactive_button();
            debug!(target: "backend/player", "info tab button: {bbox:?}");
            if let Ok(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            revealing.state = State::SwitchToInfoTab(timeout);
        }
        Lifecycle::Ended => {
            revealing.state = State::FindCards(Timeout::default());
        }
        Lifecycle::Updated(timeout) => {
            revealing.state = State::SwitchToInfoTab(timeout);
        }
    }
}

fn update_find_cards(resources: &mut Resources, revealing: &mut FamiliarsRevealingPotential) {
    let State::FindCards(timeout) = revealing.state else {
        panic!("familiars revealing potential state is not finding cards");
    };

    match next_timeout_lifecycle(timeout, 5) {
        Lifecycle::Ended => {
            let cards = resources.detector().detect_familiar_cards();
            debug!(target: "backend/player", "familiar collection cards: {}", cards.len());

            if revealing.target_icons.is_empty() {
                for (bbox, rank) in cards {
                    let rarity = match rank {
                        FamiliarRank::Rare => FamiliarRarity::Rare,
                        FamiliarRank::Epic => FamiliarRarity::Epic,
                    };
                    if revealing.reveal_rarities.iter().any(|r| *r == rarity)
                        && revealing.cards.len() < MAX_CARDS
                    {
                        revealing.cards.push(bbox);
                    }
                }
            } else {
                // Targeted re-check - ignores reveal_rarities entirely and restricts to Epic
                // rarity instead, since target_icons only ever comes from confirmed Rare->Epic
                // rank-ups - see Self::target_icons' docs for why this doesn't match by icon.
                for (bbox, rank) in cards {
                    if matches!(rank, FamiliarRank::Epic) && revealing.cards.len() < MAX_CARDS {
                        revealing.cards.push(bbox);
                    }
                }
            }

            revealing.state = if revealing.cards.is_empty() {
                revealing.success = true;
                State::Completing(Timeout::default(), false)
            } else {
                State::SelectCard(Timeout::default(), 0)
            };
        }
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            revealing.state = State::FindCards(timeout);
        }
    }
}

const SELECT_CARD_TIMEOUT: u32 = 15;

fn update_select_card(resources: &mut Resources, revealing: &mut FamiliarsRevealingPotential) {
    let State::SelectCard(timeout, index) = revealing.state else {
        panic!("familiars revealing potential state is not selecting card");
    };

    if index as usize >= revealing.cards.len() {
        revealing.success = true;
        revealing.state = State::Completing(Timeout::default(), false);
        return;
    }

    match next_timeout_lifecycle(timeout, SELECT_CARD_TIMEOUT) {
        Lifecycle::Started(timeout) => {
            let (x, y) = bbox_click_point(revealing.cards[index as usize]);
            resources.input.send_mouse(x, y, MouseKind::Click);
            revealing.state = State::SelectCard(timeout, index);
        }
        Lifecycle::Ended => {
            let locked = resources.detector().detect_familiar_info_potential_locked();
            debug!(target: "backend/player", "card {index} potential locked: {locked:?}");

            revealing.state = match locked {
                Ok(bbox) => {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    State::ConfirmUnlock(Timeout::default(), index, bbox, false)
                }
                // Already revealed (or a detection hiccup) - see State::SelectCard's own docs on
                // why this cannot be read without the locked bbox to anchor on.
                Err(_) => State::NextCard(index),
            };
        }
        Lifecycle::Updated(timeout) => {
            revealing.state = State::SelectCard(timeout, index);
        }
    }
}

const CONFIRM_UNLOCK_TIMEOUT: u32 = 20;

/// See [`State::ConfirmUnlock`]'s docs.
fn update_confirm_unlock(resources: &mut Resources, revealing: &mut FamiliarsRevealingPotential) {
    let State::ConfirmUnlock(timeout, index, region, clicked) = revealing.state else {
        panic!("familiars revealing potential state is not confirming unlock");
    };

    match next_timeout_lifecycle(timeout, CONFIRM_UNLOCK_TIMEOUT) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            let mut clicked = clicked;
            if !clicked {
                let bbox = resources.detector().detect_popup_confirm_button();
                debug!(target: "backend/player", "unlock potential confirm button: {bbox:?}");
                if let Ok(bbox) = bbox {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    clicked = true;
                }
            }
            revealing.state = State::ConfirmUnlock(timeout, index, region, clicked);
        }
        Lifecycle::Ended => {
            revealing.state = State::ReadPotential(Timeout::default(), index, region);
        }
    }
}

const READ_POTENTIAL_TIMEOUT: u32 = 15;

fn update_read_potential(resources: &mut Resources, revealing: &mut FamiliarsRevealingPotential) {
    let State::ReadPotential(timeout, index, region) = revealing.state else {
        panic!("familiars revealing potential state is not reading potential");
    };

    match next_timeout_lifecycle(timeout, READ_POTENTIAL_TIMEOUT) {
        Lifecycle::Ended => {
            // Still locked means the unlock-cost popup's Confirm did not actually go through
            // (e.g. detection hiccup, insufficient Mesos) - nothing was revealed, so there is
            // nothing to read. Deliberately not retried, to avoid risking a second, unintended
            // unlock-cost popup click against a card whose first attempt may in fact have gone
            // through but simply rendered slower than expected.
            let still_locked = resources
                .detector()
                .detect_familiar_info_potential_locked()
                .is_ok();
            if still_locked {
                debug!(target: "backend/player", "card {index} still locked after confirm, giving up on it");
                revealing.state = State::NextCard(index);
                return;
            }

            let good = resources
                .detector()
                .detect_familiar_potential_good_line(region);
            debug!(target: "backend/player", "card {index} revealed potential, good line found: {good}");
            if good
                && let Ok(icon) = resources
                    .detector()
                    .capture_region(revealing.cards[index as usize])
            {
                revealing.protected_icons.push(Arc::new(icon));
            }

            revealing.state = State::NextCard(index);
        }
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            revealing.state = State::ReadPotential(timeout, index, region);
        }
    }
}

fn update_next_card(revealing: &mut FamiliarsRevealingPotential, index: u32) {
    revealing.state = State::SelectCard(Timeout::default(), index + 1);
}

fn update_completing(
    resources: &mut Resources,
    revealing: &mut FamiliarsRevealingPotential,
    timeout: Timeout,
    completed: bool,
) {
    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            revealing.state = State::Completing(timeout, completed);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_menu_opened() {
                resources.input.send_key(KeyKind::Esc);
            }
            revealing.state = State::Completing(Timeout::default(), true);
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use opencv::core::{CV_8UC3, Scalar};

    use super::*;
    use crate::{bridge::MockInput, detect::MockDetector, ecs::Resources, player::PlayerContext};

    fn new_revealing() -> FamiliarsRevealingPotential {
        FamiliarsRevealingPotential::new(Array::new(), Vec::new())
    }

    /// A minimal valid (non-empty, typed) `Mat` - unlike `Mat::default()`, whose `Debug` impl
    /// genuinely errors on an empty/untyped matrix, tripping mockall's per-call argument-
    /// description formatting regardless of whether the expectation itself is satisfied.
    fn dummy_icon() -> Mat {
        Mat::new_rows_cols_with_default(1, 1, CV_8UC3, Scalar::all(0.0)).unwrap()
    }

    #[test]
    fn update_find_cards_full_scan_filters_by_rarity() {
        let rare_bbox = Rect::new(1, 1, 10, 10);
        let epic_bbox = Rect::new(2, 2, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(move || {
                vec![
                    (rare_bbox, FamiliarRank::Rare),
                    (epic_bbox, FamiliarRank::Epic),
                ]
            });
        let mut resources = Resources::new(None, Some(detector));

        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        let mut revealing = FamiliarsRevealingPotential::new(rarities, Vec::new());
        revealing.state = State::FindCards(Timeout {
            current: 5,
            started: true,
            ..Default::default()
        });

        update_find_cards(&mut resources, &mut revealing);

        assert_eq!(revealing.cards.len(), 1);
        assert_eq!(revealing.cards[0], rare_bbox);
        assert_matches!(revealing.state, State::SelectCard(_, 0));
    }

    #[test]
    fn update_find_cards_targeted_ignores_rarity_restricts_to_epic() {
        let rare_bbox = Rect::new(1, 1, 10, 10);
        let epic_bbox = Rect::new(2, 2, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(move || {
                vec![
                    (rare_bbox, FamiliarRank::Rare),
                    (epic_bbox, FamiliarRank::Epic),
                ]
            });
        let mut resources = Resources::new(None, Some(detector));

        // reveal_rarities is left empty below to prove targeted mode ignores it entirely - only
        // the Epic card should survive, regardless of reveal_rarities' contents.
        let mut revealing =
            FamiliarsRevealingPotential::new(Array::new(), vec![Arc::new(dummy_icon())]);
        revealing.state = State::FindCards(Timeout {
            current: 5,
            started: true,
            ..Default::default()
        });

        update_find_cards(&mut resources, &mut revealing);

        assert_eq!(revealing.cards.len(), 1);
        assert_eq!(revealing.cards[0], epic_bbox);
    }

    #[test]
    fn update_find_cards_completes_when_nothing_matches() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(Vec::new);
        let mut resources = Resources::new(None, Some(detector));

        let mut revealing = new_revealing();
        revealing.state = State::FindCards(Timeout {
            current: 5,
            started: true,
            ..Default::default()
        });

        update_find_cards(&mut resources, &mut revealing);

        assert!(revealing.cards.is_empty());
        assert_matches!(revealing.state, State::Completing(_, false));
    }

    #[test]
    fn update_familiars_revealing_potential_state_preserves_protected_icons_mid_run() {
        // Regression test: protected_icons was being cleared via std::mem::take on *every* tick,
        // not just the completing one - a good line found on any card but the very last one
        // processed was silently wiped before this state ever reached its own completion,
        // defeating the entire point of the feature.
        let mut detector = MockDetector::default();
        detector
            .expect_detect_lie_detector_shape()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        detector
            .expect_detect_lie_detector_violetta()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        let mut resources = Resources::new(Some(MockInput::default()), Some(detector));

        let mut revealing = new_revealing();
        revealing.protected_icons.push(Arc::new(dummy_icon()));
        // A non-completing transition - update_next_card just advances to the next SelectCard.
        revealing.state = State::NextCard(0);
        let mut player = PlayerEntity {
            state: Player::FamiliarsRevealingPotential(revealing),
            context: PlayerContext::default(),
        };
        player.context.config.familiar_key = Some(KeyKind::A);

        update_familiars_revealing_potential_state(&mut resources, &mut player);

        let Player::FamiliarsRevealingPotential(revealing) = &player.state else {
            panic!("expected still revealing potential, got {:?}", player.state);
        };
        assert_eq!(revealing.protected_icons.len(), 1);
        // Not handed off to PlayerContext yet either - only on actual completion.
        assert!(player.context.protected_familiar_icons.is_empty());
    }

    #[test]
    fn is_full_scan_true_when_target_icons_empty() {
        assert!(new_revealing().is_full_scan());
        assert!(
            !FamiliarsRevealingPotential::new(Array::new(), vec![Arc::new(dummy_icon())])
                .is_full_scan()
        );
    }
}
