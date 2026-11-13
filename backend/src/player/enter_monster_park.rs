use log::debug;
use opencv::core::Rect;

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    player::{
        MonsterParkEntryOutcome, PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// States of entering a Monster Park run from the entry lobby's gate.
///
/// Monster Park's dungeon-select dialog has a `Sweeps` counter with a `MAX` button - maxing it out
/// then clicking `Enter` starts a single run whose completion uses up every remaining free clear at
/// once. So one run is the whole day's Monster Park.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Presses Up to open the dungeon-select dialog.
    PressingUp(Timeout),
    /// Clicks the `MAX` button next to the `Sweeps` counter so `Enter` uses up every remaining
    /// free clear at once.
    ClickingMax(Timeout),
    /// Clicks the `Enter` button, keeping whatever dungeon tile was already selected rather than
    /// picking one.
    Confirming(Timeout),
    /// Terminal state, whether entering succeeded or was given up on (e.g. some element not
    /// found). Either way, there is nothing further this state can do.
    Done,
}

#[derive(Debug, Clone, Copy)]
pub struct EnteringMonsterPark {
    state: State,
}

impl EnteringMonsterPark {
    pub fn new() -> Self {
        Self {
            state: State::PressingUp(Timeout::default()),
        }
    }
}

impl Default for EnteringMonsterPark {
    fn default() -> Self {
        Self::new()
    }
}

/// Updates [`Player::EnteringMonsterPark`] contextual state.
pub fn update_entering_monster_park_state(resources: &mut Resources, player: &mut PlayerEntity) {
    let Player::EnteringMonsterPark(mut entering) = player.state else {
        panic!("state is not entering monster park")
    };

    match entering.state {
        State::PressingUp(_) => update_pressing_up(resources, &mut entering),
        State::ClickingMax(_) => update_clicking_max(resources, &mut entering),
        State::Confirming(_) => {
            if let Some(outcome) = update_confirming(resources, &mut entering) {
                player.context.monster_park_entry_outcome = Some(outcome);
            }
        }
        State::Done => (),
    }

    let player_next_state = if matches!(entering.state, State::Done) {
        Player::Idle
    } else {
        Player::EnteringMonsterPark(entering)
    };
    let is_terminal = matches!(player_next_state, Player::Idle);

    match next_action(&player.context) {
        Some(_) => {
            if is_terminal {
                player.context.clear_action_completed();
            }
            player.state = player_next_state;
        }
        None => player.state = Player::Idle,
    }
}

fn update_pressing_up(resources: &mut Resources, entering: &mut EnteringMonsterPark) {
    let State::PressingUp(timeout) = entering.state else {
        panic!("entering monster park state is not pressing up")
    };

    match next_timeout_lifecycle(timeout, 35) {
        Lifecycle::Started(timeout) => {
            // The dialog can already be open here - e.g. a previous entry attempt's Up press
            // actually landed but `detect_monster_park_dialog` missed it below before this state
            // got recreated and retried (see `DefaultRotator::rotate_monster_park_entry`'s
            // attempt-limit retry). Up means something different once the dialog is already up:
            // it navigates the dungeon-select tile grid instead of opening it, which would
            // silently move the selection away from whatever dungeon was actually chosen last
            // time. Skip straight to maxing sweeps instead of risking that.
            if resources.detector().detect_monster_park_dialog() {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dungeon-select dialog already open, skipping Up press"
                );
                entering.state = State::ClickingMax(Timeout::default());
                return;
            }

            resources.input.send_key(KeyKind::Up);
            entering.state = State::PressingUp(timeout);
        }
        Lifecycle::Ended => {
            if !resources.detector().detect_monster_park_dialog() {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dungeon-select dialog did not appear after pressing Up"
                );
                entering.state = State::Done;
                return;
            }

            entering.state = State::ClickingMax(Timeout::default());
        }
        Lifecycle::Updated(timeout) => entering.state = State::PressingUp(timeout),
    }
}

fn update_clicking_max(resources: &mut Resources, entering: &mut EnteringMonsterPark) {
    let State::ClickingMax(timeout) = entering.state else {
        panic!("entering monster park state is not clicking max")
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            // Not finding `MAX` isn't fatal - `Enter` still sweeps whatever count is already set
            // (at least 1), and a later attempt can pick up the rest. Same for "no free clears
            // left", where `MAX` may not do anything at all.
            match resources.detector().detect_monster_park_sweep_max_button() {
                Ok(bbox) => {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                }
                Err(_) => debug!(
                    target: "backend/player",
                    "Entering Monster Park: Sweeps MAX button not found, confirming anyway"
                ),
            }
            entering.state = State::ClickingMax(timeout);
        }
        Lifecycle::Ended => entering.state = State::Confirming(Timeout::default()),
        Lifecycle::Updated(timeout) => entering.state = State::ClickingMax(timeout),
    }
}

/// Returns the outcome once `Enter` was actually clicked and the result observed - see
/// [`MonsterParkEntryOutcome`].
fn update_confirming(
    resources: &mut Resources,
    entering: &mut EnteringMonsterPark,
) -> Option<MonsterParkEntryOutcome> {
    let State::Confirming(timeout) = entering.state else {
        panic!("entering monster park state is not confirming")
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources.detector().detect_monster_park_enter_button() else {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: Enter button not found"
                );
                entering.state = State::Done;
                return None;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            entering.state = State::Confirming(timeout);
            None
        }
        Lifecycle::Ended => {
            // The dungeon-select dialog stays open if entry didn't go through (e.g. no free
            // clears left) - clicking Enter otherwise dismisses it and starts the run. Leaving the
            // dialog open is fine since exiting closes it first.
            entering.state = State::Done;
            if resources.detector().detect_monster_park_dialog() {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dialog still open after clicking Enter, assuming no free clears left"
                );
                Some(MonsterParkEntryOutcome::NoClearsLeft)
            } else {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: entered a run with Sweeps maxed out"
                );
                Some(MonsterParkEntryOutcome::Entered)
            }
        }
        Lifecycle::Updated(timeout) => {
            entering.state = State::Confirming(timeout);
            None
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}
