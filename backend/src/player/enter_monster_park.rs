use log::debug;
use opencv::core::Rect;

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    player::{
        PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// States of entering a Monster Park run from the entry lobby's gate.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Presses Up to open the dungeon-select dialog.
    PressingUp(Timeout),
    /// Clicks the `Enter` button, keeping whatever dungeon tile was already selected from a
    /// previous run rather than picking one.
    Confirming(Timeout),
    /// Terminal state, whether entering succeeded or was given up on (e.g. no free entry, some
    /// element not found). Either way, there is nothing further this state can do.
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
        State::Confirming(_) => update_confirming(resources, &mut entering),
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
            resources.input.send_key(KeyKind::Up);
            entering.state = State::PressingUp(timeout);
        }
        Lifecycle::Ended => {
            // Whether a free entry is actually available (ticket count/free-clear-remaining OCR)
            // is unreliable to check upfront - just attempt the Enter click and treat a still-open
            // dialog afterwards as "no free entry" instead (see `update_confirming`).
            if !resources.detector().detect_monster_park_dialog() {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dungeon-select dialog did not appear after pressing Up"
                );
                entering.state = State::Done;
                return;
            }

            entering.state = State::Confirming(Timeout::default());
        }
        Lifecycle::Updated(timeout) => entering.state = State::PressingUp(timeout),
    }
}

fn update_confirming(resources: &mut Resources, entering: &mut EnteringMonsterPark) {
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
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            entering.state = State::Confirming(timeout);
        }
        Lifecycle::Ended => {
            // The dungeon-select dialog stays open if entry didn't actually go through (e.g. no
            // free entry left) - clicking Enter otherwise dismisses it.
            if resources.detector().detect_monster_park_dialog() {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dialog still open after clicking Enter, assuming no free entry available"
                );
            }
            entering.state = State::Done;
        }
        Lifecycle::Updated(timeout) => entering.state = State::Confirming(timeout),
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}
