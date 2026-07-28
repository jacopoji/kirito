use anyhow::Result;
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

/// Number of ticks to wait for the dungeon-select dialog to close before giving up and trying to
/// click the shuttle anyway.
const CLOSE_DIALOG_TIMEOUT: u32 = 40;
/// Number of ticks between repeated Esc/click attempts while waiting for the dialog(s) to close.
///
/// Unlike the clicks in `update_clicking_shuttle`/`update_clicking_yes` (harmless to repeat every
/// tick - clicking an already-open dialog's button again is a no-op), Esc is context-sensitive:
/// pressed while a dialog is open it closes that dialog, but pressed the instant nothing is open
/// anymore it opens the game's own system menu instead - and that menu blocks the shuttle click
/// far worse than the dialog it was meant to close ever did. The dialog's close animation can
/// still visually match `detect_monster_park_dialog` for a tick or two after the game itself
/// already considers it closed, so pressing again on literally the next tick risks landing in
/// exactly that gap. Waiting a full cooldown between presses gives the animation time to actually
/// finish before anything decides another press is warranted.
const CLOSE_DIALOG_RETRY_COOLDOWN_TICKS: u32 = 15;
/// Number of ticks to keep retrying shuttle detection for before giving up.
///
/// Generous on purpose - the shuttle is a social hotspot in the lobby (see the half-templates'
/// docs on partial occlusion) and other players/NPCs clustering in front of it can transiently
/// block every template for longer than a single frame.
const CLICKING_SHUTTLE_TIMEOUT: u32 = 60;
/// Number of ticks to keep retrying the dialog's `YES` button detection for before giving up.
const CLICKING_YES_TIMEOUT: u32 = 30;

/// States of exiting Monster Park via the entry lobby's shuttle NPC.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Dismisses the dungeon-select dialog, if still open, with Esc.
    ClosingDialog(Timeout),
    /// Clicks the shuttle NPC to open its "need a lift back to town?" dialog.
    ClickingShuttle(Timeout),
    /// Clicks the dialog's `YES` button to leave for town.
    ClickingYes(Timeout),
    /// Terminal state, whether leaving succeeded or was given up on (e.g. shuttle or its dialog's
    /// `YES` button not found). Either way, there is nothing further this state can do.
    Done,
}

#[derive(Debug, Clone, Copy)]
pub struct ExitingMonsterPark {
    state: State,
}

impl ExitingMonsterPark {
    pub fn new() -> Self {
        Self {
            state: State::ClosingDialog(Timeout::default()),
        }
    }
}

impl Default for ExitingMonsterPark {
    fn default() -> Self {
        Self::new()
    }
}

/// Updates [`Player::ExitingMonsterPark`] contextual state.
pub fn update_exiting_monster_park_state(resources: &mut Resources, player: &mut PlayerEntity) {
    let Player::ExitingMonsterPark(mut exiting) = player.state else {
        panic!("state is not exiting monster park")
    };

    match exiting.state {
        State::ClosingDialog(_) => update_closing_dialog(resources, &mut exiting),
        State::ClickingShuttle(_) => update_clicking_shuttle(resources, &mut exiting),
        State::ClickingYes(_) => update_clicking_yes(resources, &mut exiting),
        State::Done => (),
    }

    let player_next_state = if matches!(exiting.state, State::Done) {
        Player::Idle
    } else {
        Player::ExitingMonsterPark(exiting)
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

fn update_closing_dialog(resources: &mut Resources, exiting: &mut ExitingMonsterPark) {
    let State::ClosingDialog(timeout) = exiting.state else {
        panic!("exiting monster park state is not closing dialog")
    };

    // Running out of free entries doesn't just leave the dungeon-select dialog open (see
    // `update_confirming`'s docs) - Spiegelmann also pops his own "You cannot enter anymore today.
    // Come back tomorrow." info dialog on top of it, covering the dialog's middle including the
    // ticket label's usual screen position. `detect_monster_park_dialog` checks the dialog's title
    // banner instead - it sits in the header, above where that popup ever reaches, so it reliably
    // reads "still open" for as long as the dialog genuinely is, regardless of what's overlaid on
    // top of it (see its own docs for why the ticket label can't be used for this same check).
    // Checked every tick (cheap, and lets this exit the instant both are actually gone), but only
    // acted on periodically - see `CLOSE_DIALOG_RETRY_COOLDOWN_TICKS`'s docs for why Esc can't be
    // spammed like the clicks elsewhere in this state machine.
    let continue_button = resources.detector().detect_popup_dialog_continue_button();
    let dialog_open = resources.detector().detect_monster_park_dialog();
    if continue_button.is_err() && !dialog_open {
        exiting.state = State::ClickingShuttle(Timeout::default());
        return;
    }

    match next_timeout_lifecycle(timeout, CLOSE_DIALOG_TIMEOUT) {
        Lifecycle::Started(timeout) => {
            send_close_dialog_input(resources, continue_button);
            exiting.state = State::ClosingDialog(timeout);
        }
        Lifecycle::Updated(timeout) if timeout.current % CLOSE_DIALOG_RETRY_COOLDOWN_TICKS == 0 => {
            send_close_dialog_input(resources, continue_button);
            exiting.state = State::ClosingDialog(timeout);
        }
        Lifecycle::Updated(timeout) => {
            exiting.state = State::ClosingDialog(timeout);
        }
        Lifecycle::Ended => {
            // Give up waiting for it to close and try the shuttle click anyway instead of getting
            // stuck here forever - if it's genuinely still covering the shuttle, that click will
            // just miss and `update_clicking_shuttle` gives up on its own timeout too.
            exiting.state = State::ClickingShuttle(Timeout::default());
        }
    }
}

/// Dismisses the info popup's Next/End Chat button if it's up, otherwise Escs the dungeon-select
/// dialog itself.
fn send_close_dialog_input(resources: &mut Resources, continue_button: Result<Rect>) {
    if let Ok(bbox) = continue_button {
        let (x, y) = bbox_click_point(bbox);
        resources.input.send_mouse(x, y, MouseKind::Click);
    } else {
        resources.input.send_key(KeyKind::Esc);
    }
}

fn update_clicking_shuttle(resources: &mut Resources, exiting: &mut ExitingMonsterPark) {
    let State::ClickingShuttle(timeout) = exiting.state else {
        panic!("exiting monster park state is not clicking shuttle")
    };

    // Retries detection every tick instead of only once on entering this state - a single missed
    // frame (e.g. another player momentarily walking in front of the shuttle) shouldn't be treated
    // as "not there at all", see `CLICKING_SHUTTLE_TIMEOUT`'s docs.
    if let Ok(bbox) = resources.detector().detect_monster_park_shuttle() {
        let (x, y) = bbox_click_point(bbox);
        resources.input.send_mouse(x, y, MouseKind::Click);
        exiting.state = State::ClickingYes(Timeout::default());
        return;
    }

    match next_timeout_lifecycle(timeout, CLICKING_SHUTTLE_TIMEOUT) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            exiting.state = State::ClickingShuttle(timeout);
        }
        Lifecycle::Ended => {
            debug!(
                target: "backend/player",
                "Exiting Monster Park: shuttle not found"
            );
            exiting.state = State::Done;
        }
    }
}

fn update_clicking_yes(resources: &mut Resources, exiting: &mut ExitingMonsterPark) {
    let State::ClickingYes(timeout) = exiting.state else {
        panic!("exiting monster park state is not clicking yes")
    };

    // Same reasoning as `update_clicking_shuttle`: retry every tick rather than once.
    if let Ok(bbox) = resources
        .detector()
        .detect_monster_park_shuttle_yes_button()
    {
        let (x, y) = bbox_click_point(bbox);
        resources.input.send_mouse(x, y, MouseKind::Click);
        exiting.state = State::Done;
        return;
    }

    match next_timeout_lifecycle(timeout, CLICKING_YES_TIMEOUT) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            exiting.state = State::ClickingYes(timeout);
        }
        Lifecycle::Ended => {
            debug!(
                target: "backend/player",
                "Exiting Monster Park: shuttle dialog YES button not found"
            );
            exiting.state = State::Done;
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}
