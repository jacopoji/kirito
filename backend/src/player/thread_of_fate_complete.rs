use std::sync::{Arc, Mutex};

use log::debug;
use opencv::core::Rect;

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    models::ThreadOfFateAskProgress,
    player::{
        PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// Number of ticks between each interact-key press in [`State::AdvanceDialog`] - mirrors
/// [`super::thread_of_fate_ask::update_advance_dialog`]'s own tuning, since turning in the quest
/// opens the same kind of NPC dialog (confirmed live: clicking the completable entry does not
/// instantly complete the quest, it opens a dialog that needs the interact key pressed through it
/// first) rather than completing instantly on the initial click.
const ADVANCE_DIALOG_PRESS_INTERVAL: u32 = 15;
/// Maximum number of interact-key presses attempted in [`State::AdvanceDialog`] before giving up
/// and falling back to [`State::ConfirmCompleted`]'s own patient wait anyway - see
/// [`super::thread_of_fate_ask::ADVANCE_DIALOG_MAX_PRESSES`]'s docs for the same reasoning.
const ADVANCE_DIALOG_MAX_PRESSES: u32 = 15;

/// States of turning in the pending Thread of Fate ask quest.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Clicks the quest hint lightbulb to open the quest list.
    OpenQuestMenu(Timeout),
    /// Clicks the ask quest's now-completable entry in the quest list.
    ClickCompletableQuest(Timeout),
    /// Repeatedly presses the interact key to advance through the turn-in dialog opened by
    /// [`State::ClickCompletableQuest`] - clicking the completable entry alone does not
    /// immediately complete the quest, it opens a dialog (matching the accept flow's own
    /// [`super::thread_of_fate_ask::State::AdvanceDialog`]) that needs to be pressed through
    /// first. Unlike the accept flow, there is no toast to watch for mid-dialog - completion is
    /// only confirmed afterwards, in [`State::ConfirmCompleted`].
    AdvanceDialog(Timeout, u32),
    /// Reopens the quest list once the turn-in dialog is no longer visible (or
    /// [`State::AdvanceDialog`] gave up) and confirms completion by checking that the completable
    /// entry [`State::ClickCompletableQuest`] clicked is now gone - there is no "Quest complete!"
    /// toast for this flow (unlike the accept flow's own toast-driven confirmation), so absence
    /// from the list is the only signal available.
    ConfirmCompleted(Timeout),
    Completing(Timeout, bool),
}

#[derive(Debug, Clone)]
pub struct ThreadOfFateCompleting {
    state: State,
    character_id: Option<i64>,
    progress: Arc<Mutex<ThreadOfFateAskProgress>>,
    success: bool,
}

impl ThreadOfFateCompleting {
    pub fn new(character_id: Option<i64>, progress: Arc<Mutex<ThreadOfFateAskProgress>>) -> Self {
        Self {
            state: State::OpenQuestMenu(Timeout::default()),
            character_id,
            progress,
            success: false,
        }
    }
}

/// Updates [`Player::ThreadOfFateCompleting`] contextual state.
pub fn update_thread_of_fate_completing_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
) {
    let Player::ThreadOfFateCompleting(mut completing) = player.state.clone() else {
        panic!("state is not thread of fate completing")
    };

    let interact_key = player.context.config.interact_key;

    match completing.state {
        State::OpenQuestMenu(_) => update_open_quest_menu(resources, &mut completing),
        State::ClickCompletableQuest(_) => {
            update_click_completable_quest(resources, &mut completing)
        }
        State::AdvanceDialog(_, _) => {
            update_advance_dialog(resources, &mut completing, interact_key)
        }
        State::ConfirmCompleted(_) => update_confirm_completed(resources, &mut completing),
        State::Completing(timeout, completed) => {
            update_completing(resources, &mut completing, timeout, completed)
        }
    }

    let is_completing = matches!(completing.state, State::Completing(_, true));
    if is_completing && completing.success {
        let mut progress = completing.progress.lock().unwrap();
        progress.pending_npc = None;
        progress.pending_since_secs = None;
        drop(progress);

        resources
            .character_updates
            .mark_thread_of_fate_ask_completed(completing.character_id);
    }

    let player_next_state = if is_completing {
        Player::Idle
    } else {
        Player::ThreadOfFateCompleting(completing)
    };

    match next_action(&player.context) {
        Some(_) => {
            if is_completing {
                player.context.clear_action_completed();
            }
            player.state = player_next_state;
        }
        None => player.state = Player::Idle, // Force cancel if not from action
    }
}

fn update_open_quest_menu(resources: &mut Resources, completing: &mut ThreadOfFateCompleting) {
    let State::OpenQuestMenu(timeout) = completing.state else {
        panic!("thread of fate completing state is not opening quest menu")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources
                .detector()
                .detect_thread_of_fate_quest_light_bulb()
            else {
                debug!(target: "backend/player", "Thread of Fate complete: quest light bulb not found");
                completing.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            completing.state = State::OpenQuestMenu(timeout);
        }
        Lifecycle::Ended => {
            if resources
                .detector()
                .detect_thread_of_fate_quest_completable()
                .is_err()
            {
                debug!(target: "backend/player", "Thread of Fate complete: completable quest not found in quest list");
                completing.state = State::Completing(Timeout::default(), false);
                return;
            }
            completing.state = State::ClickCompletableQuest(Timeout::default());
        }
        Lifecycle::Updated(timeout) => completing.state = State::OpenQuestMenu(timeout),
    }
}

fn update_click_completable_quest(
    resources: &mut Resources,
    completing: &mut ThreadOfFateCompleting,
) {
    let State::ClickCompletableQuest(timeout) = completing.state else {
        panic!("thread of fate completing state is not clicking completable quest")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources
                .detector()
                .detect_thread_of_fate_quest_completable()
            else {
                debug!(target: "backend/player", "Thread of Fate complete: completable quest disappeared before clicking");
                completing.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            completing.state = State::ClickCompletableQuest(timeout);
        }
        Lifecycle::Ended => {
            completing.state = State::AdvanceDialog(Timeout::default(), 0);
        }
        Lifecycle::Updated(timeout) => completing.state = State::ClickCompletableQuest(timeout),
    }
}

fn update_advance_dialog(
    resources: &mut Resources,
    completing: &mut ThreadOfFateCompleting,
    interact_key: KeyKind,
) {
    let State::AdvanceDialog(timeout, presses) = completing.state else {
        panic!("thread of fate completing state is not advancing dialog")
    };

    match next_timeout_lifecycle(timeout, ADVANCE_DIALOG_PRESS_INTERVAL) {
        Lifecycle::Started(timeout) => {
            debug!(
                target: "backend/player",
                "Thread of Fate complete: press {presses} pressing interact key {interact_key:?}"
            );
            resources.input.send_key(interact_key);
            completing.state = State::AdvanceDialog(timeout, presses);
        }
        Lifecycle::Updated(timeout) => {
            completing.state = State::AdvanceDialog(timeout, presses);
        }
        Lifecycle::Ended => {
            // Mirrors `thread_of_fate_ask`'s own `AdvanceDialog` reasoning: `detect_special_dialog`
            // covers decline/cooldown-style messages that don't close on ESC and still need the
            // interact key pressed through them, so treat one being visible as "still in dialog"
            // instead of prematurely handing off to the list re-check below.
            let still_in_dialog = resources.detector().detect_special_dialog();
            debug!(
                target: "backend/player",
                "Thread of Fate complete: press {presses} ended, still_in_dialog={still_in_dialog}"
            );
            if !still_in_dialog {
                completing.state = State::ConfirmCompleted(Timeout::default());
                return;
            }
            if presses + 1 >= ADVANCE_DIALOG_MAX_PRESSES {
                debug!(
                    target: "backend/player",
                    "Thread of Fate complete: gave up advancing dialog after {} presses, checking quest list anyway",
                    presses + 1
                );
                completing.state = State::ConfirmCompleted(Timeout::default());
                return;
            }
            completing.state = State::AdvanceDialog(Timeout::default(), presses + 1);
        }
    }
}

fn update_confirm_completed(resources: &mut Resources, completing: &mut ThreadOfFateCompleting) {
    let State::ConfirmCompleted(timeout) = completing.state else {
        panic!("thread of fate completing state is not confirming completed")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            // The turn-in dialog closing does not by itself confirm completion - re-open the
            // quest list (there is no "Quest complete!" toast for this flow, unlike the accept
            // flow) so the `Ended` branch below can check whether the completable entry is gone.
            let Ok(bbox) = resources
                .detector()
                .detect_thread_of_fate_quest_light_bulb()
            else {
                debug!(target: "backend/player", "Thread of Fate complete: quest light bulb not found while confirming completion");
                completing.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            completing.state = State::ConfirmCompleted(timeout);
        }
        Lifecycle::Ended => {
            let still_completable = resources
                .detector()
                .detect_thread_of_fate_quest_completable()
                .is_ok();
            debug!(
                target: "backend/player",
                "Thread of Fate complete: quest list re-checked, still_completable={still_completable}"
            );
            completing.success = !still_completable;
            completing.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => completing.state = State::ConfirmCompleted(timeout),
    }
}

fn update_completing(
    resources: &mut Resources,
    completing: &mut ThreadOfFateCompleting,
    timeout: Timeout,
    completed: bool,
) {
    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            completing.state = State::Completing(timeout, completed);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_thread_of_fate_menu_opened() {
                resources.input.send_key(KeyKind::Esc);
            }
            debug!(target: "backend/player", "Thread of Fate complete completed, success={}", completing.success);
            completing.state = State::Completing(Timeout::default(), true);
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}
