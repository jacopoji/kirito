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

/// States of turning in the pending Thread of Fate ask quest.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Clicks the quest hint lightbulb to open the quest list.
    OpenQuestMenu(Timeout),
    /// Clicks the ask quest's now-completable entry in the quest list.
    ClickCompletableQuest(Timeout),
    /// Waits for the "Quest complete!" toast confirming the turn-in went through.
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

    match completing.state {
        State::OpenQuestMenu(_) => update_open_quest_menu(resources, &mut completing),
        State::ClickCompletableQuest(_) => {
            update_click_completable_quest(resources, &mut completing)
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
            completing.state = State::ConfirmCompleted(Timeout::default());
        }
        Lifecycle::Updated(timeout) => completing.state = State::ClickCompletableQuest(timeout),
    }
}

fn update_confirm_completed(resources: &mut Resources, completing: &mut ThreadOfFateCompleting) {
    let State::ConfirmCompleted(timeout) = completing.state else {
        panic!("thread of fate completing state is not confirming completed")
    };

    match next_timeout_lifecycle(timeout, 60) {
        Lifecycle::Updated(timeout) => {
            if resources.detector().detect_quest_complete_popup().is_ok() {
                debug!(target: "backend/player", "Thread of Fate complete: quest complete toast seen");
                completing.success = true;
                completing.state = State::Completing(Timeout::default(), false);
                return;
            }
            completing.state = State::ConfirmCompleted(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_quest_complete_popup().is_ok() {
                completing.success = true;
            } else {
                debug!(target: "backend/player", "Thread of Fate complete: quest complete toast not seen in time");
            }
            completing.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Started(timeout) => completing.state = State::ConfirmCompleted(timeout),
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
