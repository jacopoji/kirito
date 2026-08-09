use std::sync::{Arc, Mutex};

use log::debug;
use opencv::core::Rect;

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    models::{ThreadOfFateAskProgress, ThreadOfFateNpc},
    player::{
        PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// Number of ticks between each interact-key press in [`State::AdvanceDialog`].
///
/// Tuned down from an initial 30 (1s) after live testing showed dialog pages advance almost
/// immediately - a full second per press made the flow feel sluggish for no benefit. Nudged back
/// up slightly from an even shorter first attempt (10) after live logs showed individual tick
/// processing regularly taking 30-42ms against the nominal 33.3ms (30 FPS) budget - a short
/// interval leaves less real slack for a still-rendering page to catch up before the next check.
const ADVANCE_DIALOG_PRESS_INTERVAL: u32 = 15;
/// Offset `(x, y)` from [`Detector::detect_thread_of_fate_ask_dialog_anchor`]'s top-left corner
/// to a point near the start of the red, clickable request line below it (e.g. "Go to Ludibrium
/// and Clean?") - measured directly against a real screenshot. Deliberately near the *start* of
/// the line rather than its center, since the actual request text is per-attempt and varies in
/// length; the start position stays fixed regardless.
///
/// Live testing showed the interact key alone (which reliably advances the earlier intro dialog
/// pages) does not reliably select this particular choice - unlike those pages, actually clicking
/// the link is needed here, matching the original spec ("clicking on the red text or pressing
/// interact key").
const ASK_DIALOG_ANCHOR_CHOICE_OFFSET: (i32, i32) = (31, 39);

/// Maximum number of interact-key presses attempted in [`State::AdvanceDialog`] before giving up -
/// generous enough for the handful of intro dialog pages seen per NPC plus the final choice, while
/// still bounded so a declined ask (e.g. the NPC still being on cooldown from a previous ask) does
/// not spin forever. Raised alongside lowering [`ADVANCE_DIALOG_PRESS_INTERVAL`] to keep roughly
/// the same overall give-up ceiling (~5s).
const ADVANCE_DIALOG_MAX_PRESSES: u32 = 15;
/// Ticks [`State::WaitForAcceptedToast`] waits for the "Quest complete!" toast to appear once the
/// dialog itself has closed, before concluding the ask was declined and giving up.
///
/// Confirmed live: the toast fades in rather than appearing instantly, taking roughly a second to
/// reach full opacity - matching it requires the fully-rendered banner, so a check made too soon
/// after the dialog closes catches it mid-fade and misses. ~2s gives that fade a comfortable
/// margin instead of racing it.
const WAIT_FOR_ACCEPTED_TOAST_TICKS: u32 = 60;

/// States of asking a Thread of Fate NPC for a favor.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Clicks the quest hint lightbulb to open the quest list.
    OpenQuestMenu(Timeout),
    /// Clicks the `[Threads of Fate] Unraveling the Threads of Fate` entry in the quest list.
    ClickEntryQuest(Timeout),
    /// Waits for the Threads of Fate menu to actually open.
    ConfirmMenuOpened(Timeout),
    /// Clicks the selected NPC's icon in the relationship grid.
    ClickNpc(Timeout),
    /// Waits for the selected NPC's portrait and `Ask` button to appear.
    ConfirmNpcSelected(Timeout),
    /// Clicks the `Ask` button.
    ClickAskButton(Timeout),
    /// Repeatedly presses the interact key (or clicks the final choice, see
    /// [`ASK_DIALOG_ANCHOR_CHOICE_OFFSET`]) to advance through the intro dialog page(s), watching
    /// for the "Quest complete!" toast (accepted) each press - see [`ADVANCE_DIALOG_MAX_PRESSES`]'s
    /// docs for why this doesn't retry forever. Moves to [`State::WaitForAcceptedToast`] as soon
    /// as the dialog itself is no longer visible, rather than concluding declined immediately.
    AdvanceDialog(Timeout, u32),
    /// Waits for the "Quest complete!" toast once the dialog has closed - see
    /// [`WAIT_FOR_ACCEPTED_TOAST_TICKS`]'s docs for why this needs its own patient wait rather
    /// than being folded into [`State::AdvanceDialog`]'s own per-press timing.
    WaitForAcceptedToast(Timeout),
    /// Closes the Threads of Fate menu, whether or not the ask actually succeeded.
    CloseMenu(Timeout),
    Completing(Timeout, bool),
}

#[derive(Debug, Clone)]
pub struct ThreadOfFateAsking {
    state: State,
    npc: ThreadOfFateNpc,
    character_id: Option<i64>,
    progress: Arc<Mutex<ThreadOfFateAskProgress>>,
    /// Whether the ask was actually accepted (the "Quest complete!" toast was seen) - decided in
    /// [`State::AdvanceDialog`], consumed once [`State::Completing`] finishes.
    success: bool,
}

impl ThreadOfFateAsking {
    pub fn new(
        npc: ThreadOfFateNpc,
        character_id: Option<i64>,
        progress: Arc<Mutex<ThreadOfFateAskProgress>>,
    ) -> Self {
        Self {
            state: State::OpenQuestMenu(Timeout::default()),
            npc,
            character_id,
            progress,
            success: false,
        }
    }
}

/// Updates [`Player::ThreadOfFateAsking`] contextual state.
///
/// Note: This state does not use any [`Task`], so all detections are blocking, mirroring
/// [`super::familiars_fusion::update_familiars_fusing_state`]'s own reasoning for why that's
/// acceptable here too.
pub fn update_thread_of_fate_asking_state(resources: &mut Resources, player: &mut PlayerEntity) {
    let Player::ThreadOfFateAsking(mut asking) = player.state.clone() else {
        panic!("state is not thread of fate asking")
    };

    let interact_key = player.context.config.interact_key;

    match asking.state {
        State::OpenQuestMenu(_) => update_open_quest_menu(resources, &mut asking),
        State::ClickEntryQuest(_) => update_click_entry_quest(resources, &mut asking),
        State::ConfirmMenuOpened(_) => update_confirm_menu_opened(resources, &mut asking),
        State::ClickNpc(_) => update_click_npc(resources, &mut asking),
        State::ConfirmNpcSelected(_) => update_confirm_npc_selected(resources, &mut asking),
        State::ClickAskButton(_) => update_click_ask_button(resources, &mut asking),
        State::AdvanceDialog(_, _) => update_advance_dialog(resources, &mut asking, interact_key),
        State::WaitForAcceptedToast(_) => update_wait_for_accepted_toast(resources, &mut asking),
        State::CloseMenu(_) => update_close_menu(resources, &mut asking),
        State::Completing(timeout, completed) => {
            update_completing(resources, &mut asking, timeout, completed)
        }
    }

    let is_completing = matches!(asking.state, State::Completing(_, true));
    if is_completing && asking.success {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut progress = asking.progress.lock().unwrap();
        progress.pending_npc = Some(asking.npc);
        progress.pending_since_secs = Some(now);
        progress.count_today += 1;
        drop(progress);

        resources
            .character_updates
            .mark_thread_of_fate_ask_accepted(asking.character_id, asking.npc, now);
    }

    let player_next_state = if is_completing {
        Player::Idle
    } else {
        Player::ThreadOfFateAsking(asking)
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

fn update_open_quest_menu(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::OpenQuestMenu(timeout) = asking.state else {
        panic!("thread of fate asking state is not opening quest menu")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources
                .detector()
                .detect_thread_of_fate_quest_light_bulb()
            else {
                debug!(target: "backend/player", "Thread of Fate ask: quest light bulb not found");
                asking.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            asking.state = State::OpenQuestMenu(timeout);
        }
        Lifecycle::Ended => {
            if resources
                .detector()
                .detect_thread_of_fate_entry_quest()
                .is_err()
            {
                debug!(target: "backend/player", "Thread of Fate ask: entry quest not found in quest list");
                asking.state = State::Completing(Timeout::default(), false);
                return;
            }
            asking.state = State::ClickEntryQuest(Timeout::default());
        }
        Lifecycle::Updated(timeout) => asking.state = State::OpenQuestMenu(timeout),
    }
}

fn update_click_entry_quest(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::ClickEntryQuest(timeout) = asking.state else {
        panic!("thread of fate asking state is not clicking entry quest")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources.detector().detect_thread_of_fate_entry_quest() else {
                debug!(target: "backend/player", "Thread of Fate ask: entry quest disappeared before clicking");
                asking.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            asking.state = State::ClickEntryQuest(timeout);
        }
        Lifecycle::Ended => {
            asking.state = State::ConfirmMenuOpened(Timeout::default());
        }
        Lifecycle::Updated(timeout) => asking.state = State::ClickEntryQuest(timeout),
    }
}

fn update_confirm_menu_opened(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::ConfirmMenuOpened(timeout) = asking.state else {
        panic!("thread of fate asking state is not confirming menu opened")
    };

    match next_timeout_lifecycle(timeout, 40) {
        Lifecycle::Ended => {
            if !resources.detector().detect_thread_of_fate_menu_opened() {
                debug!(target: "backend/player", "Thread of Fate ask: menu did not open");
                asking.state = State::Completing(Timeout::default(), false);
                return;
            }
            asking.state = State::ClickNpc(Timeout::default());
        }
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            asking.state = State::ConfirmMenuOpened(timeout);
        }
    }
}

fn update_click_npc(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::ClickNpc(timeout) = asking.state else {
        panic!("thread of fate asking state is not clicking npc")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(title_bbox) = resources.detector().detect_thread_of_fate_ui_title() else {
                debug!(target: "backend/player", "Thread of Fate ask: title anchor not found, cannot locate npc {:?}", asking.npc);
                asking.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (dx, dy) = asking.npc.icon_offset_from_title();
            resources
                .input
                .send_mouse(title_bbox.x + dx, title_bbox.y + dy, MouseKind::Click);
            asking.state = State::ClickNpc(timeout);
        }
        Lifecycle::Ended => {
            asking.state = State::ConfirmNpcSelected(Timeout::default());
        }
        Lifecycle::Updated(timeout) => asking.state = State::ClickNpc(timeout),
    }
}

fn update_confirm_npc_selected(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::ConfirmNpcSelected(timeout) = asking.state else {
        panic!("thread of fate asking state is not confirming npc selected")
    };

    // Checks every tick instead of only once after a fixed wait - live testing showed the
    // portrait/button typically render almost immediately after the click, so waiting out a full
    // fixed timeout regardless felt sluggish. The timeout below is only a safety ceiling for the
    // case this never becomes true.
    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            let portrait_matches = resources
                .detector()
                .detect_thread_of_fate_npc_portrait(asking.npc);
            let has_ask_button = resources
                .detector()
                .detect_thread_of_fate_ask_button()
                .is_ok();
            if portrait_matches && has_ask_button {
                asking.state = State::ClickAskButton(Timeout::default());
                return;
            }
            asking.state = State::ConfirmNpcSelected(timeout);
        }
        Lifecycle::Ended => {
            debug!(
                target: "backend/player",
                "Thread of Fate ask: npc {:?} not confirmed selected within timeout",
                asking.npc
            );
            asking.state = State::Completing(Timeout::default(), false);
        }
    }
}

fn update_click_ask_button(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::ClickAskButton(timeout) = asking.state else {
        panic!("thread of fate asking state is not clicking ask button")
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            let Ok(bbox) = resources.detector().detect_thread_of_fate_ask_button() else {
                debug!(target: "backend/player", "Thread of Fate ask: ask button disappeared before clicking");
                asking.state = State::Completing(Timeout::default(), false);
                return;
            };
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            asking.state = State::ClickAskButton(timeout);
        }
        Lifecycle::Ended => {
            asking.state = State::AdvanceDialog(Timeout::default(), 0);
        }
        Lifecycle::Updated(timeout) => {
            // Moves on the instant the resulting dialog is visibly up, rather than always
            // waiting out the full timeout regardless - live testing showed it typically renders
            // well before the fixed wait used to elapse. Falls back to the fixed timeout above if
            // neither ever appears (harmless either way - `AdvanceDialog` still presses the
            // interact key and has its own bail-out if nothing is actually there).
            let dialog_visible = resources
                .detector()
                .detect_thread_of_fate_next_button()
                .is_ok()
                || resources
                    .detector()
                    .detect_thread_of_fate_ask_dialog_anchor()
                    .is_ok();
            if dialog_visible {
                asking.state = State::AdvanceDialog(Timeout::default(), 0);
                return;
            }
            asking.state = State::ClickAskButton(timeout);
        }
    }
}

/// Whether the ask-accepted toast is currently visible - tries the dedicated template first (see
/// [`Detector::detect_thread_of_fate_quest_started_popup`]'s docs) and falls back to the generic
/// "Quest complete!" toast, in case either one alone happens to miss.
fn detect_ask_accepted_toast(resources: &mut Resources) -> bool {
    resources
        .detector()
        .detect_thread_of_fate_quest_started_popup()
        .is_ok()
        || resources.detector().detect_quest_complete_popup().is_ok()
}

fn update_advance_dialog(
    resources: &mut Resources,
    asking: &mut ThreadOfFateAsking,
    interact_key: KeyKind,
) {
    let State::AdvanceDialog(timeout, presses) = asking.state else {
        panic!("thread of fate asking state is not advancing dialog")
    };

    match next_timeout_lifecycle(timeout, ADVANCE_DIALOG_PRESS_INTERVAL) {
        Lifecycle::Started(timeout) => {
            // The final choice page (the anchor's own page) needs an actual click on the red
            // request line - the interact key alone does not reliably select it. Every earlier
            // page (the intro dialog(s)) advances fine on the interact key alone.
            if let Ok(anchor_bbox) = resources
                .detector()
                .detect_thread_of_fate_ask_dialog_anchor()
            {
                let (dx, dy) = ASK_DIALOG_ANCHOR_CHOICE_OFFSET;
                let (x, y) = (anchor_bbox.x + dx, anchor_bbox.y + dy);
                debug!(
                    target: "backend/player",
                    "Thread of Fate ask: press {presses} clicking choice at ({x}, {y}) (anchor={anchor_bbox:?})"
                );
                resources.input.send_mouse(x, y, MouseKind::Click);
            } else {
                debug!(
                    target: "backend/player",
                    "Thread of Fate ask: press {presses} pressing interact key {interact_key:?}"
                );
                resources.input.send_key(interact_key);
            }
            asking.state = State::AdvanceDialog(timeout, presses);
        }
        Lifecycle::Updated(timeout) => {
            if detect_ask_accepted_toast(resources) {
                debug!(target: "backend/player", "Thread of Fate ask: accepted toast seen");
                asking.success = true;
                asking.state = State::CloseMenu(Timeout::default());
                return;
            }
            asking.state = State::AdvanceDialog(timeout, presses);
        }
        Lifecycle::Ended => {
            if detect_ask_accepted_toast(resources) {
                asking.success = true;
                asking.state = State::CloseMenu(Timeout::default());
                return;
            }
            // Neither still mid-dialog (anchor/next button/any other special dialog visible) nor
            // succeeded yet. `detect_special_dialog` also covers decline/cooldown messages (e.g.
            // "you need to chill out" with its own Yes button) that the earlier two checks alone
            // don't recognize - those still need the interact key pressed against them to actually
            // dismiss (they don't close on ESC, see that method's own docs), so treating them as
            // "still in dialog" keeps this loop pressing through them via the plain interact-key
            // branch above instead of prematurely handing off to `WaitForAcceptedToast` while one
            // is still open on screen. The dialog itself has served its purpose either way once
            // none of these match - hand off to `WaitForAcceptedToast` for a dedicated, patient
            // wait for the (slow-fading-in) toast rather than continuing to burn through the press
            // budget here.
            let still_in_dialog = resources
                .detector()
                .detect_thread_of_fate_ask_dialog_anchor()
                .is_ok()
                || resources
                    .detector()
                    .detect_thread_of_fate_next_button()
                    .is_ok()
                || resources.detector().detect_special_dialog();
            debug!(
                target: "backend/player",
                "Thread of Fate ask: press {presses} ended, still_in_dialog={still_in_dialog}"
            );
            if !still_in_dialog {
                asking.state = State::WaitForAcceptedToast(Timeout::default());
                return;
            }
            if presses + 1 >= ADVANCE_DIALOG_MAX_PRESSES {
                debug!(
                    target: "backend/player",
                    "Thread of Fate ask: gave up advancing dialog after {} presses, waiting for toast anyway",
                    presses + 1
                );
                asking.state = State::WaitForAcceptedToast(Timeout::default());
                return;
            }
            asking.state = State::AdvanceDialog(Timeout::default(), presses + 1);
        }
    }
}

/// Waits for the "Quest complete!" toast to fade in after [`State::AdvanceDialog`]'s dialog has
/// closed - see [`WAIT_FOR_ACCEPTED_TOAST_TICKS`]'s docs for why this needs its own dedicated,
/// patient wait rather than being decided immediately once the dialog closes.
fn update_wait_for_accepted_toast(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::WaitForAcceptedToast(timeout) = asking.state else {
        panic!("thread of fate asking state is not waiting for accepted toast")
    };

    match next_timeout_lifecycle(timeout, WAIT_FOR_ACCEPTED_TOAST_TICKS) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            if detect_ask_accepted_toast(resources) {
                debug!(target: "backend/player", "Thread of Fate ask: accepted toast seen");
                asking.success = true;
                asking.state = State::CloseMenu(Timeout::default());
                return;
            }
            asking.state = State::WaitForAcceptedToast(timeout);
        }
        Lifecycle::Ended => {
            if detect_ask_accepted_toast(resources) {
                asking.success = true;
            } else {
                debug!(target: "backend/player", "Thread of Fate ask: toast never appeared, likely declined");
            }
            asking.state = State::CloseMenu(Timeout::default());
        }
    }
}

fn update_close_menu(resources: &mut Resources, asking: &mut ThreadOfFateAsking) {
    let State::CloseMenu(timeout) = asking.state else {
        panic!("thread of fate asking state is not closing menu")
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            if resources.detector().detect_thread_of_fate_menu_opened() {
                resources.input.send_key(KeyKind::Esc);
            }
            asking.state = State::CloseMenu(timeout);
        }
        Lifecycle::Ended => {
            asking.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => asking.state = State::CloseMenu(timeout),
    }
}

fn update_completing(
    resources: &mut Resources,
    asking: &mut ThreadOfFateAsking,
    timeout: Timeout,
    completed: bool,
) {
    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            asking.state = State::Completing(timeout, completed);
        }
        Lifecycle::Ended => {
            // Last-resort cleanup - see `detect_esc_settings`'s own inclusion of
            // `detect_thread_of_fate_menu_opened` for the primary safety net (panic overriding
            // this state mid-flow).
            if resources.detector().detect_thread_of_fate_menu_opened() {
                resources.input.send_key(KeyKind::Esc);
            }
            debug!(target: "backend/player", "Thread of Fate ask completed, success={}", asking.success);
            asking.state = State::Completing(Timeout::default(), true);
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}
