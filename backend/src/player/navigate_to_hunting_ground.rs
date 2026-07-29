use log::{debug, info};
use opencv::core::Rect;

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    player::{
        NavigateToHuntingGround, PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// World map dropdown box offsets `(x, y, width, height)`, relative to the detected `WORLD MAP`
/// title anchor's top-left corner, indexed by dropdown slot (`0` = region, `1` = first
/// [`NavigateToHuntingGround::dropdown_path`] entry, `2` = second). Dropdown boxes render at
/// fixed-width, consecutive slots regardless of how many are actually active - long option text
/// is truncated with `..` rather than growing the box - so these stay valid regardless of which
/// region/options are selected.
const DROPDOWN_SLOT_OFFSETS: [(i32, i32, i32, i32); 3] =
    [(94, 36, 107, 20), (204, 36, 92, 20), (303, 36, 101, 20)];
/// Generous scan height below an opened dropdown box to find its option list - the exact number
/// of options or the list's real height isn't relied upon,
/// [`crate::detect::Detector::detect_world_map_label`] fuzzy-matches whichever text row within
/// this region matches.
const DROPDOWN_OPTIONS_HEIGHT: i32 = 300;
const MAP_CONTENT_OFFSET: (i32, i32, i32, i32) = (6, 73, 630, 465);

/// States of navigating to a daily quest hunting ground via the in-game world map.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Presses the world map key and waits for the world map UI to open.
    OpeningWorldMap(Timeout),
    /// Clicks the minimap's "Click to open the World Map" button and waits for the world map UI
    /// to open.
    ///
    /// Fallback for when [`NavigateToHuntingGround`]'s configured world map key is either not set
    /// or didn't actually open the world map - unlike the key, this button always exists
    /// regardless of user keybind configuration, so it's used to recover rather than aborting
    /// outright.
    ClickingMinimapButton(Timeout),
    /// Clicks the dropdown box at `slot` (see [`DROPDOWN_SLOT_OFFSETS`]) to open its option list.
    /// `Rect` is the world map title anchor.
    OpeningDropdown(Timeout, Rect, usize),
    /// Finds and clicks the option matching `region` (slot `0`) or the corresponding
    /// `dropdown_path` entry (slot `1`, `2`, ...).
    SelectingOption(Timeout, Rect, usize),
    /// Double-clicks the target location's node at its verified `location_point` offset from the
    /// anchor.
    SelectingLocation(Timeout, Rect),
    /// Finds and double-clicks the target sub-location's node label.
    ///
    /// Only reached when double-clicking the location's node landed on an intermediate view
    /// instead of the teleport-confirm popup (e.g. two hunting grounds sharing one node).
    SelectingSubLocation(Timeout, Rect),
    /// Clicks the teleport-confirm popup's `Confirm` button.
    ConfirmingTeleport(Timeout),
    /// Terminal state, whether navigation succeeded or was given up on (e.g. world map key not
    /// set, some element not found). Either way, there is nothing further this state can do.
    ///
    /// Carries whether navigation actually reached its destination - see
    /// [`crate::player::state::PlayerContext::take_daily_quest_navigate_failed`] for why this
    /// can't just be inferred from the priority action clearing.
    Done(bool),
}

#[derive(Debug, Clone)]
pub struct NavigatingToHuntingGround {
    state: State,
    target: NavigateToHuntingGround,
}

impl NavigatingToHuntingGround {
    pub fn new(target: NavigateToHuntingGround) -> Self {
        Self {
            state: State::OpeningWorldMap(Timeout::default()),
            target,
        }
    }

    /// Total number of dropdown slots to click through: region plus every
    /// [`NavigateToHuntingGround::dropdown_path`] entry.
    #[inline]
    fn total_dropdown_slots(&self) -> usize {
        1 + self.target.dropdown_path.len()
    }

    /// The state to enter once the world map is open and `anchor` is known - either the first
    /// dropdown slot not already covered by [`NavigateToHuntingGround::skip_dropdown_slots`], or
    /// straight to selecting the location if every slot is already covered.
    #[inline]
    fn state_after_world_map_open(&self, anchor: Rect) -> State {
        if self.target.skip_dropdown_slots >= self.total_dropdown_slots() {
            State::SelectingLocation(Timeout::default(), anchor)
        } else {
            State::OpeningDropdown(Timeout::default(), anchor, self.target.skip_dropdown_slots)
        }
    }

    /// The text to search for in the dropdown option list at `slot`.
    #[inline]
    fn dropdown_slot_text(&self, slot: usize) -> Option<String> {
        if slot == 0 {
            Some(self.target.region.to_string())
        } else {
            self.target.dropdown_path.get(slot - 1).cloned()
        }
    }
}

/// Updates [`Player::NavigatingToHuntingGround`] contextual state.
pub fn update_navigating_to_hunting_ground_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
) {
    let Player::NavigatingToHuntingGround(mut navigating) = player.state.clone() else {
        panic!("state is not navigating to hunting ground")
    };

    match navigating.state {
        State::OpeningWorldMap(_) => {
            let world_map_key = player.context.config.world_map_key;
            update_opening_world_map(resources, world_map_key, &mut navigating);
        }
        State::ClickingMinimapButton(_) => {
            update_clicking_minimap_button(resources, &mut navigating)
        }
        State::OpeningDropdown(_, anchor, slot) => {
            update_opening_dropdown(resources, &mut navigating, anchor, slot)
        }
        State::SelectingOption(_, anchor, slot) => {
            update_selecting_option(resources, &mut navigating, anchor, slot)
        }
        State::SelectingLocation(_, anchor) => {
            update_selecting_location(resources, &mut navigating, anchor)
        }
        State::SelectingSubLocation(_, anchor) => {
            update_selecting_sub_location(resources, &mut navigating, anchor)
        }
        State::ConfirmingTeleport(_) => update_confirming_teleport(resources, &mut navigating),
        State::Done(_) => (),
    }

    let succeeded = match navigating.state {
        State::Done(succeeded) => Some(succeeded),
        _ => None,
    };
    let player_next_state = if succeeded.is_some() {
        Player::Idle
    } else {
        Player::NavigatingToHuntingGround(navigating)
    };
    let is_terminal = matches!(player_next_state, Player::Idle);
    if succeeded == Some(false) {
        player.context.set_daily_quest_navigate_failed();
    }

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

fn update_opening_world_map(
    resources: &mut Resources,
    world_map_key: Option<KeyKind>,
    navigating: &mut NavigatingToHuntingGround,
) {
    let State::OpeningWorldMap(timeout) = navigating.state else {
        panic!("navigating to hunting ground state is not opening world map")
    };

    let Some(world_map_key) = world_map_key else {
        debug!(
            target: "backend/player",
            "Navigating to hunting ground: world map key is not set, \
             falling back to clicking the minimap button"
        );
        navigating.state = State::ClickingMinimapButton(Timeout::default());
        return;
    };

    match next_timeout_lifecycle(timeout, 35) {
        Lifecycle::Started(timeout) => {
            resources.input.send_key(world_map_key);
            navigating.state = State::OpeningWorldMap(timeout);
        }
        Lifecycle::Ended => {
            let Ok(anchor) = resources.detector().detect_world_map_title() else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: world map did not open after pressing key, \
                     falling back to clicking the minimap button"
                );
                navigating.state = State::ClickingMinimapButton(Timeout::default());
                return;
            };
            navigating.state = navigating.state_after_world_map_open(anchor);
        }
        Lifecycle::Updated(timeout) => navigating.state = State::OpeningWorldMap(timeout),
    }
}

fn update_clicking_minimap_button(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
) {
    let State::ClickingMinimapButton(timeout) = navigating.state else {
        panic!("navigating to hunting ground state is not clicking minimap button")
    };

    match next_timeout_lifecycle(timeout, 35) {
        Lifecycle::Started(timeout) => {
            let Ok(button) = resources.detector().detect_minimap_world_map_button() else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: aborted because minimap world map button was \
                     not found"
                );
                navigating.state = State::Done(false);
                return;
            };
            let (cx, cy) = rect_click_point(button);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::ClickingMinimapButton(timeout);
        }
        Lifecycle::Ended => {
            let Ok(anchor) = resources.detector().detect_world_map_title() else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: world map did not open after clicking the \
                     minimap button"
                );
                navigating.state = State::Done(false);
                return;
            };
            navigating.state = navigating.state_after_world_map_open(anchor);
        }
        Lifecycle::Updated(timeout) => navigating.state = State::ClickingMinimapButton(timeout),
    }
}

fn update_opening_dropdown(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
    anchor: Rect,
    slot: usize,
) {
    let State::OpeningDropdown(timeout, ..) = navigating.state else {
        panic!("navigating to hunting ground state is not opening dropdown")
    };
    let Some(&(x, y, width, height)) = DROPDOWN_SLOT_OFFSETS.get(slot) else {
        debug!(
            target: "backend/player",
            "Navigating to hunting ground: dropdown slot {slot} is not supported"
        );
        abort_and_close_world_map(resources, navigating);
        return;
    };

    match next_timeout_lifecycle(timeout, 5) {
        Lifecycle::Started(timeout) => {
            debug!(target: "backend/player", "Navigating to hunting ground: opening dropdown slot {slot}");
            let box_rect = Rect::new(anchor.x + x, anchor.y + y, width, height);
            let (cx, cy) = rect_click_point(box_rect);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::OpeningDropdown(timeout, anchor, slot);
        }
        Lifecycle::Ended => {
            navigating.state = State::SelectingOption(Timeout::default(), anchor, slot);
        }
        Lifecycle::Updated(timeout) => {
            navigating.state = State::OpeningDropdown(timeout, anchor, slot);
        }
    }
}

fn update_selecting_option(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
    anchor: Rect,
    slot: usize,
) {
    let State::SelectingOption(timeout, ..) = navigating.state else {
        panic!("navigating to hunting ground state is not selecting option")
    };

    match next_timeout_lifecycle(timeout, 6) {
        Lifecycle::Started(timeout) => {
            let Some(text) = navigating.dropdown_slot_text(slot) else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: no dropdown option configured for slot {slot}"
                );
                abort_and_close_world_map(resources, navigating);
                return;
            };
            let (x, y, width, _) = DROPDOWN_SLOT_OFFSETS[slot];
            let options_roi = Rect::new(anchor.x + x, anchor.y + y, width, DROPDOWN_OPTIONS_HEIGHT);

            let Ok(option) = resources
                .detector()
                .detect_world_map_label(options_roi, &text)
            else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: dropdown option `{text}` not found"
                );
                abort_and_close_world_map(resources, navigating);
                return;
            };
            debug!(target: "backend/player", "Navigating to hunting ground: selected `{text}` for slot {slot}");
            let (cx, cy) = rect_click_point(option);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::SelectingOption(timeout, anchor, slot);
        }
        Lifecycle::Ended => {
            debug!(target: "backend/player", "Navigating to hunting ground: done with slot {slot}");
            navigating.state = if slot + 1 < navigating.total_dropdown_slots() {
                State::OpeningDropdown(Timeout::default(), anchor, slot + 1)
            } else {
                State::SelectingLocation(Timeout::default(), anchor)
            };
        }
        Lifecycle::Updated(timeout) => {
            navigating.state = State::SelectingOption(timeout, anchor, slot);
        }
    }
}

fn update_selecting_location(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
    anchor: Rect,
) {
    let State::SelectingLocation(timeout, _) = navigating.state else {
        panic!("navigating to hunting ground state is not selecting location")
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            let (px, py) = navigating.target.location_point;
            let (cx, cy) = (anchor.x + px, anchor.y + py);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::SelectingLocation(timeout, anchor);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_popup_confirm_button().is_ok() {
                navigating.state = State::ConfirmingTeleport(Timeout::default());
            } else if navigating.target.sub_location_label.is_some() {
                navigating.state = State::SelectingSubLocation(Timeout::default(), anchor);
            } else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: no teleport confirm popup after selecting location"
                );
                abort_and_close_world_map(resources, navigating);
            }
        }
        Lifecycle::Updated(timeout) => {
            navigating.state = State::SelectingLocation(timeout, anchor);
        }
    }
}

fn update_selecting_sub_location(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
    anchor: Rect,
) {
    let State::SelectingSubLocation(timeout, _) = navigating.state else {
        panic!("navigating to hunting ground state is not selecting sub location")
    };
    let Some(sub_location_label) = navigating.target.sub_location_label.clone() else {
        abort_and_close_world_map(resources, navigating);
        return;
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            let (x, y, width, height) = MAP_CONTENT_OFFSET;
            let content_roi = Rect::new(anchor.x + x, anchor.y + y, width, height);
            let Ok(node) = resources
                .detector()
                .detect_world_map_label(content_roi, &sub_location_label)
            else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: sub-location `{sub_location_label}` not found"
                );
                abort_and_close_world_map(resources, navigating);
                return;
            };
            let (cx, cy) = rect_click_point(node);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::SelectingSubLocation(timeout, anchor);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_popup_confirm_button().is_ok() {
                navigating.state = State::ConfirmingTeleport(Timeout::default());
            } else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: no teleport confirm popup after selecting sub-location"
                );
                abort_and_close_world_map(resources, navigating);
            }
        }
        Lifecycle::Updated(timeout) => {
            navigating.state = State::SelectingSubLocation(timeout, anchor);
        }
    }
}

fn update_confirming_teleport(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
) {
    let State::ConfirmingTeleport(timeout) = navigating.state else {
        panic!("navigating to hunting ground state is not confirming teleport")
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            let Ok(button) = resources.detector().detect_popup_confirm_button() else {
                debug!(
                    target: "backend/player",
                    "Navigating to hunting ground: Confirm button not found"
                );
                abort_and_close_world_map(resources, navigating);
                return;
            };
            let (cx, cy) = rect_click_point(button);
            resources.input.send_mouse(cx, cy, MouseKind::Click);
            navigating.state = State::ConfirmingTeleport(timeout);
        }
        Lifecycle::Ended => {
            info!(target: "backend/player", "Navigating to hunting ground: teleport confirmed");
            navigating.state = State::Done(true);
        }
        Lifecycle::Updated(timeout) => {
            navigating.state = State::ConfirmingTeleport(timeout);
        }
    }
}

#[inline]
fn rect_click_point(rect: Rect) -> (i32, i32) {
    let x = rect.x + rect.width / 2;
    let y = rect.y + rect.height / 2;
    (x, y)
}

/// Presses Esc to close the world map before giving up on navigation.
///
/// Only called from states that already carry an `anchor` (i.e. after
/// [`crate::detect::Detector::detect_world_map_title`] has confirmed the world map is open) - a
/// failure at that point still leaves the world map open on screen, which keeps covering the
/// minimap and breaking minimap-dependent detection (and anything relying on it, e.g. panic mode)
/// for the rest of the run instead of just aborting this one navigation attempt.
#[inline]
fn abort_and_close_world_map(
    resources: &mut Resources,
    navigating: &mut NavigatingToHuntingGround,
) {
    resources.input.send_key(KeyKind::Esc);
    navigating.state = State::Done(false);
}
