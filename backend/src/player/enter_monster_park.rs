use log::debug;
use opencv::core::{MatTraitConst, Point, Rect};

use super::{Player, timeout::Timeout};
use crate::{
    bridge::{KeyKind, MouseKind},
    ecs::Resources,
    player::{
        PlayerEntity, next_action,
        timeout::{Lifecycle, next_timeout_lifecycle},
    },
};

/// Column x-positions and last-row y-offset for the Monster Park dungeon-select dialog's tile
/// grid, relative to the detected ticket label's top-left corner. Measured empirically from the
/// dialog's fixed layout.
const TILE_COL_X_OFFSETS: [i32; 2] = [122, 337];
const LAST_ROW_Y_OFFSET: i32 = 323;
/// Row spacing between consecutive grid rows.
const ROW_SPACING: i32 = 65;
/// Half-width/height of the region checked around each grid slot's expected center for the
/// empty-slot placeholder - generous enough to comfortably tolerate the same on-screen drift the
/// fixed offsets above already allow for, while still being small enough that nothing else in
/// the dialog plausibly falls within it.
const TILE_CHECK_HALF_SIZE: i32 = 40;
/// Number of columns in the dungeon-select dialog's tile grid.
const GRID_COLUMNS: i32 = 2;
/// Total number of Monster Park dungeon entries currently in the game (Chu Chu Island,
/// Lachelein, Mysterious Forest of Arcana, Morass, Esfera, Sellas, Moonbridge, Labyrinth of
/// Suffering, Limina) - an odd count in [`GRID_COLUMNS`], so the last row has only one tile (left
/// column), with the right column left empty. This is fixed game content, not tied to the
/// character in any way - update it if a future game update changes the dungeon list.
const TOTAL_DUNGEON_COUNT: i32 = 9;
/// Total grid slots, including the grid's own permanently-empty filler slot when
/// [`TOTAL_DUNGEON_COUNT`] doesn't evenly fill whole rows (currently: it doesn't, so this is one
/// more than [`TOTAL_DUNGEON_COUNT`]). That filler slot always reads as the empty-slot
/// placeholder regardless of anything else, which is exactly what makes the scan in
/// [`last_active_dungeon_index`] always terminate - if nothing else does, that guaranteed final
/// slot will.
const TOTAL_GRID_SLOTS: i32 =
    TOTAL_DUNGEON_COUNT + (GRID_COLUMNS - TOTAL_DUNGEON_COUNT % GRID_COLUMNS) % GRID_COLUMNS;

/// Computes the on-screen center of dungeon-select grid slot `index` (0-based reading order:
/// top-to-bottom, left-to-right, including the permanently-empty filler slot at
/// [`TOTAL_DUNGEON_COUNT`] if `index` reaches it), from the ticket label's detected position.
fn dungeon_tile_point(ticket_label: Rect, index: i32) -> Point {
    let anchor = ticket_label.tl();
    let last_row_index = (TOTAL_DUNGEON_COUNT - 1) / GRID_COLUMNS;
    let row = index / GRID_COLUMNS;
    let col = index % GRID_COLUMNS;
    let x = anchor.x + TILE_COL_X_OFFSETS[col as usize];
    let y = anchor.y + LAST_ROW_Y_OFFSET + (row - last_row_index) * ROW_SPACING;
    Point::new(x, y)
}

/// Computes the on-screen region checked for the empty-slot placeholder at grid slot `index`.
fn dungeon_tile_region(ticket_label: Rect, index: i32) -> Rect {
    let center = dungeon_tile_point(ticket_label, index);
    Rect::new(
        center.x - TILE_CHECK_HALF_SIZE,
        center.y - TILE_CHECK_HALF_SIZE,
        TILE_CHECK_HALF_SIZE * 2,
        TILE_CHECK_HALF_SIZE * 2,
    )
}

/// Scans grid slots `0..total_slots` in reading order and returns the index of the last
/// available dungeon: the slot immediately before the first one `is_locked` reports as showing
/// the empty-slot placeholder.
///
/// Checking one known slot position at a time - rather than searching the whole dialog (or worse,
/// the whole captured frame) and collecting every match - avoids two problems a real run hit:
/// matches against unrelated UI elements far outside the dialog entirely, and several overlapping
/// duplicate matches for a single true tile that were awkward to reliably tell apart from
/// genuinely distinct ones. A tightly-scoped, single-tile-sized region has no room for either.
///
/// Falls back to [`TOTAL_DUNGEON_COUNT`]` - 1` (the last real dungeon) if `is_locked` never
/// reports true for any slot - this shouldn't happen in practice since
/// [`TOTAL_GRID_SLOTS`] always includes at least the grid's own permanently-empty filler slot,
/// which reads as the placeholder unconditionally.
fn last_active_dungeon_index(total_slots: i32, mut is_locked: impl FnMut(i32) -> bool) -> i32 {
    for index in 0..total_slots {
        if is_locked(index) {
            return index - 1;
        }
    }
    TOTAL_DUNGEON_COUNT - 1
}

/// States of entering a Monster Park run from the entry lobby's gate.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Presses Up to open the dungeon-select dialog.
    PressingUp(Timeout),
    /// Verifies a free entry is available, then clicks the highest-level unlocked dungeon tile.
    ///
    /// The `Rect` is the ticket label's detected position, used as the anchor for locating the
    /// dungeon tile grid.
    SelectingDungeon(Timeout, Rect),
    /// Clicks the `Enter` button.
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
        State::SelectingDungeon(_, _) => update_selecting_dungeon(resources, &mut entering),
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
            // is unreliable to check upfront - just attempt the dungeon selection and Enter click,
            // and treat a still-open dialog afterwards as "no free entry" instead (see
            // `update_confirming`).
            let Ok(ticket_label) = resources.detector().detect_monster_park_ticket_label() else {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: dungeon-select dialog did not appear after pressing Up"
                );
                entering.state = State::Done;
                return;
            };

            entering.state = State::SelectingDungeon(Timeout::default(), ticket_label);
        }
        Lifecycle::Updated(timeout) => entering.state = State::PressingUp(timeout),
    }
}

fn update_selecting_dungeon(resources: &mut Resources, entering: &mut EnteringMonsterPark) {
    let State::SelectingDungeon(timeout, ticket_label) = entering.state else {
        panic!("entering monster park state is not selecting dungeon")
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) => {
            let last_active_index = last_active_dungeon_index(TOTAL_GRID_SLOTS, |index| {
                let tile = dungeon_tile_region(ticket_label, index);
                resources
                    .detector()
                    .detect_monster_park_dungeon_tile_locked(tile)
            });
            let point = dungeon_tile_point(ticket_label, last_active_index.max(0));
            debug!(
                target: "backend/player",
                "Entering Monster Park: ticket label {ticket_label:?}, last active dungeon index \
                 {last_active_index}, computed dungeon tile at {point:?}"
            );

            // The point above is computed from fixed offsets, not detected directly, so it isn't
            // guaranteed to land inside the captured frame. Clicking outside the game window's
            // client area moves the real OS cursor there and can defocus the window entirely, so
            // bail out instead of clicking blind if that would happen.
            let frame_size = resources.detector().mat().size().unwrap();
            if point.x < 0
                || point.y < 0
                || point.x >= frame_size.width
                || point.y >= frame_size.height
            {
                debug!(
                    target: "backend/player",
                    "Entering Monster Park: computed dungeon tile point {point:?} is outside \
                     the captured frame {frame_size:?}, aborting instead of risking a click \
                     outside the game window"
                );
                entering.state = State::Done;
                return;
            }

            resources
                .input
                .send_mouse(point.x, point.y, MouseKind::Click);
            entering.state = State::SelectingDungeon(timeout, ticket_label);
        }
        Lifecycle::Ended => {
            entering.state = State::Confirming(Timeout::default());
        }
        Lifecycle::Updated(timeout) => {
            entering.state = State::SelectingDungeon(timeout, ticket_label);
        }
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
            if resources
                .detector()
                .detect_monster_park_ticket_label()
                .is_ok()
            {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_locked_stops_at_the_guaranteed_filler_slot() {
        // No slot ever reports locked except the grid's own permanently-empty filler slot at the
        // very end - every real dungeon (0..=8) is available, so the answer is the last one,
        // index 8 (Limina).
        let index = last_active_dungeon_index(TOTAL_GRID_SLOTS, |i| i == TOTAL_DUNGEON_COUNT);
        assert_eq!(index, TOTAL_DUNGEON_COUNT - 1);
    }

    #[test]
    fn one_real_dungeon_unavailable() {
        // Index 8 (Limina) itself reports locked - the last available is the one right before
        // it, index 7 (Labyrinth of Suffering).
        let index = last_active_dungeon_index(TOTAL_GRID_SLOTS, |i| i == TOTAL_DUNGEON_COUNT - 1);
        assert_eq!(index, TOTAL_DUNGEON_COUNT - 2);
    }

    #[test]
    fn two_trailing_dungeons_unavailable() {
        // Indices 7 and 8 (Labyrinth of Suffering and Limina) report locked - the scan stops at
        // the first one it encounters (7), so the last available is index 6 (Moonbridge).
        let index = last_active_dungeon_index(TOTAL_GRID_SLOTS, |i| i >= 7);
        assert_eq!(index, 6);
    }

    #[test]
    fn never_locked_falls_back_to_the_last_real_dungeon() {
        // Defensive case: if `is_locked` never reports true at all (detection missed even the
        // guaranteed filler slot), still default to the last real dungeon rather than an
        // unbounded/invalid index.
        let index = last_active_dungeon_index(TOTAL_GRID_SLOTS, |_| false);
        assert_eq!(index, TOTAL_DUNGEON_COUNT - 1);
    }

    #[test]
    fn dungeon_tile_point_places_known_slots_correctly() {
        let ticket_label = Rect::new(500, 300, 200, 20);
        let anchor = ticket_label.tl();

        // Index 0 (Chu Chu Island): first row, left column.
        let first = dungeon_tile_point(ticket_label, 0);
        assert_eq!(first.x, anchor.x + TILE_COL_X_OFFSETS[0]);
        assert_eq!(first.y, anchor.y + LAST_ROW_Y_OFFSET - 4 * ROW_SPACING);

        // Index 8 (Limina): last row, left column.
        let last = dungeon_tile_point(ticket_label, TOTAL_DUNGEON_COUNT - 1);
        assert_eq!(last.x, anchor.x + TILE_COL_X_OFFSETS[0]);
        assert_eq!(last.y, anchor.y + LAST_ROW_Y_OFFSET);

        // Index 9: the permanently-empty filler slot, last row, right column.
        let filler = dungeon_tile_point(ticket_label, TOTAL_DUNGEON_COUNT);
        assert_eq!(filler.x, anchor.x + TILE_COL_X_OFFSETS[1]);
        assert_eq!(filler.y, anchor.y + LAST_ROW_Y_OFFSET);
    }
}
