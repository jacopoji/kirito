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

/// Fallback column x-positions and last-row y-offset for the Monster Park dungeon-select
/// dialog's tile grid, relative to the detected ticket label's top-left corner. Used only when
/// there is no detected locked tile at all to calibrate the grid from (e.g. every dungeon
/// happens to be unlocked). Measured empirically from the dialog's fixed layout.
const FALLBACK_TILE_COL_X_OFFSETS: [i32; 2] = [122, 337];
const FALLBACK_LAST_ROW_Y_OFFSET: i32 = 323;
/// Fallback row spacing, used only when fewer than two distinct locked-tile rows were detected
/// to calibrate spacing from directly.
const FALLBACK_ROW_SPACING: i32 = 65;

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
            let locked_tiles = resources
                .detector()
                .detect_monster_park_locked_dungeon_tiles();
            let point = last_active_dungeon_tile(&locked_tiles, ticket_label);

            // The point above can be calibrated from fallback offsets instead of an actually
            // detected tile (see `last_active_dungeon_tile`), so it isn't guaranteed to land
            // inside the captured frame. Clicking outside the game window's client area moves
            // the real OS cursor there and can defocus the window entirely, so bail out instead
            // of clicking blind if that would happen.
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

/// Finds the center of the last unlocked dungeon tile in reading order (top-to-bottom,
/// left-to-right).
///
/// Monster Park's dungeon tiles are always locked in a contiguous block at the end of the grid,
/// so the last active tile is simply the one immediately before the first locked tile. The
/// grid's column x-positions and row spacing are calibrated from the actually detected locked
/// tiles themselves rather than fixed offsets from the ticket label - a fixed offset just one
/// row off from the dialog's real on-screen position lands squarely on a locked tile instead of
/// the intended active one.
fn last_active_dungeon_tile(locked_tiles: &[Rect], ticket_label: Rect) -> Point {
    const CLUSTER_TOLERANCE: i32 = 10;

    let anchor = ticket_label.tl();
    if locked_tiles.is_empty() {
        // Nothing to calibrate from at all - fall back to the dialog's known reference layout
        // and assume the very last grid slot.
        return Point::new(
            anchor.x + FALLBACK_TILE_COL_X_OFFSETS[1],
            anchor.y + FALLBACK_LAST_ROW_Y_OFFSET,
        );
    }

    let centers: Vec<Point> = locked_tiles
        .iter()
        .map(|tile| Point::new(tile.x + tile.width / 2, tile.y + tile.height / 2))
        .collect();

    // Sorting alone can't tell which side a lone detected column is on - e.g. a single locked
    // tile sitting in the right column would otherwise get mislabeled as `left_x`, making the
    // "is left column also locked" check below trivially true and skipping an extra row. Use the
    // known reference columns purely to classify sides, keeping the actually-detected x for
    // whichever side was observed.
    let ref_left_x = anchor.x + FALLBACK_TILE_COL_X_OFFSETS[0];
    let ref_right_x = anchor.x + FALLBACK_TILE_COL_X_OFFSETS[1];

    let mut col_xs: Vec<i32> = centers.iter().map(|point| point.x).collect();
    col_xs.sort_unstable();
    col_xs.dedup_by(|a, b| (*a - *b).abs() < CLUSTER_TOLERANCE);
    let (left_x, right_x) = match col_xs.as_slice() {
        [single] => {
            if (single - ref_left_x).abs() <= (single - ref_right_x).abs() {
                (*single, ref_right_x)
            } else {
                (ref_left_x, *single)
            }
        }
        [first, second, ..] => (*first, *second),
        [] => (ref_left_x, ref_right_x),
    };

    let mut row_ys: Vec<i32> = centers.iter().map(|point| point.y).collect();
    row_ys.sort_unstable();
    row_ys.dedup_by(|a, b| (*a - *b).abs() < CLUSTER_TOLERANCE);
    let first_locked_row_y = row_ys[0];
    let row_spacing = row_ys
        .get(1)
        .map(|&y| y - first_locked_row_y)
        .unwrap_or(FALLBACK_ROW_SPACING);

    let left_locked_in_first_row = centers.iter().any(|point| {
        (point.y - first_locked_row_y).abs() < CLUSTER_TOLERANCE
            && (point.x - left_x).abs() < CLUSTER_TOLERANCE
    });

    if !left_locked_in_first_row {
        // Left column of the first locked row is still active - it's the last active tile.
        Point::new(left_x, first_locked_row_y)
    } else {
        // Both columns of the first locked row are locked - the last active tile is the right
        // column of the row above.
        Point::new(right_x, first_locked_row_y - row_spacing)
    }
}
