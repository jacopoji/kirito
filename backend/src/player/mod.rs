use actions::next_action;
use adjust::{Adjusting, update_adjusting_state};
use cash_shop::{CashShop, update_cash_shop_state};
use double_jump::{DoubleJumping, update_double_jumping_state};
use fall::update_falling_state;
use familiars_fusion::update_familiars_fusing_state;
use familiars_potential::update_familiars_revealing_potential_state;
use familiars_swap::{FamiliarsSwapping, update_familiars_swapping_state};
use grapple::update_grappling_state;
use idle::update_idle_state;
use jump::update_jumping_state;
use moving::{MOVE_TIMEOUT, Moving, MovingIntermediates, update_moving_state};
use opencv::core::Point;
use panic::update_panicking_state;
use solve_rune::{SolvingRune, update_solving_rune_state};
use stall::update_stalling_state;
use state::LastMovement;
use strum::Display;
use timeout::Timeout;
use unstuck::update_unstucking_state;
use up_jump::{UpJumping, update_up_jumping_state};
use use_key::{UseKey, update_use_key_state};

use crate::{
    bridge::KeyKind,
    buff::BuffEntities,
    ecs::Resources,
    minimap::{Minimap, MinimapEntity},
    models::ActionKeyDirection,
    player::{
        enter_monster_park::{EnteringMonsterPark, update_entering_monster_park_state},
        exchange_booster::{ExchangingBooster, update_exchanging_booster_state},
        exit_monster_park::{ExitingMonsterPark, update_exiting_monster_park_state},
        fall::Falling,
        grapple::Grappling,
        navigate_to_hunting_ground::{
            NavigatingToHuntingGround, update_navigating_to_hunting_ground_state,
        },
        solve_shape::{SolvingShape, update_solving_shape_state},
        solve_violetta::{SolvingVioletta, update_solving_violetta_state},
        thread_of_fate_ask::{ThreadOfFateAsking, update_thread_of_fate_asking_state},
        thread_of_fate_complete::{ThreadOfFateCompleting, update_thread_of_fate_completing_state},
        unstuck::Unstucking,
        use_booster::{UsingBooster, update_using_booster_state},
    },
};

mod actions;
mod adjust;
mod cash_shop;
mod double_jump;
mod enter_monster_park;
mod exchange_booster;
mod exit_monster_park;
mod fall;
mod familiars_fusion;
mod familiars_potential;
mod familiars_swap;
mod grapple;
mod idle;
mod jump;
mod moving;
mod navigate_to_hunting_ground;
mod panic;
mod solve_rune;
mod solve_shape;
mod solve_violetta;
mod stall;
mod state;
mod thread_of_fate_ask;
mod thread_of_fate_complete;
mod timeout;
mod unstuck;
mod up_jump;
mod use_booster;
mod use_key;

pub use actions::*;
pub use {
    double_jump::DOUBLE_JUMP_THRESHOLD, familiars_fusion::FamiliarsFusing,
    familiars_potential::FamiliarsRevealingPotential, grapple::GRAPPLING_MAX_THRESHOLD,
    grapple::GRAPPLING_THRESHOLD, panic::Panicking, state::MonsterParkEntryOutcome,
    state::PlayerContext, state::Quadrant,
};

/// Minimum y distance from the destination required to perform a jump.
pub const JUMP_THRESHOLD: i32 = 7;

#[derive(Debug)]
pub struct PlayerEntity {
    pub state: Player,
    pub context: PlayerContext,
}

/// The player contextual states.
#[derive(Clone, Debug, Display)]
#[allow(clippy::large_enum_variant)] // There is only ever a single instance of Player
pub enum Player {
    /// Detects player on the minimap.
    Detecting,
    /// Does nothing state.
    ///
    /// Acts as entry to other state when there is a [`PlayerAction`].
    Idle,
    /// Uses key.
    UseKey(UseKey),
    /// Movement-related coordinator state.
    Moving(Point, bool, Option<MovingIntermediates>),
    /// Performs walk or small adjustment x-wise action.
    Adjusting(Adjusting),
    /// Performs double jump action.
    DoubleJumping(DoubleJumping),
    /// Performs a grappling action.
    Grappling(Grappling),
    /// Performs a normal jump.
    Jumping(Moving),
    /// Performs an up jump action.
    UpJumping(UpJumping),
    Falling(Falling),
    /// Unstucks when inside non-detecting position or because of [`PlayerState::unstuck_counter`].
    Unstucking(Unstucking),
    /// Stalls for time and return to [`Player::Idle`] or [`PlayerState::stalling_timeout_state`].
    Stalling(Timeout, u32),
    /// Tries to solve a rune.
    SolvingRune(SolvingRune),
    /// Tries to solve lie detector's transparent shape.
    #[strum(to_string = "SolvingShape({0})")]
    SolvingShape(SolvingShape),
    #[strum(to_string = "SolvingVioletta({0})")]
    SolvingVioletta(SolvingVioletta),
    /// Enters the cash shop then exit after 10 seconds.
    CashShopThenExit(CashShop),
    #[strum(to_string = "FamiliarsSwapping({0})")]
    FamiliarsSwapping(FamiliarsSwapping),
    #[strum(to_string = "FamiliarsFusing({0})")]
    FamiliarsFusing(FamiliarsFusing),
    #[strum(to_string = "FamiliarsRevealingPotential({0})")]
    FamiliarsRevealingPotential(FamiliarsRevealingPotential),
    Panicking(Panicking),
    UsingBooster(UsingBooster),
    ExchangingBooster(ExchangingBooster),
    EnteringMonsterPark(EnteringMonsterPark),
    ExitingMonsterPark(ExitingMonsterPark),
    NavigatingToHuntingGround(NavigatingToHuntingGround),
    ThreadOfFateAsking(ThreadOfFateAsking),
    ThreadOfFateCompleting(ThreadOfFateCompleting),
}

impl Player {
    #[inline]
    pub fn can_override_current_state(&self, cur_pos: Option<Point>) -> bool {
        const OVERRIDABLE_DISTANCE: i32 = DOUBLE_JUMP_THRESHOLD / 2;

        match self {
            Player::Detecting
            | Player::Idle
            | Player::ThreadOfFateAsking(_)
            | Player::ThreadOfFateCompleting(_) => true,
            Player::Moving(dest, _, _) => {
                if let Some(pos) = cur_pos {
                    (dest.x - pos.x).abs() >= OVERRIDABLE_DISTANCE
                } else {
                    true
                }
            }
            Player::DoubleJumping(DoubleJumping {
                moving,
                forced: false,
                ..
            })
            | Player::Adjusting(Adjusting { moving, .. }) => {
                let (distance, _) =
                    moving.x_distance_direction_from(true, cur_pos.unwrap_or(moving.pos));
                distance >= OVERRIDABLE_DISTANCE
            }
            Player::Grappling(Grappling { moving, .. })
            | Player::Jumping(moving)
            | Player::UpJumping(UpJumping { moving, .. })
            | Player::Falling(Falling { moving, .. }) => moving.completed,
            Player::SolvingRune(_)
            | Player::CashShopThenExit(_)
            | Player::Unstucking(_)
            | Player::DoubleJumping(DoubleJumping { forced: true, .. })
            | Player::UseKey(_)
            | Player::FamiliarsSwapping(_)
            | Player::FamiliarsFusing(_)
            | Player::FamiliarsRevealingPotential(_)
            | Player::Panicking(_)
            | Player::UsingBooster(_)
            | Player::ExchangingBooster(_)
            | Player::EnteringMonsterPark(_)
            | Player::ExitingMonsterPark(_)
            | Player::NavigatingToHuntingGround(_)
            | Player::SolvingShape(_)
            | Player::SolvingVioletta(_)
            | Player::Stalling(_, _) => false,
        }
    }
}

/// Releases any key `state` may currently be holding down (`send_key_down` without a matching
/// `send_key_up` yet), for when `state` is about to be discarded outside its own normal
/// completion path.
///
/// A handful of states hold a key down across several ticks instead of a single press-and-release
/// - `Adjusting`/`DoubleJumping`/`Stalling`/`Unstucking` hold Left or Right, `UpJumping` holds Up,
///   and `UseKey` holds its main key and/or link key (commonly a modifier, for combo skills) when
///   `key_hold_ticks > 0` or `link_key` is [`crate::bridge::LinkKeyKind::Along`]. All of them only
///   release what they're holding once their own state naturally completes - `run_system` has a
///   few places that instead overwrite `player.state` directly (player detection failing
///   entirely, or a pending reset to `Idle`), bypassing that cleanup and leaving the key held at
///   the OS level indefinitely. If that's a modifier key, anything else that later sends an
///   unrelated key (e.g. `Unstucking`'s own Esc-dismiss) can be misread by Windows or the game as
///   a shortcut combo - this is the direct fix for exactly that class of bug.
///
/// Not gated on whether a hold is actually in progress right now for the direction/Up-holding
/// states - releasing a key that isn't actually down is a harmless no-op at the input layer
/// (`bridge`'s `send_key_up` is a plain, unconditional key-up simulation), so it's simpler and
/// safer to release unconditionally than to track each state's precise internal timing here too.
fn release_keys_held_by(resources: &mut Resources, state: &Player) {
    match state {
        Player::Adjusting(_)
        | Player::DoubleJumping(_)
        | Player::Stalling(_, _)
        | Player::Unstucking(_) => {
            resources.input.send_key_up(KeyKind::Left);
            resources.input.send_key_up(KeyKind::Right);
        }
        Player::UpJumping(_) => {
            resources.input.send_key_up(KeyKind::Up);
        }
        Player::UseKey(use_key) => {
            for key in use_key.held_keys() {
                resources.input.send_key_up(key);
            }
        }
        _ => {}
    }
}

pub fn run_system(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap: &MinimapEntity,
    buffs: &BuffEntities,
) {
    if player.context.rune_cash_shop {
        resources.input.send_key_up(KeyKind::Up);
        resources.input.send_key_up(KeyKind::Down);
        resources.input.send_key_up(KeyKind::Left);
        resources.input.send_key_up(KeyKind::Right);
        player.context.rune_cash_shop = false;
        player.context.reset_to_idle_next_update = false;
        player.state = Player::CashShopThenExit(CashShop::new());
        return;
    }

    let did_update =
        player
            .context
            .update_state(resources, player.state.clone(), minimap.state, buffs);
    if !did_update && !resources.operation.halting() {
        // When the player detection fails, the possible causes are:
        // - Player moved inside the edges of the minimap
        // - Other UIs overlapping the minimap
        //
        // `update_non_positional_context` is here to continue updating
        // `Player::Unstucking` returned from below when the player
        // is inside the edges of the minimap. And also `Player::CashShopThenExit`.
        if update_non_positional_state(resources, player, minimap.state, true) {
            return;
        }

        let is_stucking = match minimap.state {
            Minimap::Detecting => false,
            Minimap::Idle(idle) => !idle.partially_overlapping,
        };
        if is_stucking {
            release_keys_held_by(resources, &player.state);
            let random = player.context.track_unstucking_transitioned();
            let blink = random && player.context.track_unstucking_gamba();
            let unstucking = Unstucking::new_movement(Timeout::default(), random, blink);
            player.state = Player::Unstucking(unstucking);
            player.context.last_known_direction = ActionKeyDirection::Any;
            return;
        }

        release_keys_held_by(resources, &player.state);
        player.state = Player::Detecting;
        return;
    }

    if player.context.reset_to_idle_next_update {
        release_keys_held_by(resources, &player.state);
        player.context.reset_to_idle_next_update = false;
        player.state = Player::Idle;
    }
    if player.context.reset_stalling_buffer_states_next_update {
        player.context.reset_stalling_buffer_states_next_update = false;
        player.context.clear_stalling_buffer_states(resources);
    }

    if !update_non_positional_state(resources, player, minimap.state, false) {
        update_positional_state(resources, player, minimap.state);
    }
}

/// Updates the contextual state that does not require the player current position.
///
/// Returns `true` if state is updated.
#[inline]
fn update_non_positional_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
    failed_to_detect_player: bool,
) -> bool {
    match player.state {
        Player::UseKey(_) => update_use_key_state(resources, player, minimap_state),
        Player::FamiliarsSwapping(_) => {
            update_familiars_swapping_state(resources, player);
        }
        Player::FamiliarsFusing(_) => {
            update_familiars_fusing_state(resources, player);
        }
        Player::FamiliarsRevealingPotential(_) => {
            update_familiars_revealing_potential_state(resources, player);
        }
        Player::Unstucking(_) => {
            update_unstucking_state(resources, player, minimap_state);
        }
        Player::Stalling(timeout, max_timeout) => {
            if failed_to_detect_player {
                return false;
            }

            update_stalling_state(player, timeout, max_timeout);
        }
        Player::SolvingRune(_) => {
            if failed_to_detect_player {
                return false;
            }

            update_solving_rune_state(resources, player);
        }
        Player::SolvingShape(_) => update_solving_shape_state(resources, player),
        Player::SolvingVioletta(_) => update_solving_violetta_state(resources, player),
        Player::CashShopThenExit(cash_shop) => {
            update_cash_shop_state(resources, player, cash_shop, failed_to_detect_player);
        }
        Player::Panicking(panicking) => {
            update_panicking_state(resources, player, minimap_state, panicking);
        }
        Player::UsingBooster(_) => update_using_booster_state(resources, player),
        Player::ExchangingBooster(_) => update_exchanging_booster_state(resources, player),
        Player::EnteringMonsterPark(_) => update_entering_monster_park_state(resources, player),
        Player::ExitingMonsterPark(_) => update_exiting_monster_park_state(resources, player),
        Player::NavigatingToHuntingGround(_) => {
            update_navigating_to_hunting_ground_state(resources, player, minimap_state)
        }
        Player::ThreadOfFateAsking(_) => {
            update_thread_of_fate_asking_state(resources, player);
        }
        Player::ThreadOfFateCompleting(_) => {
            update_thread_of_fate_completing_state(resources, player);
        }
        Player::Detecting
        | Player::Idle
        | Player::Moving(_, _, _)
        | Player::Adjusting(_)
        | Player::DoubleJumping(_)
        | Player::Grappling(_)
        | Player::Jumping(_)
        | Player::UpJumping(_)
        | Player::Falling(_) => return false,
    }

    true
}

/// Updates the contextual state that requires the player current position.
#[inline]
fn update_positional_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
) {
    match player.state {
        Player::Detecting => player.state = Player::Idle,
        Player::Idle => update_idle_state(resources, player, minimap_state),
        Player::Moving(_, _, _) => update_moving_state(resources, player, minimap_state),
        Player::Adjusting(_) => update_adjusting_state(resources, player, minimap_state),
        Player::DoubleJumping(_) => update_double_jumping_state(resources, player, minimap_state),
        Player::Grappling(_) => update_grappling_state(resources, player, minimap_state),
        Player::UpJumping(_) => update_up_jumping_state(resources, player, minimap_state),
        Player::Jumping(moving) => update_jumping_state(resources, player, moving),
        Player::Falling(Falling { .. }) => update_falling_state(resources, player, minimap_state),
        Player::UseKey(_)
        | Player::Unstucking(_)
        | Player::Stalling(_, _)
        | Player::SolvingRune(_)
        | Player::FamiliarsSwapping(_)
        | Player::FamiliarsFusing(_)
        | Player::FamiliarsRevealingPotential(_)
        | Player::Panicking(_)
        | Player::UsingBooster(_)
        | Player::ExchangingBooster(_)
        | Player::EnteringMonsterPark(_)
        | Player::ExitingMonsterPark(_)
        | Player::NavigatingToHuntingGround(_)
        | Player::ThreadOfFateAsking(_)
        | Player::ThreadOfFateCompleting(_)
        | Player::SolvingShape(_)
        | Player::SolvingVioletta(_)
        | Player::CashShopThenExit(_) => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use mockall::predicate::eq;

    use super::*;
    use crate::{
        bridge::{LinkKeyKind, MockInput},
        models::{ActionKeyDirection, ActionKeyWith, WaitAfterBuffered},
        player::Key,
    };

    #[test]
    fn release_keys_held_by_direction_holding_states_releases_left_and_right() {
        for state in [
            Player::Adjusting(Adjusting::new(Moving::new(
                Point::new(0, 0),
                Point::new(0, 0),
                false,
                None,
            ))),
            Player::Stalling(Timeout::default(), 0),
            Player::Unstucking(Unstucking::new_esc()),
        ] {
            let mut keys = MockInput::new();
            keys.expect_send_key_up().with(eq(KeyKind::Left)).once();
            keys.expect_send_key_up().with(eq(KeyKind::Right)).once();
            let mut resources = Resources::new(Some(keys), None);

            release_keys_held_by(&mut resources, &state);
        }
    }

    #[test]
    fn release_keys_held_by_up_jumping_releases_up() {
        let moving = Moving::new(Point::new(0, 0), Point::new(0, 20), false, None);
        let mut new_resources = Resources::new(None, None);
        let up_jumping = UpJumping::new(moving, &mut new_resources, &PlayerContext::default());

        let mut keys = MockInput::new();
        keys.expect_send_key_up().with(eq(KeyKind::Up)).once();
        let mut resources = Resources::new(Some(keys), None);

        release_keys_held_by(&mut resources, &Player::UpJumping(up_jumping));
    }

    #[test]
    fn release_keys_held_by_use_key_with_along_link_key_releases_both_keys() {
        let use_key = UseKey::from_key(Key {
            key: KeyKind::A,
            key_hold_ticks: 0,
            key_hold_buffered_to_wait_after: false,
            link_key: LinkKeyKind::Along(KeyKind::Alt),
            count: 1,
            position: None,
            direction: ActionKeyDirection::Any,
            with: ActionKeyWith::Any,
            wait_before_use_ticks: 0,
            wait_before_use_ticks_random_range: 0,
            wait_after_use_ticks: 0,
            wait_after_use_ticks_random_range: 0,
            wait_after_buffered: WaitAfterBuffered::None,
        });

        let mut keys = MockInput::new();
        keys.expect_send_key_up().with(eq(KeyKind::A)).once();
        keys.expect_send_key_up().with(eq(KeyKind::Alt)).once();
        let mut resources = Resources::new(Some(keys), None);

        release_keys_held_by(&mut resources, &Player::UseKey(use_key));
    }

    #[test]
    fn release_keys_held_by_idle_does_nothing() {
        let mut keys = MockInput::new();
        keys.expect_send_key_up().never();
        let mut resources = Resources::new(Some(keys), None);

        release_keys_held_by(&mut resources, &Player::Idle);
    }
}
