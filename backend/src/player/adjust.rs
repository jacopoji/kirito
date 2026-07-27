use std::cmp::Ordering;

use log::debug;

use super::{
    Key, PlayerAction,
    moving::Moving,
    timeout::{Lifecycle, next_timeout_lifecycle},
    use_key::UseKey,
};
use crate::{
    ActionKeyDirection, ActionKeyWith,
    bridge::{InputKeyOptions, KeyKind},
    ecs::Resources,
    minimap::Minimap,
    player::{
        Player, PlayerEntity,
        actions::{MID_AIR_ATTACK_MIN_TICKS, update_from_auto_mob_action},
        double_jump::DoubleJumping,
        moving::MOVE_TIMEOUT,
        next_action,
        state::LastMovement,
        timeout::{ChangeAxis, MovingLifecycle, Timeout, next_moving_lifecycle_with_axis},
    },
};

/// Minimum x distance from the destination required to perform small movement.
pub const ADJUSTING_SHORT_THRESHOLD: i32 = 1;

/// Minimum x distance from the destination required to walk.
pub const ADJUSTING_MEDIUM_THRESHOLD: i32 = 3;

/// Same role as [`ADJUSTING_MEDIUM_THRESHOLD`], but used instead of it whenever [`Moving::exact`]
/// is true - the distance at which continuous-hold "medium adjust" walking gives way to bounded,
/// tap-and-recheck "short adjust" movement.
///
/// Continuous-hold walking releases the direction key based on a position check made once per
/// tick, but that release can only take effect starting the *next* tick - by which point the
/// character has already moved through however far it travels in one tick's worth of continuous
/// walking. If that per-tick distance exceeds how far `x_distance` still had left to close,
/// continuous-hold walking necessarily overshoots the destination, sometimes bouncing back and
/// forth a couple of times before finally landing inside `ADJUSTING_SHORT_THRESHOLD`'s range.
/// `ADJUSTING_MEDIUM_THRESHOLD` (3px) is well inside that per-tick travel distance, so exact
/// destinations were paying for this overshoot-then-correct dance on every approach. Switching to
/// bounded, fixed-duration taps at a wider distance instead gives the approach room to decelerate
/// before overshooting becomes physically possible within a single tick, rather than correcting
/// for it afterward.
///
/// Only actually exercised by Monster Park's portal/gate movement today - `exact: true` is not
/// currently set anywhere else in the codebase - so this can be tuned independently of
/// `ADJUSTING_MEDIUM_THRESHOLD` without affecting any other movement. The value is an empirical
/// starting point; adjust based on in-game observation if overshoot still happens (raise it) or
/// the approach feels unnecessarily slow (lower it).
const ADJUSTING_EXACT_SWITCH_THRESHOLD: i32 = 8;

const ADJUSTING_SHORT_TIMEOUT: u32 = MOVE_TIMEOUT + 3;

/// Number of short-adjust tap cycles to attempt reaching exact (0px) alignment before giving up
/// and completing anyway.
///
/// A single 80ms tap's effect on position isn't guaranteed to be finer than
/// [`ADJUSTING_SHORT_THRESHOLD`] - if it consistently overshoots past the target by more than
/// that, `x_distance` can oscillate around 0 without ever landing on it exactly, and since this
/// path already exempts itself from the normal move timeout (see `adjusting_started` below), that
/// oscillation had no bound and could continue indefinitely. Giving up after a bounded number of
/// cycles hands control back to the caller (e.g. the rotator), which re-issues a fresh approach
/// rather than retrying the same tap in place forever.
const MAX_SHORT_ADJUST_ATTEMPTS: u32 = 8;

#[derive(Clone, Copy, Debug)]
pub struct Adjusting {
    pub moving: Moving,
    adjust_timeout: Timeout,
    short_adjust_attempts: u32,
}

impl Adjusting {
    pub fn new(moving: Moving) -> Self {
        Self {
            moving,
            adjust_timeout: Timeout::default(),
            short_adjust_attempts: 0,
        }
    }

    fn moving(self, moving: Moving) -> Adjusting {
        Adjusting { moving, ..self }
    }

    fn update_adjusting(&mut self, resources: &mut Resources, key: Option<KeyKind>) {
        self.adjust_timeout =
            match next_timeout_lifecycle(self.adjust_timeout, ADJUSTING_SHORT_TIMEOUT) {
                Lifecycle::Started(timeout) => {
                    resources.input.send_key_up(KeyKind::Left);
                    resources.input.send_key_up(KeyKind::Right);
                    if let Some(key) = key {
                        resources
                            .input
                            .send_key_with_options(key, InputKeyOptions::default().down_ms(80));
                    }
                    timeout
                }
                Lifecycle::Ended => {
                    self.short_adjust_attempts = self.short_adjust_attempts.saturating_add(1);
                    Timeout::default()
                }
                Lifecycle::Updated(timeout) => timeout,
            };
    }
}

/// Updates the [`Player::Adjusting`] contextual state.
///
/// This state just walks towards the destination. If [`Moving::exact`] is true,
/// then it will perform small movement to ensure the `x` is as close as possible.
pub fn update_adjusting_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
) {
    let Player::Adjusting(adjusting) = player.state else {
        panic!("state is not adjusting")
    };
    let context = &mut player.context;
    let cur_pos = context.last_known_pos.expect("in positional state");

    let moving = adjusting.moving;
    let is_intermediate = moving.is_destination_intermediate();

    match next_moving_lifecycle_with_axis(moving, cur_pos, MOVE_TIMEOUT, ChangeAxis::Both) {
        MovingLifecycle::Started(moving) => {
            context.last_movement = Some(LastMovement::Adjusting);
            player.state = Player::Adjusting(adjusting.moving(moving));
        }
        MovingLifecycle::Ended(moving) => {
            resources.input.send_key_up(KeyKind::Right);
            resources.input.send_key_up(KeyKind::Left);
            player.state = Player::Moving(moving.dest, moving.exact, moving.intermediates);
        }
        MovingLifecycle::Updated(mut moving) => {
            let mut adjusting = adjusting;
            let threshold = context.double_jump_threshold(is_intermediate);
            let (x_distance, x_direction) = moving.x_distance_direction_from(true, moving.pos);

            if !context.config.disable_double_jumping && x_distance >= threshold {
                player.state = Player::Moving(moving.dest, moving.exact, moving.intermediates);
                return;
            }

            // Movement logics
            if !moving.completed {
                let adjusting_started = adjusting.adjust_timeout.started;
                if adjusting_started {
                    // Do not allow timing out if adjusting is in-progress
                    moving.timeout.current = moving.timeout.current.saturating_sub(1);
                }

                // Exact destinations switch to bounded tap correction earlier - see
                // `ADJUSTING_EXACT_SWITCH_THRESHOLD`'s docs for why.
                let medium_threshold = if moving.exact {
                    ADJUSTING_EXACT_SWITCH_THRESHOLD
                } else {
                    ADJUSTING_MEDIUM_THRESHOLD
                };
                let should_adjust_medium = !adjusting_started && x_distance >= medium_threshold;
                let should_adjust_short =
                    adjusting_started || (moving.exact && x_distance >= ADJUSTING_SHORT_THRESHOLD);
                debug!(
                    target: "backend/player",
                    "adjusting pos={:?} dest={:?} x_distance={x_distance} x_direction={x_direction} \
                     should_adjust_medium={should_adjust_medium} should_adjust_short={should_adjust_short}",
                    moving.pos, moving.dest
                );
                let direction = match x_direction.cmp(&0) {
                    Ordering::Greater => {
                        Some((KeyKind::Right, KeyKind::Left, ActionKeyDirection::Right))
                    }
                    Ordering::Less => {
                        Some((KeyKind::Left, KeyKind::Right, ActionKeyDirection::Left))
                    }
                    _ => None,
                };

                match (should_adjust_medium, should_adjust_short, direction) {
                    (true, _, Some((down_key, up_key, dir))) => {
                        resources.input.send_key_up(up_key);
                        resources.input.send_key_down(down_key);
                        context.last_known_direction = dir;
                    }
                    (false, true, Some((down_key, _, dir))) => {
                        adjusting.update_adjusting(resources, Some(down_key));
                        context.last_known_direction = dir;
                    }
                    _ => {
                        if adjusting_started {
                            adjusting.update_adjusting(resources, None);
                        } else {
                            resources.input.send_key_up(KeyKind::Left);
                            resources.input.send_key_up(KeyKind::Right);
                            moving = moving.completed(true);
                        }
                    }
                }
            }

            // Computes and sets initial next state first
            let next_moving = if !moving.completed {
                moving
            } else if moving.exact
                && x_distance >= ADJUSTING_SHORT_THRESHOLD
                && adjusting.short_adjust_attempts < MAX_SHORT_ADJUST_ATTEMPTS
            {
                // Exact adjusting incomplete
                moving.completed(false).timeout_current(0)
            } else {
                moving.timeout_current(MOVE_TIMEOUT)
            };
            player.state = Player::Adjusting(adjusting.moving(next_moving));

            update_from_action(resources, player, minimap_state, moving);
        }
    }
}

fn update_from_action(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
    moving: Moving,
) {
    const USE_KEY_Y_THRESHOLD: i32 = 2;

    let cur_pos = moving.pos;
    let context = &player.context;
    let (x_distance, x_direction) = moving.x_distance_direction_from(false, cur_pos);
    let (y_distance, y_direction) = moving.y_distance_direction_from(false, cur_pos);

    match next_action(context) {
        Some(PlayerAction::Key(
            key @ Key {
                with: ActionKeyWith::DoubleJump,
                direction,
                ..
            },
        )) => {
            if !moving.completed || y_distance > 0 {
                return;
            }

            player.state = if matches!(direction, ActionKeyDirection::Any)
                || direction == context.last_known_direction
            {
                Player::DoubleJumping(DoubleJumping::new(
                    moving.timeout(Timeout::default()).completed(false),
                    true,
                    false,
                ))
            } else {
                Player::UseKey(UseKey::from_key(key))
            };
        }
        Some(PlayerAction::Key(
            key @ Key {
                with: ActionKeyWith::Any,
                ..
            },
        )) => {
            if moving.completed && y_distance <= USE_KEY_Y_THRESHOLD {
                player.state = Player::UseKey(UseKey::from_key(key));
            }
        }
        Some(PlayerAction::AutoMob(mob)) => update_from_auto_mob_action(
            resources,
            player,
            minimap_state,
            mob,
            (x_distance, x_direction),
            (y_distance, y_direction),
            moving.timeout.total >= MID_AIR_ATTACK_MIN_TICKS,
        ),
        None
        | Some(
            PlayerAction::Key(Key {
                with: ActionKeyWith::Stationary,
                ..
            })
            | PlayerAction::SolveRune
            | PlayerAction::Move(_),
        ) => (),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use mockall::predicate::eq;
    use opencv::core::Point;

    use super::*;
    use crate::{
        bridge::MockInput,
        player::{Player, PlayerContext},
    };

    fn mock_player_entity(pos: Point) -> PlayerEntity {
        let mut context = PlayerContext::default();
        context.last_known_pos = Some(pos);

        PlayerEntity {
            state: Player::Idle,
            context,
        }
    }

    #[test]
    fn update_adjusting_state_started() {
        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 10, y: 0 };
        let mut player = mock_player_entity(pos);
        player.context.is_stationary = true;
        player.state = Player::Adjusting(Adjusting::new(Moving::new(pos, dest, false, None)));

        let mut resources = Resources::new(None, None);

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        // Should remain in Adjusting state (started branch moves it into Adjusting with moving started)
        assert_matches!(player.state, Player::Adjusting(_));
        assert_matches!(player.context.last_movement, Some(LastMovement::Adjusting));
    }

    #[test]
    fn update_adjusting_state_updated_performs_medium_adjustment_right() {
        let mut keys = MockInput::default();
        // Expect right to be pressed down and left to be released
        keys.expect_send_key_up().with(eq(KeyKind::Left)).once();
        keys.expect_send_key_down().with(eq(KeyKind::Right)).once();

        let mut resources = Resources::new(Some(keys), None);

        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 5, y: 0 }; // x_distance = 5 (>= medium threshold = 3)
        let mut player = mock_player_entity(pos);
        player.state = Player::Adjusting(Adjusting::new(
            Moving::new(pos, dest, false, None).timeout_started(true),
        ));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::Adjusting(_));
        assert_eq!(
            player.context.last_known_direction,
            ActionKeyDirection::Right
        );
    }

    #[test]
    fn update_adjusting_state_updated_performs_medium_adjustment_left() {
        let mut keys = MockInput::default();
        keys.expect_send_key_up().with(eq(KeyKind::Right)).once();
        keys.expect_send_key_down().with(eq(KeyKind::Left)).once();

        let mut resources = Resources::new(Some(keys), None);

        let pos = Point { x: 10, y: 0 };
        let dest = Point { x: 0, y: 0 }; // x_distance = 10
        let mut player = mock_player_entity(pos);
        player.state = Player::Adjusting(Adjusting::new(
            Moving::new(pos, dest, false, None).timeout_started(true),
        ));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::Adjusting(_));
        assert_eq!(
            player.context.last_known_direction,
            ActionKeyDirection::Left
        );
    }

    #[test]
    fn update_adjusting_state_updated_exact_uses_short_adjust_below_exact_switch_threshold() {
        let mut keys = MockInput::default();
        // Below `ADJUSTING_EXACT_SWITCH_THRESHOLD` (8) but at/above `ADJUSTING_MEDIUM_THRESHOLD`
        // (3) - a non-exact move would still be continuous-hold walking here, but an exact one
        // should already have switched to a bounded tap instead.
        keys.expect_send_key_up().with(eq(KeyKind::Left)).once();
        keys.expect_send_key_up().with(eq(KeyKind::Right)).once();
        keys.expect_send_key_with_options()
            .with(
                eq(KeyKind::Right),
                eq(InputKeyOptions::default().down_ms(80)),
            )
            .once();

        let mut resources = Resources::new(Some(keys), None);

        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 5, y: 0 }; // x_distance = 5
        let mut player = mock_player_entity(pos);
        player.state = Player::Adjusting(Adjusting::new(
            Moving::new(pos, dest, true, None).timeout_started(true),
        ));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                adjust_timeout: Timeout { started: true, .. },
                ..
            })
        );
        assert_eq!(
            player.context.last_known_direction,
            ActionKeyDirection::Right
        );
    }

    #[test]
    fn update_adjusting_state_updated_completes_when_no_direction_and_no_adjustment() {
        let mut keys = MockInput::default();
        keys.expect_send_key_up().with(eq(KeyKind::Left)).once();
        keys.expect_send_key_up().with(eq(KeyKind::Right)).once();

        let mut resources = Resources::new(Some(keys), None);

        let pos = Point { x: 5, y: 0 };
        let dest = Point { x: 5, y: 0 }; // same position, no direction
        let mut player = mock_player_entity(pos);
        player.state = Player::Adjusting(Adjusting::new(
            Moving::new(pos, dest, false, None).timeout_started(true),
        ));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                moving: Moving {
                    completed: true,
                    ..
                },
                ..
            })
        );
    }

    #[test]
    fn update_adjusting_state_updated_short_adjustment_started() {
        let mut keys = MockInput::default();
        keys.expect_send_key_up().with(eq(KeyKind::Left)).once();
        keys.expect_send_key_up().with(eq(KeyKind::Right)).once();
        keys.expect_send_key_with_options()
            .with(
                eq(KeyKind::Right),
                eq(InputKeyOptions::default().down_ms(80)),
            )
            .once();

        let mut resources = Resources::new(Some(keys), None);

        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 1, y: 0 }; // exact = true, x_distance = 1
        let mut player = mock_player_entity(pos);
        player.state = Player::Adjusting(Adjusting::new(
            Moving::new(pos, dest, true, None).timeout_started(true),
        ));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                adjust_timeout: Timeout { started: true, .. },
                ..
            })
        );
        assert_eq!(
            player.context.last_known_direction,
            ActionKeyDirection::Right
        );
    }

    #[test]
    fn update_adjusting_state_updated_timeout_freezes_when_adjusting_started() {
        let mut resources = Resources::new(None, None);
        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 1, y: 0 };
        let mut player = mock_player_entity(pos);

        let moving = Moving::new(pos, dest, true, None)
            .timeout_current(3)
            .timeout_started(true);
        let mut adjusting = Adjusting::new(moving);
        adjusting.adjust_timeout = Timeout {
            current: 1,
            started: true,
            ..Default::default()
        };
        player.state = Player::Adjusting(adjusting);

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                moving: Moving {
                    timeout: Timeout { current: 3, .. },
                    ..
                },
                adjust_timeout: Timeout { current: 2, .. },
                ..
            })
        );
    }

    #[test]
    fn update_adjusting_state_updated_complted_exact_not_close_enough_keeps_adjusting() {
        let mut resources = Resources::new(None, None);
        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 1, y: 0 };
        let mut player = mock_player_entity(pos);

        let moving = Moving::new(pos, dest, true, None)
            .completed(true)
            .timeout_current(4)
            .timeout_started(true);
        player.state = Player::Adjusting(Adjusting::new(moving));

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                moving: Moving {
                    completed: false,
                    timeout: Timeout {
                        current: 0,
                        started: true,
                        ..
                    },
                    ..
                },
                ..
            })
        );
    }

    #[test]
    fn update_adjusting_state_updated_completed_exact_gives_up_after_max_attempts() {
        let mut resources = Resources::new(None, None);
        let pos = Point { x: 0, y: 0 };
        let dest = Point { x: 1, y: 0 };
        let mut player = mock_player_entity(pos);

        let moving = Moving::new(pos, dest, true, None)
            .completed(true)
            .timeout_current(4)
            .timeout_started(true);
        let mut adjusting = Adjusting::new(moving);
        adjusting.short_adjust_attempts = MAX_SHORT_ADJUST_ATTEMPTS;
        player.state = Player::Adjusting(adjusting);

        update_adjusting_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::Adjusting(Adjusting {
                moving: Moving {
                    completed: true,
                    timeout: Timeout {
                        current: MOVE_TIMEOUT,
                        ..
                    },
                    ..
                },
                ..
            })
        );
    }

    // TODO: add tests for on_action
}
