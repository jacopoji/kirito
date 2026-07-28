use log::debug;

use super::{
    Key, Player, PlayerContext,
    actions::update_from_ping_pong_action,
    moving::Moving,
    timeout::{MovingLifecycle, next_moving_lifecycle_with_axis},
    use_key::UseKey,
};
use crate::{
    ActionKeyWith,
    bridge::{InputKeyDownOptions, KeyKind},
    ecs::Resources,
    minimap::Minimap,
    player::{
        MOVE_TIMEOUT, PlayerAction, PlayerEntity,
        actions::{MID_AIR_ATTACK_MIN_TICKS, update_from_auto_mob_action},
        next_action, state::LastMovement, timeout::ChangeAxis,
    },
};

/// Number of ticks to wait before spamming jump key.
const SPAM_DELAY: u32 = 7;

/// Number of ticks to wait before sending [`UpJumpingKind::UpArrow`]/[`UpJumpingKind::JumpKey`]'s
/// follow-up key press.
///
/// Must stay below the ~4-6 ticks a plain jump's own rising velocity takes to cross
/// [`UP_JUMPED_Y_VELOCITY_THRESHOLD`] on its own (observed from logs), otherwise the follow-up
/// press loses the race against that false-positive completion check and never gets sent.
const SECOND_PRESS_DELAY: u32 = 2;

/// Number of ticks to wait before spamming jump key for lesser travel distance.
const SOFT_SPAM_DELAY: u32 = 12;

const TIMEOUT: u32 = MOVE_TIMEOUT + 3;

/// Player's `y` velocity to be considered as up jumped.
const UP_JUMPED_Y_VELOCITY_THRESHOLD: f32 = 1.3;

/// Player's `x` velocity to be considered as near stationary.
const X_NEAR_STATIONARY_THRESHOLD: f32 = 0.28;

/// Player's `y` velocity to be considered as near stationary.
const Y_NEAR_STATIONARY_VELOCITY_THRESHOLD: f32 = 0.4;

/// Minimum distance required to perform an up jump using teleport key with jump.
const TELEPORT_WITH_JUMP_THRESHOLD: i32 = 19;

/// Minimum distance required to perform an up jump using teleport key with jump when teleport
/// increase buff is enabled.
const EXTENDED_TELEPORT_WITH_JUMP_THRESHOLD: i32 = 20;

/// Minimum distance required to perform an up jump and then teleport.
const UP_JUMP_AND_TELEPORT_THRESHOLD: i32 = 23;

const SOFT_UP_JUMP_THRESHOLD: i32 = 16;

// Upstream issue #159 / PR #161 ("Improve up jump"): the old logic used a single boolean heuristic
// and could pick jump-then-teleport when a direct teleport sufficed, or worse, perform the
// up-jump half of a "up jump + teleport" combo and never fire the follow-up teleport at all. This
// three-state machine plus the threshold constants above exist so exactly one of "just teleport" /
// "jump then teleport" / "up-jump then teleport" is chosen by distance, and an up-jump always
// transitions back through `MageState::Teleporting` so the teleport actually gets sent.
#[derive(Debug, Clone, Copy)]
struct Mage {
    state: MageState,
    teleport_with_jump_threshold: i32,
}

#[derive(Debug, Clone, Copy)]
enum MageState {
    Teleporting,
    UpJumping,
    Flying,
}

#[derive(Debug, Clone, Copy)]
enum UpJumpingKind {
    Mage(Mage),
    UpArrow,
    JumpKey,
    SpecificKey,
}

#[derive(Debug, Clone, Copy)]
pub struct UpJumping {
    pub moving: Moving,
    /// The kind of up jump.
    kind: UpJumpingKind,
    /// Number of ticks to wait before sending jump key(s).
    spam_delay: u32,
    /// Whether auto-mobbing should wait for up jump completion in non-intermediate destination.
    auto_mob_wait_completion: bool,
    /// Whether [`UpJumpingKind::UpArrow`]/[`UpJumpingKind::JumpKey`]'s follow-up key press has
    /// been sent.
    ///
    /// A plain jump's own rising velocity crosses [`UP_JUMPED_Y_VELOCITY_THRESHOLD`] on its own
    /// within a handful of ticks, well before [`Self::spam_delay`] - so gating the follow-up
    /// press on that same velocity check meant it almost never actually got sent, and the up
    /// jump combo silently degraded into a plain jump. This flag lets the follow-up press be
    /// sent on its own short timer regardless of velocity, and reserves the velocity check for
    /// deciding completion only after that press has gone out.
    second_press_sent: bool,
}

impl UpJumping {
    pub fn new(moving: Moving, resources: &mut Resources, player_context: &PlayerContext) -> Self {
        let (y_distance, _) = moving.y_distance_direction_from(true, moving.pos);
        let spam_delay = if !player_context.config.up_jump_specific_key_should_jump
            && y_distance <= SOFT_UP_JUMP_THRESHOLD
        {
            SOFT_SPAM_DELAY
        } else {
            SPAM_DELAY
        };
        let auto_mob_wait_completion =
            player_context.has_auto_mob_action_only() && resources.rng.random_bool(0.5);
        let kind = up_jumping_kind(
            player_context.config.up_jump_key,
            player_context.config.teleport_key.is_some(),
            player_context.config.has_extended_teleport_range,
        );

        Self {
            moving,
            kind,
            spam_delay,
            auto_mob_wait_completion,
            second_press_sent: false,
        }
    }

    #[inline]
    fn moving(mut self, moving: Moving) -> UpJumping {
        self.moving = moving;
        self
    }
}

/// Whether abandoning the up jump right now to attack instead would still land every key press
/// [`update_up_jump`] considers required for `up_jumping`'s kind, rather than cutting it short.
///
/// - [`UpJumpingKind::UpArrow`]/[`UpJumpingKind::JumpKey`] need their delayed follow-up press
///   ([`UpJumping::second_press_sent`]) to actually reach full height - before that's sent,
///   abandoning would silently degrade the jump into a much shorter plain one.
/// - [`UpJumpingKind::Mage`] still owes its teleport half while in [`MageState::UpJumping`], and
///   [`MageState::Flying`] continuously holds a key that isn't necessarily one of the four arrow
///   keys released on a mid-air attack interrupt - neither is safe to cut short.
/// - [`UpJumpingKind::SpecificKey`] is a single fixed key with nothing further to send once
///   pressed, unless it's flying (same held-key concern as `Mage`'s `Flying`).
#[inline]
fn up_jump_mid_air_attack_safe(up_jumping: &UpJumping, is_flight: bool) -> bool {
    match up_jumping.kind {
        UpJumpingKind::UpArrow | UpJumpingKind::JumpKey => up_jumping.second_press_sent,
        UpJumpingKind::Mage(mage) => matches!(mage.state, MageState::Teleporting),
        UpJumpingKind::SpecificKey => !is_flight,
    }
}

/// Updates the [`Player::UpJumping`] contextual state.
///
/// This state can only be transitioned via [`Player::Moving`] when the
/// player has reached the destination x-wise. Before performing an up jump, it will check for
/// stationary state and whether the player is currently near a portal. If the player is near
/// a portal, this action is aborted. The up jump action is made to be adapted for various classes
/// that has different up jump key combination.
pub fn update_up_jumping_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
) {
    let Player::UpJumping(mut up_jumping) = player.state else {
        panic!("state is not up jumping");
    };
    let up_jump_key = player.context.config.up_jump_key;
    let jump_key = player.context.config.jump_key;
    let should_jump = player.context.config.up_jump_specific_key_should_jump;
    let is_flight = player.context.config.up_jump_is_flight;

    match next_moving_lifecycle_with_axis(
        up_jumping.moving,
        player
            .context
            .last_known_pos
            .expect("in positional context"),
        TIMEOUT,
        ChangeAxis::Vertical,
    ) {
        MovingLifecycle::Started(moving) => {
            // Stall until near stationary
            let (x_velocity, y_velocity) = player.context.velocity;
            if x_velocity > X_NEAR_STATIONARY_THRESHOLD
                || y_velocity > Y_NEAR_STATIONARY_VELOCITY_THRESHOLD
            {
                let moving = moving.timeout_started(false);
                let up_jumping = up_jumping.moving(moving);

                player.state = Player::UpJumping(up_jumping);
                return;
            }

            // Upstream issue #196: on some maps a rune/target can sit close enough to an exit
            // portal that lining up the x-coordinate to up-jump/teleport would carry the player
            // through the portal, producing an infinite align-teleport-through-portal-walk-back
            // loop. Aborting to Idle when already inside a portal's bounds breaks that loop. This
            // is a deliberate trade-off (favors not triggering the portal over always reaching a
            // target in this spot) with no code-only fix identified upstream — don't remove it
            // without a replacement, or the loop comes back.
            let is_inside_portal = match minimap_state {
                Minimap::Idle(idle) => idle.is_position_inside_portal(moving.pos),
                _ => false,
            };
            if is_inside_portal {
                player.state = Player::Idle;
                player.context.clear_action_completed();
                return;
            }

            let (y_distance, _) = moving.y_distance_direction_from(true, moving.pos);
            if let UpJumpingKind::Mage(mage) = &mut up_jumping.kind {
                mage.state = if is_flight {
                    MageState::Flying
                } else if y_distance >= UP_JUMP_AND_TELEPORT_THRESHOLD {
                    MageState::UpJumping
                } else {
                    MageState::Teleporting
                };
            }

            player.context.last_movement = Some(LastMovement::UpJumping);
            player.state = Player::UpJumping(up_jumping.moving(moving));

            match up_jumping.kind {
                UpJumpingKind::Mage(mage) => {
                    resources.input.send_key_down(KeyKind::Up);
                    let can_jump =
                        y_distance >= mage.teleport_with_jump_threshold && up_jump_key.is_none();
                    if is_flight || can_jump {
                        resources.input.send_key(jump_key);
                    }
                }
                UpJumpingKind::UpArrow => {
                    resources.input.send_key(jump_key);
                }
                UpJumpingKind::JumpKey => {
                    resources.input.send_key_down(KeyKind::Up);
                    resources.input.send_key(jump_key);
                }
                UpJumpingKind::SpecificKey => {
                    resources.input.send_key_down(KeyKind::Up);
                    if is_flight || should_jump {
                        resources.input.send_key(jump_key);
                    }
                }
            }
        }
        MovingLifecycle::Ended(moving) => {
            player.state = Player::Moving(moving.dest, moving.exact, moving.intermediates);
            resources.input.send_key_up(KeyKind::Up);
        }
        MovingLifecycle::Updated(mut moving) => {
            let (y_distance, y_direction) = moving.y_distance_direction_from(true, moving.pos);
            update_up_jump(
                resources,
                &player.context,
                &mut moving,
                &mut up_jumping,
                y_distance,
                y_direction,
            );

            // Sets initial next state first
            player.state = Player::UpJumping(up_jumping.moving(moving));
            update_from_action(
                resources,
                player,
                minimap_state,
                up_jumping,
                moving,
                y_direction,
            );
        }
    }
}

fn update_from_action(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
    up_jumping: UpJumping,
    moving: Moving,
    y_direction: i32,
) {
    let cur_pos = moving.pos;

    match next_action(&player.context) {
        Some(PlayerAction::AutoMob(mob)) => {
            if moving.completed && moving.is_destination_intermediate() && y_direction <= 0 {
                resources.input.send_key_up(KeyKind::Up);
                player.state = Player::Moving(moving.dest, moving.exact, moving.intermediates);
                return;
            }

            if up_jumping.auto_mob_wait_completion && !moving.completed {
                return;
            }

            // `up_jump_mid_air_attack_safe` alone only guarantees the jump's required key
            // sequence has gone out - not that the jump has actually gained any height yet, so
            // pairing it with `MID_AIR_ATTACK_MIN_TICKS` (as double_jump/fall/adjust already do
            // for their own mid-air attack checks) keeps an attack from interrupting the jump
            // before it has actually started closing the gap, which otherwise looks like
            // attacking in place under a mob still well out of reach above.
            let mid_air_attack_safe =
                up_jump_mid_air_attack_safe(&up_jumping, player.context.config.up_jump_is_flight)
                    && moving.timeout.total >= MID_AIR_ATTACK_MIN_TICKS;
            let (x_distance, x_direction) = moving.x_distance_direction_from(false, cur_pos);
            let (y_distance, y_direction) = moving.y_distance_direction_from(false, cur_pos);
            update_from_auto_mob_action(
                resources,
                player,
                minimap_state,
                mob,
                (x_distance, x_direction),
                (y_distance, y_direction),
                mid_air_attack_safe,
            )
        }

        // Upstream PR #56 ("Improve `UseWith` `Any` for fall and up jump"): same reasoning as the
        // matching branch in fall.rs — without it, `Any`-with key actions never fire mid up-jump.
        Some(PlayerAction::Key(
            key @ Key {
                with: ActionKeyWith::Any,
                ..
            },
        )) => {
            if moving.completed && y_direction <= 0 {
                player.state = Player::UseKey(UseKey::from_key(key));
            }
        }

        Some(PlayerAction::PingPong(ping_pong)) => {
            if !moving.completed
                || !resources
                    .rng
                    .random_perlin_bool(cur_pos.x, cur_pos.y, resources.tick, 0.7)
            {
                return;
            }

            update_from_ping_pong_action(resources, player, minimap_state, ping_pong, cur_pos);
        }

        Some(
            PlayerAction::Key(Key {
                with: ActionKeyWith::Stationary | ActionKeyWith::DoubleJump,
                ..
            })
            | PlayerAction::Move(_)
            | PlayerAction::SolveRune,
        )
        | None => (),
        _ => unreachable!(),
    }
}

fn update_up_jump(
    resources: &mut Resources,
    context: &PlayerContext,
    moving: &mut Moving,
    up_jumping: &mut UpJumping,
    y_distance: i32,
    y_direction: i32,
) {
    let jump_key = context.config.jump_key;
    let up_jump_key = context.config.up_jump_key;
    let should_jump = context.config.up_jump_specific_key_should_jump;
    let is_flight = context.config.up_jump_is_flight;

    if moving.completed {
        resources.input.send_key_up(KeyKind::Up);
        return;
    }

    match &mut up_jumping.kind {
        UpJumpingKind::Mage(mage) => {
            update_mage_up_jump(
                resources,
                context,
                moving,
                mage,
                up_jumping.spam_delay,
                y_distance,
                y_direction,
            );
        }
        UpJumpingKind::UpArrow | UpJumpingKind::JumpKey => {
            // The follow-up press is sent on its own short timer, independent of velocity - a
            // plain jump's own rising velocity crosses UP_JUMPED_Y_VELOCITY_THRESHOLD on its own
            // within a handful of ticks, so gating this on velocity meant the follow-up press
            // almost never actually went out before the check below concluded "already jumped".
            if !up_jumping.second_press_sent {
                if moving.timeout.total >= SECOND_PRESS_DELAY {
                    debug!(
                        target: "backend/player",
                        "up jump sending follow-up key at timeout {}, velocity.1 {}",
                        moving.timeout.total, context.velocity.1
                    );
                    if matches!(up_jumping.kind, UpJumpingKind::UpArrow) {
                        resources.input.send_key(KeyKind::Up);
                    } else {
                        resources.input.send_key(jump_key);
                    }
                    up_jumping.second_press_sent = true;
                }
                // Don't also fall through to the completion/fallback-spam check below on the
                // same tick - `context.velocity` reflects the position from before this tick's
                // press, so it can't yet show the effect of a press just sent this tick.
                return;
            }

            if context.velocity.1 > UP_JUMPED_Y_VELOCITY_THRESHOLD {
                debug!(
                    target: "backend/player",
                    "up jump completed at timeout {} (spam_delay {}), velocity.1 {}",
                    moving.timeout.total, up_jumping.spam_delay, context.velocity.1
                );
                moving.completed = true;
            } else if moving.timeout.total >= up_jumping.spam_delay {
                // Fallback: the follow-up press above didn't register as a boost in time, keep
                // spamming as before.
                debug!(
                    target: "backend/player",
                    "up jump spamming key at timeout {} (spam_delay {}), velocity.1 {}",
                    moving.timeout.total, up_jumping.spam_delay, context.velocity.1
                );
                if matches!(up_jumping.kind, UpJumpingKind::UpArrow) {
                    resources.input.send_key(KeyKind::Up);
                } else {
                    resources.input.send_key(jump_key);
                }
            }
        }
        UpJumpingKind::SpecificKey => {
            if !is_flight {
                if !should_jump || moving.timeout.total >= up_jumping.spam_delay {
                    resources
                        .input
                        .send_key(up_jump_key.expect("has up jump key"));
                    moving.completed = true;
                }
            } else {
                update_flying(
                    resources,
                    moving,
                    y_direction,
                    up_jump_key.expect("has up jump key"),
                );
            }
        }
    }
}

fn update_mage_up_jump(
    resources: &mut Resources,
    context: &PlayerContext,
    moving: &mut Moving,
    mage: &mut Mage,
    spam_delay: u32,
    y_distance: i32,
    y_direction: i32,
) {
    let jump_key = context.config.jump_key;
    let up_jump_key = context.config.up_jump_key;
    let teleport_key = context.config.teleport_key.expect("has teleport key");

    match mage.state {
        MageState::Teleporting => {
            if y_direction > 0 && y_distance < mage.teleport_with_jump_threshold {
                resources.input.send_key(teleport_key);
                moving.completed = true;
            }
        }
        MageState::UpJumping => match up_jump_key {
            Some(key) => {
                resources.input.send_key(key);
                mage.state = MageState::Teleporting;
            }
            None => {
                if context.velocity.1 <= UP_JUMPED_Y_VELOCITY_THRESHOLD {
                    if moving.timeout.total >= spam_delay {
                        resources.input.send_key(jump_key);
                    }
                } else {
                    mage.state = MageState::Teleporting;
                }
            }
        },
        MageState::Flying => update_flying(
            resources,
            moving,
            y_direction,
            up_jump_key.unwrap_or(teleport_key),
        ),
    }
}

#[inline]
fn update_flying(resources: &mut Resources, moving: &mut Moving, y_direction: i32, key: KeyKind) {
    if y_direction > 0 {
        resources
            .input
            .send_key_down_with_options(key, InputKeyDownOptions::default().repeatable());
    } else {
        resources.input.send_key_up(key);
        moving.completed = true;
    }
}

#[inline]
fn up_jumping_kind(
    up_jump_key: Option<KeyKind>,
    has_teleport_key: bool,
    has_extended_teleport_range: bool,
) -> UpJumpingKind {
    match (up_jump_key, has_teleport_key) {
        (Some(_), true) | (None, true) => UpJumpingKind::Mage(Mage {
            state: MageState::Teleporting, // Overwrite later
            teleport_with_jump_threshold: if has_extended_teleport_range {
                EXTENDED_TELEPORT_WITH_JUMP_THRESHOLD
            } else {
                TELEPORT_WITH_JUMP_THRESHOLD
            },
        }),
        (Some(KeyKind::Up), false) => UpJumpingKind::UpArrow,
        (None, false) => UpJumpingKind::JumpKey,
        (Some(_), false) => UpJumpingKind::SpecificKey,
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use opencv::core::Point;

    use super::*;
    use crate::bridge::{KeyKind, MockInput};
    use crate::ecs::Resources;
    use crate::player::{Player, PlayerEntity};

    fn setup_player(up_jumping: UpJumping) -> PlayerEntity {
        let mut player = PlayerEntity {
            state: Player::UpJumping(up_jumping),
            context: PlayerContext::default(),
        };
        player.context.last_known_pos = Some(Point::new(0, 0));
        player.context.config.jump_key = KeyKind::Space;
        player
    }

    #[test]
    fn update_up_jumping_state_started_jump_key_presses_up_and_jump() {
        let moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        let mut keys = MockInput::new();
        keys.expect_send_key_down()
            .withf(|k| *k == KeyKind::Up)
            .once();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_started_up_arrow_presses_jump_only() {
        let moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::UpArrow,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_started_specific_key_presses_up_only() {
        let moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::SpecificKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        player.context.config.up_jump_key = Some(KeyKind::C);
        let mut keys = MockInput::new();
        keys.expect_send_key_down()
            .withf(|k| *k == KeyKind::Up)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_started_mage_up_and_jump() {
        let moving = Moving::new(Point::new(0, 0), Point::new(0, 25), true, None);
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::Mage(Mage {
                state: MageState::Teleporting,
                teleport_with_jump_threshold: TELEPORT_WITH_JUMP_THRESHOLD,
            }),
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        player.context.config.teleport_key = Some(KeyKind::Shift);
        let mut keys = MockInput::new();
        keys.expect_send_key_down()
            .withf(|k| *k == KeyKind::Up)
            .once();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_velocity_marks_completed() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.timeout.started = true;
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: true, // Follow-up press already sent, velocity now decides
        });
        player.context.velocity = (0.0, 2.0); // Y velocity above threshold
        let mut resources = Resources::new(None, None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(
            player.state,
            Player::UpJumping(UpJumping {
                moving: Moving {
                    completed: true,
                    ..
                },
                ..
            })
        );
    }

    #[test]
    fn update_up_jumping_state_updated_before_second_press_delay_no_keys_sent() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.timeout.started = true;
        // `next_moving_lifecycle_with_axis` increments `total` by 1 before `update_up_jump` reads
        // it (the position doesn't change tick over tick here, so `current` isn't reset and this
        // is a plain `Updated` lifecycle) - preset two below the threshold so it lands one below
        // after that increment, genuinely testing "before threshold" instead of landing on it.
        moving.timeout.total = SECOND_PRESS_DELAY.saturating_sub(2); // before threshold
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        let mut keys = MockInput::new();
        keys.expect_send_key().never();
        keys.expect_send_key_down().never();
        keys.expect_send_key_up().never();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_sends_follow_up_key_after_delay() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.timeout.started = true;
        moving.timeout.total = SECOND_PRESS_DELAY; // exactly at threshold
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_spam_fallback_after_follow_up_press() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.timeout.started = true;
        moving.timeout.total = SPAM_DELAY; // follow-up already sent, now past spam_delay too
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: true,
        });
        player.context.velocity = (0.0, 0.0); // Y velocity still below threshold
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_spam_specific_key_after_delay() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.timeout.started = true;
        moving.timeout.total = SPAM_DELAY;
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::SpecificKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        player.context.config.up_jump_key = Some(KeyKind::C);
        let mut keys = MockInput::new();
        keys.expect_send_key().withf(|k| *k == KeyKind::C).once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_mage_spam_jump_after_delay() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 25), true, None);
        moving.timeout.started = true;
        moving.timeout.total = SPAM_DELAY;
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::Mage(Mage {
                state: MageState::UpJumping,
                teleport_with_jump_threshold: TELEPORT_WITH_JUMP_THRESHOLD,
            }),
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        player.context.config.jump_key = KeyKind::Space;
        player.context.config.teleport_key = Some(KeyKind::Shift);
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| *k == KeyKind::Space)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }

    #[test]
    fn update_up_jumping_state_updated_completed_and_releases_up() {
        let mut moving = Moving::new(Point::new(0, 0), Point::new(0, 20), true, None);
        moving.completed = true;
        moving.timeout.started = true;
        let mut player = setup_player(UpJumping {
            moving,
            kind: UpJumpingKind::JumpKey,
            spam_delay: SPAM_DELAY,
            auto_mob_wait_completion: false,
            second_press_sent: false,
        });
        let mut keys = MockInput::new();
        keys.expect_send_key_up()
            .withf(|k| *k == KeyKind::Up)
            .once();
        let mut resources = Resources::new(Some(keys), None);

        update_up_jumping_state(&mut resources, &mut player, Minimap::Detecting);

        assert_matches!(player.state, Player::UpJumping(_));
    }
}
