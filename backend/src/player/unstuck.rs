use opencv::core::Point;

use super::timeout::{Lifecycle, Timeout, next_timeout_lifecycle};
use crate::{
    bridge::KeyKind,
    ecs::Resources,
    minimap::Minimap,
    player::{MOVE_TIMEOUT, Player, PlayerAction, PlayerEntity, next_action},
};

#[derive(Debug, Clone, Copy)]
enum UnstuckingKind {
    Esc,
    Movement {
        timeout: Timeout,
        random: bool,
        blink: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Unstucking {
    kind: UnstuckingKind,
}

impl Unstucking {
    pub fn new_esc() -> Self {
        Self {
            kind: UnstuckingKind::Esc,
        }
    }

    pub fn new_movement(timeout: Timeout, random: bool, blink: bool) -> Self {
        Self {
            kind: UnstuckingKind::Movement {
                timeout,
                random,
                blink,
            },
        }
    }

    fn movement(mut self, timeout: Timeout, random: bool, blink: bool) -> Unstucking {
        self.kind = UnstuckingKind::Movement {
            timeout,
            random,
            blink,
        };
        self
    }
}

/// A threshold to consider spamming falling action
///
/// This is when the player is inside the top edge of minimap. At least for higher level maps, this
/// seems rare but one possible map is The Forest Of Earth in Arcana.
const Y_IGNORE_THRESHOLD: i32 = 18;

/// Updates the [`Player::Unstucking`] contextual state
///
/// This state can only be transitioned to when [`PlayerState::unstuck_counter`] reached the fixed
/// threshold, when the player moved into the edges of the minimap or rotator detected an UI
/// element blocking the player.
/// If [`PlayerState::unstuck_consecutive_counter`] has not reached the threshold and the player
/// moved into the left/right/top edges of the minimap, it will try to move
/// out as appropriate. It will also try to press ESC key to exit any dialog.
///
/// Each initial transition to [`Player::Unstucking`] increases
/// the [`PlayerState::unstuck_consecutive_counter`] by one. If the threshold is reached, this
/// state will just jump in random direction (GAMBA MODE). If GAMBA MODE itself keeps being
/// entered without the player position changing (e.g. the player wandered off the minimap
/// entirely and normal movement cannot recover it), it will additionally press
/// [`PlayerConfiguration::blink_key`], if configured, as a last resort.
pub fn update_unstucking_state(
    resources: &mut Resources,
    player: &mut PlayerEntity,
    minimap_state: Minimap,
) {
    let Player::Unstucking(unstucking) = player.state else {
        panic!("state is not unstucking");
    };

    // A blocking dialog/menu (e.g. the change-channel popup) can itself prevent the minimap
    // from being detected, so this must run before the `Minimap::Idle` check below instead of
    // being gated behind it - otherwise it would never get a chance to dismiss the very thing
    // that is blocking detection. This also applies regardless of `UnstuckingKind`: the bot can
    // land here from automatic stuck detection (`Movement`) just as easily as from an explicit
    // `Unstuck` action (`Esc`), and a leftover dialog can be the actual cause of being stuck.
    //
    // A "special" NPC dialog (see `Detector::detect_special_dialog`'s docs, e.g. a Threads of
    // Fate dialog left open by some interrupted flow) does not close on ESC at all - pressing
    // ESC against one just wastes a tick while the player stays stuck behind it. Its own interact
    // key is what actually dismisses it, checked first since ESC would otherwise be a no-op here.
    if resources.detector().detect_special_dialog() {
        resources.input.send_key(player.context.config.interact_key);
    } else if resources.detector().detect_esc_settings()
        || resources.detector().detect_change_channel_menu_opened()
        || resources.detector().detect_quest_menu_opened()
    {
        resources.input.send_key(KeyKind::Esc);
    }

    let Minimap::Idle(idle) = minimap_state else {
        player.state = Player::Detecting;
        return;
    };

    match unstucking.kind {
        UnstuckingKind::Esc => match next_action(&player.context) {
            Some(PlayerAction::Unstuck) => {
                player.context.clear_action_completed();
                player.state = Player::Detecting;
            }
            Some(_) | None => {
                player.state = Player::Detecting;
            }
        },
        UnstuckingKind::Movement {
            timeout,
            random,
            blink,
        } => {
            let context = &mut player.context;
            let pos = context
                .last_known_pos
                .map(|pos| Point::new(pos.x, idle.bbox.height - pos.y));
            let random = random || pos.is_none();

            match next_timeout_lifecycle(timeout, MOVE_TIMEOUT) {
                Lifecycle::Started(timeout) => {
                    if blink && let Some(key) = context.config.blink_key {
                        resources.input.send_key(key);
                    }

                    let to_right = match (random, pos) {
                        (true, _) => resources.rng.random_bool(0.5),
                        (_, Some(Point { y, .. })) if y <= Y_IGNORE_THRESHOLD => {
                            player.state =
                                Player::Unstucking(unstucking.movement(timeout, random, blink));
                            return;
                        }
                        (_, Some(Point { x, .. })) => x <= idle.bbox.width / 2,
                        (_, None) => unreachable!(),
                    };
                    if to_right {
                        resources.input.send_key_down(KeyKind::Right);
                    } else {
                        resources.input.send_key_up(KeyKind::Left);
                    }

                    player.state = Player::Unstucking(unstucking.movement(timeout, random, blink));
                }
                Lifecycle::Ended => {
                    resources.input.send_key_up(KeyKind::Right);
                    resources.input.send_key_up(KeyKind::Left);
                    player.state = Player::Detecting;
                }
                Lifecycle::Updated(timeout) => {
                    let send_space = match (random, pos) {
                        (true, _) => true,
                        (_, Some(pos)) if pos.y > Y_IGNORE_THRESHOLD => true,
                        _ => false,
                    };
                    // Coin-flip instead of pressing every tick - repeated stuck detection can
                    // chain many of these states back to back, and a jump key firing on every
                    // single ~33ms tick with zero variance is a distinctly bot-like input pattern
                    // (this was observed triggering the in-game lie detector check). Skipping
                    // roughly half the ticks still keeps the player jumping regularly enough to
                    // recover while breaking up the perfectly uniform cadence.
                    if send_space && resources.rng.random_bool(0.5) {
                        resources.input.send_key(context.config.jump_key);
                    }

                    player.state = Player::Unstucking(unstucking.movement(timeout, random, blink));
                }
            }
        }
    }
}
