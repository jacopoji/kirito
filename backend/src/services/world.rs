use super::EventContext;
use crate::{
    OperationUpdate,
    ecs::WorldEvent,
    notification::NotificationKind,
    player::{PanicTo, Panicking, Player},
    services::{EventHandler, operation::Halt},
};

pub struct WorldEventHandler;

impl EventHandler<WorldEvent> for WorldEventHandler {
    fn handle(&mut self, context: &mut EventContext<'_>, event: WorldEvent) {
        match event {
            WorldEvent::RunTimerEnded => {
                context
                    .operation_service
                    .queue_halt(true, Halt { go_to_town: true });
                if context
                    .settings_service
                    .settings()
                    .notifications
                    .notify_on_run_timer_end
                {
                    context
                        .resources
                        .notification
                        .schedule_notification(NotificationKind::RunTimerEnded);
                }
            }
            WorldEvent::PlayerDied => {
                if context.settings_service.settings().stop_on_player_die {
                    context
                        .operation_service
                        .queue_halt(true, Halt { go_to_town: false });
                }
            }
            WorldEvent::MinimapChanged => {
                if context.resources.operation.halting() {
                    return;
                }

                // Monster Park intentionally changes maps every time it uses a portal, so the
                // "unexpected map change" response below (notification, and optionally halting
                // to go to town) would otherwise fire on every single portal transition.
                if context.rotator.is_monster_park_active() {
                    return;
                }

                // Navigating to a daily quest's hunting ground intentionally changes maps too -
                // opening the world map covers the minimap (triggering re-detection) and
                // teleporting actually changes it. Without this, the watchdog below would send
                // the player back to town mid-navigation on every single daily quest entry.
                if context.rotator.is_navigating_daily_quest() {
                    return;
                }

                let is_panicking = matches!(
                    context.world.player.state,
                    Player::Panicking(Panicking {
                        to: PanicTo::Channel,
                        ..
                    })
                );
                if is_panicking {
                    return;
                }

                context
                    .resources
                    .notification
                    .schedule_notification(NotificationKind::FailOrMapChange);
                context.operation_service.abort_halt();

                if !context
                    .settings_service
                    .settings()
                    .stop_on_fail_or_change_map
                {
                    return;
                }

                context
                    .operation_service
                    .queue_halt(false, Halt { go_to_town: true });
            }
            WorldEvent::CaptureFailed => {
                if context.resources.operation.halting() {
                    return;
                }

                if context
                    .settings_service
                    .settings()
                    .stop_on_fail_or_change_map
                {
                    context
                        .operation_service
                        .update(context.resources, OperationUpdate::TemporaryHalt);
                }
                context
                    .resources
                    .notification
                    .schedule_notification(NotificationKind::FailOrMapChange);
            }
            WorldEvent::LieDetectorShapeAppeared => {
                if !context.resources.operation.halting() {
                    context
                        .resources
                        .notification
                        .schedule_notification(NotificationKind::LieDetectorShapeAppear);
                    play_lie_detector_alert_sound(context);
                }
            }
            WorldEvent::LieDetectorViolettaAppeared => {
                if !context.resources.operation.halting() {
                    context
                        .resources
                        .notification
                        .schedule_notification(NotificationKind::LieDetectorViolettaAppear);
                    play_lie_detector_alert_sound(context);
                }
            }
            WorldEvent::EliteBossAppeared => {
                if !context.resources.operation.halting() {
                    context
                        .resources
                        .notification
                        .schedule_notification(NotificationKind::EliteBossAppear);
                }
            }
            WorldEvent::LieDetectorShapeSolved => {
                if context
                    .settings_service
                    .settings()
                    .stop_on_lie_detector_shape_solve
                {
                    context
                        .operation_service
                        .queue_halt(true, Halt { go_to_town: true });
                }
            }
        }
    }
}

fn play_lie_detector_alert_sound(context: &EventContext<'_>) {
    let settings = context.settings_service.settings();
    if settings.notifications.play_sound_on_lie_detector_appear {
        let sound_path = settings.notifications.lie_detector_sound_path.clone();
        drop(settings);
        platforms::sound::play_alert(Some(sound_path.as_str()));
    }
}
