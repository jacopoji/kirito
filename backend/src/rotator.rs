use std::{
    assert_matches::debug_assert_matches,
    collections::VecDeque,
    fmt::Debug,
    ops::RangeInclusive,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use log::{debug, info};
#[cfg(test)]
use mockall::{automock, concretize};
use opencv::core::{Point, Rect};
use ordered_hash_map::OrderedHashMap;

use crate::{
    Bound,
    array::Array,
    bridge::{KeyKind, LinkKeyKind},
    buff::{Buff, BuffKind},
    detect::{Detector, QuickSlotsHexaBooster, SolErda},
    ecs::{Resources, World},
    minimap::{Minimap, MinimapContext},
    models::{
        Action, ActionCondition, ActionKey, ActionKeyDirection, ActionKeyWith, ActionMove,
        DailyQuestEntry, DailyQuestNavigation, EliteBossBehavior, ExchangeHexaBoosterCondition,
        Familiars, FamiliarsFusion, FamiliarsPotential, MobbingKey, Position,
        ThreadOfFateAskProgress, ThreadOfFateNpc, WaitAfterBuffered,
    },
    notification::NotificationKind,
    operation::OperationState,
    pathing::Platform,
    player::{
        AUTO_MOB_USE_KEY_X_THRESHOLD, AutoMob, Booster, ExchangeBooster, FamiliarsFuse,
        FamiliarsRevealPotential, FamiliarsSwap, GRAPPLING_THRESHOLD, Key, MonsterParkEntryOutcome,
        Move, NavigateToHuntingGround, Panic, PanicTo, PingPong, PingPongDirection, Player,
        PlayerAction, PlayerContext, PlayerEntity, Quadrant, ThreadOfFateAsk,
        ThreadOfFateAskComplete, UseBooster,
    },
    run::MS_PER_TICK,
    skill::{Skill, SkillKind},
    task::{Task, Update, update_detection_task},
};

const AUTO_MOB_SAME_QUAD_THRESHOLD: u32 = 5;

/// Width of the OCR region scanned below a matched "Quest complete!" toast (see
/// [`DefaultRotator::rotate_daily_quest`]) for its second line (e.g. `"[Daily Quest] Chu Chu
/// Island"`) - generous to fit the longest hunting ground names.
const DAILY_QUEST_COMPLETE_LABEL_ROI_WIDTH: i32 = 300;
/// Extra height, beyond the matched toast's own height, of the OCR region scanned for its second
/// line.
const DAILY_QUEST_COMPLETE_LABEL_ROI_PADDING: i32 = 24;

/// Poll interval for [`DefaultRotator::daily_quest_complete_task`] - kept short since a miss
/// (template not found) is cheap and the toast itself only renders briefly.
const DAILY_QUEST_COMPLETE_POLL_MILLIS: u64 = 1000;

/// `auto_mob_use_key_when_pathing_update_millis` applied while platform-pathing a daily quest
/// entry that has recorded platforms (see [`crate::models::DailyQuestId::platforms`]) - a plain
/// fixed default since, unlike a user-authored [`crate::models::Map`], there's no per-hunting-
/// ground value configured for this.
const DAILY_QUEST_AUTO_MOB_USE_KEY_UPDATE_MILLIS: u64 = 50;

/// Number of leading dropdown slots (region, then each `dropdown_path` entry in order) `current`
/// shares with `previous` - see [`NavigateToHuntingGround::skip_dropdown_slots`].
///
/// The world map keeps its dropdown selections as-is across being closed and reopened, so
/// consecutive daily quest entries sharing a region (and possibly deeper dropdown path) don't
/// need to re-select that shared prefix - it's already showing from the previous entry's
/// navigation.
fn shared_dropdown_prefix_len(
    previous: &DailyQuestNavigation,
    current: &DailyQuestNavigation,
) -> usize {
    if previous.region != current.region {
        return 0;
    }

    1 + previous
        .dropdown_path
        .iter()
        .zip(current.dropdown_path.iter())
        .take_while(|(prev, cur)| prev == cur)
        .count()
}

/// [`Condition`] evaluation result.
#[derive(Debug)]
enum ConditionResult {
    /// The action will be queued.
    Queue,
    /// The action is skipped.
    Skip,
    /// The action is skipped but `last_queued_time` is updated.
    Ignore,
}

type ConditionFn = Box<dyn FnMut(&Resources, &World, &PriorityActionQueueInfo) -> ConditionResult>;

/// Predicate for when a priority action can be queued.
struct Condition(ConditionFn);

impl std::fmt::Debug for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn Fn(...)")
    }
}

/// A priority action that can override a normal action.
///
/// This includes all non-[`ActionCondition::Any`] actions.
///
/// When a player is in the middle of doing a normal action, this type of action
/// can override most of the player's current state and forced to perform this action.
/// However, it cannot override player states that are considered "terminal". These states
/// include stalling, using key and forced double jumping. It also cannot override linked action.
///
/// When this type of action has [`Self::queue_to_front`] set, it will be queued to the
/// front and override other non-[`Self::queue_to_front`] priority action. The overriden
/// action is simply placed back to the queue in front. It is mostly useful for action such as
/// `press attack after x seconds even in the middle of moving`.
#[derive(Debug)]
struct PriorityAction {
    /// The predicate for when this action should be queued.
    condition: Condition,
    /// The kind the above predicate was derived from.
    condition_kind: Option<ActionCondition>,
    /// The inner action.
    inner: RotatorAction,
    /// The metadata about this action.
    metadata: Option<ActionMetadata>,
    /// Whether to queue this action to the front of [`Rotator::priority_actions_queue`].
    queue_to_front: bool,
    /// Whether/when this action should not be newly queued while a daily quest entry is pending
    /// (see [`DefaultRotator::daily_quest_index`]).
    suppress_during_daily_quest: DailyQuestSuppression,
    queue_info: PriorityActionQueueInfo,
}

/// When a [`PriorityAction`] should be held back from being newly queued while a daily quest
/// entry is pending (see [`DefaultRotator::daily_quest_index`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum DailyQuestSuppression {
    /// Never suppressed - matters regardless of what the player is currently doing. Used for
    /// safety/anti-bot actions (unstuck, rune/shape/violetta solving).
    Never,
    /// Suppressed only while still navigating to the current entry's hunting ground, not yet
    /// arrived and mobbing (see [`DefaultRotator::daily_quest_arrived`]).
    ///
    /// Used for fixed actions not tied to any particular map position (e.g. a periodic
    /// character-level buff/skill, see [`priority_action`]'s caller in
    /// [`DefaultRotator::build_actions`]) - these stay meaningful once actually mobbing at the
    /// hunting ground, unlike ones bound to a spot on the character's regular farming map.
    WhileNavigating,
    /// Suppressed for the entire daily quest run, navigating or mobbing alike.
    ///
    /// Regular grinding assists (buffs, boosters, familiar swapping, elite boss/panic channel
    /// changing) and map-tied fixed actions (an `ErdaShowerOffCooldown` skill, or an `EveryMillis`
    /// one bound to a specific position) are tied to the currently selected farming map and either
    /// make no sense or are actively disruptive while the player is off navigating to and mobbing
    /// a daily quest hunting ground instead - worse, a map-tied action can keep re-queuing
    /// back-to-back indefinitely, starving [`DefaultRotator::rotate_daily_quest`] of the "no
    /// priority action" tick it needs to ever start navigating.
    WhileActive,
}

#[derive(Debug, Default)]
struct PriorityActionQueueInfo {
    /// Whether this action is being ignored.
    ///
    /// While ignored, [`Self::last_queued_time`] will be updated to [`Instant::now`].
    /// The action is ignored for as long as it is still in the queue or the player
    /// is still executing it.
    ignoring: bool,
    /// The last [`Instant`] when this action was queued
    last_queued_time: Option<Instant>,
}

/// Action metadata to help identifying action type.
#[derive(Debug, Copy, Clone)]
enum ActionMetadata {
    UseBooster,
    Buff { kind: BuffKind },
}

/// The action that will be passed to the player.
///
/// There are [`RotatorAction::Single`] and [`RotatorAction::Linked`] actions.
/// With [`RotatorAction::Linked`] action is a linked list of actions. [`RotatorAction::Linked`]
/// action is executed in order, until completion and cannot be replaced by any other
/// type of actions.
#[derive(Clone, Debug)]
enum RotatorAction {
    Single(PlayerAction),
    Linked(LinkedAction),
}

/// A linked list of actions.
#[derive(Clone, Debug)]
struct LinkedAction {
    inner: PlayerAction,
    next: Option<Box<LinkedAction>>,
}

/// The rotator's rotation mode.
#[derive(Default, Debug, Clone, Copy)]
pub enum RotatorMode {
    StartToEnd,
    #[default]
    StartToEndThenReverse,
    AutoMobbing(MobbingKey, Bound),
    PingPong(MobbingKey, Bound),
    Idle,
}

#[derive(Debug, Default, Clone)]
pub struct RotatorBuildArgs {
    pub mode: RotatorMode,
    pub character_level: u32,
    pub character_actions: Vec<Action>,
    pub map_actions: Vec<Action>,
    pub buffs: Vec<(BuffKind, KeyKind)>,
    pub familiars: Familiars,
    pub familiars_fusion: FamiliarsFusion,
    pub familiars_potential: FamiliarsPotential,
    pub familiar_essence_key: KeyKind,
    pub elite_boss_behavior: EliteBossBehavior,
    pub elite_boss_behavior_key: KeyKind,
    pub hexa_booster_exchange_condition: ExchangeHexaBoosterCondition,
    pub hexa_booster_exchange_amount: u32,
    pub hexa_booster_exchange_all: bool,
    pub enable_panic_mode: bool,
    pub enable_rune_solving: bool,
    pub enable_transparent_shape_solving: bool,
    pub enable_violetta_solving: bool,
    pub enable_reset_normal_actions_on_erda: bool,
    pub enable_using_generic_booster: bool,
    pub enable_using_hexa_booster: bool,
    /// Daily quest hunting-ground checklist to run before the map's normal [`RotatorMode`].
    ///
    /// See [`DefaultRotator::rotate_daily_quest`].
    pub daily_quest_entries: Vec<DailyQuestEntry>,
    /// Mobbing key shared by every daily quest entry.
    pub daily_quest_mobbing_key: MobbingKey,
    /// The id of the character [`Self::daily_quest_entries`] belongs to, if saved.
    ///
    /// See [`DefaultRotator::daily_quest_character_id`].
    pub daily_quest_character_id: Option<i64>,
    /// See [`crate::models::Character::daily_quest_use_key_while_double_jumping`].
    pub daily_quest_use_key_while_double_jumping: bool,
    /// See [`crate::models::Character::monster_park_enabled`].
    ///
    /// Pre-empts [`Self::mode`] the same way [`Self::daily_quest_entries`] does, whenever it and
    /// they aren't both simultaneously wanting to run - see [`DefaultRotator::rotate_action`].
    pub monster_park_enabled: bool,
    /// See [`crate::models::Character::thread_of_fate_ask_enabled`].
    pub thread_of_fate_ask_enabled: bool,
    /// See [`crate::models::Character::thread_of_fate_ask_npc`].
    pub thread_of_fate_ask_npc: ThreadOfFateNpc,
    /// See [`crate::models::Character::thread_of_fate_ask_progress`].
    pub thread_of_fate_ask_progress: ThreadOfFateAskProgress,
    /// The id of the character [`Self::thread_of_fate_ask_progress`] belongs to, if saved.
    ///
    /// Mirrors [`Self::daily_quest_character_id`] - used to persist progress against the right
    /// character even if the UI's selected character changes mid-run.
    pub thread_of_fate_ask_character_id: Option<i64>,
}

/// Handles rotating provided [`PlayerAction`]s.
#[cfg_attr(test, automock)]
pub trait Rotator: Debug + 'static {
    #[cfg_attr(test, concretize)]
    fn build_actions(&mut self, args: RotatorBuildArgs);

    /// Whether a daily quest entry is currently being travelled to (see
    /// [`DefaultRotator::daily_quest_navigating`], [`DefaultRotator::daily_quest_arrived`]).
    ///
    /// Used by the world event handler to exempt the map changes this causes (opening the world
    /// map covers the minimap; teleporting actually changes it) from the "unexpected map change"
    /// watchdog, the same way Monster Park is exempted for its own portal-driven map changes (see
    /// [`Self::is_monster_park_active`]). `false` again once the entry's hunting ground is
    /// actually reached and mobbing starts, so a map change during mobbing (e.g. failing a lie
    /// detector check) still trips the watchdog.
    fn is_navigating_daily_quest(&self) -> bool;

    /// Whether Monster Park farming is currently armed (see
    /// [`crate::models::Character::monster_park_enabled`]).
    ///
    /// Used by the world event handler to exempt the frequent portal-driven map changes a run
    /// causes from the "unexpected map change" watchdog, the same way
    /// [`Self::is_navigating_daily_quest`] is exempted for daily quest navigation.
    fn is_monster_park_active(&self) -> bool;

    /// Whether [`RotatorMode::Idle`] is the currently active mode.
    ///
    /// Idle stands in place and reacts to nothing about other players - panic mode is force
    /// disabled for it (see [`DefaultRotator::build_actions`]) - so this is also used to suppress
    /// the Guildie/Stranger/Friend "appear" webhook notifications, which would otherwise still
    /// fire off [`crate::minimap`]'s detection alone even though the bot cannot and will not act
    /// on it.
    fn is_idle_mode_active(&self) -> bool;

    /// Resets priority and normal actions queues.
    ///
    /// This does not remove previously built actions.
    fn reset_queue(&mut self);

    /// Clears the Thread of Fate ask "blocked by daily reset" guard (see
    /// [`ThreadOfFateAskProgress::blocked_by_reset`]).
    ///
    /// Called when the bot (re)starts running after being halted - the desync the guard protects
    /// against is known to clear up once the bot has been stopped and started again.
    fn clear_thread_of_fate_ask_reset_block(&mut self);

    /// Injects an action to be executed.
    ///
    /// This can be useful for one-time action that needs to be run in response to some external
    /// event (e.g. chat). But should work co-operatively with previously built actions instead of
    /// directly overwriting through [`PlayerState::set_priority_action`].
    fn inject_action(&mut self, action: PlayerAction);

    /// Rotates actions previously built with [`Self::build_actions`].
    ///
    /// If [`Operation`] is currently halting, it does not rotate the built actions but only the
    /// side-loaded actions added by [`Self::inject_action`].
    fn rotate_action(&mut self, resources: &mut Resources, world: &mut World);
}

/// Snapshot of platform-pathing state overwritten while a daily quest run is applying its own
/// hunting ground's platforms - see [`DefaultRotator::daily_quest_saved_pathing`].
#[derive(Clone, Debug)]
struct DailyQuestSavedPathing {
    platforms: Vec<Platform>,
    auto_mob_platforms_pathing: bool,
    auto_mob_platforms_pathing_up_jump_only: bool,
    auto_mob_use_key_when_pathing: bool,
    auto_mob_use_key_when_pathing_update_millis: u64,
    auto_mob_use_key_while_double_jumping: bool,
    rune_platforms_pathing: bool,
    rune_platforms_pathing_up_jump_only: bool,
}

/// A step of [`DefaultRotator::rotate_monster_park`]'s portal-stuck "wiggle" recovery - see
/// [`DefaultRotator::monster_park_portal_wiggled`]'s docs for what it's for.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PortalWiggleStep {
    Left,
    Right,
    Up,
}

/// Number of ticks [`DefaultRotator::advance_monster_park_portal_escape`] holds the direction key
/// down for before releasing it. See [`DefaultRotator::monster_park_portal_escape`]'s docs.
const PORTAL_ESCAPE_HOLD_TICKS: u32 = 20;
/// The tick (counting down from [`PORTAL_ESCAPE_HOLD_TICKS`]) at which
/// [`DefaultRotator::advance_monster_park_portal_escape`] presses the jump key - roughly midway
/// through the hold, so the player is already moving before jumping instead of jumping in place
/// first.
const PORTAL_ESCAPE_JUMP_AT_TICKS_REMAINING: u32 = PORTAL_ESCAPE_HOLD_TICKS / 2;

/// See [`DefaultRotator::monster_park_sweep`].
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
enum MonsterParkSweep {
    /// No run entered yet.
    #[default]
    NotEntered,
    /// `Enter` was clicked with `Sweeps` maxed out and the dialog closed, but the player hasn't
    /// been confirmed outside the entry lobby yet (e.g. still loading in).
    Entered,
    /// Confirmed outside the entry lobby after entering - the run is underway, and being back in
    /// the lobby means it's over.
    RunStarted,
    /// Entered and finished, or the dialog stayed open after `Enter` (no free clears left).
    Finished,
}

#[derive(Default, Debug)]
pub struct DefaultRotator {
    normal_actions: Vec<(u32, RotatorAction)>,
    normal_queuing_linked_action: Option<(u32, Box<LinkedAction>)>,
    normal_index: usize,
    /// Whether [`Self::normal_actions`] is being accessed from the end
    normal_actions_backward: bool,
    normal_actions_reset_on_erda: bool,
    normal_rotate_mode: RotatorMode,
    /// The character's configured level, used by [`DefaultRotator::rotate_monster_park_entry`]
    /// to decide which gate to use.
    character_level: u32,
    /// See [`RotatorBuildArgs::monster_park_enabled`].
    monster_park_enabled: bool,
    /// See [`RotatorBuildArgs::thread_of_fate_ask_character_id`].
    thread_of_fate_ask_character_id: Option<i64>,
    /// Live, shared copy of [`RotatorBuildArgs::thread_of_fate_ask_progress`], reseeded from it on
    /// every [`Self::build_actions`] call.
    ///
    /// Shared (via [`Arc<Mutex<_>>`]) with the priority action's own condition closure and, once
    /// queued, with the resulting [`crate::player::ThreadOfFateAsk`]/
    /// [`crate::player::ThreadOfFateAskComplete`] action - so a successful accept/complete is
    /// immediately visible to later scheduling decisions without waiting for actions to be
    /// rebuilt, mirroring how [`crate::ecs::CharacterUpdates`] shares state between deep tick code
    /// and the outer service layer.
    thread_of_fate_ask_progress: Arc<Mutex<ThreadOfFateAskProgress>>,

    /// The [`Task`] used when [`Self::normal_rotate_mode`] is [`RotatorMode::AutoMobbing`]
    auto_mob_task: Option<Task<Result<Vec<Point>>>>,
    /// Tracks number of times a mob detection has been completed inside the same quad.
    ///
    /// This limits the number of detections can be done inside the same quad as to help player
    /// advances to the next quad.
    auto_mob_quadrant_consecutive_count: Option<(Quadrant, u32)>,
    /// Tracks the number of consecutive ticks [`DefaultRotator::rotate_monster_park`] detected no
    /// enemies on the minimap.
    ///
    /// A single tick's detection can miss enemies that are actually still there (e.g. transient
    /// capture noise, dots overlapping), so this requires several consecutive empty detections
    /// before concluding the map is actually clear and it's time to head to the portal. Resets to
    /// 0 as soon as any enemy is detected again.
    monster_park_no_enemy_count: u32,
    // (Formerly `monster_park_stage_cleared_confirmed`/`monster_park_stage_cleared_wait_since`:
    // gating "safe to approach the portal" on the game's CLEAR/run-complete banner. Removed -
    // that banner was observed matching falsely, presumably from visual similarity to the
    // stage-start "STAGE N" banner shown at the very same moment a fresh stage begins, which
    // could latch a stage as "cleared" before a single monster had been fought. Replaced by
    // pressing Up at the portal immediately once the no-enemy-dot debounce below is satisfied,
    // with an elite-boss-HP-bar check (see `is_at_portal`'s handling in
    // `DefaultRotator::rotate_monster_park`, checked before ever pressing Up) as the backstop
    // against the actual scenario the banner check existed for: a boss standing on the portal,
    // alive but occluded from the dot scan. That check needs no extra state of its own - it
    // reuses `monster_park_portal_attempts`.)
    /// Background task continuously re-scanning the minimap for the Monster Park portal, until a
    /// position is actually found.
    ///
    /// The portal's minimap icon is present from the very start of a stage - well before all
    /// enemies are cleared - so this runs every tick right alongside
    /// [`Self::monster_park_enemies_task`] instead of only once the map looks clear, to catch a
    /// position as early as possible (see [`Self::monster_park_last_portal`]'s docs for why that
    /// matters). Unlike the enemies task, this stops being polled entirely as soon as
    /// [`Self::monster_park_last_portal`] is `Some` - the portal doesn't move mid-stage, so there
    /// is nothing left to gain from paying for this non-trivial-cost detection call every tick for
    /// however much of the stage remains (which can be several minutes), and resumes only once a
    /// genuine stage transition clears that field again.
    ///
    /// Portal position needs to be re-scanned every tick (while still unknown) rather than through
    /// the sticky `idle.portals()` cache (see [`DefaultRotator::rotate_monster_park`] for why), but
    /// that detection is a synchronous, non-trivial-cost call - running it inline on the tick
    /// thread stretches out that tick's wall-clock time, which is exactly what was found to break
    /// [`Self::monster_park_enemies_task`]'s tight-timing movement (up jump combos) before it was
    /// moved to a background task. The player approaching the portal relies on the same kind of
    /// tight timing (short-adjust's fixed-duration key taps, see `Adjusting::update_adjusting`),
    /// so this is scanned the same way for the same reason.
    monster_park_portal_task: Option<Task<Result<Vec<Rect>>>>,
    /// The last successfully detected portal Rect from [`Self::monster_park_portal_task`].
    ///
    /// A mob dot can sit on top of or right next to the portal icon early in a stage, and once
    /// the player is standing on/near it their own minimap marker can occlude it too - either way
    /// a scan can briefly come up empty right when a position is needed. Scanning continuously
    /// from the very start of a stage (rather than waiting until no enemies are left) means the
    /// portal only has to be seen clearly once, at any point during the whole stage, for this to
    /// have a usable position; falling back to that instead of demanding a fresh hit every tick
    /// tolerates both kinds of occlusion without reintroducing staleness across a real map change.
    ///
    /// Cleared only by [`Self::reset_monster_park_run_state`] on a genuine stage transition (see
    /// the `monster_park_stage_bbox` check in [`DefaultRotator::rotate_monster_park`]) - *not* on
    /// a fresh enemy target being found, since the portal's position doesn't move mid-stage and
    /// clearing it there previously reintroduced the exact occlusion deadlock this field exists to
    /// avoid (see [`Self::monster_park_no_portal_stuck_since`] for what that looked like).
    monster_park_last_portal: Option<Rect>,
    /// The [`Instant`] [`DefaultRotator::rotate_monster_park`] first found the map clear of
    /// enemies but with [`Self::monster_park_last_portal`] still `None` (the portal has never
    /// been detected even once), since the last genuine progress.
    ///
    /// Guards a deadlock [`Self::monster_park_last_portal`]'s own occlusion fallback can't help
    /// with: if the player's minimap marker happens to be sitting right on top of where the
    /// portal icon would be the very first time this state is reached, there is no earlier
    /// successful detection to fall back on, so every scan comes up empty - and nothing else in
    /// that branch moves the player, so the occlusion never clears on its own. Past
    /// [`MONSTER_PARK_NO_PORTAL_STUCK_TIMEOUT`], nudge the player via the same wiggle tap
    /// sequence used for a portal-stuck episode (see [`Self::monster_park_portal_wiggle`]) to
    /// clear the occlusion and give the next scan a real chance to see through.
    ///
    /// Reset to `None` by [`Self::reset_monster_park_run_state`] (a real map transition) and as
    /// soon as a portal is actually detected (genuine progress - the whole point of this timer).
    monster_park_no_portal_stuck_since: Option<Instant>,
    /// Number of consecutive times [`Self::monster_park_no_portal_stuck_since`]'s timeout has
    /// fired without a portal ever actually turning up afterward.
    ///
    /// The wiggle above is a cheap first resort for the minimap-occlusion theory, but if it keeps
    /// timing out anyway, occlusion probably isn't the real problem - something else is blocking
    /// (same reasoning as [`Self::monster_park_portal_give_up_cycles`]). Past
    /// [`MONSTER_PARK_NO_PORTAL_WIGGLE_LIMIT`], escalate to [`Self::monster_park_portal_escape`]
    /// instead of wiggling forever. Reset alongside [`Self::monster_park_no_portal_stuck_since`].
    monster_park_no_portal_wiggle_count: u32,
    /// Tracks the number of consecutive times [`DefaultRotator::rotate_monster_park`] has pressed
    /// Up while standing at the portal without the map actually changing.
    ///
    /// A real map transition always resets Monster Park's state via [`Self::reset_queue`] (see
    /// [`Self::monster_park_no_enemy_count`]), so if this keeps climbing it means the last Up
    /// press genuinely isn't advancing the map - most likely because the "no enemies left" call
    /// was wrong (e.g. the player's own minimap marker was occluding a still-alive enemy's dot)
    /// rather than an actual portal-usage failure. Past [`MONSTER_PARK_PORTAL_ATTEMPT_LIMIT`],
    /// give up pressing Up and go back to re-scanning for enemies instead of retrying forever.
    monster_park_portal_attempts: u32,
    /// Tracks the number of consecutive times [`Self::monster_park_portal_attempts`] has hit its
    /// give-up limit without any genuine progress happening in between (a new enemy target
    /// acquired, the player moving off the portal, or an actual map transition - all of which
    /// reset this back to 0, same as [`Self::monster_park_portal_attempts`] itself).
    ///
    /// If giving up and re-scanning keeps leading right back to the same stuck-at-portal
    /// situation, retrying that cycle forever isn't going to help - most likely the player is
    /// genuinely wedged. Past [`MONSTER_PARK_PORTAL_GIVE_UP_CYCLE_LIMIT`], escalate via
    /// [`Self::escalate_monster_park_portal_stuck`] instead of looping between "give up" and
    /// "immediately re-find the same portal" indefinitely.
    monster_park_portal_give_up_cycles: u32,
    /// The `target_y` last computed for the portal-approach `Move` (see
    /// [`DefaultRotator::rotate_monster_park`]'s `same_platform` logic), together with how many
    /// consecutive times that Move has been reissued targeting that exact same height without
    /// ever reaching the portal.
    ///
    /// `same_platform` freezes `target_y` to the player's current height instead of the portal's
    /// real detected height, on the assumption the player is already on the right platform - if
    /// that assumption is wrong (the position read when making the decision doesn't hold up once
    /// `Player::Moving` re-evaluates a moment later, e.g. a double jump ends up firing after all),
    /// the frozen height can be one the player physically can't reach, and re-deriving `target_y`
    /// the same way on every retry reproduces the identical wrong value forever. Once the count
    /// exceeds [`MONSTER_PARK_PORTAL_TARGET_Y_REPEAT_LIMIT`], `target_y` is forced to the portal's
    /// real detected height instead, guaranteeing eventual convergence regardless of why the
    /// frozen value was wrong. Reset to `(portal_center_y, 0)` whenever `target_y` actually changes
    /// between reissues (genuine progress) and cleared by [`Self::reset_monster_park_run_state`].
    monster_park_portal_target_y: Option<(i32, u32)>,
    /// Set by [`Self::escalate_monster_park_portal_stuck`] to force the *next* portal-approach
    /// `Move` to target the portal's real detected height, bypassing `same_platform` entirely -
    /// consumed (reset to `false`) the instant that Move is computed, so it only overrides once
    /// per escalation rather than permanently.
    ///
    /// Exists because `same_platform` and `is_at_portal` (see [`DefaultRotator::rotate_monster_park`])
    /// share the same vertical tolerance (`PORTAL_Y_THRESHOLD`) for two different questions - "is
    /// the player already on the portal's platform" vs. "is the player close enough to use the
    /// portal now". When the real gap between the player and the portal happens to sit just under
    /// that tolerance (a genuine one-platform difference, not detection noise), `same_platform`
    /// wrongly freezes `target_y` at the player's current, too-low height, and `is_at_portal` then
    /// *also* reads true at that same wrong height - so the player never leaves the "at portal,
    /// pressing Up" branch long enough to ever revisit the height decision.
    /// [`Self::monster_park_portal_target_y`]'s own repeat-limit escape hatch lives entirely in
    /// the "not at portal" `Move` branch and can't fire from inside that deadlock - observed live
    /// as an over-7-minute stuck episode: wiggle and escape both nudge the player just enough to
    /// eventually clear `is_at_portal`, but neither actually climbs, so the player lands right back
    /// on the same too-low platform and the cycle repeats. Setting this flag whenever an escalation
    /// fires guarantees the very next `Move` targets the real height instead of re-freezing the
    /// same wrong one - if a genuine gap exists, `Player::Moving`'s own jump logic then actually
    /// climbs it; if there wasn't one, targeting the real height is a no-op. Cleared alongside the
    /// other portal-stuck counters whenever a fresh enemy target is acquired (the old stuck episode
    /// is moot) and by [`Self::reset_monster_park_run_state`] on a genuine stage transition.
    monster_park_portal_force_real_height: bool,
    /// The [`Instant`] [`Self::rotate_monster_park`] first found a portal-approach `Move` action
    /// in flight (`player_context.has_normal_action()` true, [`Self::monster_park_target`] `None`)
    /// since the last genuine progress.
    ///
    /// `same_platform` (see [`Self::monster_park_portal_target_y`]'s docs) can freeze `target_y`
    /// to a height the player physically can't reach. Once that `Move` is dispatched,
    /// [`Self::rotate_monster_park`] does not run again until the player finishes or gives up on
    /// it (same as [`Self::monster_park_target_stuck_since`]'s docs) - but giving up on a `Move`
    /// built from a wrong frozen height just means [`Player::Moving`]'s own repeated-state
    /// recovery (`abort_action_on_state_repeat`) keeps firing without ever clearing the action, so
    /// the exact same wrong `Move` gets reissued forever and this function never gets a chance to
    /// recompute it. This is the wall-clock backstop: past
    /// [`MONSTER_PARK_PORTAL_APPROACH_STUCK_TIMEOUT`] without reaching the portal, give up on the
    /// in-flight `Move` via [`PlayerContext::reset_normal_action`] regardless, letting the next
    /// tick recompute a fresh (and, via [`Self::monster_park_portal_target_y`]'s own repeat limit,
    /// eventually correct) target.
    ///
    /// Reset to `None` by [`Self::reset_monster_park_run_state`] and whenever a fresh
    /// portal-approach `Move` is actually (re)dispatched (genuine progress - a new attempt gets a
    /// fresh budget).
    monster_park_portal_approach_stuck_since: Option<Instant>,
    /// The [`Instant`] [`DefaultRotator::rotate_monster_park`] first found the player at the
    /// portal (`is_at_portal`) since the last genuine progress.
    ///
    /// The attempt/give-up-cycle counters above ([`Self::monster_park_portal_attempts`],
    /// [`Self::monster_park_portal_give_up_cycles`]) are reset by the mere `is_at_portal` check
    /// reading false for a single tick - which pixel-level detection noise on the tight
    /// `PORTAL_ARRIVED_X_THRESHOLD` can cause without the player having moved anywhere - so a run
    /// of such flicker can keep those counters from ever reaching their limit and escalating,
    /// leaving the player pressing Up at the portal indefinitely. This field is a wall-clock
    /// backstop that isn't reset by that flicker: it starts counting the instant the player is
    /// first seen at the portal and, unlike the counters above, keeps counting across it. Past
    /// [`MONSTER_PARK_PORTAL_STUCK_TIMEOUT`], escalate via [`Self::escalate_monster_park_portal_stuck`]
    /// regardless of what the attempt counters say.
    ///
    /// Reset to `None` (i.e. genuine progress) by [`Self::reset_monster_park_run_state`] (a real
    /// map transition) and whenever a fresh enemy target is actually committed to (see the
    /// `monster_park_pending_target` debounce in [`Self::rotate_monster_park`]) - both mean
    /// whatever was happening at the portal is no longer relevant. Deliberately *not* reset by
    /// `is_at_portal` reading false on its own, for the flicker reason above.
    monster_park_portal_stuck_since: Option<Instant>,
    /// In-progress step of the portal-stuck "wiggle" recovery, as `(step, ticks remaining before
    /// that step's key is sent)`. See [`Self::monster_park_portal_wiggled`] for when/why this
    /// gets started, and [`Self::advance_monster_park_portal_wiggle`] for how it's driven forward.
    ///
    /// `None` whenever no wiggle is in progress. While `Some`,
    /// [`Self::rotate_monster_park`] does nothing else - the wiggle deliberately moves the player
    /// off `is_at_portal` for a moment (that's the point), so evaluating any of the normal
    /// portal/enemy logic mid-wiggle would just fight it.
    monster_park_portal_wiggle: Option<(PortalWiggleStep, u32)>,
    /// The key ([`KeyKind::Left`] or [`KeyKind::Right`]) that actually closes the horizontal gap
    /// for the current portal-stuck episode, if known - used by
    /// [`Self::advance_monster_park_portal_wiggle`] to bias [`PortalWiggleStep::Right`]'s tap
    /// (the last lateral one before Up) toward whichever side resolves the misalignment, instead
    /// of a fixed direction.
    ///
    /// Set by [`Self::escalate_monster_park_portal_stuck`] from `is_at_portal`'s own known offset
    /// sign - the player being stuck there is often exactly one pixel outside
    /// `PORTAL_ARRIVED_X_THRESHOLD`'s tolerance, and a symmetric Left-then-Right tap sequence can
    /// cancel itself back out to that same pixel every time, never actually landing on the exact
    /// spot Up needs. `None` for the "portal not found" recovery instead, which has no such known
    /// direction to bias towards, so the wiggle there keeps its original fixed Left-then-Right
    /// order (see the call site).
    monster_park_portal_wiggle_toward_key: Option<KeyKind>,
    /// Whether the "wiggle" recovery (tap Left, then Right, then Up - see
    /// [`Self::monster_park_portal_wiggle`]) has already been tried once for the current
    /// portal-stuck episode.
    ///
    /// A cheaper first resort than [`Self::monster_park_portal_escape`], which actually walks the
    /// player away from the portal and costs time getting back to it, in case whatever's blocking
    /// Up from registering (a settled-but-just-barely-off pixel position, a stuck input state)
    /// clears on its own from the nudge. Consulted by [`Self::escalate_monster_park_portal_stuck`]:
    /// `false` starts the wiggle instead of escaping outright; on completion (see
    /// [`Self::advance_monster_park_portal_wiggle`]) the normal Up-press retry counters/timer are
    /// reset back to fresh, giving the wiggle a real chance to have actually fixed things before
    /// concluding it didn't. This flag itself is *not* reset by that, so if the fresh budget is
    /// exhausted too, the second escalation goes straight to the escape instead of wiggling
    /// forever. Reset to `false` alongside everything else in
    /// [`Self::reset_monster_park_run_state`] and whenever a fresh enemy target is committed to -
    /// both mean genuine progress happened, unrelated to whatever the wiggle was trying to fix.
    monster_park_portal_wiggled: bool,
    /// In-progress step of the portal-stuck "escape" recovery, as `(direction key held, ticks
    /// remaining)`. Started by [`Self::escalate_monster_park_portal_stuck`] once the cheaper
    /// [`Self::monster_park_portal_wiggle`] has already been tried and the player is still stuck,
    /// and by the "portal not found" recovery once its own wiggle budget
    /// (`MONSTER_PARK_NO_PORTAL_WIGGLE_LIMIT`) is exhausted.
    ///
    /// A bespoke, raw-input recovery like the wiggle above - sent directly via [`Resources::input`]
    /// rather than through [`PlayerAction::Unstuck`], which only presses Esc to dismiss a blocking
    /// dialog ([`Player::Unstucking`]'s `Esc` kind) and does nothing when the player is genuinely
    /// wedged with no dialog on screen (the actual Monster Park case this exists for - a real
    /// settings/channel-change dialog is already handled independently by the always-on
    /// `unstuck_priority_action`). Unlike the wiggle's brief taps, this holds a direction for
    /// longer and presses the jump key partway through, to force real displacement away from
    /// whatever position the player is wedged at.
    ///
    /// `None` whenever no escape is in progress. While `Some`, [`Self::rotate_monster_park`] does
    /// nothing else, same reasoning as [`Self::monster_park_portal_wiggle`].
    monster_park_portal_escape: Option<(KeyKind, u32)>,
    /// The position of the enemy dot [`DefaultRotator::rotate_monster_park`] last targeted with
    /// [`PlayerAction::AutoMob`].
    ///
    /// Once a normal action is set, the rotator does not run again until the player finishes or
    /// gives up on it - so without tracking this separately, a mob that dies mid-navigation (e.g.
    /// killed by another hit or a party member) is never noticed and the player keeps trying to
    /// reach/attack a position with nothing there until the movement's own repeated-state abort
    /// eventually fires, which can take a while and does not clear the action anyway. Comparing
    /// fresh (see [`Self::monster_park_enemies`]) detections against this lets the action be
    /// abandoned as soon as the mob is confirmed gone. Cleared alongside
    /// [`Self::monster_park_no_enemy_count`].
    monster_park_target: Option<Point>,
    /// Tracks the number of consecutive checks the enemy at [`Self::monster_park_target`] was not
    /// found in a fresh detection.
    ///
    /// Debounced for the same reason as [`Self::monster_park_no_enemy_count`] - a single check's
    /// miss can be transient capture noise or dot overlap rather than the mob actually being dead.
    monster_park_target_missing_count: u32,
    /// The [`Instant`] [`Self::rotate_monster_park`] started pursuing the current
    /// [`Self::monster_park_target`] since it was last (re)committed.
    ///
    /// An enemy dot that's still detected every scan is never caught by
    /// [`Self::monster_park_target_missing_count`] above - correct for a genuinely alive,
    /// reachable enemy, but some dots turn out to be unreachable by normal movement (e.g. sitting
    /// on a disconnected platform). In that case the player's own movement state machine retries
    /// the same unreachable position forever: its repeated-state recovery
    /// (`Player::Unstucking`) resets the movement machinery but never clears the action itself
    /// (see [`Self::monster_park_target`]'s docs), so nothing above would ever notice. This is a
    /// wall-clock backstop against exactly that, observed lasting several minutes in practice -
    /// past [`MONSTER_PARK_TARGET_STUCK_TIMEOUT`] pursuing the same committed target without
    /// reaching it, give up regardless of the dot still being detected.
    ///
    /// Reset to `None` whenever a target is freshly committed to or redirected (genuine
    /// progress - see the acquisition and drift-redirect sites in [`Self::rotate_monster_park`]),
    /// whenever [`Self::monster_park_target`] is cleared for any reason, and by
    /// [`Self::reset_monster_park_run_state`].
    monster_park_target_stuck_since: Option<Instant>,
    /// Enemy dot positions [`Self::rotate_monster_park`] gave up pursuing this stage because
    /// [`Self::monster_park_target_stuck_since`] exceeded [`MONSTER_PARK_TARGET_STUCK_TIMEOUT`]
    /// without ever reaching them, matched with the same tolerance as
    /// `MONSTER_PARK_TARGET_MATCH_THRESHOLD`.
    ///
    /// Giving up on an unreachable dot without remembering it here would just make it immediately
    /// eligible to be re-picked as the nearest enemy again next tick - it never moves, never dies,
    /// never leaves [`Self::monster_park_enemies`] - so the player would retry-then-give-up in an
    /// endless cycle instead of moving on to the stage's other, reachable enemies. Excluded from
    /// candidate selection in [`Self::rotate_monster_park`]. Cleared alongside the rest of the
    /// run-scoped state in [`Self::reset_monster_park_run_state`] - a fresh stage may have its own
    /// layout where the same relative position is perfectly reachable.
    monster_park_unreachable_targets: Vec<Point>,
    /// A newly-spotted enemy dot not yet committed to as [`Self::monster_park_target`], along with
    /// how many consecutive scans it's been seen in.
    ///
    /// [`Self::monster_park_enemies_task`] refreshes far more often than the old throttled
    /// synchronous check did (as fast as the detector can go, not a few times a second), which
    /// makes a single-scan false positive (transient capture noise, aliasing) much more likely to
    /// be caught and immediately committed to before it disappears on the very next scan. Requires
    /// the same dot to reappear across a couple of scans before actually committing to chase it -
    /// mirrors [`Self::monster_park_target_missing_count`]'s reasoning but for acquiring a target
    /// instead of giving one up.
    monster_park_pending_target: Option<(Point, u32)>,
    /// Background task continuously re-scanning the minimap for Monster Park enemy dots.
    ///
    /// The detection itself is a synchronous, non-trivial-cost call - running it inline on the
    /// tick thread (even throttled) stretches out that tick's wall-clock time enough to interfere
    /// with movement that depends on tight key-press timing (e.g. an up jump combo needing its
    /// second key press to land within a short window). Running it via [`update_detection_task`]
    /// moves that cost onto a background thread entirely: the tick loop only ever polls a channel
    /// for whatever the latest completed scan found (essentially free), and a new scan is kicked
    /// off again immediately once the previous one lands (`repeat_delay_millis: 0` - mirrors
    /// [`Self::auto_mob_task`]'s use of the same pattern), so [`Self::monster_park_enemies`] stays
    /// as fresh as the detector can make it without ever stalling a tick.
    monster_park_enemies_task: Option<Task<Result<Vec<Rect>>>>,
    /// Enemy dot positions from the most recently completed [`Self::monster_park_enemies_task`]
    /// scan, already converted to the bottom-left, center-of-dot coordinate convention used
    /// throughout [`DefaultRotator::rotate_monster_park`].
    ///
    /// Left unchanged while a scan is still in flight or fails, so a single slow/failed scan
    /// doesn't blank out an otherwise-valid cache.
    monster_park_enemies: Vec<Point>,
    /// The minimap `bbox` last seen by [`DefaultRotator::rotate_monster_park`], used to detect a
    /// genuine stage transition (see the bbox comparison at the top of that function for the full
    /// reasoning). `None` whenever no run is in progress - reset alongside everything else in
    /// [`Self::reset_monster_park_run_state`].
    monster_park_stage_bbox: Option<Rect>,
    /// A hash of the "Stage N: <subtitle>" text shown directly above the minimap, last seen by
    /// [`DefaultRotator::rotate_monster_park`] - a second, independent signal for detecting a
    /// genuine stage transition, alongside [`Self::monster_park_stage_bbox`].
    ///
    /// `monster_park_stage_bbox`'s own doc admits it isn't fully bulletproof: two consecutive
    /// stages can occasionally share similar enough minimap chrome that the anchor-mismatch
    /// check behind it doesn't trip, leaving `bbox` unchanged even though the stage genuinely
    /// did change - which was observed leaving the player pressing Up at a previous, already-used
    /// portal position for over a minute before self-correcting. The stage indicator text always
    /// changes on a genuine transition (it names the specific stage/subtitle), so comparing its
    /// hash tick-to-tick catches exactly the case the bbox check can miss. `None` whenever no run
    /// is in progress or the region couldn't be read this tick - reset alongside everything else
    /// in [`Self::reset_monster_park_run_state`], left unchanged (not cleared) on a single failed
    /// read so transient detection noise doesn't spuriously look like "no signal yet".
    ///
    /// Only trusted as "changed" once [`Self::monster_park_stage_indicator_pending`] has confirmed
    /// the new value across several ticks in a row - see that field's docs for why a bare
    /// tick-to-tick difference isn't enough on its own to fire a reset.
    monster_park_stage_indicator_hash: Option<u64>,
    /// A candidate replacement for [`Self::monster_park_stage_indicator_hash`], as `(candidate
    /// hash, consecutive ticks seen)` - debounces the hash-changed signal before it's trusted
    /// enough to reset run state.
    ///
    /// The crop this hash is computed from was calibrated offline against reference screenshots
    /// and never confirmed against a live capture (see [`Self::monster_park_stage_indicator_hash`]'s
    /// docs on where it was disabled for exactly that reason) - if the calibration is off for a
    /// given resolution/UI scale, the crop can land partly on dynamic minimap content (moving
    /// dots, the player marker) instead of the static "Stage N" text, changing the hash almost
    /// every tick. Firing a reset on a single tick-to-tick difference would turn that
    /// miscalibration into a reset almost every tick - far more damaging than the stuck-portal bug
    /// this hash exists to catch (see [`DefaultRotator::rotate_monster_park`]'s bbox-comparison
    /// docs for that bug).
    ///
    /// Requiring the same candidate to reappear for several ticks running tells the two cases
    /// apart without needing offline calibration confirmation first: a miscalibrated crop sitting
    /// on genuinely dynamic content produces an essentially-random new value practically every
    /// tick, so it is exceedingly unlikely to repeat identically two ticks in a row, let alone
    /// `MONSTER_PARK_STAGE_INDICATOR_DEBOUNCE_COUNT` (a local const in
    /// [`DefaultRotator::rotate_monster_park`]) - while a real stage transition's text
    /// renders once and then stays fixed for the rest of the (multi-minute) stage, so it clears
    /// the same debounce within a couple of ticks. `None` whenever no candidate is currently being
    /// confirmed - reset alongside everything else in [`Self::reset_monster_park_run_state`], and
    /// whenever a freshly-read hash matches the already-confirmed
    /// [`Self::monster_park_stage_indicator_hash`] (nothing to confirm).
    monster_park_stage_indicator_pending: Option<(u64, u32)>,
    /// Whether the stage's own countdown timer (top-center HUD, "Time Left: M:SS") has been
    /// confirmed visible at least once since the last genuine stage transition.
    ///
    /// A fresh stage transition can leave a brief window where the minimap already reads `Idle`
    /// with a perfectly plausible `bbox` - passing the checks that gate the rest of
    /// [`DefaultRotator::rotate_monster_park`] - before the stage has actually finished loading in
    /// (its monsters rendered, its own timer started). Enemy scans during that window
    /// legitimately read empty, and without this gate that reads exactly like "the stage is
    /// already clear" to the no-enemy-debounce logic further down - the game's CLEAR banner was
    /// previously used here too and observed matching falsely (presumably from visual similarity
    /// to the stage-start "STAGE N" banner shown at the very same moment), latching a stage as
    /// cleared before a single monster had been fought and leaving the player idle indefinitely
    /// (no enemies to chase, no portal found yet either) - which is also why that banner is no
    /// longer trusted for this decision at all (see the no-enemy handling in
    /// [`DefaultRotator::rotate_monster_park`]). Requiring the timer to have been seen first
    /// closes the loading-window version of the same problem:
    /// unlike the minimap `bbox`/enemy dots, the timer only ever appears once the stage has
    /// genuinely, fully started.
    ///
    /// Deliberately sticky (checked once per stage, not re-verified every tick) rather than
    /// gating on live visibility continuously - the timer is a persistent HUD element for the
    /// stage's entire duration, so a later tick where it's transiently occluded (e.g. by a dialog
    /// or effect) should not itself pause an already-legitimately-running stage. Reset to `false`
    /// alongside everything else in [`Self::reset_monster_park_run_state`].
    monster_park_timer_seen: bool,
    /// The [`Instant`] [`DefaultRotator::rotate_monster_park`] first started waiting on
    /// [`Self::monster_park_timer_seen`] for the current stage.
    ///
    /// Every other "wait for X" mechanism in Monster Park has a bounded give-up timeout - this
    /// needs one too, for the same reason: if the timer is never actually going to appear (a
    /// detection mismatch, or the player genuinely stuck short of a real stage, e.g. the
    /// dungeon-select dialog silently failing to advance), waiting on it forever with no escape
    /// is a strictly worse outcome than the premature-clear false positive it exists to prevent.
    /// Past [`MONSTER_PARK_TIMER_WAIT_TIMEOUT`], proceed as if the timer had been seen rather than
    /// stall indefinitely. Reset to `None` alongside everything else in
    /// [`Self::reset_monster_park_run_state`], and as soon as the timer is actually seen (genuine
    /// progress - the whole point of this timer).
    monster_park_timer_wait_since: Option<Instant>,
    /// Tracks the number of consecutive times [`DefaultRotator::rotate_monster_park_entry`] has
    /// attempted to hand off to [`PlayerAction::EnterMonsterPark`] from the gate without ever
    /// actually leaving the entry lobby (e.g. no free entry left for the day, so the dungeon-select
    /// dialog stays open / the Enter click does nothing).
    ///
    /// Reset back to 0 as soon as a fresh, synchronous check confirms the player has genuinely
    /// left the lobby (see the Monster Park dispatch in [`Self::rotate_action`]), so
    /// this only ever measures a truly unbroken streak of failed attempts, not attempts spread
    /// across otherwise-successful runs. Past [`MONSTER_PARK_ENTRY_ATTEMPT_LIMIT`], stop retrying,
    /// leave via [`PlayerAction::ExitMonsterPark`]'s shuttle instead of clicking through the same
    /// failing dialog, and disable Monster Park for the rest of the session.
    monster_park_entry_attempts: u32,
    /// Set when [`Self::rotate_monster_park_entry`] gives up on entering after
    /// [`MONSTER_PARK_ENTRY_ATTEMPT_LIMIT`] and injects [`PlayerAction::ExitMonsterPark`], cleared
    /// only once a fresh, synchronous `detect_monster_park_entry_map` check confirms the player has
    /// actually left the lobby.
    ///
    /// `monster_park_enabled` itself must stay `true` for that entire window - flipping it off at
    /// the same moment the exit is injected (rather than once it's confirmed to have worked) would
    /// disable both the retry dispatch in [`Self::rotate_action`] (gated on `monster_park_enabled`)
    /// and the live entry-lobby safety check that blocks daily quest/normal-mode navigation
    /// (`blocked_by_monster_park_run`, also gated on it) for the rest of the session - including for
    /// this exit attempt itself. Since `Player::ExitingMonsterPark`'s shuttle/dialog detection can
    /// silently give up without actually leaving (e.g. the YES button
    /// obscured by other players clustered around the shuttle - a known hazard, see
    /// `exit_monster_park.rs`'s docs), disabling early would let normal rotation (auto-mobbing,
    /// daily quest world-map navigation) start right there in the crowded shared lobby instead of
    /// back in town. Keeping `monster_park_enabled` true until departure is confirmed lets this
    /// flag drive a retry loop instead: while still detected at the lobby,
    /// [`Self::rotate_monster_park_entry`] keeps re-triggering the give-up branch (re-injecting the
    /// exit) every time it's reached again, rather than settling for one silent attempt.
    monster_park_giving_up: bool,
    /// Progress of today's single Monster Park run.
    ///
    /// The dungeon-select dialog's `Sweeps` -> `MAX` -> `Enter` starts one run whose completion
    /// uses up every remaining free clear, so once that run is over (back in the entry lobby after
    /// having been seen outside it) or the dialog reports no free clears left, there is nothing
    /// more to do and [`Self::rotate_monster_park_entry`] leaves via the shuttle right away instead
    /// of walking back to the gate.
    monster_park_sweep: MonsterParkSweep,
    /// Set alongside disabling `monster_park_enabled` once a give-up exit is confirmed (see
    /// [`Self::monster_park_giving_up`]'s docs) - distinguishes "Monster Park was armed and used
    /// up for the day" from `monster_park_enabled` simply never having been `true` this session
    /// (not configured at all), which reads identically on `monster_park_enabled` alone.
    ///
    /// Consulted by [`Self::rotate_action`]'s final fallback: once both Monster Park and daily
    /// quests are exhausted, there's no configured place left to send the player, and falling
    /// through to `normal_rotate_mode` runs it wherever the player happens to be standing right
    /// out of Monster Park's shuttle (a town, not wherever that mode's positions/bound were
    /// configured for) - halting instead avoids that instead of auto-mobbing blind among
    /// whatever other players happen to be there. Left `false` (so normal rotation behaves
    /// exactly as before) whenever Monster Park was never armed to begin with. Reset back to
    /// `false` on the next full rotator rebuild (`build_actions`), same as any other
    /// session/config restart.
    monster_park_exhausted: bool,
    /// Whether the player is currently believed to be at Monster Park's entry lobby (walking to
    /// the gate or driving the dungeon-select dialog via [`Player::EnteringMonsterPark`]), as
    /// opposed to actually sweeping a run's stages.
    ///
    /// Updated from the same fresh, synchronous `detect_monster_park_entry_map` check the
    /// Monster Park dispatch in [`Self::rotate_action`] already does to decide
    /// between [`Self::rotate_monster_park_entry`] and [`Self::rotate_monster_park`] - not
    /// re-detected here, since that call is a non-trivial-cost full-frame template match not
    /// worth paying for twice a tick. Consulted by [`Self::rotate_priority_actions`] to suppress
    /// map/character `EveryMillis` fixed actions only while genuinely in the lobby (where they'd
    /// reference a farming-map position the player isn't at) and let them run normally once a
    /// run's stages are actually being swept.
    monster_park_at_entry_lobby: bool,
    /// Whether a Monster Park run is currently underway - from the moment
    /// [`Self::rotate_monster_park_entry`] hands off to [`PlayerAction::EnterMonsterPark`] at the
    /// gate, through the dungeon-select dialog and every stage swept by [`Self::rotate_monster_park`],
    /// until [`Self::rotate_monster_park_entry`] freshly confirms the player is genuinely back at
    /// the entry lobby.
    ///
    /// Unlike [`Self::monster_park_at_entry_lobby`] (which only updates from the live
    /// `detect_monster_park_entry_map` check gated at the Monster Park dispatch in
    /// [`Self::rotate_action`], itself skipped whenever an action is in flight or - before this
    /// field existed - preempted entirely by daily quest navigation hijacking control the instant
    /// the player stepped past the lobby), this is set explicitly at the hand-off and read
    /// unconditionally by [`Self::rotate_action`]'s daily quest gate. Without it, the world map
    /// isn't reachable anywhere in a Monster Park run (dungeon-select dialog or an actual stage,
    /// same as the lobby itself), but the gate had no way to know that once the player was no
    /// longer literally standing in the lobby - so pending daily quests would immediately try to
    /// open the world map mid-run and fail every attempt.
    monster_park_run_in_progress: bool,

    /// Configured daily quest entries for [`Self::rotate_daily_quest`], from
    /// [`RotatorBuildArgs::daily_quest_entries`].
    ///
    /// Only assigned in [`Self::build_actions`] - unlike the other `daily_quest_*` fields below,
    /// this and [`Self::daily_quest_index`] represent actual checklist progress and must survive
    /// a pause/resume (see [`Self::reset_queue`]), not just in-flight per-tick tracking.
    daily_quest_entries: Vec<DailyQuestEntry>,
    /// Mobbing key shared by every daily quest entry, from
    /// [`RotatorBuildArgs::daily_quest_mobbing_key`].
    daily_quest_mobbing_key: MobbingKey,
    /// The id of the character [`Self::daily_quest_entries`] belongs to, from
    /// [`RotatorBuildArgs::daily_quest_character_id`].
    ///
    /// Threaded through to [`crate::ecs::CharacterUpdates::mark_daily_quest_completed`] so a
    /// completion is persisted against the character this run is actually for, not whichever
    /// character happens to be selected in `CharacterService` at the moment the tick loop drains
    /// it - those can diverge if the user switches the selected character in the UI while a
    /// previous character's daily quest run is still in flight.
    daily_quest_character_id: Option<i64>,
    /// See [`RotatorBuildArgs::daily_quest_use_key_while_double_jumping`].
    daily_quest_use_key_while_double_jumping: bool,
    /// Index into [`Self::daily_quest_entries`] of the currently active entry.
    ///
    /// Once this reaches `daily_quest_entries.len()`, all configured dailies for this run have
    /// been completed and [`Self::rotate_action`] falls through to the map's normal
    /// [`RotatorMode`] instead of calling [`Self::rotate_daily_quest`].
    daily_quest_index: usize,
    /// Whether [`PlayerAction::NavigateToHuntingGround`] has already been issued for the entry at
    /// [`Self::daily_quest_index`].
    ///
    /// Reset on [`Self::reset_queue`] (e.g. on pause) since a pause also aborts any in-flight
    /// navigation - unlike [`Self::daily_quest_index`], this is transient per-attempt tracking.
    daily_quest_navigating: bool,
    /// Number of consecutive times navigation to the entry at [`Self::daily_quest_index`] has
    /// failed (see [`Self::rotate_daily_quest`]'s handling of
    /// [`PlayerContext::take_daily_quest_navigate_failed`]).
    ///
    /// A single failure is often just transient (e.g. a network stutter dropping a frame the
    /// world map UI needed to render in time) rather than the hunting ground genuinely being
    /// unreachable, so failures are retried up to [`DAILY_QUEST_NAVIGATE_ATTEMPT_LIMIT`] times
    /// before giving up and skipping the entry for this run. Reset to `0` on a successful arrival
    /// and whenever the entry is finally skipped, as well as on [`Self::reset_queue`].
    daily_quest_navigate_attempts: u32,
    /// Whether the entry at [`Self::daily_quest_index`] has been reached and mobbing has started,
    /// i.e. [`Self::daily_quest_navigating`] is `true` but travel itself has finished.
    ///
    /// [`Rotator::is_navigating_daily_quest`] only exempts map changes while this is `false` - the
    /// world map/teleport-driven changes it exists to exempt only happen during travel itself.
    /// Once mobbing is underway, a map change (e.g. the player getting bounced to a different map
    /// after failing a lie detector check) is exactly the kind of unexpected event the "unexpected
    /// map change" watchdog in [`crate::services::world`] should still catch. Reset alongside
    /// [`Self::daily_quest_navigating`] on navigation failure, quest completion, and
    /// [`Self::reset_queue`].
    daily_quest_arrived: bool,
    /// Background task scanning for the "Quest complete!" toast confirmed as a daily quest (see
    /// [`Self::rotate_daily_quest`]), same reasoning as [`Self::monster_park_portal_task`] - run
    /// off-thread rather than called inline every tick so a match still pays for OCR without
    /// stalling everything else (movement, mob detection, minimap re-detection) along with it.
    daily_quest_complete_task: Option<Task<Result<()>>>,
    /// Whether [`Self::daily_quest_complete_task`] has ever matched the current entry.
    ///
    /// The toast only renders for a few seconds, so a miss must not clear a previous hit -
    /// latched `true` until [`Self::rotate_daily_quest`] resets it on completion, or navigation
    /// failure/pause resets it early.
    daily_quest_complete_popup_detected: bool,
    /// Whether [`Self::daily_quest_complete_task`] has reported a miss for the current entry since
    /// arriving at it.
    ///
    /// A hit is only trusted once this is `true` (see [`Self::rotate_daily_quest`]) - without it,
    /// the previous entry's own completion toast, still fading on screen through the teleport to
    /// this one, would immediately satisfy [`Self::daily_quest_complete_popup_detected`] here too.
    /// Reset alongside it.
    daily_quest_complete_popup_confirmed_absent: bool,
    /// The minimap platforms and [`PlayerContext::config`] platform-pathing settings as they were
    /// just before the first daily quest entry of this run applied its own (see
    /// [`DailyQuestId::platforms`](crate::models::DailyQuestId::platforms)).
    ///
    /// The daily quest solver has no [`crate::models::Map`] of its own to carry this
    /// configuration, so [`Self::rotate_daily_quest`] borrows the regular auto-mobbing plumbing
    /// directly - `Some` for the duration of a daily quest run and restored once
    /// [`Self::daily_quest_index`] reaches [`Self::daily_quest_entries`]'s length, so farming under
    /// the character's actual map isn't left running with a hunting ground's platforms afterward.
    daily_quest_saved_pathing: Option<DailyQuestSavedPathing>,

    priority_actions: OrderedHashMap<u32, PriorityAction>,
    /// The currently executing [`RotatorAction::Linked`] action
    priority_queuing_linked_action: Option<(u32, Box<LinkedAction>)>,
    /// A [`VecDeque`] of [`PriorityAction`] ids
    ///
    /// Populates from [`Self::priority_actions`] when its predicate for queuing is true
    priority_actions_queue: VecDeque<u32>,
    /// Side-loaded one-time priority actions.
    ///
    /// These are actions injected externally and to be executed as appropriate with the current
    /// [`Self::priority_actions_queue`]. These actions are run only once and do not have an ID.
    priority_actions_side_queue: VecDeque<RotatorAction>,

    /// See [`crate::models::FamiliarsFusion::attack_key`]. `None` when disabled.
    ///
    /// Read once per tick by [`Self::rotate_familiars_fusing_attack`] to tap a key while
    /// [`Player::FamiliarsFusing`] is in progress, independent of the priority-action queue
    /// (which never runs while fusion is active - see [`Player::can_override_current_state`]).
    familiars_fuse_attack_key: Option<KeyKind>,
    /// See [`crate::models::FamiliarsFusion::attack_interval_millis`].
    familiars_fuse_attack_interval_millis: u64,
    /// See [`crate::models::FamiliarsFusion::attack_interval_millis_random_range`].
    familiars_fuse_attack_interval_millis_random_range: u64,
    /// The next [`Instant`] [`Self::rotate_familiars_fusing_attack`] is due to tap
    /// [`Self::familiars_fuse_attack_key`] again. `None` both before the first tap of a run and
    /// whenever [`Player::FamiliarsFusing`] isn't the current state (reset there so a fresh run
    /// doesn't inherit a stale delay from an earlier one).
    familiars_fuse_attack_next_at: Option<Instant>,
}

impl DefaultRotator {
    #[inline]
    fn reset_normal_actions_queue(&mut self) {
        self.normal_index = 0;
        self.normal_queuing_linked_action = None;
    }

    /// Rotates the actions inside the [`Self::priority_actions`]
    ///
    /// This function does not pass the action to the player but only pushes the action to
    /// [`Self::priority_actions_queue`]. It is responsible for checking queuing condition.
    fn rotate_priority_actions(&mut self, resources: &mut Resources, world: &mut World) {
        #[derive(Debug)]
        enum ResolveConflict {
            None,
            #[allow(dead_code)]
            Replace {
                id: u32,
            },
            Ignore,
        }

        /// Checks if the provided `id` is a priority linked action in queue or executing.
        #[inline]
        fn is_priority_linked_action_queuing_or_executing(
            rotator: &DefaultRotator,
            player_context: &PlayerContext,
            id: u32,
        ) -> bool {
            let queuing_id = rotator
                .priority_queuing_linked_action
                .as_ref()
                .map(|(action_id, _)| *action_id);
            if Some(id) == queuing_id {
                return true;
            }

            let Some(action_id) = player_context.priority_action_id() else {
                return false;
            };
            if action_id != id {
                return false;
            }

            rotator
                .priority_actions
                .get(&id)
                .is_some_and(|action| matches!(action.inner, RotatorAction::Linked(_)))
        }

        /// Checks if the player or the queue has
        /// a [`ActionCondition::ErdaShowerOffCooldown`] action.
        #[inline]
        fn has_erda_action_queuing_or_executing(
            rotator: &DefaultRotator,
            player_context: &PlayerContext,
        ) -> bool {
            if let Some(id) = player_context.priority_action_id()
                && let Some(action) = rotator.priority_actions.get(&id)
                && matches!(
                    action.condition_kind,
                    Some(ActionCondition::ErdaShowerOffCooldown)
                )
            {
                return true;
            }

            rotator.priority_actions_queue.iter().any(|id| {
                let condition = rotator
                    .priority_actions
                    .get(id)
                    .and_then(|action| action.condition_kind);
                matches!(condition, Some(ActionCondition::ErdaShowerOffCooldown))
            })
        }

        fn resolve_conflict_from_metadata(
            rotator: &DefaultRotator,
            player_context: &PlayerContext,
            metadata: ActionMetadata,
        ) -> ResolveConflict {
            match metadata {
                ActionMetadata::UseBooster => {
                    if player_context
                        .priority_action_id()
                        .and_then(|id| rotator.priority_actions.get(&id))
                        .and_then(|action| action.metadata)
                        .is_some_and(|metadata| matches!(metadata, ActionMetadata::UseBooster))
                    {
                        info!(target: "backend/rotator", "ignored booster usage due to conflict with another booster kind");
                        return ResolveConflict::Ignore;
                    }

                    for id in rotator.priority_actions_queue.iter() {
                        let action = rotator.priority_actions.get(id).expect("exists");
                        if matches!(action.metadata, Some(ActionMetadata::UseBooster)) {
                            info!(target: "backend/rotator", "ignored booster usage due to conflict with another booster kind");
                            return ResolveConflict::Ignore;
                        }
                    }
                }
                ActionMetadata::Buff {
                    kind: BuffKind::ExpCouponX2 | BuffKind::ExpCouponX3 | BuffKind::ExpCouponX4,
                } => {
                    // TODO:
                }
                ActionMetadata::Buff {
                    kind: BuffKind::WealthAcquisitionPotion | BuffKind::SmallWealthAcquisitionPotion,
                } => {
                    // TODO:
                }
                ActionMetadata::Buff {
                    kind: BuffKind::ExpAccumulationPotion | BuffKind::SmallExpAccumulationPotion,
                } => {
                    // TODO:
                }
                ActionMetadata::Buff { .. } => (),
            }

            ResolveConflict::None
        }

        // Keeps ignoring while there is any type of erda condition action inside the queue
        let has_erda_action = has_erda_action_queuing_or_executing(self, &world.player.context);
        let ids = self.priority_actions.keys().copied().collect::<Vec<_>>();
        let mut did_queue_erda_action = false;
        // See `DailyQuestSuppression`'s docs for why these are held back from being newly queued
        // while a daily quest entry is still pending.
        let daily_quest_active = self.daily_quest_index < self.daily_quest_entries.len();
        // A superset of `is_navigating_daily_quest` (which additionally requires
        // `daily_quest_navigating`) - also covers the brief window where an entry is active but
        // navigation hasn't been issued yet, since there's still nothing meaningful to interrupt
        // for there either.
        //
        // Excludes the case where an active Monster Park run is the one holding this entry back
        // (see `rotate_action`'s `blocked_by_monster_park_run`) - `rotate_daily_quest` never runs
        // while that's true, so `daily_quest_arrived` would otherwise stay stuck at `false` for the
        // Monster Park run's entire duration (which can be minutes, not the brief window this is
        // meant to cover), wrongly suppressing character-level `EveryMillis` fixed actions the
        // whole time even though nothing is actually being navigated to.
        let monster_park_blocking_daily_quest = self.monster_park_enabled
            && (self.monster_park_run_in_progress || self.monster_park_at_entry_lobby);
        let daily_quest_navigating =
            daily_quest_active && !self.daily_quest_arrived && !monster_park_blocking_daily_quest;
        // `monster_park_at_entry_lobby` is only meaningful while Monster Park is actually armed -
        // guard on that too rather than trusting it alone, since it isn't reset back to `false`
        // when Monster Park is disabled (see the field's docs).
        let monster_park_at_entry_lobby =
            self.monster_park_at_entry_lobby && self.monster_park_enabled;

        for id in ids {
            // Ignores for as long as the action is a linked action that is queuing
            // or executing
            let has_linked_action =
                is_priority_linked_action_queuing_or_executing(self, &world.player.context, id);
            let action = self.priority_actions.get_mut(&id).expect("action id exist");

            action.queue_info.ignoring = match action.condition_kind {
                Some(ActionCondition::ErdaShowerOffCooldown) => {
                    has_erda_action || has_linked_action
                }
                Some(ActionCondition::Linked) | Some(ActionCondition::EveryMillis(_)) | None => {
                    world
                        .player
                        .context // The player currently executing action
                        .priority_action_id()
                        .is_some_and(|action_id| action_id == id)
                        || self // The action is in queue
                            .priority_actions_queue
                            .iter()
                            .any(|action_id| *action_id == id)
                        || has_linked_action
                }
                Some(ActionCondition::Any) => unreachable!(),
            };
            if action.queue_info.ignoring {
                action.queue_info.last_queued_time = Some(Instant::now());
                continue;
            }
            match action.suppress_during_daily_quest {
                DailyQuestSuppression::Never => {}
                DailyQuestSuppression::WhileNavigating if daily_quest_navigating => continue,
                DailyQuestSuppression::WhileActive if daily_quest_active => continue,
                DailyQuestSuppression::WhileNavigating | DailyQuestSuppression::WhileActive => {}
            }
            // Position-less Monster Park `EveryMillis` fixed actions are registered
            // unconditionally (see `build_actions`'s `EveryMillis` arm) but should still lapse
            // while genuinely at the entry lobby - nothing meaningful to use them on there either.
            if monster_park_at_entry_lobby
                && matches!(action.condition_kind, Some(ActionCondition::EveryMillis(_)))
            {
                continue;
            }

            let condition_fn = &mut action.condition.0;
            let result = condition_fn(resources, world, &mut action.queue_info);
            match result {
                ConditionResult::Queue => {
                    let conflict = if let Some(metadata) = action.metadata {
                        resolve_conflict_from_metadata(self, &world.player.context, metadata)
                    } else {
                        ResolveConflict::None
                    };
                    // Reborrow mutably here so the above `resolve_conflict_from_metadata`
                    // can do immutable borrow.
                    let action = self.priority_actions.get_mut(&id).expect("action id exist");

                    match conflict {
                        ResolveConflict::None => {
                            if action.queue_to_front {
                                self.priority_actions_queue.push_front(id);
                            } else {
                                self.priority_actions_queue.push_back(id);
                            }
                            action.queue_info.last_queued_time = Some(Instant::now());

                            if !did_queue_erda_action {
                                did_queue_erda_action = matches!(
                                    action.condition_kind,
                                    Some(ActionCondition::ErdaShowerOffCooldown)
                                );
                            }
                        }
                        ResolveConflict::Replace { id: replace_id } => {
                            if let Some(replace_id) = self
                                .priority_actions_queue
                                .iter_mut()
                                .find(|id| **id == replace_id)
                            {
                                *replace_id = id;
                            }

                            action.queue_info.last_queued_time = Some(Instant::now());
                        }
                        ResolveConflict::Ignore => {
                            action.queue_info.last_queued_time = Some(Instant::now());
                        }
                    }
                }
                ConditionResult::Skip => (),
                ConditionResult::Ignore => {
                    action.queue_info.last_queued_time = Some(Instant::now());
                }
            }
        }

        if did_queue_erda_action && self.normal_actions_reset_on_erda {
            self.reset_normal_actions_queue();
            world.player.context.reset_normal_action();
        }
    }

    /// Rotates the actions inside the [`Self::priority_actions_queue`].
    ///
    /// If there is any on-going linked action:
    /// - For normal action, it will wait until the action is completed by the normal rotation.
    /// - For priority action, it will rotate and wait until all the actions are executed.
    ///
    /// After that, it will rotate actions inside [`Self::priority_actions_queue`].
    fn rotate_priority_actions_queue(&mut self, player: &mut PlayerEntity) {
        /// Checks if the player is queuing or executing a normal [`RotatorAction::Linked`] action.
        ///
        /// This prevents [`Self::rotate_priority_actions_queue`] from overriding the normal
        /// linked action.
        #[inline]
        fn has_normal_linked_action_queuing_or_executing(
            rotator: &DefaultRotator,
            player_context: &PlayerContext,
        ) -> bool {
            if rotator.normal_queuing_linked_action.is_some() {
                return true;
            }
            player_context.normal_action_id().is_some_and(|id| {
                rotator.normal_actions.iter().any(|(action_id, action)| {
                    *action_id == id && matches!(action, RotatorAction::Linked(_))
                })
            })
        }

        /// Checks if the player is executing a priority [`RotatorAction::Linked`] action.
        ///
        /// This does not check the queuing linked action because this check is to allow the linked
        /// action to be rotated in [`Self::rotate_priority_actions_queue`].
        #[inline]
        fn has_priority_linked_action_executing(
            rotator: &DefaultRotator,
            player_context: &PlayerContext,
        ) -> bool {
            player_context.priority_action_id().is_some_and(|id| {
                rotator
                    .priority_actions
                    .get(&id)
                    .is_some_and(|action| matches!(action.inner, RotatorAction::Linked(_)))
            })
        }

        if self.priority_actions_queue.is_empty()
            && self.priority_actions_side_queue.is_empty()
            && self.priority_queuing_linked_action.is_none()
        {
            return;
        }
        if !player
            .state
            .can_override_current_state(player.context.last_known_pos)
            || has_normal_linked_action_queuing_or_executing(self, &player.context)
            || has_priority_linked_action_executing(self, &player.context)
            || has_side_loaded_action_executing(&player.context)
        {
            return;
        }

        if self.rotate_queuing_linked_action(&mut player.context, true) {
            return;
        }
        if self.rotate_side_priority_action(&mut player.context) {
            return;
        }

        let player_has_queue_to_front = player
            .context
            .priority_action_id()
            .and_then(|id| {
                self.priority_actions
                    .get(&id)
                    .map(|action| action.queue_to_front)
            })
            .unwrap_or_default();
        if player_has_queue_to_front {
            return;
        }

        let Some(id) = self.priority_actions_queue.pop_front_if(|id| {
            self.priority_actions
                .get(id)
                .is_none_or(|action| !player.context.has_priority_action() || action.queue_to_front)
        }) else {
            return;
        };
        let Some(action) = self.priority_actions.get(&id) else {
            return;
        };

        match action.inner.clone() {
            RotatorAction::Single(inner) => {
                if action.queue_to_front {
                    if let Some(id) = player.context.replace_priority_action(Some(id), inner) {
                        self.priority_actions_queue.push_front(id);
                    }
                } else {
                    player.context.set_priority_action(Some(id), inner);
                }
            }
            RotatorAction::Linked(linked) => {
                if action.queue_to_front
                    && let Some(id) = player.context.take_priority_action()
                {
                    self.priority_actions_queue.push_front(id);
                }
                self.priority_queuing_linked_action = Some((id, Box::new(linked)));
                self.rotate_queuing_linked_action(&mut player.context, true);
            }
        }
    }

    /// Taps [`Self::familiars_fuse_attack_key`] at a randomized interval while
    /// [`Player::FamiliarsFusing`] is in progress, so the character keeps fighting back mobs
    /// instead of standing idle through fusion's up-to-14/28-round retry loop.
    ///
    /// Deliberately bypasses the priority-action queue entirely rather than going through
    /// [`PlayerContext::set_priority_action`] - [`Player::can_override_current_state`] marks
    /// fusion non-overridable because replacing `player.state` mid-flight would lose its
    /// in-progress `round`/`attempted`/`targets` tracking, so there is no way to "pause and
    /// resume" it through that mechanism. This instead sends a raw, fire-and-forget key tap, the
    /// same shape fusion's own menu-opening step already uses to deactivate the `Familiar` buff
    /// before opening the menu - a plain key press doesn't touch the mouse or menu focus, so it's
    /// safe to interleave with fusion's own UI clicks. Called
    /// unconditionally every tick from [`Self::rotate_action`]; downstream dispatch (daily
    /// quest/Monster Park/normal mode) already no-ops while fusion's priority action is in
    /// flight, so nothing else needs to skip or `return` around this.
    ///
    /// `FamiliarsFusing::is_safe_for_side_key_press` additionally withholds the tap for as long as
    /// the Inventory fodder top-up dialog is open, since typing the "how many do you want to
    /// use?" quantity is the one place a stray key press could land in a text field instead of
    /// being interpreted as a normal keybind.
    fn rotate_familiars_fusing_attack(&mut self, resources: &mut Resources, world: &mut World) {
        let Some(key) = self.familiars_fuse_attack_key else {
            return;
        };
        let Player::FamiliarsFusing(fusing) = &world.player.state else {
            self.familiars_fuse_attack_next_at = None;
            return;
        };
        if !fusing.is_safe_for_side_key_press() {
            return;
        }

        let now = Instant::now();
        if self
            .familiars_fuse_attack_next_at
            .is_none_or(|at| now >= at)
        {
            debug!(target: "backend/rotator", "familiars fusing side action: attack key {key:?}");
            resources.input.send_key(key);

            let base = self.familiars_fuse_attack_interval_millis;
            let range = self.familiars_fuse_attack_interval_millis_random_range;
            let delay_ms = if range == 0 {
                base
            } else {
                resources
                    .rng
                    .random_range(base.saturating_sub(range)..=base + range)
            };
            self.familiars_fuse_attack_next_at = Some(now + Duration::from_millis(delay_ms));
        }
    }

    /// Runs the daily quest checklist at [`Self::daily_quest_index`], if any is still pending.
    ///
    /// For the active entry, this navigates to its hunting ground (via
    /// [`PlayerAction::NavigateToHuntingGround`]), then delegates to
    /// [`Self::rotate_auto_mobbing`] using the entry's own [`DailyQuestEntry::bound`] and
    /// [`DailyQuestEntry::mobbing_key`] - the mobbing behavior itself is identical to regular
    /// auto-mobbing, only which map and area differ. Advances to the next entry once the
    /// "Quest complete!" toast (see [`Detector::detect_quest_complete_popup`]) is confirmed.
    ///
    /// Called from [`Self::rotate_action`] before the map's normal [`RotatorMode`] dispatch -
    /// once every configured entry is completed, subsequent calls fall through to that normal
    /// dispatch instead, so the bot then farms the current map as configured.
    fn rotate_daily_quest(
        &mut self,
        resources: &mut Resources,
        player_context: &mut PlayerContext,
        minimap_context: &mut MinimapContext,
        minimap_state: Minimap,
    ) {
        let Some(entry) = self
            .daily_quest_entries
            .get(self.daily_quest_index)
            .cloned()
        else {
            return;
        };

        if !self.daily_quest_navigating {
            if player_context.has_normal_action() || player_context.has_priority_action() {
                return;
            }

            if self.daily_quest_saved_pathing.is_none() {
                self.daily_quest_saved_pathing = Some(DailyQuestSavedPathing {
                    platforms: minimap_context.platforms().to_vec(),
                    auto_mob_platforms_pathing: player_context.config.auto_mob_platforms_pathing,
                    auto_mob_platforms_pathing_up_jump_only: player_context
                        .config
                        .auto_mob_platforms_pathing_up_jump_only,
                    auto_mob_use_key_when_pathing: player_context
                        .config
                        .auto_mob_use_key_when_pathing,
                    auto_mob_use_key_when_pathing_update_millis: player_context
                        .config
                        .auto_mob_use_key_when_pathing_update_millis,
                    auto_mob_use_key_while_double_jumping: player_context
                        .config
                        .auto_mob_use_key_while_double_jumping,
                    rune_platforms_pathing: player_context.config.rune_platforms_pathing,
                    rune_platforms_pathing_up_jump_only: player_context
                        .config
                        .rune_platforms_pathing_up_jump_only,
                });
            }
            // The daily quest solver has no `Map` of its own to draw platforms on - apply this
            // entry's recorded platforms (if any) the same way selecting a regular farming map
            // would, so auto-mobbing here can path around the hunting ground's layout instead of
            // blindly grappling/jumping toward mobs it can't reach directly. Entries without
            // recorded platforms yet are left exactly as before (pathing disabled).
            let platforms = entry.id.platforms();
            let has_platforms = !platforms.is_empty();
            minimap_context.set_platforms(platforms.into_iter().map(Platform::from).collect());
            // Each entry can have a completely different hunting ground layout, but this doesn't
            // go through the regular map-change path that would otherwise clear
            // `player_context`'s learned auto-mob platform/quadrant tracking (see
            // `PlayerContext::auto_mob_reset_tracking`'s docs) - without this, reachable-y/ignored
            // x-range data learned on a previous entry's platforms gets reused against this one's
            // different geometry, sending auto-mob after destinations that aren't on any real
            // platform here and looping forever retrying them.
            player_context.auto_mob_reset_tracking();
            player_context.config.auto_mob_platforms_pathing = has_platforms;
            player_context
                .config
                .auto_mob_platforms_pathing_up_jump_only = false;
            player_context.config.auto_mob_use_key_when_pathing = has_platforms;
            player_context
                .config
                .auto_mob_use_key_when_pathing_update_millis =
                DAILY_QUEST_AUTO_MOB_USE_KEY_UPDATE_MILLIS;
            // Character-level, not per-hunting-ground - applies the same to every entry, same
            // reasoning as `daily_quest_mobbing_key`.
            player_context.config.auto_mob_use_key_while_double_jumping =
                self.daily_quest_use_key_while_double_jumping;
            // A rune can spawn on a daily hunting ground same as any regular map, and rune-solving
            // is never suppressed during daily quest navigation (see `solve_rune_priority_action`'s
            // `DailyQuestSuppression::Never`) - so these need the same reset-per-entry treatment as
            // the auto-mob pathing fields above, using this entry's own platforms rather than
            // whatever was left over from the character's regular farming map.
            player_context.config.rune_platforms_pathing = has_platforms;
            player_context.config.rune_platforms_pathing_up_jump_only = false;

            let navigation = entry.id.navigation();
            let skip_dropdown_slots = self
                .daily_quest_index
                .checked_sub(1)
                .map(|prev_index| self.daily_quest_entries[prev_index].id.navigation())
                .map(|previous| shared_dropdown_prefix_len(&previous, &navigation))
                .unwrap_or(0);
            debug!(
                target: "backend/rotator",
                "Daily Quest: navigating to {} (skipping {skip_dropdown_slots} already-selected \
                 dropdown slot(s))",
                navigation.location_label
            );
            player_context.set_priority_action(
                None,
                PlayerAction::NavigateToHuntingGround(NavigateToHuntingGround {
                    region: navigation.region,
                    dropdown_path: navigation.dropdown_path,
                    location_point: navigation.location_point,
                    sub_location_label: navigation.sub_location_label,
                    skip_dropdown_slots,
                }),
            );
            self.daily_quest_navigating = true;
            return;
        }

        if player_context.has_priority_action() {
            // Still navigating
            return;
        }

        if player_context.take_daily_quest_navigate_failed() {
            // Navigation didn't actually reach the hunting ground (e.g. world map key not
            // configured, an expected UI element wasn't found) - `has_priority_action()` clears
            // the same way on failure as it does on success, so without this check the code
            // below would otherwise start auto-mobbing with this entry's bound wherever the
            // player currently is instead of recognizing the attempt never got there.
            //
            // Often just a transient hiccup (e.g. network stutter causing a UI element to not
            // render in time for detection) rather than the hunting ground genuinely being
            // unreachable, so retry a handful of times - re-issuing the same
            // `NavigateToHuntingGround` from scratch - before giving up on this entry for the
            // run. `daily_quest_navigating` is left as-is (`true`) so the top of this function's
            // `!self.daily_quest_navigating` branch isn't what drives the retry - it's reset to
            // `false` below so the very next call re-enters that branch and re-issues navigation.
            const DAILY_QUEST_NAVIGATE_ATTEMPT_LIMIT: u32 = 3;
            self.daily_quest_navigate_attempts += 1;
            if self.daily_quest_navigate_attempts < DAILY_QUEST_NAVIGATE_ATTEMPT_LIMIT {
                info!(
                    target: "backend/rotator",
                    "Daily Quest: failed to navigate to {} (attempt {}/{}), retrying",
                    entry.id, self.daily_quest_navigate_attempts, DAILY_QUEST_NAVIGATE_ATTEMPT_LIMIT
                );
                self.daily_quest_navigating = false;
                self.daily_quest_arrived = false;
                self.auto_mob_task = None;
                self.auto_mob_quadrant_consecutive_count = None;
                self.daily_quest_complete_task = None;
                self.daily_quest_complete_popup_detected = false;
                self.daily_quest_complete_popup_confirmed_absent = false;
                return;
            }

            // Skip this entry for the current run rather than mis-hunting or retrying forever.
            info!(
                target: "backend/rotator",
                "Daily Quest: failed to navigate to {} {} times in a row, skipping for this run",
                entry.id, self.daily_quest_navigate_attempts
            );
            self.daily_quest_index += 1;
            self.daily_quest_navigating = false;
            self.daily_quest_arrived = false;
            self.daily_quest_navigate_attempts = 0;
            self.auto_mob_task = None;
            self.auto_mob_quadrant_consecutive_count = None;
            self.daily_quest_complete_task = None;
            self.daily_quest_complete_popup_detected = false;
            self.daily_quest_complete_popup_confirmed_absent = false;
            return;
        }

        // Reached this point only once travel has actually finished successfully - from here on,
        // any further map change is no longer part of expected navigation.
        self.daily_quest_arrived = true;
        self.daily_quest_navigate_attempts = 0;

        if !self.daily_quest_complete_popup_detected {
            match update_detection_task(
                resources,
                DAILY_QUEST_COMPLETE_POLL_MILLIS,
                &mut self.daily_quest_complete_task,
                move |detector| {
                    let popup = detector.detect_quest_complete_popup()?;
                    let label_roi = Rect::new(
                        popup.x,
                        popup.y + popup.height,
                        DAILY_QUEST_COMPLETE_LABEL_ROI_WIDTH,
                        popup.height + DAILY_QUEST_COMPLETE_LABEL_ROI_PADDING,
                    );
                    detector.detect_world_map_label(label_roi, "[Daily Quest]")?;
                    Ok(())
                },
            ) {
                // A hit only counts once a miss has been observed since arriving - otherwise
                // this can't tell apart the current entry's own toast from the *previous*
                // entry's toast still fading on screen (it lingers a few seconds, which can
                // outlast the teleport to the next entry - observed marking Odium, Shangri-La
                // and Arteria "complete" within 1 second of arriving, off Hotel Arcus's
                // leftover toast, without a single mob killed at any of them).
                Update::Ok(()) if self.daily_quest_complete_popup_confirmed_absent => {
                    debug!(
                        target: "backend/rotator",
                        "Daily Quest: detected \"Quest complete!\" toast for {}", entry.id
                    );
                    self.daily_quest_complete_popup_detected = true;
                }
                Update::Err(_) => {
                    self.daily_quest_complete_popup_confirmed_absent = true;
                }
                Update::Ok(()) | Update::Pending => {}
            }
        }

        if self.daily_quest_complete_popup_detected {
            info!(
                target: "backend/rotator",
                "Daily Quest: completed {}", entry.id
            );
            resources
                .character_updates
                .mark_daily_quest_completed(self.daily_quest_character_id, entry.id);
            // `rotate_auto_mobbing` below may still have a normal action in flight chasing this
            // entry's last mob - left alone, the top-of-function `daily_quest_navigating == false`
            // branch's `has_normal_action()` guard keeps seeing that stale action and refuses to
            // dispatch navigation to the next entry, so the player just keeps trying (and, if that
            // mob's position isn't actually reachable, endlessly retrying/unstucking) a chase that
            // no longer matters now that this entry is done. `clear_actions_aborted(true)` also
            // resets the player to `Idle` on the next update instead of waiting for the current
            // attempt to fail on its own first.
            player_context.clear_actions_aborted(true);
            self.daily_quest_index += 1;
            self.daily_quest_navigating = false;
            self.daily_quest_arrived = false;
            self.auto_mob_task = None;
            self.auto_mob_quadrant_consecutive_count = None;
            self.daily_quest_complete_task = None;
            self.daily_quest_complete_popup_detected = false;
            self.daily_quest_complete_popup_confirmed_absent = false;
            if self.daily_quest_index >= self.daily_quest_entries.len() {
                if let Some(saved) = self.daily_quest_saved_pathing.take() {
                    minimap_context.set_platforms(saved.platforms);
                    player_context.config.auto_mob_platforms_pathing =
                        saved.auto_mob_platforms_pathing;
                    player_context
                        .config
                        .auto_mob_platforms_pathing_up_jump_only =
                        saved.auto_mob_platforms_pathing_up_jump_only;
                    player_context.config.auto_mob_use_key_when_pathing =
                        saved.auto_mob_use_key_when_pathing;
                    player_context
                        .config
                        .auto_mob_use_key_when_pathing_update_millis =
                        saved.auto_mob_use_key_when_pathing_update_millis;
                    player_context.config.auto_mob_use_key_while_double_jumping =
                        saved.auto_mob_use_key_while_double_jumping;
                    player_context.config.rune_platforms_pathing = saved.rune_platforms_pathing;
                    player_context.config.rune_platforms_pathing_up_jump_only =
                        saved.rune_platforms_pathing_up_jump_only;
                }
                resources
                    .notification
                    .schedule_notification(NotificationKind::DailyQuestCompleted);
                // There is no per-map navigation back to the character's actual farming map yet
                // (unlike daily quest hunting grounds, those aren't fixed content - each is
                // user-defined and would need its own recorded navigation, one per map the user
                // wants to hunt at). Returning to town is a safe, always-reachable parking spot
                // instead of continuing to auto-mob the last daily's hunting ground under the
                // farming map's `RotatorMode`, which would run with the wrong bound entirely.
                self.inject_action(PlayerAction::Panic(Panic { to: PanicTo::Town }));
                // Halting (rather than resetting the queue and leaving it `Running`) stops the
                // bot once there, matching what the manual "stop and go to town" button does -
                // `rotate_action`'s halting branch still drains side-loaded actions like the one
                // just injected above, so the walk to town still completes despite halting here
                // immediately instead of waiting for arrival first.
                resources.operation.state = OperationState::Halting;
            }
            return;
        }

        self.rotate_auto_mobbing(
            resources,
            player_context,
            minimap_state,
            self.daily_quest_mobbing_key,
            entry.id.bound(),
        );
    }

    fn rotate_auto_mobbing(
        &mut self,
        resources: &mut Resources,
        player_context: &mut PlayerContext,
        minimap_state: Minimap,
        key: MobbingKey,
        bound: Bound,
    ) {
        if player_context.has_normal_action() {
            return;
        }

        let Minimap::Idle(idle) = minimap_state else {
            return;
        };
        let Some(pos) = player_context.last_known_pos else {
            return;
        };
        let bound = bound.into();

        let name = player_context.name();
        let points =
            match update_detection_task(resources, 0, &mut self.auto_mob_task, move |detector| {
                detector.detect_mobs(idle.bbox, bound, pos, name)
            }) {
                Update::Ok(points) => points,
                Update::Err(err) => {
                    debug!(target: "backend/rotator", "auto mob detection failed: {err:?}");
                    return;
                }
                Update::Pending => return,
            };
        let points = points
            .iter()
            .filter_map(|point| {
                let y = idle.bbox.height - point.y;
                let point = if y <= pos.y || (y - pos.y).abs() <= GRAPPLING_THRESHOLD {
                    Some(Point::new(point.x, y))
                } else {
                    None
                };
                debug!(target: "backend/rotator", "auto mob raw position {point:?}");
                point.and_then(|point| {
                    player_context.auto_mob_pick_reachable_y_position(
                        resources,
                        minimap_state,
                        point,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut use_pathing_point = false;

        if let Some(last_quad) = player_context.auto_mob_last_quadrant()
            && !points.is_empty()
        {
            if self
                .auto_mob_quadrant_consecutive_count
                .is_none_or(|(quad, _)| quad != last_quad)
            {
                self.auto_mob_quadrant_consecutive_count = Some((last_quad, 0));
            }
            let (_, count) = self
                .auto_mob_quadrant_consecutive_count
                .as_mut()
                .expect("is some");

            *count += 1;
            if *count >= AUTO_MOB_SAME_QUAD_THRESHOLD {
                *count = 0;
                use_pathing_point = true;
            }
        }

        let mut is_pathing = use_pathing_point;
        let point = if use_pathing_point {
            player_context.auto_mob_pathing_point(resources, minimap_state, bound)
        } else {
            resources
                .rng
                .random_choose(points.into_iter())
                .unwrap_or_else(|| {
                    is_pathing = true;
                    player_context.auto_mob_pathing_point(resources, minimap_state, bound)
                })
        };
        let key_hold_ticks = (key.key_hold_millis / MS_PER_TICK) as u32;
        let wait_before_ticks = (key.wait_before_millis / MS_PER_TICK) as u32;
        let wait_before_ticks_random_range =
            (key.wait_before_millis_random_range / MS_PER_TICK) as u32;
        let wait_after_ticks = (key.wait_after_millis / MS_PER_TICK) as u32;
        let wait_after_ticks_random_range =
            (key.wait_after_millis_random_range / MS_PER_TICK) as u32;
        let position = Position {
            x: point.x,
            x_random_range: 0,
            y: point.y,
            allow_adjusting: false,
        };

        player_context.set_normal_action(
            None,
            PlayerAction::AutoMob(AutoMob {
                key: key.key.into(),
                key_hold_ticks,
                link_key: key.link_key.into(),
                count: key.count.max(1),
                with: key.with,
                wait_before_ticks,
                wait_before_ticks_random_range,
                wait_after_ticks,
                wait_after_ticks_random_range,
                position,
                is_pathing,
                is_monster_park: false,
            }),
        );
    }

    /// Updates [`Self::monster_park_portal_wiggle`] by one step.
    ///
    /// Each call either counts down the delay before the current step's key, or - once that
    /// delay elapses - sends the key and advances to the next step (with a freshly randomized
    /// delay of its own, so the gaps between taps aren't a uniform, distinctly bot-like cadence).
    /// Sent as raw input (like [`Player::EnteringMonsterPark`]'s dialog navigation and
    /// [`Player::Unstucking`]'s movement) rather than through a [`PlayerAction`], since this is a
    /// bespoke recovery tap, not a real scripted game action - it doesn't need the player action
    /// system's animation-aware timing.
    fn advance_monster_park_portal_wiggle(&mut self, resources: &mut Resources) {
        let (step, ticks_remaining) = self
            .monster_park_portal_wiggle
            .expect("no portal wiggle in progress");
        if ticks_remaining > 0 {
            self.monster_park_portal_wiggle = Some((step, ticks_remaining - 1));
            return;
        }

        // `Right` is the corrective tap (see `monster_park_portal_wiggle_toward_key`'s docs) -
        // defaulting to `KeyKind::Right` when the direction isn't known keeps the "portal not
        // found" recovery's original fixed Left-then-Right order unchanged.
        let toward_key = self
            .monster_park_portal_wiggle_toward_key
            .unwrap_or(KeyKind::Right);
        let away_key = if toward_key == KeyKind::Right {
            KeyKind::Left
        } else {
            KeyKind::Right
        };
        let key = match step {
            PortalWiggleStep::Left => away_key,
            PortalWiggleStep::Right => toward_key,
            PortalWiggleStep::Up => KeyKind::Up,
        };
        resources.input.send_key(key);
        debug!(target: "backend/rotator", "Monster Park: portal wiggle - pressing {key:?}");

        // Random human-like gap before the next tap.
        const WIGGLE_STEP_DELAY_TICKS: RangeInclusive<u32> = 5..=15;
        self.monster_park_portal_wiggle = match step {
            PortalWiggleStep::Left => Some((
                PortalWiggleStep::Right,
                resources.rng.random_range(WIGGLE_STEP_DELAY_TICKS),
            )),
            PortalWiggleStep::Right => Some((
                PortalWiggleStep::Up,
                resources.rng.random_range(WIGGLE_STEP_DELAY_TICKS),
            )),
            PortalWiggleStep::Up => None,
        };
    }

    /// Updates [`Self::monster_park_portal_escape`] by one step: holds the direction key down for
    /// [`PORTAL_ESCAPE_HOLD_TICKS`], presses the jump key partway through, then releases it.
    ///
    /// Sent as raw input, same reasoning as [`Self::advance_monster_park_portal_wiggle`] - this is
    /// a bespoke recovery, not a real scripted game action.
    fn advance_monster_park_portal_escape(&mut self, resources: &mut Resources, jump_key: KeyKind) {
        let (key, ticks_remaining) = self
            .monster_park_portal_escape
            .expect("no portal escape in progress");

        if ticks_remaining == PORTAL_ESCAPE_HOLD_TICKS {
            resources.input.send_key_down(key);
        } else if ticks_remaining == PORTAL_ESCAPE_JUMP_AT_TICKS_REMAINING {
            resources.input.send_key(jump_key);
        }

        if ticks_remaining == 0 {
            resources.input.send_key_up(key);
            self.monster_park_portal_escape = None;
            return;
        }

        self.monster_park_portal_escape = Some((key, ticks_remaining - 1));
    }

    /// Escalates a portal-stuck episode - tries the cheaper "wiggle" recovery once (see
    /// [`Self::monster_park_portal_wiggled`]'s docs), or starts the "escape" recovery (see
    /// [`Self::monster_park_portal_escape`]'s docs) if that's already been tried and the player is
    /// still stuck.
    ///
    /// `toward_key` is the key that closes the current horizontal gap to the portal, if known -
    /// see [`Self::monster_park_portal_wiggle_toward_key`]'s docs. The escape moves *away* from it
    /// (the wedge is more likely on the far side than the near one) when known, otherwise picks a
    /// direction at random.
    ///
    /// Also arms [`Self::monster_park_portal_force_real_height`] - see that field's docs for why
    /// an escalation is exactly the signal that the player may be a genuine platform short of the
    /// portal rather than just noisily flickering off `is_at_portal`.
    fn escalate_monster_park_portal_stuck(
        &mut self,
        resources: &mut Resources,
        toward_key: Option<KeyKind>,
    ) {
        self.monster_park_portal_stuck_since = None;
        self.monster_park_portal_attempts = 0;
        self.monster_park_portal_give_up_cycles = 0;
        self.monster_park_portal_force_real_height = true;

        if self.monster_park_portal_wiggled {
            info!(
                target: "backend/rotator",
                "Monster Park: still stuck at the portal after wiggling, trying to escape"
            );
            self.monster_park_portal_wiggled = false;
            let away_key = match toward_key {
                Some(KeyKind::Right) => KeyKind::Left,
                Some(_) => KeyKind::Right,
                None => {
                    if resources.rng.random_bool(0.5) {
                        KeyKind::Left
                    } else {
                        KeyKind::Right
                    }
                }
            };
            self.monster_park_portal_escape = Some((away_key, PORTAL_ESCAPE_HOLD_TICKS));
            return;
        }

        debug!(
            target: "backend/rotator",
            "Monster Park: stuck at the portal, trying to wiggle (tap Left, Right, Up) before \
             escaping"
        );
        self.monster_park_portal_wiggled = true;
        self.monster_park_portal_wiggle_toward_key = toward_key;
        self.monster_park_portal_wiggle = Some((PortalWiggleStep::Left, 0));
    }

    /// Builds and dispatches a Monster Park [`PlayerAction::AutoMob`] targeting `point`.
    ///
    /// Shared by [`Self::rotate_monster_park`]'s fresh-target acquisition and its still-pursuing
    /// re-target (see the latter's docs on why an already in-flight pursuit sometimes needs to be
    /// redirected at a new position instead of just updating [`Self::monster_park_target`]).
    fn dispatch_monster_park_auto_mob(
        player_context: &mut PlayerContext,
        key: MobbingKey,
        point: Point,
    ) {
        let key_hold_ticks = (key.key_hold_millis / MS_PER_TICK) as u32;
        let wait_before_ticks = (key.wait_before_millis / MS_PER_TICK) as u32;
        let wait_before_ticks_random_range =
            (key.wait_before_millis_random_range / MS_PER_TICK) as u32;
        let wait_after_ticks = (key.wait_after_millis / MS_PER_TICK) as u32;
        let wait_after_ticks_random_range =
            (key.wait_after_millis_random_range / MS_PER_TICK) as u32;
        let position = Position {
            x: point.x,
            x_random_range: 0,
            y: point.y,
            allow_adjusting: false,
        };

        player_context.set_normal_action(
            None,
            PlayerAction::AutoMob(AutoMob {
                key: key.key.into(),
                key_hold_ticks,
                link_key: key.link_key.into(),
                count: key.count.max(1),
                with: key.with,
                wait_before_ticks,
                wait_before_ticks_random_range,
                wait_after_ticks,
                wait_after_ticks_random_range,
                position,
                is_pathing: false,
                is_monster_park: true,
            }),
        );
    }

    fn rotate_monster_park(
        &mut self,
        resources: &mut Resources,
        player_context: &mut PlayerContext,
        minimap_state: Minimap,
        key: MobbingKey,
    ) {
        let Minimap::Idle(idle) = minimap_state else {
            return;
        };

        // The "Use key mid double jump" checkbox (ui/src/dailies/mod.rs) is documented as shared
        // by Monster Park, same as `daily_quest_mobbing_key` just below - but unlike the mobbing
        // key, nothing was actually wiring it into `player_context.config` here, so Monster Park's
        // copy was left to whatever leaked in from the last regular map loaded instead of this
        // setting. Reapplied every tick (cheap, idempotent) so it can never go stale.
        player_context.config.auto_mob_use_key_while_double_jumping =
            self.daily_quest_use_key_while_double_jumping;

        // A genuine stage transition (the common case) reliably changes the minimap's bbox - it
        // requires a full anchor-mismatch -> re-detection cycle to get a new `MinimapIdle`, and
        // `bbox` only ever comes from a fresh `detect_minimap()` call, never per-tick jitter within
        // the same `Minimap::Idle` session (see `update_idle_state` in `minimap.rs`). Comparing it
        // tick-to-tick is a direct, reliable "did the map actually change" signal - unlike relying
        // solely on a new enemy being found soon to implicitly invalidate stale portal/enemy state
        // (the original design for `monster_park_last_portal`), which can race: if the map starts
        // with a brief window where enemies genuinely aren't detected yet, and the no-enemy
        // debounce was already saturated from the previous stage's tail end, the very first tick(s)
        // on the new stage can jump straight to "check portal" using the *previous* stage's now
        // meaningless last-known portal position - sending the player to try to reach/climb to a
        // spot that has nothing to do with this stage, exactly the kind of stuck-retrying-forever
        // loop this is meant to prevent.
        //
        // Not fully bulletproof - consecutive stages can occasionally share similar enough minimap
        // chrome that the anchor-mismatch check doesn't trip and `bbox` doesn't change (see
        // `monster_park_portal_task`'s docs) - but it catches the common case instead of relying
        // on the enemy-found fallback alone.
        //
        // The stage indicator text ("Stage N: <subtitle>", directly above the minimap) is a second,
        // independent signal for the exact case the bbox check can miss - see
        // `monster_park_stage_indicator_hash`'s docs, and `monster_park_stage_indicator_pending`'s
        // docs for why a bare tick-to-tick hash difference is debounced across several ticks before
        // being trusted enough to fire a reset here, rather than acted on immediately.
        const MONSTER_PARK_STAGE_INDICATOR_DEBOUNCE_COUNT: u32 = 3;
        let stage_indicator_hash = resources
            .detector()
            .detect_minimap_monster_park_stage_indicator_hash(idle.bbox);
        let stage_indicator_changed = match stage_indicator_hash {
            // No reading this tick - transient detection noise, not a signal either way. Leave
            // any in-progress debounce candidate untouched rather than restarting it from scratch.
            None => false,
            // Fresh reading matches the confirmed baseline (or there's no confirmed baseline yet
            // to compare against, e.g. the very first tick of a run) - nothing to debounce.
            Some(hash)
                if self
                    .monster_park_stage_indicator_hash
                    .is_none_or(|confirmed| confirmed == hash) =>
            {
                self.monster_park_stage_indicator_pending = None;
                false
            }
            Some(hash) => {
                let count = match self.monster_park_stage_indicator_pending {
                    Some((candidate, count)) if candidate == hash => count + 1,
                    _ => 1,
                };
                if count >= MONSTER_PARK_STAGE_INDICATOR_DEBOUNCE_COUNT {
                    self.monster_park_stage_indicator_pending = None;
                    true
                } else {
                    self.monster_park_stage_indicator_pending = Some((hash, count));
                    false
                }
            }
        };
        if self.monster_park_stage_bbox != Some(idle.bbox) || stage_indicator_changed {
            self.reset_monster_park_run_state();
            self.monster_park_stage_bbox = Some(idle.bbox);
        }
        if let Some(hash) = stage_indicator_hash {
            self.monster_park_stage_indicator_hash = Some(hash);
        }

        // A portal-stuck wiggle in progress takes over entirely until it's done - it deliberately
        // moves the player off `is_at_portal` for a moment, so letting any of the normal
        // enemy/portal logic below run concurrently would just fight it. Checked after the bbox
        // transition above so a genuine stage change mid-wiggle (rare, but e.g. if the last Up tap
        // of a previous attempt actually lands right as this one starts) still resets everything,
        // including the wiggle itself, instead of being missed until the wiggle finishes.
        if self.monster_park_portal_wiggle.is_some() {
            self.advance_monster_park_portal_wiggle(resources);
            return;
        }

        // A portal-stuck "escape" in progress takes over entirely until it's done - same
        // reasoning as the wiggle above.
        if self.monster_park_portal_escape.is_some() {
            self.advance_monster_park_portal_escape(resources, player_context.config.jump_key);
            return;
        }

        // Don't trust "no enemies" as meaningful until the stage's own timer has been seen at
        // least once - see `monster_park_timer_seen`'s docs for the false-clear this prevents.
        // Checked after the wiggle/escape early-returns above (those are recovery in progress from
        // a *previous* stage's tail end and should still run to completion) but before anything
        // that reads `monster_park_enemies`/the CLEAR banner.
        if !self.monster_park_timer_seen {
            if resources.detector().detect_monster_park_timer_visible() {
                self.monster_park_timer_seen = true;
                self.monster_park_timer_wait_since = None;
            } else {
                const MONSTER_PARK_TIMER_WAIT_TIMEOUT: Duration = Duration::from_secs(20);
                let waiting_since = *self
                    .monster_park_timer_wait_since
                    .get_or_insert_with(Instant::now);
                if waiting_since.elapsed() >= MONSTER_PARK_TIMER_WAIT_TIMEOUT {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: stage timer still not visible after \
                         {MONSTER_PARK_TIMER_WAIT_TIMEOUT:?}, proceeding anyway"
                    );
                    self.monster_park_timer_seen = true;
                } else {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: stage timer not visible yet, waiting for the stage to \
                         finish loading in"
                    );
                    return;
                }
            }
        }

        // Tolerance for matching a fresh detection back to a previously seen enemy dot, loose
        // enough to absorb detection jitter and the dot's own size (6x6) without being so loose it
        // matches a different, nearby enemy. Shared by both the still-pursuing-target check below
        // and the new-target acquisition debounce further down.
        //
        // This only ever needs to cover *one* scan interval's worth of movement, not the enemy's
        // total displacement since first acquired - the still-pursuing check below updates
        // `monster_park_target` to the latest matched position every time it succeeds (see there),
        // so the reference point tracks a wandering-but-alive enemy instead of staying frozen at
        // its initial detection. An earlier, tighter value (5) was picked to stop a dead target's
        // stale position from matching a different, still-alive neighboring dot (observed as close
        // as ~8px apart in a single stage) - but logs showed ordinary walking enemies moving ~6px
        // between consecutive scans, well past that, so the tighter threshold instead made the
        // still-pursuing check "lose" a perfectly alive, simply-moving enemy almost every scan,
        // discarding and immediately re-picking the same target in a busy-loop that never actually
        // lands a hit. Restored to comfortably cover that per-scan movement; the rarer neighbor-mob
        // mismatch this reopens is far less costly than that busy-loop.
        const MONSTER_PARK_TARGET_MATCH_THRESHOLD: i32 = 10;

        // Enemy dots are re-scanned continuously in the background (see
        // `monster_park_enemies_task`'s docs) - both the stale-target re-validation below and the
        // fresh-target pick further down read from this same always-refreshing cache instead of
        // each making their own synchronous detector call.
        match update_detection_task(
            resources,
            0,
            &mut self.monster_park_enemies_task,
            move |detector| Ok(detector.detect_minimap_monster_park_enemies(idle.bbox)),
        ) {
            Update::Ok(rects) => {
                self.monster_park_enemies = rects
                    .into_iter()
                    .map(|rect| {
                        // Flip coordinate to bottom-left, same convention as portals
                        let y = idle.bbox.height - rect.br().y;
                        Point::new(rect.x + rect.width / 2, y + rect.height / 2)
                    })
                    .collect();
            }
            Update::Err(_) | Update::Pending => (),
        }

        // The portal's minimap icon is present from the very start of a stage, well before the
        // map is clear of enemies (see `monster_park_last_portal`'s docs) - re-scanned right
        // alongside the enemy dots above instead of only once no enemies are left, so a position
        // caught early (before the player could plausibly be standing on top of it) is already
        // cached by the time it's actually needed. `idle.portals()` isn't used for the same
        // staleness/anchor reasons `monster_park_enemies_task`'s docs give for enemies.
        //
        // Unlike the enemy scan above, this stops entirely once a position is actually cached -
        // the portal doesn't move mid-stage (see `monster_park_last_portal`'s docs on when it's
        // cleared), so continuing to pay for this non-trivial-cost detection call every single
        // tick for the rest of a stage (which can run several minutes) would be pure waste with
        // nothing left to find.
        if self.monster_park_last_portal.is_none() {
            match update_detection_task(
                resources,
                0,
                &mut self.monster_park_portal_task,
                move |detector| Ok(detector.detect_minimap_portals(idle.bbox)),
            ) {
                Update::Ok(rects) => {
                    let detected_portal = rects
                        .into_iter()
                        .map(|rect| {
                            // Flip coordinate to bottom-left, same convention as enemies
                            let y = idle.bbox.height - rect.br().y;
                            Rect::new(rect.x, y, rect.width, rect.height)
                        })
                        .next();
                    if detected_portal.is_some() {
                        self.monster_park_last_portal = detected_portal;
                    }
                }
                Update::Err(_) | Update::Pending => (),
            }
        }

        // First check if player has a normal action, if so, let it complete. But if it is
        // pursuing a specific enemy (`monster_park_target`), keep re-validating that the enemy
        // is still there - otherwise a mob that dies mid-navigation (e.g. killed by a party
        // member or splash damage from another hit) goes unnoticed and the player keeps trying
        // to reach/attack an empty position until the movement's own repeated-state recovery
        // eventually fires, which does not even clear the action (see `abort_action_on_state_repeat`).
        if player_context.has_normal_action() {
            let Some(target) = self.monster_park_target else {
                // Not pursuing a specific enemy - a portal-approach `Move` is in flight instead.
                // See `monster_park_portal_approach_stuck_since`'s docs: that `Move` can be built
                // from a wrong frozen `target_y`, in which case it never reaches the portal and
                // this function never gets to recompute it on its own. Give up on a wall-clock
                // timeout regardless of what the movement machinery is doing internally.
                const MONSTER_PARK_PORTAL_APPROACH_STUCK_TIMEOUT: Duration =
                    Duration::from_secs(15);
                let stuck_since = *self
                    .monster_park_portal_approach_stuck_since
                    .get_or_insert_with(Instant::now);
                if stuck_since.elapsed() >= MONSTER_PARK_PORTAL_APPROACH_STUCK_TIMEOUT {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: portal approach stuck for over \
                         {MONSTER_PARK_PORTAL_APPROACH_STUCK_TIMEOUT:?}, giving up on the \
                         in-flight move to recompute"
                    );
                    self.monster_park_portal_approach_stuck_since = None;
                    player_context.reset_normal_action();
                } else {
                    // TODO: diagnostic only, remove once the "stuck idle" report is root-caused -
                    // this branch otherwise waits out the full 15s silently.
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: normal action {:?} in flight (no target), waiting \
                         (elapsed {:?})",
                        player_context.normal_action_name(),
                        stuck_since.elapsed()
                    );
                }
                return;
            };

            // A single scan can miss an enemy that is actually still there (transient capture
            // noise, dots overlapping), so this requires a consecutive repeat miss before
            // concluding the targeted enemy is actually dead - mirrors
            // `monster_park_no_enemy_count`'s reasoning below.
            const MONSTER_PARK_TARGET_MISSING_DEBOUNCE_COUNT: u32 = 3;

            // Picks the closest candidate within the match window (rather than just any) and
            // adopts it as the new `monster_park_target` - a wandering-but-alive enemy otherwise
            // keeps drifting further from its *original* acquisition point every scan, and once
            // that cumulative drift alone exceeds the match threshold, it would eventually get
            // wrongly declared missing even though every single scan along the way matched fine.
            // Rebasing onto the latest seen position each time means the threshold only ever has
            // to cover one scan interval's worth of movement, not the whole pursuit's total drift.
            let closest_present = self
                .monster_park_enemies
                .iter()
                .copied()
                .filter(|point| {
                    (point.x - target.x).abs() <= MONSTER_PARK_TARGET_MATCH_THRESHOLD
                        && (point.y - target.y).abs() <= MONSTER_PARK_TARGET_MATCH_THRESHOLD
                })
                .min_by_key(|point| (point.x - target.x).pow(2) + (point.y - target.y).pow(2));
            if let Some(point) = closest_present {
                // The in-flight `AutoMob` action's `position` is fixed at dispatch time and does
                // not itself track a moving enemy - only `monster_park_target` above does. Left at
                // that, an enemy that walks (horizontally - enemies don't move vertically, they're
                // bound to their platform) far enough from the position the player is actually
                // walking/jumping toward can make arriving there and attacking land nowhere near
                // the enemy's real current spot - a silent whiff, not the "targeted enemy no
                // longer detected" case this function otherwise guards against, since the enemy
                // *is* still being detected the whole time, just not where the action is heading.
                //
                // Only checked on `x` for that reason - `y` isn't expected to change for the same
                // still-alive enemy, so comparing it here would only risk a spurious redirect from
                // detection noise rather than catching anything real.
                //
                // Mirrors `AUTO_MOB_USE_KEY_X_THRESHOLD`, the same `x` distance
                // `update_from_auto_mob_action` uses to decide whether an attack would actually
                // connect: if the freshly matched position has drifted far enough from the old one
                // that an attack from there wouldn't land on the enemy's new spot either, redirect
                // the pursuit now via `dispatch_monster_park_auto_mob` (which resets the player to
                // `Idle` on the next update - see `set_normal_action`'s docs - so this takes effect
                // immediately) instead of waiting to arrive and whiff. Below the threshold, the
                // ongoing pursuit is left alone; it'll still land within tolerance.
                let drifted_out_of_reach =
                    (point.x - target.x).abs() > AUTO_MOB_USE_KEY_X_THRESHOLD;
                self.monster_park_target = Some(point);
                self.monster_park_target_missing_count = 0;
                if drifted_out_of_reach {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: targeted enemy walked from {target:?} to {point:?}, \
                         redirecting pursuit"
                    );
                    Self::dispatch_monster_park_auto_mob(player_context, key, point);
                    self.monster_park_target_stuck_since = None;
                    return;
                }

                // See `monster_park_target_stuck_since`'s docs: the enemy still being detected
                // every scan (the `closest_present` match above) only proves it's alive, not that
                // it's actually reachable - this is the backstop for the latter.
                const MONSTER_PARK_TARGET_STUCK_TIMEOUT: Duration = Duration::from_secs(15);
                let stuck_since = self
                    .monster_park_target_stuck_since
                    .get_or_insert_with(Instant::now);
                if stuck_since.elapsed() < MONSTER_PARK_TARGET_STUCK_TIMEOUT {
                    return;
                }

                debug!(
                    target: "backend/rotator",
                    "Monster Park: gave up on enemy at {point:?} after over \
                     {MONSTER_PARK_TARGET_STUCK_TIMEOUT:?} without reaching it, marking \
                     unreachable for this stage"
                );
                player_context.reset_normal_action();
                self.monster_park_target = None;
                self.monster_park_target_missing_count = 0;
                self.monster_park_target_stuck_since = None;
                self.monster_park_unreachable_targets.push(point);
                // Falls through to re-scan for a new target below instead of waiting for next
                // tick, same as the "no longer detected" abandon path below.
            } else {
                self.monster_park_target_missing_count =
                    self.monster_park_target_missing_count.saturating_add(1);
                if self.monster_park_target_missing_count
                    < MONSTER_PARK_TARGET_MISSING_DEBOUNCE_COUNT
                {
                    return;
                }

                debug!(target: "backend/rotator", "Monster Park: targeted enemy at {target:?} no longer detected, abandoning to re-scan");
                player_context.reset_normal_action();
                self.monster_park_target = None;
                self.monster_park_target_missing_count = 0;
                self.monster_park_target_stuck_since = None;
                // Falls through to re-scan for a new target below instead of waiting for next
                // tick.
            }
        }

        // Check if player has a priority action for portal usage
        if player_context.has_priority_action() {
            // TODO: diagnostic only, remove once the "stuck idle with a priority action pinned"
            // report is root-caused - names which priority action is blocking progress and for
            // how long, since this branch otherwise returns completely silently.
            debug!(
                target: "backend/rotator",
                "Monster Park: has priority action {:?}, waiting",
                player_context.priority_action_name()
            );
            return;
        }

        // After the final stage, a reward dialog (e.g. Spiegelmann's Monster Park completion
        // screen) blocks the player until dismissed. Detect its Next/End Chat button and press
        // the interact key to progress past it, instead of endlessly waiting for mobs or a
        // portal that no longer exist once the run is actually over.
        if resources
            .detector()
            .detect_popup_dialog_continue_button()
            .is_ok()
        {
            let key = Key {
                key: player_context.config.interact_key,
                key_hold_ticks: 0,
                key_hold_buffered_to_wait_after: false,
                link_key: LinkKeyKind::None,
                count: 1,
                position: None,
                direction: ActionKeyDirection::Any,
                with: ActionKeyWith::Stationary,
                wait_before_use_ticks: 5,
                wait_before_use_ticks_random_range: 0,
                wait_after_use_ticks: 0,
                wait_after_use_ticks_random_range: 0,
                wait_after_buffered: WaitAfterBuffered::None,
            };
            player_context.set_priority_action(None, PlayerAction::Key(key));
            debug!(target: "backend/rotator", "Monster Park: reward dialog detected, pressing interact key");
            return;
        }

        // A single tick's detection can miss enemies that are actually still on the minimap
        // (transient capture noise, dots overlapping each other), so require a couple of
        // consecutive empty detections before concluding the map is genuinely clear and it's time
        // to look for the portal - otherwise a single flaky tick sends the player off mid-sweep.
        // Kept as low as that still tolerates - each extra tick here is dead time added to every
        // single stage transition of a run. Declared here (an item, so visible for the rest of the
        // function regardless of textual position) since its use further down needs it.
        //
        // The game's own CLEAR/run-complete banner used to short-circuit this debounce (skipping
        // straight to "safe to approach the portal" the instant it was seen) - removed after it
        // was observed matching falsely, presumably from visual similarity to the stage-start
        // "STAGE N" banner shown at the very same moment a fresh stage begins, latching a stage as
        // cleared before a single monster had been fought. The debounce below, plus the elite-boss
        // check at the portal's attempt limit further down (for the one real scenario the banner
        // existed to protect - an occluded boss standing on the portal), replace it entirely.
        const MONSTER_PARK_NO_ENEMY_DEBOUNCE_COUNT: u32 = 2;

        let Some(pos) = player_context.last_known_pos else {
            return;
        };

        // Monster Park shows every remaining enemy as a red dot directly on the minimap, so
        // their position is ground truth - unlike regular auto mobbing there is no need for
        // camera-based mob detection, reachable-y correction, or ignore-xs/quadrant exploration
        // heuristics tied to a single map's geometry, none of which get reset across the several
        // maps a Monster Park run moves through.
        debug!(target: "backend/rotator", "Monster Park: {} enemy dot(s) in cache: {:?}", self.monster_park_enemies.len(), self.monster_park_enemies);
        let nearest_enemy = {
            self.monster_park_enemies
                .iter()
                .copied()
                // Skip dots already proven unreachable this stage (see
                // `monster_park_unreachable_targets`'s docs) - otherwise the very next tick would
                // just re-pick and re-chase the same dot [`Self::rotate_monster_park`] just gave
                // up on.
                .filter(|point| {
                    !self
                        .monster_park_unreachable_targets
                        .iter()
                        .any(|unreachable| {
                            (point.x - unreachable.x).abs() <= MONSTER_PARK_TARGET_MATCH_THRESHOLD
                                && (point.y - unreachable.y).abs()
                                    <= MONSTER_PARK_TARGET_MATCH_THRESHOLD
                        })
                })
                .min_by_key(|point| (point.x - pos.x).pow(2) + (point.y - pos.y).pow(2))
        };

        if let Some(point) = nearest_enemy {
            let was_no_enemy = self.monster_park_no_enemy_count > 0;
            self.monster_park_no_enemy_count = 0;

            // The background scan refreshes far more often than the old throttled synchronous
            // check did (as fast as the detector can go, not a few times a second), which makes a
            // single-scan false positive (transient capture noise, aliasing) much more likely to
            // be caught and committed to before it disappears on the very next scan. Require the
            // same dot to reappear across several scans before actually chasing it.
            //
            // Only applied once a "no enemy" gap has actually started (`was_no_enemy`) - false-
            // positive dots (appearing at scattered, unrelated positions well after a stage looked
            // clear - likely detection noise, not real enemies) were only ever observed in that
            // situation, consistently lasting for exactly 2 scans before vanishing. During ordinary
            // continuous combat (killing one dot, immediately picking up the next with no gap
            // between), every pick has instead been a genuine enemy, so gating the same strict
            // debounce there too was pure lost DPS - up to ~2 extra seconds of standing still
            // between every single kill for no benefit, compounding heavily over a whole stage.
            const MONSTER_PARK_NEW_TARGET_DEBOUNCE_COUNT: u32 = 4;
            const MONSTER_PARK_NEW_TARGET_DEBOUNCE_COUNT_ACTIVE: u32 = 1;
            let new_target_debounce_count = if was_no_enemy {
                MONSTER_PARK_NEW_TARGET_DEBOUNCE_COUNT
            } else {
                MONSTER_PARK_NEW_TARGET_DEBOUNCE_COUNT_ACTIVE
            };
            let seen_count = match self.monster_park_pending_target {
                Some((candidate, seen_count))
                    if (candidate.x - point.x).abs() <= MONSTER_PARK_TARGET_MATCH_THRESHOLD
                        && (candidate.y - point.y).abs() <= MONSTER_PARK_TARGET_MATCH_THRESHOLD =>
                {
                    seen_count + 1
                }
                _ => 1,
            };
            self.monster_park_pending_target = Some((point, seen_count));
            if seen_count < new_target_debounce_count {
                return;
            }

            // Deliberately NOT clearing `monster_park_last_portal` here - unlike the counters
            // below, it's not stale just because a new enemy showed up (see the field's docs: its
            // position doesn't change mid-stage, only genuine progress is a real stage transition,
            // handled by the `monster_park_stage_bbox` check instead).
            self.monster_park_no_portal_stuck_since = None;
            self.monster_park_no_portal_wiggle_count = 0;
            self.monster_park_portal_attempts = 0;
            self.monster_park_portal_give_up_cycles = 0;
            self.monster_park_portal_target_y = None;
            self.monster_park_portal_force_real_height = false;
            self.monster_park_portal_approach_stuck_since = None;
            self.monster_park_portal_stuck_since = None;
            self.monster_park_portal_wiggle = None;
            self.monster_park_portal_wiggle_toward_key = None;
            self.monster_park_portal_wiggled = false;
            self.monster_park_portal_escape = None;
            self.monster_park_target = Some(point);
            self.monster_park_target_missing_count = 0;
            self.monster_park_target_stuck_since = None;
            Self::dispatch_monster_park_auto_mob(player_context, key, point);
            return;
        }

        // No candidate to debounce against anymore - the enemy pursued next, whenever one shows
        // up, starts its own fresh count.
        self.monster_park_pending_target = None;

        self.monster_park_no_enemy_count = self.monster_park_no_enemy_count.saturating_add(1);
        if self.monster_park_no_enemy_count < MONSTER_PARK_NO_ENEMY_DEBOUNCE_COUNT {
            return;
        }
        // No longer relevant once the map is considered clear of enemies - clear so a future
        // normal action (moving to the portal) is never mistaken for still pursuing this enemy.
        self.monster_park_target = None;
        self.monster_park_target_missing_count = 0;
        self.monster_park_target_stuck_since = None;

        // The dot-based debounce just above only proves no enemy *dot matched a scan* - not that
        // the stage is actually clear: an enemy (most often the stage's own boss, which can end up
        // standing right on the portal) can be alive but invisible to the scan because its minimap
        // dot is occluded by the portal icon. Rather than stall here attacking blind on a guess (the
        // previous design, gated on the game's CLEAR banner - removed, see the debounce comment
        // above for why), head for the portal immediately: `nearest_enemy` above is recomputed
        // fresh every tick from the same continuously-refreshing scan, so a real enemy reappearing
        // mid-approach - occluded one becoming visible again, or a genuinely new one - is picked up
        // and chased on its own next tick same as any other, no special-casing needed here. The
        // occluded-boss-on-the-portal case this used to guard against is instead caught once at the
        // portal: see the elite-boss-HP-bar check in `is_at_portal`'s handling further down,
        // checked before ever pressing Up there.
        //
        // No enemies left on the minimap - `monster_park_last_portal` was already kept fresh by
        // the scan run alongside the enemy dots above (every tick, not just once the map looks
        // clear - see the field's docs for why), including whatever it last saw while enemies
        // were still around. Once the player is standing on/near the portal, their own minimap
        // marker can occlude the icon and make a scan come up briefly empty right when it matters
        // most to recognize arrival - relying on this already-cached value instead of demanding a
        // fresh hit this exact tick tolerates that without reintroducing staleness across a real
        // map change (see the field's docs for how).
        let Some(portal) = self.monster_park_last_portal else {
            // No portal found, Monster Park is complete for this map
            debug!(target: "backend/rotator", "Monster Park: No mobs and no portal found, map complete");

            // See `monster_park_no_portal_stuck_since`'s docs: if the portal has never once been
            // detected here, the player's own minimap marker may be permanently occluding it, and
            // nothing else in this branch moves the player to clear that. Escalate on a timeout
            // instead of looping this log line forever.
            const MONSTER_PARK_NO_PORTAL_STUCK_TIMEOUT: Duration = Duration::from_secs(5);
            const MONSTER_PARK_NO_PORTAL_WIGGLE_LIMIT: u32 = 3;
            let stuck_since = *self
                .monster_park_no_portal_stuck_since
                .get_or_insert_with(Instant::now);
            if stuck_since.elapsed() >= MONSTER_PARK_NO_PORTAL_STUCK_TIMEOUT {
                self.monster_park_no_portal_stuck_since = None;
                self.monster_park_no_portal_wiggle_count =
                    self.monster_park_no_portal_wiggle_count.saturating_add(1);
                if self.monster_park_no_portal_wiggle_count > MONSTER_PARK_NO_PORTAL_WIGGLE_LIMIT {
                    info!(
                        target: "backend/rotator",
                        "Monster Park: still no portal found after {} wiggle(s), trying to escape",
                        self.monster_park_no_portal_wiggle_count
                    );
                    self.monster_park_no_portal_wiggle_count = 0;
                    // No known correction direction here either - see
                    // `monster_park_portal_wiggle_toward_key`'s docs.
                    let away_key = if resources.rng.random_bool(0.5) {
                        KeyKind::Left
                    } else {
                        KeyKind::Right
                    };
                    self.monster_park_portal_escape = Some((away_key, PORTAL_ESCAPE_HOLD_TICKS));
                } else {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: no portal found for over \
                         {MONSTER_PARK_NO_PORTAL_STUCK_TIMEOUT:?}, wiggling to clear possible \
                         minimap occlusion"
                    );
                    // No known correction direction here (unlike `escalate_monster_park_portal_stuck`'s
                    // callers) - the portal itself hasn't even been found yet, so keep the
                    // original fixed Left-then-Right order.
                    self.monster_park_portal_wiggle_toward_key = None;
                    self.monster_park_portal_wiggle = Some((PortalWiggleStep::Left, 0));
                }
            }
            return;
        };
        // A portal was actually detected - clear the no-portal-found deadlock guard above, same
        // "genuine progress" reasoning as the rest of this function's state.
        self.monster_park_no_portal_stuck_since = None;
        self.monster_park_no_portal_wiggle_count = 0;

        // Calculate portal center position
        // Portal coordinates are already in bottom-left coordinate system
        //
        // The portal icon's bounding box is consistently detected a few pixels above the ground
        // the player actually needs to stand on (the icon's swirl graphic extends upward from its
        // usable base), so its raw vertical center overshoots the real target height. Subtract
        // that fixed offset so both the arrival check and movement targeting below aim at the
        // actual reachable ground instead of a point slightly above it.
        const PORTAL_GROUND_Y_OFFSET: i32 = 6;
        let portal_center_x = portal.x + portal.width / 2;
        let portal_center_y = portal.y + portal.height / 2 - PORTAL_GROUND_Y_OFFSET;
        let portal_center = Point::new(portal_center_x, portal_center_y);

        // Check if player is already at the portal. `is_position_inside_portal` uses the
        // detected portal Rect, which is padded well beyond the icon itself (see
        // PORTAL_EXPAND_SIZE) - that was letting this branch fire while the player was still
        // several pixels short of the portal's actual usable spot, wasting Up presses that
        // don't do anything. Require being genuinely close to the portal's center instead.
        //
        // The portal icon's detected height can also be off by a handful of pixels from the
        // player's actual reachable ground height (icon padding/centering imprecision). Left
        // alone, that was making the move step below ask for a short jump to close a "gap" that
        // isn't really there - since there's no real ledge to land on, the jump doesn't change
        // the player's height at all, so it retries the identical jump over and over until it
        // hits the movement-repeat abort. Both checks below use the same tolerance so a height
        // difference small enough to be treated as "same platform" when moving is also accepted
        // as "arrived" once there, instead of asking to move and then refusing to call it close
        // enough forever.
        //
        // X needs to be genuinely exact (not just "close enough") - the in-game portal trigger
        // is narrow enough that a few pixels off is enough for Up to do nothing, unlike Y where
        // being on the right platform is what matters. See `Adjusting::short_adjust_attempts`
        // for how the movement side avoids looping forever chasing this precision.
        //
        // Was `1` (a 1px tolerance) rather than `0` - observed live settling the player exactly
        // 1px short of `portal_center_x` after auto-mob combat pulled them off the portal and
        // back (a resting spot the wiggle/escape recoveries below also kept landing back on).
        // With a 1px tolerance that read as "arrived," so the rotator just kept pressing Up
        // there forever - Up genuinely does nothing at that offset in-game - instead of ever
        // issuing the ordinary `Move`/`Adjusting` walk (the `else` branch below) that would have
        // closed the last pixel exactly. `0` routes that case back through the normal movement
        // path instead of trusting a "close enough" read that isn't actually close enough.
        //
        // Wider than it looks: this isn't just "close enough to not bother re-targeting height."
        // Player::Moving's own jump trigger kicks in for *any* y-gap of 4px or more (see
        // `JUMPABLE_RANGE` in moving.rs), well inside what this offset/detection can drift by
        // between ticks (transient minimap-tracking noise, a recalibration, or a portal skin with
        // slightly different icon padding than `PORTAL_GROUND_Y_OFFSET` was measured against).
        // A too-tight threshold here lets that drift slip past as "different platform," freezing
        // a stale target height into a Move that then wastes several seconds jumping in place at
        // a gap that was never really there before self-recovering. Comfortably clearing the
        // largest gap seen in practice (and then some) keeps that from tripping in normal play,
        // while staying far below any genuine platform-to-platform gap (e.g. `GRAPPLING_THRESHOLD`
        // is 24).
        const PORTAL_ARRIVED_X_THRESHOLD: i32 = 0;
        const PORTAL_Y_THRESHOLD: i32 = 14;
        let is_at_portal = (pos.x - portal_center_x).abs() <= PORTAL_ARRIVED_X_THRESHOLD
            && (pos.y - portal_center_y).abs() <= PORTAL_Y_THRESHOLD;

        if is_at_portal {
            // The key that would close the current (sub-`PORTAL_ARRIVED_X_THRESHOLD`) horizontal
            // gap, passed to `escalate_monster_park_portal_stuck` below so its wiggle recovery
            // biases toward actually fixing this specific misalignment - see
            // `monster_park_portal_wiggle_toward_key`'s docs for why a fixed direction isn't
            // enough. Arbitrarily defaults to `Right` when already exactly centered (`portal_center_x
            // == pos.x`); the direction is meaningless in that case since there is no gap to close.
            let portal_wiggle_toward_key = Some(if portal_center_x >= pos.x {
                KeyKind::Right
            } else {
                KeyKind::Left
            });

            // Checked first, before ever pressing Up - this used to be gated behind
            // `MONSTER_PARK_PORTAL_ATTEMPT_LIMIT` failed Up presses below instead, which wasted a
            // few Up presses (each with its own `wait_before_use_ticks`) every time the boss was
            // still simply occluded. The game's boss HP bar is its own authoritative "is something
            // still alive" signal, independent
            // of whether that boss's minimap dot is currently visible to `nearest_enemy` upstream
            // (most often occluded by the portal icon, since the boss tends to end up standing
            // right on it). If it's still up, the stage genuinely isn't clear yet, so there's no
            // point ever pressing Up here - keep attacking blind at the current position instead.
            //
            // Also skips the stuck-timer/attempt bookkeeping below entirely while the boss is
            // confirmed alive, rather than letting it run in the background: that timer doesn't
            // care why `is_at_portal` has stayed true, so a boss fight lasting past
            // `MONSTER_PARK_PORTAL_STUCK_TIMEOUT` was observed tripping the wiggle/escape recovery
            // - which shoves the player off the portal - in the middle of an still-ongoing fight.
            // `monster_park_portal_stuck_since` is reset here too so it starts fresh once the boss
            // actually dies, instead of resuming from a clock that was already expired from
            // ticking untouched throughout the fight.
            if resources.detector().detect_elite_boss_bar() {
                debug!(
                    target: "backend/rotator",
                    "Monster Park: boss HP bar still visible at the portal, attacking in place \
                     instead of pressing Up"
                );
                self.monster_park_portal_attempts = 0;
                self.monster_park_portal_stuck_since = None;
                Self::dispatch_monster_park_auto_mob(player_context, key, pos);
                return;
            }

            // Hard wall-clock backstop, checked before the attempt/give-up-cycle counters below:
            // those counters are reset by a single tick of `is_at_portal` flickering false (see
            // the `else` branch), which pixel-level detection noise on `PORTAL_ARRIVED_X_THRESHOLD`
            // can cause without the player actually having moved - so a run of that flicker can
            // keep the counters from ever reaching their limit and escalating, leaving the player
            // pressing Up at the portal indefinitely (what this is guarding against). This timer
            // isn't reset by that same flicker (see `monster_park_portal_stuck_since`'s docs), so
            // it guarantees escalation within a bounded time regardless of how the counters below
            // are behaving.
            const MONSTER_PARK_PORTAL_STUCK_TIMEOUT: Duration = Duration::from_secs(10);
            let stuck_since = *self
                .monster_park_portal_stuck_since
                .get_or_insert_with(Instant::now);
            if stuck_since.elapsed() >= MONSTER_PARK_PORTAL_STUCK_TIMEOUT {
                debug!(
                    target: "backend/rotator",
                    "Monster Park: stuck at the portal for over {MONSTER_PARK_PORTAL_STUCK_TIMEOUT:?}"
                );
                self.escalate_monster_park_portal_stuck(resources, portal_wiggle_toward_key);
                return;
            }

            // A real map transition always resets `monster_park_portal_attempts` back to 0 via
            // `reset_queue` (triggered by the map-change event rebuilding the rotator). If Up has
            // been pressed several times in a row while still reading as "at the portal" and that
            // never happened, the map genuinely isn't changing - most likely because the earlier
            // "no enemies left" conclusion was wrong (e.g. the player's own minimap marker was
            // occluding a still-alive enemy's dot) rather than the Up press itself failing.
            const MONSTER_PARK_PORTAL_ATTEMPT_LIMIT: u32 = 3;
            // Past this many give-up-and-re-scan cycles in a row with no progress, re-scanning
            // again isn't going to break the cycle by itself - escalate instead of repeating it
            // forever. See `Self::monster_park_portal_give_up_cycles`'s docs.
            const MONSTER_PARK_PORTAL_GIVE_UP_CYCLE_LIMIT: u32 = 3;
            if self.monster_park_portal_attempts >= MONSTER_PARK_PORTAL_ATTEMPT_LIMIT {
                debug!(
                    target: "backend/rotator",
                    "Monster Park: pressed Up {} times without the map changing, re-checking for enemies",
                    self.monster_park_portal_attempts
                );
                self.monster_park_no_enemy_count = 0;
                // Deliberately NOT clearing `monster_park_last_portal` here. The player is
                // standing on/near the portal right now - exactly the situation where their own
                // minimap marker is most likely occluding the portal icon (see the field's docs).
                // Clearing it would force the next attempt to depend on a fresh scan that's likely
                // to keep failing for that same reason, and since nothing else re-establishes it
                // while the player never moves off the portal, `last_portal` could stay `None`
                // forever - permanently blocking the `let Some(portal) = ... else { return }` gate
                // below on every future tick. That would make this whole retry/escalation path
                // (including `monster_park_portal_give_up_cycles`'s `Unstuck` escalation) silently
                // unreachable, leaving the player stuck in `Player::Idle` forever instead of
                // retrying Up. Keeping the last known position lets `is_at_portal` keep evaluating
                // true independent of whether a fresh scan succeeds, so retries actually continue.
                self.monster_park_portal_attempts = 0;
                self.monster_park_portal_target_y = None;
                self.monster_park_portal_give_up_cycles += 1;
                if self.monster_park_portal_give_up_cycles
                    >= MONSTER_PARK_PORTAL_GIVE_UP_CYCLE_LIMIT
                {
                    debug!(
                        target: "backend/rotator",
                        "Monster Park: stuck at the same portal for {} give-up cycles in a row",
                        self.monster_park_portal_give_up_cycles
                    );
                    self.escalate_monster_park_portal_stuck(resources, portal_wiggle_toward_key);
                    return;
                }
                // An ordinary (non-elite) mob standing exactly on the portal is invisible to both
                // the minimap dot scan above (occluded by the portal icon, same as the elite-boss
                // case) and `detect_elite_boss_bar` below (elite-only - a regular mob never shows
                // an HP bar), so nothing else here would ever catch it, and Up would keep doing
                // nothing for as long as it's alive. Try attacking blind in place before the next
                // give-up cycle resumes pressing Up - harmless if wrong (same reasoning as the
                // elite-boss branch: with nothing actually there, this just whiffs once), but
                // clears the case where something is.
                debug!(
                    target: "backend/rotator",
                    "Monster Park: attacking in place in case an undetected mob is occluding the \
                     portal"
                );
                Self::dispatch_monster_park_auto_mob(player_context, key, pos);
                return;
            }
            self.monster_park_portal_attempts += 1;

            // Press Up in place instead of targeting the portal's detected center as a movement
            // destination - a `position` here would route through Player::Moving first, and any
            // remaining pixel of y mismatch is enough to cross the up-jump threshold and send
            // the player jumping in place instead of using the portal.
            let key = Key {
                key: KeyKind::Up,
                key_hold_ticks: 0,
                key_hold_buffered_to_wait_after: false,
                link_key: LinkKeyKind::None,
                count: 1,
                position: None,
                direction: ActionKeyDirection::Any,
                with: ActionKeyWith::Stationary,
                wait_before_use_ticks: 5,
                wait_before_use_ticks_random_range: 0,
                wait_after_use_ticks: 0,
                wait_after_use_ticks_random_range: 0,
                wait_after_buffered: WaitAfterBuffered::None,
            };
            player_context.set_priority_action(None, PlayerAction::Key(key));
            debug!(target: "backend/rotator", "Monster Park: Player at portal, pressing Up key");
        } else {
            // Not at the portal (yet, or pushed off it), so the attempt streak from a previous
            // arrival no longer applies.
            self.monster_park_portal_attempts = 0;
            self.monster_park_portal_give_up_cycles = 0;

            // Player is not at portal, move towards it. Only target the detected height when
            // it's far enough from the player's current height to plausibly be a real platform
            // change - otherwise just walk there horizontally at the player's current height,
            // per the reasoning above.
            //
            // That shortcut only makes sense once the player is already horizontally near the
            // portal - `pos.y` at this exact tick could otherwise be read while still mid-jump/
            // mid-grapple on a completely different, distant platform that happens to sit within
            // `PORTAL_Y_THRESHOLD` of the portal's height by coincidence. Freezing that transient
            // reading as the target then has the player climb back to it once they actually reach
            // the portal's platform and its real (different) height takes over, instead of just
            // heading straight for the portal's real height from the start.
            //
            // Gated tightly (much tighter than `DOUBLE_JUMP_THRESHOLD`) rather than on "close
            // enough that Player::Moving itself wouldn't need a double jump" as originally
            // reasoned - live testing showed that boundary isn't actually safe: a double jump can
            // still end up firing between this decision and `Player::Moving`'s own fresh
            // recomputation a moment later (position drift across that gap), which invalidates
            // the "already on this platform" assumption `target_y` relies on. Once that happens,
            // the double jump (and whatever climb follows) can land short of the frozen target
            // height, and the remaining small gap can fall into a range `Player::Jumping` can't
            // reliably close, retrying indefinitely. Tightening this substantially shrinks that
            // window; it doesn't eliminate it, hence the escape hatch below.
            const SAME_PLATFORM_X_THRESHOLD: i32 = 3;
            // Consumed immediately (one-shot) - see `monster_park_portal_force_real_height`'s docs
            // for why an escalation forces exactly the next `Move` to the real height rather than
            // permanently disabling `same_platform`.
            let force_real_height = std::mem::take(&mut self.monster_park_portal_force_real_height);
            let same_platform = !force_real_height
                && (portal_center_x - pos.x).abs() < SAME_PLATFORM_X_THRESHOLD
                && (portal_center_y - pos.y).abs() <= PORTAL_Y_THRESHOLD;
            let mut target_y = if same_platform {
                pos.y
            } else {
                portal_center_y
            };
            if force_real_height {
                debug!(
                    target: "backend/rotator",
                    "Monster Park: forcing real detected height {portal_center_y} after a portal-\
                     stuck escalation, bypassing same_platform"
                );
            }

            // Escape hatch: if the same (wrong) target_y keeps getting reissued without ever
            // reaching the portal, re-deriving it the same way again isn't going to help - force
            // the real detected height instead, guaranteeing convergence regardless of why the
            // frozen value was wrong in the first place. See `monster_park_portal_target_y`'s docs.
            const MONSTER_PARK_PORTAL_TARGET_Y_REPEAT_LIMIT: u32 = 3;
            let repeats = match self.monster_park_portal_target_y {
                Some((last_target_y, repeats)) if last_target_y == target_y => repeats + 1,
                _ => 0,
            };
            if repeats >= MONSTER_PARK_PORTAL_TARGET_Y_REPEAT_LIMIT {
                debug!(
                    target: "backend/rotator",
                    "Monster Park: target y {target_y} reissued {repeats} times without reaching \
                     the portal, forcing real detected height {portal_center_y} instead"
                );
                target_y = portal_center_y;
                self.monster_park_portal_target_y = None;
                // `monster_park_portal_stuck_since` deliberately ignores ordinary `is_at_portal`
                // flicker (see its docs) so it keeps ticking through the whole "frozen at a
                // slightly-wrong height, within `PORTAL_Y_THRESHOLD` so `is_at_portal` reads true
                // the whole time, pressing Up, giving up, re-targeting the same wrong height
                // again" dance above - by design, so genuinely endless flailing still gets caught.
                // But by the time this escape hatch fires, that clock has typically already been
                // running the entire dance and can be at or past `MONSTER_PARK_PORTAL_STUCK_TIMEOUT`
                // already - so the very next tick the player finally arrives at the *correct*
                // height, `is_at_portal` reads true, and the stuck check immediately escalates to
                // wiggle/give-up before ever pressing Up at the position that could actually work.
                // Forcing the real height here is a genuine change in strategy that's likely to
                // succeed, unlike the flicker this timer is meant to see through - it earns a
                // fresh budget instead of inheriting a clock that was already expired for a
                // different, now-abandoned target height.
                self.monster_park_portal_stuck_since = None;
            } else {
                self.monster_park_portal_target_y = Some((target_y, repeats));
            }

            let position = Position {
                x: portal_center_x,
                x_random_range: 0,
                y: target_y,
                allow_adjusting: true,
            };
            player_context.set_normal_action(
                None,
                PlayerAction::Move(Move {
                    position,
                    wait_after_move_ticks: 0,
                    is_monster_park: true,
                }),
            );
            // A fresh attempt gets a fresh stuck budget - see
            // `monster_park_portal_approach_stuck_since`'s docs.
            self.monster_park_portal_approach_stuck_since = None;
            debug!(target: "backend/rotator", "Monster Park: Moving to portal at {:?} (target y {target_y})", portal_center);
        }
    }

    fn rotate_monster_park_entry(
        &mut self,
        player_context: &mut PlayerContext,
        minimap_state: Minimap,
    ) {
        // This is only ever called once the caller (`rotate_action`'s Monster Park
        // dispatch) has freshly, synchronously confirmed the player is genuinely back in the
        // entry lobby - i.e. the previous run (if any) has definitely ended. Reset the run-scoped
        // Monster Park state right here rather than waiting until actually standing at the gate:
        // a run's end and the next run's start happen without a rotator rebuild in between (same
        // "MP" map config the whole time), so without resetting somewhere in between, the previous
        // run's leftover state (last portal position, an already-saturated no-enemy counter) could
        // be mistaken for this run's own progress once enemy scanning restarts. Resetting as soon
        // as the lobby is confirmed - instead of only right before the gate hand-off - closes that
        // gap immediately rather than relying on `rotate_monster_park`'s own early-return checks to
        // stay safe for however long it takes to walk to the gate. Harmless to repeat on every
        // call while still walking there (already-zeroed fields, cheap).
        self.reset_monster_park_run_state();
        // Mirrors the reasoning above: confirmed back at the lobby means no run is in progress
        // anymore, regardless of whether one just ended or this is an earlier, still-failed
        // attempt to get past the gate. See the field's docs for why this can't just be inferred
        // from `monster_park_at_entry_lobby`.
        self.monster_park_run_in_progress = false;

        if player_context.has_normal_action() {
            return;
        }
        if player_context.has_priority_action() {
            return;
        }

        // Being back in the lobby after the run started means the single maxed-out-sweeps run
        // (see `monster_park_sweep`'s docs) is over.
        if self.monster_park_sweep == MonsterParkSweep::RunStarted {
            self.monster_park_sweep = MonsterParkSweep::Finished;
        }
        // Nothing left to do here today - leave right away, same as the attempt-limit give-up
        // below. Repeated on every call while still in the lobby, so a shuttle exit that silently
        // failed gets retried rather than walking back to the gate.
        if self.monster_park_sweep == MonsterParkSweep::Finished {
            info!(
                target: "backend/rotator",
                "Monster Park: all free clears used up, leaving via the shuttle and disabling \
                 Monster Park for the rest of the session"
            );
            self.inject_action(PlayerAction::ExitMonsterPark);
            self.monster_park_giving_up = true;
            return;
        }

        let Minimap::Idle(_) = minimap_state else {
            return;
        };
        let Some(pos) = player_context.last_known_pos else {
            return;
        };

        // Fixed gate x positions on Monster Park's entry lobby map. This hub's layout is static
        // game content shared by everyone, not something users configure per grinding map, so
        // these are hardcoded rather than read from a configurable bound like other modes.
        const GATE_X_UNDER_LEVEL_260: i32 = 109;
        const GATE_X_LEVEL_260_AND_ABOVE: i32 = 120;
        const GATE_Y: i32 = 0;
        const GATE_ARRIVED_THRESHOLD: i32 = 1;

        let target_x = if self.character_level < 260 {
            GATE_X_UNDER_LEVEL_260
        } else {
            GATE_X_LEVEL_260_AND_ABOVE
        };
        let is_at_gate = (pos.x - target_x).abs() <= GATE_ARRIVED_THRESHOLD
            && (pos.y - GATE_Y).abs() <= GATE_ARRIVED_THRESHOLD;

        if !is_at_gate {
            let position = Position {
                x: target_x,
                x_random_range: 0,
                y: GATE_Y,
                allow_adjusting: true,
            };
            player_context.set_normal_action(
                None,
                PlayerAction::Move(Move {
                    position,
                    wait_after_move_ticks: 0,
                    is_monster_park: false,
                }),
            );
            debug!(target: "backend/rotator", "Monster Park Entry: moving to gate at {:?}", position);
            return;
        }

        // Past this many consecutive attempts that never actually leave the lobby, further
        // retries aren't going to do anything different - most likely the day's free entries
        // (Monster Park is capped per character per day) are used up, so the dungeon-select
        // dialog staying open / the Enter click doing nothing will just keep repeating forever.
        // Stop clicking through it and leave via the shuttle instead of busy-looping for the rest
        // of the session.
        const MONSTER_PARK_ENTRY_ATTEMPT_LIMIT: u32 = 3;
        if self.monster_park_entry_attempts >= MONSTER_PARK_ENTRY_ATTEMPT_LIMIT {
            info!(
                target: "backend/rotator",
                "Monster Park: entry failed {} times in a row (likely no free entries left today), \
                 leaving via the shuttle and disabling Monster Park for the rest of the session",
                self.monster_park_entry_attempts
            );
            // Injected as a side-loaded action (rather than a normal/priority one set directly)
            // so it takes effect regardless of whatever this rotator would otherwise have decided
            // to do next - `rotate_priority_actions` drains it unconditionally on the very next
            // tick, same as `PlayerAction::Unstuck` elsewhere.
            self.inject_action(PlayerAction::ExitMonsterPark);
            // There being no free entries left is a whole-day condition, not a transient failure -
            // retrying would just repeat this exact same 3-attempt failure loop forever, so this is
            // still meant to fall through to daily quests and this rotator's normal mode for the
            // rest of the session instead of leaving the character idle. But `monster_park_enabled`
            // isn't cleared here - see `monster_park_giving_up`'s docs for why disabling it before
            // the exit is actually confirmed to have worked would let normal rotation start right
            // there in the lobby instead of back in town. It's cleared once that's confirmed,
            // below. Reset back to `true` on the next full rotator rebuild (`build_actions`), same
            // as any other session/config restart.
            self.monster_park_giving_up = true;
            return;
        }
        self.monster_park_entry_attempts += 1;

        // At the gate - hand off to `Player::EnteringMonsterPark`, which presses Up and drives
        // the whole dungeon-select dialog from here. This rotator's job stops at getting the
        // player to the right spot. Run state was already reset when this function was first
        // entered (see the top of the function), so nothing further to clear here.
        player_context.set_priority_action(None, PlayerAction::EnterMonsterPark);
        // From here through however many stages the run has, the world map isn't reachable - see
        // the field's docs for why the daily quest gate needs this instead of relying on a live
        // lobby check.
        self.monster_park_run_in_progress = true;
        debug!(target: "backend/rotator", "Monster Park Entry: at gate, entering Monster Park");
    }

    /// Resets the run-scoped Monster Park state tracked across [`Self::rotate_monster_park`]
    /// ticks (enemy/portal caches and their debounce counters).
    ///
    /// Called on a full rotator rebuild ([`Self::reset_queue`]), as soon as the player is confirmed
    /// back in the entry lobby ([`Self::rotate_monster_park_entry`]'s entry point), and on every
    /// detected stage transition mid-run ([`Self::rotate_monster_park`]'s bbox comparison) - state
    /// from whatever stage was previously active (last portal position, an already-saturated
    /// no-enemy counter, cached enemy dots) is meaningless for a new stage/run and would otherwise
    /// be mistaken for this one's own progress.
    #[inline]
    fn reset_monster_park_run_state(&mut self) {
        self.monster_park_no_enemy_count = 0;
        self.monster_park_portal_task = None;
        self.monster_park_last_portal = None;
        self.monster_park_no_portal_stuck_since = None;
        self.monster_park_no_portal_wiggle_count = 0;
        self.monster_park_portal_attempts = 0;
        self.monster_park_portal_give_up_cycles = 0;
        self.monster_park_portal_target_y = None;
        self.monster_park_portal_force_real_height = false;
        self.monster_park_portal_approach_stuck_since = None;
        self.monster_park_portal_stuck_since = None;
        self.monster_park_portal_wiggle = None;
        self.monster_park_portal_wiggle_toward_key = None;
        self.monster_park_portal_wiggled = false;
        self.monster_park_portal_escape = None;
        self.monster_park_target = None;
        self.monster_park_target_missing_count = 0;
        self.monster_park_target_stuck_since = None;
        self.monster_park_unreachable_targets.clear();
        self.monster_park_pending_target = None;
        self.monster_park_enemies_task = None;
        self.monster_park_enemies.clear();
        self.monster_park_stage_bbox = None;
        self.monster_park_stage_indicator_hash = None;
        self.monster_park_stage_indicator_pending = None;
        self.monster_park_timer_seen = false;
        self.monster_park_timer_wait_since = None;
    }

    fn rotate_ping_pong(
        &mut self,
        player_context: &mut PlayerContext,
        minimap_state: Minimap,
        key: MobbingKey,
        bound: Bound,
    ) {
        if player_context.has_normal_action() {
            return;
        }

        let Minimap::Idle(idle) = minimap_state else {
            return;
        };
        let Some(pos) = player_context.last_known_pos else {
            return;
        };

        let bbox = idle.bbox;
        let dist_left = pos.x - bbox.x;
        let dist_right = (bbox.x + bbox.width) - pos.x;
        let direction = if dist_left > dist_right {
            PingPongDirection::Left
        } else {
            PingPongDirection::Right
        };
        let bound = Rect::new(
            bound.x,
            bbox.height - (bound.y + bound.height),
            bound.width,
            bound.height,
        );

        player_context.set_normal_action(
            None,
            PlayerAction::PingPong(PingPong {
                key: key.key.into(),
                key_hold_ticks: (key.key_hold_millis / MS_PER_TICK) as u32,
                link_key: key.link_key.into(),
                count: key.count.max(1),
                with: key.with,
                wait_before_ticks: (key.wait_before_millis / MS_PER_TICK) as u32,
                wait_before_ticks_random_range: (key.wait_before_millis_random_range / MS_PER_TICK)
                    as u32,
                wait_after_ticks: (key.wait_after_millis / MS_PER_TICK) as u32,
                wait_after_ticks_random_range: (key.wait_after_millis_random_range / MS_PER_TICK)
                    as u32,
                bound,
                direction,
            }),
        );
    }

    fn rotate_start_to_end(&mut self, player_context: &mut PlayerContext) {
        if player_context.has_normal_action() || self.normal_actions.is_empty() {
            return;
        }
        if self.rotate_queuing_linked_action(player_context, false) {
            return;
        }

        debug_assert!(self.normal_index < self.normal_actions.len());
        let (id, action) = self.normal_actions[self.normal_index].clone();
        self.normal_index = (self.normal_index + 1) % self.normal_actions.len();
        match action {
            RotatorAction::Single(action) => {
                player_context.set_normal_action(Some(id), action);
            }
            RotatorAction::Linked(action) => {
                self.normal_queuing_linked_action = Some((id, Box::new(action)));
                self.rotate_queuing_linked_action(player_context, false);
            }
        }
    }

    fn rotate_start_to_end_then_reverse(&mut self, player_context: &mut PlayerContext) {
        if player_context.has_normal_action() || self.normal_actions.is_empty() {
            return;
        }
        if self.rotate_queuing_linked_action(player_context, false) {
            return;
        }

        let len = self.normal_actions.len();
        if (self.normal_index + 1) == len {
            self.normal_actions_backward = !self.normal_actions_backward;
            self.normal_index = 0;
        }

        debug_assert!(self.normal_index < self.normal_actions.len());

        let i = if self.normal_actions_backward {
            (len - self.normal_index).saturating_sub(1)
        } else {
            self.normal_index
        };
        let (id, action) = self.normal_actions[i].clone();

        self.normal_index = (self.normal_index + 1) % len;
        match action {
            RotatorAction::Single(action) => {
                player_context.set_normal_action(Some(id), action);
            }
            RotatorAction::Linked(action) => {
                self.normal_queuing_linked_action = Some((id, Box::new(action)));
                self.rotate_queuing_linked_action(player_context, false);
            }
        }
    }

    #[inline]
    fn rotate_queuing_linked_action(
        &mut self,
        player_context: &mut PlayerContext,
        is_priority: bool,
    ) -> bool {
        let linked_action = if is_priority {
            &mut self.priority_queuing_linked_action
        } else {
            &mut self.normal_queuing_linked_action
        };
        if linked_action.is_none() {
            return false;
        }
        let (id, action) = linked_action.take().unwrap();
        *linked_action = action.next.map(|action| (id, action));
        if is_priority {
            player_context.set_priority_action(Some(id), action.inner);
        } else {
            player_context.set_normal_action(Some(id), action.inner);
        }
        true
    }

    #[inline]
    fn rotate_side_priority_action(&mut self, player_context: &mut PlayerContext) -> bool {
        if let Some(action) = self.priority_actions_side_queue.pop_front() {
            debug_assert!(!player_context.has_priority_action());
            match action {
                RotatorAction::Single(action) => {
                    player_context.set_priority_action(None, action);
                }
                RotatorAction::Linked(_) => unreachable!(),
            }
            return true;
        }

        false
    }
}

impl Rotator for DefaultRotator {
    #[inline]
    fn is_navigating_daily_quest(&self) -> bool {
        self.daily_quest_navigating && !self.daily_quest_arrived
    }

    #[inline]
    fn is_monster_park_active(&self) -> bool {
        self.monster_park_enabled
    }

    #[inline]
    fn is_idle_mode_active(&self) -> bool {
        matches!(self.normal_rotate_mode, RotatorMode::Idle)
    }

    #[cfg_attr(test, concretize)]
    fn build_actions(&mut self, args: RotatorBuildArgs) {
        info!(target: "backend/rotator", "preparing actions {args:?}");
        let RotatorBuildArgs {
            mode,
            character_level,
            character_actions,
            map_actions,
            buffs,
            familiars,
            familiars_fusion,
            familiars_potential,
            familiar_essence_key,
            elite_boss_behavior,
            elite_boss_behavior_key,
            hexa_booster_exchange_condition,
            hexa_booster_exchange_amount,
            hexa_booster_exchange_all,
            enable_panic_mode,
            enable_rune_solving,
            enable_transparent_shape_solving,
            enable_violetta_solving,
            enable_reset_normal_actions_on_erda,
            enable_using_generic_booster,
            enable_using_hexa_booster,
            daily_quest_entries,
            daily_quest_mobbing_key,
            daily_quest_character_id,
            daily_quest_use_key_while_double_jumping,
            monster_park_enabled,
            thread_of_fate_ask_enabled,
            thread_of_fate_ask_npc,
            thread_of_fate_ask_progress,
            thread_of_fate_ask_character_id,
        } = args;
        self.reset_queue();
        self.normal_actions.clear();
        self.normal_rotate_mode = mode;
        self.character_level = character_level;
        self.normal_actions_reset_on_erda = enable_reset_normal_actions_on_erda;
        self.priority_actions.clear();
        self.daily_quest_entries = daily_quest_entries;
        self.daily_quest_mobbing_key = daily_quest_mobbing_key;
        self.daily_quest_character_id = daily_quest_character_id;
        self.daily_quest_use_key_while_double_jumping = daily_quest_use_key_while_double_jumping;
        self.monster_park_enabled = monster_park_enabled;
        {
            // `thread_of_fate_ask_progress` is the persisted copy re-read from the character on
            // every rebuild (e.g. an unrelated settings change while the bot keeps running), which
            // never itself carries `blocked_by_reset` true - that guard only ever gets set live,
            // in-memory (see `ThreadOfFateAskProgress::reset_if_new_day`'s docs), and is only ever
            // cleared by `clear_thread_of_fate_ask_reset_block` once the bot is actually restarted.
            // Blindly overwriting the whole struct here would silently drop it on the very next
            // unrelated rebuild, so it's carried forward instead - but only while still building
            // for the same character; switching characters must not carry one's block onto another.
            let mut thread_of_fate_ask_progress = thread_of_fate_ask_progress;
            if self.thread_of_fate_ask_character_id == thread_of_fate_ask_character_id {
                let previously_blocked = self
                    .thread_of_fate_ask_progress
                    .lock()
                    .unwrap()
                    .blocked_by_reset;
                thread_of_fate_ask_progress.blocked_by_reset = previously_blocked;
            }
            self.thread_of_fate_ask_character_id = thread_of_fate_ask_character_id;
            *self.thread_of_fate_ask_progress.lock().unwrap() = thread_of_fate_ask_progress;
        }
        self.daily_quest_saved_pathing = None;
        self.daily_quest_index = 0;

        // Monster Park is a tight, timed sweep-then-portal loop across several maps - buffs,
        // boosters and Erda Shower off-cooldown actions are disruptive there (e.g. stopping to
        // cast mid-sweep, or having to re-detect them fresh in every new map) and not worth
        // interrupting the loop for, so this mode (including its entry lobby) ignores them
        // entirely.
        let is_monster_park = monster_park_enabled;
        // Idle stands in place and does nothing besides priority actions unrelated to combat
        // upkeep (e.g. Thread of Fate "Ask", lie detector solving) - buffs and boosters have
        // nothing to prepare for here, so skip them same as Monster Park skips them mid-sweep.
        let is_idle = matches!(mode, RotatorMode::Idle);

        // Low priority
        if enable_using_generic_booster && !is_monster_park && !is_idle {
            self.priority_actions.insert(
                next_action_id(),
                use_booster_priority_action(Booster::Generic),
            );
        }

        if enable_using_hexa_booster && !is_monster_park && !is_idle {
            self.priority_actions
                .insert(next_action_id(), use_booster_priority_action(Booster::Hexa));
        }

        if !is_idle
            && !matches!(
                hexa_booster_exchange_condition,
                ExchangeHexaBoosterCondition::None
            )
        {
            self.priority_actions.insert(
                next_action_id(),
                exchange_hexa_booster_priority_action(
                    hexa_booster_exchange_condition,
                    hexa_booster_exchange_amount,
                    hexa_booster_exchange_all,
                ),
            );
        }

        // Same reasoning as the boosters above - not worth interrupting Monster Park's tight
        // sweep-then-portal loop for.
        //
        // Skipped entirely while fusion is also enabled: `Player::FamiliarsFusing` unconditionally
        // chains into a `Player::FamiliarsSwapping` run on every completion (successful rank-up,
        // no eligible target, or failure alike - see its own docs on why), so fusion's own
        // `fusion_check_millis` timer already guarantees periodic swapping. Registering this as a
        // second, independent priority action on its own `swap_check_millis` timer raced against
        // that chain: the chained swap never touches this action's `last_queued_time` (it bypasses
        // the priority-action queue entirely), so a few seconds after the chained swap closed the
        // familiar menu, this timer would elapse on its own schedule and reopen the menu again for
        // a completely redundant swap pass.
        if familiars.enable_familiars_swapping
            && !is_monster_park
            && !familiars_fusion.enable_familiars_fusion
        {
            self.priority_actions.insert(
                next_action_id(),
                familiars_swap_priority_action(
                    FamiliarsSwap {
                        swappable_slots: familiars.swappable_familiars,
                        swappable_rarities: Array::from_iter(familiars.swappable_rarities.clone()),
                    },
                    familiars.swap_check_millis,
                ),
            );
        }

        // Same reasoning as swap/reveal-potential above - not worth interrupting Monster Park's
        // tight sweep-then-portal loop for.
        if familiars_fusion.enable_familiars_fusion && !is_monster_park {
            self.priority_actions.insert(
                next_action_id(),
                familiars_fusion_priority_action(
                    FamiliarsFuse {
                        fusable_rarities: Array::from_iter(
                            familiars_fusion.fusable_rarities.clone(),
                        ),
                        swappable_slots: familiars.swappable_familiars,
                        swappable_rarities: Array::from_iter(familiars.swappable_rarities),
                    },
                    familiars_fusion.fusion_check_millis,
                ),
            );
        }

        // Skipped entirely while fusion or swap is enabled - same reasoning as swap's own skip
        // above: `Player::FamiliarsSwapping` now conditionally chains into
        // `Player::FamiliarsRevealingPotential` on its own completion (see
        // `PlayerContext::familiars_potential_due`), and that path is reached regardless of
        // whether swap got there standalone or chained from fusion. Registering this as a second,
        // independent priority action on its own timer would only race against that chain the
        // same way an independent swap action would race against fusion's own chain into swap.
        if familiars_potential.enable_familiars_potential
            && !is_monster_park
            && !familiars.enable_familiars_swapping
            && !familiars_fusion.enable_familiars_fusion
        {
            self.priority_actions.insert(
                next_action_id(),
                familiars_reveal_potential_priority_action(
                    FamiliarsRevealPotential {
                        reveal_rarities: Array::from_iter(
                            familiars_potential.reveal_rarities.clone(),
                        ),
                    },
                    familiars_potential.potential_check_millis,
                ),
            );
        }

        if thread_of_fate_ask_enabled {
            self.priority_actions.insert(
                next_action_id(),
                thread_of_fate_ask_accept_priority_action(
                    thread_of_fate_ask_npc,
                    self.thread_of_fate_ask_character_id,
                    self.thread_of_fate_ask_progress.clone(),
                ),
            );
            self.priority_actions.insert(
                next_action_id(),
                thread_of_fate_ask_complete_priority_action(
                    self.thread_of_fate_ask_character_id,
                    self.thread_of_fate_ask_progress.clone(),
                ),
            );
        }
        self.familiars_fuse_attack_key = familiars_fusion
            .attack_key
            .enabled
            .then_some(familiars_fusion.attack_key.key.into());
        self.familiars_fuse_attack_interval_millis = familiars_fusion.attack_interval_millis;
        self.familiars_fuse_attack_interval_millis_random_range =
            familiars_fusion.attack_interval_millis_random_range;
        self.familiars_fuse_attack_next_at = None;

        // Mid priority
        let mut i = 0;
        let character_actions_len = character_actions.len();
        let actions = [character_actions, map_actions].concat();
        while i < actions.len() {
            let action_index = i;
            let action = actions[i];
            let condition = action.condition();
            let queue_to_front = match action {
                Action::Move(_) => false,
                Action::Key(ActionKey { queue_to_front, .. }) => queue_to_front.unwrap_or_default(),
            };
            // Whether this action came from `map_actions` rather than `character_actions` - the
            // former is tied to whatever the user's regular farming map is set up with, which has
            // nothing to do with Monster Park's stages or a daily quest's hunting ground, even for
            // a position-less `Action::Key` (e.g. a per-map periodic skill rotation keybind with
            // no fixed spot - still configured specifically for that map, unlike a genuinely
            // character-level periodic buff/skill). See the `EveryMillis` arm below for where this
            // distinction actually matters.
            let is_map_action = action_index >= character_actions_len;
            let (action, offset) = rotator_action(action, i, &actions);
            debug_assert!(i != 0 || !matches!(condition, ActionCondition::Linked));
            // Should not move i below the match because it could cause
            // infinite loop due to auto mobbing ignoring Any condition
            i += offset;
            match condition {
                // Map-level actions (see `is_map_action` above) are skipped entirely during
                // Monster Park, for the whole mode session and not just the entry lobby, since
                // there's no meaningful "farming map position" at any point during a run.
                //
                // Character-level `EveryMillis` ones (periodic character-level buffs/skills) are
                // registered unconditionally instead: `Self::rotate_priority_actions` suppresses
                // them at runtime only while [`Self::monster_park_at_entry_lobby`] is true, so they
                // still lapse while at the entry lobby (nothing meaningful to use them on there
                // either) but keep firing normally once a run's stages are actually being swept,
                // instead of lapsing for a run's entire duration.
                ActionCondition::EveryMillis(_) => {
                    if !is_monster_park || !is_map_action {
                        self.priority_actions.insert(
                            next_action_id(),
                            priority_action(action, condition, queue_to_front, is_map_action),
                        );
                    }
                }
                // Always map-tied (an Erda Shower skill bound to a spot on the farming map), so
                // this one is skipped entirely during Monster Park same as before.
                ActionCondition::ErdaShowerOffCooldown => {
                    if !is_monster_park {
                        self.priority_actions.insert(
                            next_action_id(),
                            priority_action(action, condition, queue_to_front, is_map_action),
                        );
                    }
                }
                ActionCondition::Any => {
                    if matches!(self.normal_rotate_mode, RotatorMode::AutoMobbing(_, _)) {
                        continue;
                    }
                    self.normal_actions.push((next_action_id(), action))
                }
                ActionCondition::Linked => unreachable!(),
            }
        }

        // High priority
        if enable_rune_solving {
            self.priority_actions
                .insert(next_action_id(), solve_rune_priority_action());
        }
        if enable_transparent_shape_solving {
            self.priority_actions
                .insert(next_action_id(), solve_transparent_shape_priority_action());
        }
        if enable_violetta_solving {
            self.priority_actions
                .insert(next_action_id(), solve_violetta_priority_action());
        }

        match elite_boss_behavior {
            EliteBossBehavior::None => (),
            EliteBossBehavior::CycleChannel => {
                self.priority_actions.insert(
                    next_action_id(),
                    elite_boss_change_channel_priority_action(),
                );
            }
            EliteBossBehavior::UseKey => {
                self.priority_actions.insert(
                    next_action_id(),
                    elite_boss_use_key_priority_action(elite_boss_behavior_key),
                );
            }
        }

        if enable_panic_mode && !is_idle {
            self.priority_actions
                .insert(next_action_id(), panic_priority_action());
        }

        if !is_idle
            && buffs
                .iter()
                .any(|(buff, _)| matches!(buff, BuffKind::Familiar))
        {
            self.priority_actions.insert(
                next_action_id(),
                familiar_essence_replenish_priority_action(familiar_essence_key),
            );
        }
        if !is_monster_park && !is_idle {
            for (i, key) in buffs.iter().copied() {
                self.priority_actions
                    .insert(next_action_id(), buff_priority_action(i, key));
            }
        }

        self.priority_actions
            .insert(next_action_id(), unstuck_priority_action());
    }

    #[inline]
    fn reset_queue(&mut self) {
        self.normal_actions_backward = false;
        self.reset_normal_actions_queue();
        self.priority_actions_queue.clear();
        self.priority_queuing_linked_action = None;
        self.auto_mob_task = None;
        self.auto_mob_quadrant_consecutive_count = None;
        self.reset_monster_park_run_state();
        self.monster_park_entry_attempts = 0;
        self.monster_park_giving_up = false;
        self.monster_park_sweep = MonsterParkSweep::NotEntered;
        self.monster_park_exhausted = false;
        self.monster_park_at_entry_lobby = false;
        self.monster_park_run_in_progress = false;
        self.daily_quest_navigating = false;
        self.daily_quest_arrived = false;
        self.daily_quest_navigate_attempts = 0;
        self.daily_quest_complete_task = None;
        self.daily_quest_complete_popup_detected = false;
        self.daily_quest_complete_popup_confirmed_absent = false;
    }

    #[inline]
    fn clear_thread_of_fate_ask_reset_block(&mut self) {
        self.thread_of_fate_ask_progress
            .lock()
            .unwrap()
            .clear_reset_block();
    }

    #[inline]
    fn inject_action(&mut self, action: PlayerAction) {
        self.priority_actions_side_queue
            .push_back(RotatorAction::Single(action));
    }

    #[inline]
    fn rotate_action(&mut self, resources: &mut Resources, world: &mut World) {
        if resources.operation.halting() {
            if !has_side_loaded_action_executing(&world.player.context) {
                self.rotate_side_priority_action(&mut world.player.context);
            }
            return;
        }

        self.rotate_priority_actions(resources, world);
        self.rotate_priority_actions_queue(&mut world.player);
        self.rotate_familiars_fusing_attack(resources, world);

        if self.daily_quest_index < self.daily_quest_entries.len() {
            // Opening the world map to navigate to a hunting ground isn't possible anywhere in a
            // Monster Park run - not just standing in the entry lobby, but through the
            // dungeon-select dialog and every stage swept once past it - the game doesn't allow it
            // there. `monster_park_run_in_progress` covers that whole span explicitly (see its
            // docs); the live `detect_monster_park_entry_map` check below additionally covers the
            // narrower case of the player merely standing in the lobby without a run underway yet
            // (e.g. between Monster Park runs) so `rotate_monster_park_entry` gets a chance to walk
            // them out first - daily quest navigation resumes normally once neither is true. The
            // live check is only made when nothing is already in flight, same reasoning as the
            // Monster Park dispatch below for why that's the only time it matters.
            let blocked_by_monster_park_run = self.monster_park_enabled
                && (self.monster_park_run_in_progress
                    || (!self.daily_quest_navigating
                        && !world.player.context.has_normal_action()
                        && !world.player.context.has_priority_action()
                        && resources.detector().detect_monster_park_entry_map()));
            if !blocked_by_monster_park_run {
                // `monster_park_at_entry_lobby` can be left stale at `true` from an entry attempt
                // that got preempted right here: the Monster Park dispatch below sets it the
                // instant a fresh detection confirms the lobby, then immediately hands off a
                // normal action (walking to the gate) - and a normal action in flight is exactly
                // what makes `blocked_by_monster_park_run`'s live re-check above skip itself, so
                // control can land back on daily quest dispatch (e.g. once `daily_quest_navigating`
                // flips true) without that branch ever running again to correct it back to `false`.
                // Left alone, `rotate_priority_actions` would then suppress every character/map
                // `EveryMillis` fixed action (see `monster_park_at_entry_lobby`'s docs) for the
                // rest of this daily quest run, even though the player is actually out mobbing a
                // hunting ground, not standing at the lobby. Clearing it here - the one place
                // dispatch is actually committing to daily quests instead - is the only point
                // guaranteed to run whenever that's true, regardless of how it got stale.
                self.monster_park_at_entry_lobby = false;
                self.rotate_daily_quest(
                    resources,
                    &mut world.player.context,
                    &mut world.minimap.context,
                    world.minimap.state,
                );
                return;
            }
        }

        if self.monster_park_enabled {
            // Picked up every tick (not just at the lobby) since the dungeon may already have
            // loaded in by the time the entering state reports back.
            match world.player.context.take_monster_park_entry_outcome() {
                Some(MonsterParkEntryOutcome::Entered) => {
                    self.monster_park_sweep = MonsterParkSweep::Entered;
                }
                Some(MonsterParkEntryOutcome::NoClearsLeft) => {
                    self.monster_park_sweep = MonsterParkSweep::Finished;
                }
                None => (),
            }

            // `detect_monster_park_entry_map` is a synchronous, full-frame template match -
            // non-trivial cost, unlike the small-ROI minimap scans inside a run that were
            // already moved to a background task (see `monster_park_enemies_task`'s docs for
            // why that mattered: a synchronous detector call stretching out a tick was found
            // to break tight-timing movement). It only needs to be checked when actually
            // deciding what to do next - while either function has an action in flight, both
            // already no-op immediately once they see it (`rotate_monster_park_entry` on
            // `has_normal_action`/`has_priority_action`, `rotate_monster_park` on the same
            // checks once its own state - `monster_park_target` - rules out that action being
            // its own), so which of the two gets called doesn't change behavior. Skipping the
            // check and falling through to `rotate_monster_park` (whose only unconditional
            // work is polling the already-async enemy scan) avoids paying for it on every tick
            // an action is already running, which is most of them.
            let has_action = world.player.context.has_normal_action()
                || world.player.context.has_priority_action();
            // `None` while an action is in flight - the detector call below is skipped then
            // (see the cost note above), so `monster_park_at_entry_lobby` just keeps whatever
            // value it was last set to, which can't have gone stale mid-action.
            let at_entry_lobby =
                (!has_action).then(|| resources.detector().detect_monster_park_entry_map());
            if matches!(at_entry_lobby, Some(true)) {
                self.monster_park_at_entry_lobby = true;
                self.rotate_monster_park_entry(&mut world.player.context, world.minimap.state);
            } else {
                if let Some(at_entry_lobby) = at_entry_lobby {
                    // A fresh, synchronous check (not skipped for having an action in flight)
                    // just confirmed we're genuinely past the lobby - drop the entry retry
                    // streak now that this attempt cycle is truly behind us, so a transient
                    // hiccup on some earlier run doesn't count towards a later, unrelated
                    // failure streak.
                    self.monster_park_entry_attempts = 0;
                    self.monster_park_at_entry_lobby = at_entry_lobby;
                    if self.monster_park_sweep == MonsterParkSweep::Entered {
                        self.monster_park_sweep = MonsterParkSweep::RunStarted;
                    }
                    // The give-up exit (see `monster_park_giving_up`'s docs) is only now confirmed
                    // to have actually worked - safe to disable Monster Park for the rest of the
                    // session. Until this fires, `monster_park_enabled` stays `true` so the retry
                    // dispatch above and `blocked_by_monster_park_run`'s live check both keep
                    // running instead of assuming the exit succeeded.
                    if self.monster_park_giving_up {
                        self.monster_park_giving_up = false;
                        self.monster_park_enabled = false;
                        self.monster_park_exhausted = true;
                        debug!(
                            target: "backend/rotator",
                            "Monster Park: confirmed left the entry lobby after giving up on \
                             entering, Monster Park now disabled for the rest of the session"
                        );
                    }
                }
                self.rotate_monster_park(
                    resources,
                    &mut world.player.context,
                    world.minimap.state,
                    self.daily_quest_mobbing_key,
                );
            }
            return;
        }

        // Both Monster Park and daily quests are spent for the day (reaching here at all
        // guarantees the daily-quest block above already returned, so `daily_quest_index` is past
        // its entries) - see `monster_park_exhausted`'s docs for why running `normal_rotate_mode`
        // blind, right where Monster Park's shuttle happened to drop the player, is worse than
        // just stopping here.
        if self.monster_park_exhausted {
            return;
        }

        match self.normal_rotate_mode {
            RotatorMode::StartToEnd => self.rotate_start_to_end(&mut world.player.context),
            RotatorMode::StartToEndThenReverse => {
                self.rotate_start_to_end_then_reverse(&mut world.player.context)
            }
            RotatorMode::AutoMobbing(key, bound) => self.rotate_auto_mobbing(
                resources,
                &mut world.player.context,
                world.minimap.state,
                key,
                bound,
            ),
            RotatorMode::PingPong(key, bound) => {
                self.rotate_ping_pong(&mut world.player.context, world.minimap.state, key, bound)
            }
            RotatorMode::Idle => {}
        }
    }
}

#[inline]
fn has_side_loaded_action_executing(player_context: &PlayerContext) -> bool {
    player_context.has_priority_action() && player_context.priority_action_id().is_none()
}

/// Creates a [`RotatorAction`] with `start_action` as the initial action
///
/// If `start_action` is linked, this function returns [`RotatorAction::Linked`] with [`usize`] as
/// the offset from `start_index` to the next non-linked action.
/// Otherwise, this returns [`RotatorAction::Single`] with [`usize`] offset of 1.
#[inline]
fn rotator_action(
    start_action: Action,
    start_index: usize,
    actions: &[Action],
) -> (RotatorAction, usize) {
    if start_index == actions.len() - 1 {
        // Last action cannot be a linked action
        return (RotatorAction::Single(start_action.into()), 1);
    }
    if start_index + 1 < actions.len() {
        match actions[start_index + 1] {
            Action::Move(ActionMove {
                condition: ActionCondition::Linked,
                ..
            })
            | Action::Key(ActionKey {
                condition: ActionCondition::Linked,
                ..
            }) => (),
            _ => return (RotatorAction::Single(start_action.into()), 1),
        }
    }
    let mut head = LinkedAction {
        inner: start_action.into(),
        next: None,
    };
    let mut current = &mut head;
    let mut offset = 1;
    for action in actions.iter().skip(start_index + 1) {
        match action {
            Action::Move(ActionMove {
                condition: ActionCondition::Linked,
                ..
            })
            | Action::Key(ActionKey {
                condition: ActionCondition::Linked,
                ..
            }) => {
                let action = LinkedAction {
                    inner: (*action).into(),
                    next: None,
                };
                current.next = Some(Box::new(action));
                current = current.next.as_mut().unwrap();
                offset += 1;
            }
            _ => break,
        }
    }
    (RotatorAction::Linked(head), offset)
}

#[inline]
fn priority_action(
    action: RotatorAction,
    condition: ActionCondition,
    queue_to_front: bool,
    is_map_action: bool,
) -> PriorityAction {
    debug_assert_matches!(
        condition,
        ActionCondition::EveryMillis(_) | ActionCondition::ErdaShowerOffCooldown
    );
    // Character-level `EveryMillis` actions (e.g. a periodic character-level buff/skill) aren't
    // tied to the farming map, so they're still meaningful once actually mobbing at a daily
    // quest's hunting ground - only held back while still navigating there. Everything else
    // (map-level ones - including position-less ones, e.g. a per-map periodic skill rotation
    // keybind - and `ErdaShowerOffCooldown` which is always map-tied) is suppressed for the whole
    // daily quest run, same as the rest of the regular grinding assists.
    let suppress_during_daily_quest = if is_map_action {
        DailyQuestSuppression::WhileActive
    } else {
        DailyQuestSuppression::WhileNavigating
    };
    PriorityAction {
        inner: action,
        condition: Condition(Box::new(move |_, world, info| {
            if should_queue_fixed_action(world, info.last_queued_time, condition) {
                ConditionResult::Queue
            } else {
                ConditionResult::Skip
            }
        })),
        condition_kind: Some(condition),
        metadata: None,
        queue_to_front,
        suppress_during_daily_quest,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

/// Creates a [`PlayerAction::Key`] priority action to replenish familiar essence
/// when it is detected as depleted.
///
/// The action will only queue if:
/// - Enough time has passed since the last queue attempt.
/// - The familiar buff is currently active.
/// - Familiar essence is detected as depleted.
///
/// If the essence is not depleted, the action will be marked as [`ConditionResult::Ignore`]
/// and temporarily ignored in subsequent queue do to `last_queued_time` being updated.
#[inline]
fn familiar_essence_replenish_priority_action(key: KeyKind) -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn = move |detector: Arc<dyn Detector>| -> Result<bool> {
        Ok(detector.detect_familiar_essence_depleted())
    };

    PriorityAction {
        condition: Condition(Box::new(move |resources, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 20000) {
                return ConditionResult::Skip;
            }

            if !matches!(world.buffs[BuffKind::Familiar].state, Buff::Yes) {
                return ConditionResult::Skip;
            }

            match update_detection_task(resources, 10000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
                Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::Key(Key {
            key,
            key_hold_ticks: 0,
            key_hold_buffered_to_wait_after: false,
            link_key: LinkKeyKind::None,
            count: 1,
            position: None,
            direction: ActionKeyDirection::Any,
            with: ActionKeyWith::Any,
            wait_before_use_ticks: 5,
            wait_before_use_ticks_random_range: 0,
            wait_after_use_ticks: 0,
            wait_after_use_ticks_random_range: 0,
            wait_after_buffered: WaitAfterBuffered::None,
        })),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn familiars_swap_priority_action(swap: FamiliarsSwap, swap_check_millis: u64) -> PriorityAction {
    PriorityAction {
        condition: Condition(Box::new(move |_, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, swap_check_millis.into()) {
                return ConditionResult::Skip;
            }

            if world
                .player
                .context
                .is_familiars_swap_fail_count_limit_reached()
            {
                return ConditionResult::Skip;
            }

            ConditionResult::Queue
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::FamiliarsSwap(swap)),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn familiars_fusion_priority_action(
    fuse: FamiliarsFuse,
    fusion_check_millis: u64,
) -> PriorityAction {
    PriorityAction {
        condition: Condition(Box::new(move |_, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, fusion_check_millis.into()) {
                return ConditionResult::Skip;
            }

            if world
                .player
                .context
                .is_familiars_fuse_fail_count_limit_reached()
            {
                return ConditionResult::Skip;
            }

            ConditionResult::Queue
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::FamiliarsFuse(fuse)),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn familiars_reveal_potential_priority_action(
    reveal: FamiliarsRevealPotential,
    potential_check_millis: u64,
) -> PriorityAction {
    PriorityAction {
        condition: Condition(Box::new(move |_, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, potential_check_millis.into()) {
                return ConditionResult::Skip;
            }

            if world
                .player
                .context
                .is_familiars_potential_fail_count_limit_reached()
            {
                return ConditionResult::Skip;
            }

            ConditionResult::Queue
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::FamiliarsRevealPotential(reveal)),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

/// How often the Thread of Fate ask/complete priority actions re-check their eligibility -
/// neither needs tick-level precision (asks are gated by a daily count and completions by a ~30
/// minute wait), so this is generous purely to avoid needless locking every tick.
const THREAD_OF_FATE_ASK_CHECK_MILLIS: u64 = 60_000;

/// Creates a [`PlayerAction::ThreadOfFateAsk`] priority action that triggers when `progress`
/// reports an ask can be accepted (see [`ThreadOfFateAskProgress::can_accept`]) - no ask currently
/// pending and today's limit not yet reached.
#[inline]
fn thread_of_fate_ask_accept_priority_action(
    npc: ThreadOfFateNpc,
    character_id: Option<i64>,
    progress: Arc<Mutex<ThreadOfFateAskProgress>>,
) -> PriorityAction {
    let action_progress = progress.clone();
    PriorityAction {
        condition: Condition(Box::new(move |_, _, info| {
            if !at_least_millis_passed_since(
                info.last_queued_time,
                THREAD_OF_FATE_ASK_CHECK_MILLIS.into(),
            ) {
                return ConditionResult::Skip;
            }

            let mut progress = progress.lock().unwrap();
            progress.reset_if_new_day();
            if progress.can_accept() {
                ConditionResult::Queue
            } else {
                ConditionResult::Skip
            }
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::ThreadOfFateAsk(ThreadOfFateAsk {
            npc,
            character_id,
            progress: action_progress,
        })),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

/// Creates a [`PlayerAction::ThreadOfFateAskComplete`] priority action that triggers when
/// `progress` reports the pending ask is ready to turn in (see
/// [`ThreadOfFateAskProgress::is_ready_to_complete`]).
#[inline]
fn thread_of_fate_ask_complete_priority_action(
    character_id: Option<i64>,
    progress: Arc<Mutex<ThreadOfFateAskProgress>>,
) -> PriorityAction {
    let action_progress = progress.clone();
    PriorityAction {
        condition: Condition(Box::new(move |_, _, info| {
            if !at_least_millis_passed_since(
                info.last_queued_time,
                THREAD_OF_FATE_ASK_CHECK_MILLIS.into(),
            ) {
                return ConditionResult::Skip;
            }

            let progress = progress.lock().unwrap();
            if progress.is_ready_to_complete() {
                ConditionResult::Queue
            } else {
                ConditionResult::Skip
            }
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::ThreadOfFateAskComplete(
            ThreadOfFateAskComplete {
                character_id,
                progress: action_progress,
            },
        )),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

/// Creates a [`PlayerAction::SolveRune`] priority action that triggers when a rune is available.
///
/// This action queues if all the following conditions are met:
/// - The player is not currently validating a rune.
/// - Enough time has passed since the last queue attempt.
/// - The minimap is in the [`Minimap::Idle`] state.
/// - A rune is present on the minimap.
/// - The player currently has no rune buff.
#[inline]
fn solve_rune_priority_action() -> PriorityAction {
    PriorityAction {
        condition: Condition(Box::new(|_, world, info| {
            if world.player.context.is_validating_rune() {
                return ConditionResult::Ignore;
            }

            if !at_least_millis_passed_since(info.last_queued_time, 10000) {
                return ConditionResult::Skip;
            }

            if let Minimap::Idle(idle) = world.minimap.state
                && idle.rune().is_some()
                && matches!(world.buffs[BuffKind::Rune].state, Buff::No)
            {
                return ConditionResult::Queue;
            }

            ConditionResult::Skip
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::SolveRune),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::Never,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn solve_transparent_shape_priority_action() -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn = move |detector: Arc<dyn Detector>| -> Result<bool> {
        Ok(detector.detect_lie_detector_shape().is_ok())
    };

    PriorityAction {
        condition: Condition(Box::new(move |resources, _, _| {
            if resources.detector.is_none() {
                return ConditionResult::Ignore;
            }

            match update_detection_task(resources, 3000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
                Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::SolveShape),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::Never,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn solve_violetta_priority_action() -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn = move |detector: Arc<dyn Detector>| -> Result<bool> {
        Ok(detector.detect_lie_detector_violetta().is_ok())
    };

    PriorityAction {
        condition: Condition(Box::new(move |resources, _, _| {
            if resources.detector.is_none() {
                return ConditionResult::Ignore;
            }

            match update_detection_task(resources, 3000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
                Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        metadata: None,
        inner: RotatorAction::Single(PlayerAction::SolveVioletta),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::Never,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

/// Creates a [`PlayerAction::Key`] priority action to cast a specific buff when it's not active.
///
/// The action queues if:
/// - Enough time has passed since the last queue attempt.
/// - The minimap is in the [`Minimap::Idle`] state.
/// - The specified buff is currently missing.
#[inline]
fn buff_priority_action(buff: BuffKind, key: KeyKind) -> PriorityAction {
    macro_rules! skip_if_has_buff {
        ($world:expr, $variant:ident $(| $variants:ident)*) => {{
            $(
                if !matches!($world.buffs[BuffKind::$variants].state, Buff::No) {
                    return ConditionResult::Skip;
                }
            )*
            if !matches!($world.buffs[BuffKind::$variant].state, Buff::No) {
                return ConditionResult::Skip;
            }
        }};
    }

    PriorityAction {
        condition: Condition(Box::new(move |_, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 20000) {
                return ConditionResult::Skip;
            }
            if !matches!(world.minimap.state, Minimap::Idle(_)) {
                return ConditionResult::Skip;
            }

            match buff {
                BuffKind::SmallWealthAcquisitionPotion => {
                    skip_if_has_buff!(world, WealthAcquisitionPotion)
                }
                BuffKind::WealthAcquisitionPotion => {
                    skip_if_has_buff!(world, SmallWealthAcquisitionPotion)
                }
                BuffKind::SmallExpAccumulationPotion => {
                    skip_if_has_buff!(world, ExpAccumulationPotion)
                }
                BuffKind::ExpAccumulationPotion => {
                    skip_if_has_buff!(world, SmallExpAccumulationPotion)
                }
                BuffKind::ExpCouponX2 => {
                    skip_if_has_buff!(world, ExpCouponX3 | ExpCouponX4)
                }
                BuffKind::ExpCouponX3 => {
                    skip_if_has_buff!(world, ExpCouponX2 | ExpCouponX4)
                }
                BuffKind::ExpCouponX4 => {
                    skip_if_has_buff!(world, ExpCouponX3 | ExpCouponX2)
                }
                BuffKind::BonusExpCoupon => {
                    skip_if_has_buff!(world, MvpBonusExpCoupon)
                }
                _ => (),
            }

            if matches!(world.buffs[buff].state, Buff::No) {
                ConditionResult::Queue
            } else {
                ConditionResult::Skip
            }
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::Key(Key {
            key,
            key_hold_ticks: 0,
            key_hold_buffered_to_wait_after: false,
            link_key: LinkKeyKind::None,
            count: 1,
            position: None,
            direction: ActionKeyDirection::Any,
            with: ActionKeyWith::Stationary,
            wait_before_use_ticks: 10,
            wait_before_use_ticks_random_range: 0,
            wait_after_use_ticks: 10,
            wait_after_use_ticks_random_range: 0,
            wait_after_buffered: WaitAfterBuffered::None,
        })),
        metadata: Some(ActionMetadata::Buff { kind: buff }),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn panic_priority_action() -> PriorityAction {
    PriorityAction {
        condition: Condition(Box::new(|_, world, info| match world.minimap.state {
            Minimap::Detecting => ConditionResult::Skip,
            Minimap::Idle(idle) => {
                if !idle.has_any_other_player() || info.last_queued_time.is_none() {
                    return ConditionResult::Ignore;
                }

                if at_least_millis_passed_since(info.last_queued_time, 15000) {
                    ConditionResult::Queue
                } else {
                    ConditionResult::Skip
                }
            }
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::Panic(Panic {
            to: PanicTo::Channel,
        })),
        metadata: None,
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn elite_boss_change_channel_priority_action() -> PriorityAction {
    let mut condition = elite_boss_condition();

    PriorityAction {
        condition: Condition(Box::new(move |resources, _, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 15000) {
                return ConditionResult::Skip;
            }

            condition(resources)
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::Panic(Panic {
            to: PanicTo::Channel,
        })),
        metadata: None,
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn elite_boss_use_key_priority_action(key: KeyKind) -> PriorityAction {
    let mut condition = elite_boss_condition();

    PriorityAction {
        condition: Condition(Box::new(move |resources, _, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 15000) {
                return ConditionResult::Skip;
            }

            condition(resources)
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::Key(Key {
            key,
            key_hold_ticks: 0,
            key_hold_buffered_to_wait_after: false,
            link_key: LinkKeyKind::None,
            count: 1,
            position: None,
            direction: ActionKeyDirection::Any,
            with: ActionKeyWith::Stationary,
            wait_before_use_ticks: 10,
            wait_before_use_ticks_random_range: 0,
            wait_after_use_ticks: 10,
            wait_after_use_ticks_random_range: 0,
            wait_after_buffered: WaitAfterBuffered::None,
        })),
        metadata: None,
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

fn elite_boss_condition() -> impl FnMut(&Resources) -> ConditionResult {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn =
        move |detector: Arc<dyn Detector>| -> Result<bool> { Ok(detector.detect_elite_boss_bar()) };

    move |resources| {
        if resources.detector.is_none() {
            return ConditionResult::Ignore;
        }

        match update_detection_task(resources, 5000, &mut task, task_fn) {
            Update::Ok(true) => ConditionResult::Queue,
            Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
            Update::Pending => ConditionResult::Skip,
        }
    }
}

#[inline]
fn use_booster_priority_action(kind: Booster) -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn =
        move |detector: Arc<dyn Detector>| -> Result<bool> { Ok(!detector.detect_timer_visible()) };

    PriorityAction {
        condition: Condition(Box::new(move |resources, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 20000) {
                return ConditionResult::Skip;
            }

            if world
                .player
                .context
                .is_booster_fail_count_limit_reached(kind)
            {
                return ConditionResult::Ignore;
            }

            if resources.detector.is_none() {
                return ConditionResult::Ignore;
            }

            match update_detection_task(resources, 10000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
                Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::UseBooster(UseBooster { kind })),
        metadata: Some(ActionMetadata::UseBooster),
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn exchange_hexa_booster_priority_action(
    condition: ExchangeHexaBoosterCondition,
    amount: u32,
    all: bool,
) -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn = move |detector: Arc<dyn Detector>| -> Result<bool> {
        let booster = detector.detect_quick_slots_hexa_booster()?;
        if !matches!(booster, QuickSlotsHexaBooster::Unavailable) {
            return Ok(false);
        }

        let sol_erda = detector.detect_hexa_sol_erda()?;
        let queue = match condition {
            ExchangeHexaBoosterCondition::None => unreachable!(),
            ExchangeHexaBoosterCondition::Full => {
                matches!(sol_erda, SolErda::Full)
            }
            ExchangeHexaBoosterCondition::AtLeastOne => {
                matches!(sol_erda, SolErda::AtLeastOne | SolErda::Full)
            }
        };

        Ok(queue)
    };

    PriorityAction {
        condition: Condition(Box::new(move |resources, _, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 20000) {
                return ConditionResult::Skip;
            }

            if resources.detector.is_none() {
                return ConditionResult::Skip;
            }

            match update_detection_task(resources, 10000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Err(_) | Update::Ok(false) => ConditionResult::Ignore,
                Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::ExchangeBooster(ExchangeBooster {
            amount,
            all,
        })),
        metadata: None,
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::WhileActive,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn unstuck_priority_action() -> PriorityAction {
    let mut task: Option<Task<Result<bool>>> = None;
    let task_fn =
        move |detector: Arc<dyn Detector>| -> Result<bool> { Ok(detector.detect_esc_settings()) };

    PriorityAction {
        condition: Condition(Box::new(move |resources, world, info| {
            if !at_least_millis_passed_since(info.last_queued_time, 3000) {
                return ConditionResult::Skip;
            }

            if !world.player.state.can_override_current_state(None) {
                return ConditionResult::Skip;
            }

            if resources.detector.is_none() {
                return ConditionResult::Skip;
            }

            if world.player.context.is_dead() {
                return ConditionResult::Skip;
            }

            match update_detection_task(resources, 3000, &mut task, task_fn) {
                Update::Ok(true) => ConditionResult::Queue,
                Update::Ok(false) | Update::Err(_) | Update::Pending => ConditionResult::Skip,
            }
        })),
        condition_kind: None,
        inner: RotatorAction::Single(PlayerAction::Unstuck),
        metadata: None,
        queue_to_front: true,
        suppress_during_daily_quest: DailyQuestSuppression::Never,
        queue_info: PriorityActionQueueInfo::default(),
    }
}

#[inline]
fn at_least_millis_passed_since(last_queued_time: Option<Instant>, millis: u128) -> bool {
    last_queued_time
        .map(|instant| Instant::now().duration_since(instant).as_millis() >= millis)
        .unwrap_or(true)
}

#[inline]
fn should_queue_fixed_action(
    world: &World,
    last_queued_time: Option<Instant>,
    condition: ActionCondition,
) -> bool {
    let millis_should_passed = match condition {
        ActionCondition::EveryMillis(millis) => millis as u128,
        ActionCondition::ErdaShowerOffCooldown => 20000,
        ActionCondition::Linked | ActionCondition::Any => unreachable!(),
    };
    if !at_least_millis_passed_since(last_queued_time, millis_should_passed) {
        return false;
    }
    if matches!(condition, ActionCondition::ErdaShowerOffCooldown)
        && !matches!(world.skills[SkillKind::ErdaShower].state, Skill::Idle(_, _))
    {
        return false;
    }
    true
}

fn next_action_id() -> u32 {
    static NEXT_ID: AtomicU32 = AtomicU32::new(0);

    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use std::{
        assert_matches::assert_matches,
        time::{Duration, Instant},
    };

    use opencv::core::{Point, Vec4b};
    use strum::IntoEnumIterator;
    use tokio::{task::yield_now, time::timeout};

    use super::*;
    use crate::{
        Position,
        bridge::MockInput,
        buff::{BuffContext, BuffEntity, BuffKind},
        detect::MockDetector,
        minimap::{MinimapContext, MinimapEntity, MinimapIdle},
        models::SwappableFamiliars,
        player::{FamiliarsFusing, Player},
        skill::{SkillContext, SkillEntity, SkillKind},
    };

    fn mock_fusing_player() -> PlayerEntity {
        PlayerEntity {
            state: Player::FamiliarsFusing(FamiliarsFusing::new(
                Array::new(),
                SwappableFamiliars::default(),
                Array::new(),
            )),
            context: PlayerContext::default(),
        }
    }

    const COOLDOWN_BETWEEN_QUEUE_MILLIS: u128 = 20_000;
    const NORMAL_ACTION: Action = Action::Move(ActionMove {
        position: Position {
            x: 0,
            x_random_range: 0,
            y: 0,
            allow_adjusting: false,
        },
        condition: ActionCondition::Any,
        wait_after_move_millis: 0,
    });
    const PRIORITY_ACTION: Action = Action::Move(ActionMove {
        position: Position {
            x: 0,
            x_random_range: 0,
            y: 0,
            allow_adjusting: false,
        },
        condition: ActionCondition::ErdaShowerOffCooldown,
        wait_after_move_millis: 0,
    });

    fn mock_world() -> World {
        World {
            minimap: MinimapEntity {
                state: Minimap::Detecting,
                context: MinimapContext::default(),
            },
            player: PlayerEntity {
                state: Player::Idle,
                context: PlayerContext::default(),
            },
            skills: SkillKind::iter()
                .map(|kind| SkillEntity {
                    state: Skill::Detecting,
                    context: SkillContext::new(kind),
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            buffs: BuffKind::iter()
                .map(|kind| BuffEntity {
                    state: Buff::No,
                    context: BuffContext::new(kind),
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    #[test]
    fn rotator_familiars_fusing_attack_sends_key_when_enabled_and_safe() {
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::A))
            .once()
            .return_const(());
        let mut resources = Resources::new(Some(keys), None);
        let mut world = mock_world();
        world.player = mock_fusing_player();

        let mut rotator = DefaultRotator {
            familiars_fuse_attack_key: Some(KeyKind::A),
            familiars_fuse_attack_interval_millis: 5000,
            ..Default::default()
        };

        rotator.rotate_familiars_fusing_attack(&mut resources, &mut world);

        assert!(rotator.familiars_fuse_attack_next_at.is_some());
    }

    #[test]
    fn rotator_familiars_fusing_attack_skips_when_disabled() {
        let mut resources = Resources::new(Some(MockInput::new()), None);
        let mut world = mock_world();
        world.player = mock_fusing_player();

        let mut rotator = DefaultRotator {
            familiars_fuse_attack_key: None,
            ..Default::default()
        };

        rotator.rotate_familiars_fusing_attack(&mut resources, &mut world);

        assert!(rotator.familiars_fuse_attack_next_at.is_none());
    }

    #[test]
    fn rotator_familiars_fusing_attack_skips_when_not_fusing() {
        let mut resources = Resources::new(Some(MockInput::new()), None);
        // `mock_world()` defaults to `Player::Idle`, not `Player::FamiliarsFusing`.
        let mut world = mock_world();

        let mut rotator = DefaultRotator {
            familiars_fuse_attack_key: Some(KeyKind::A),
            familiars_fuse_attack_next_at: Some(Instant::now() + Duration::from_secs(60)),
            ..Default::default()
        };

        rotator.rotate_familiars_fusing_attack(&mut resources, &mut world);

        assert!(rotator.familiars_fuse_attack_next_at.is_none());
    }

    #[test]
    fn rotator_familiars_fusing_attack_throttled_until_next_at_elapses() {
        let mut resources = Resources::new(Some(MockInput::new()), None);
        let mut world = mock_world();
        world.player = mock_fusing_player();

        let mut rotator = DefaultRotator {
            familiars_fuse_attack_key: Some(KeyKind::A),
            familiars_fuse_attack_next_at: Some(Instant::now() + Duration::from_secs(60)),
            ..Default::default()
        };

        // No expectation set on the mock input - a call to send_key would panic, proving the tap
        // is withheld while still throttled.
        rotator.rotate_familiars_fusing_attack(&mut resources, &mut world);
    }

    #[test]
    fn rotator_at_least_millis_passed_since() {
        let now = Instant::now();
        assert!(at_least_millis_passed_since(None, 1000));
        assert!(at_least_millis_passed_since(
            Some(now - Duration::from_millis(2000)),
            1000
        ));
        assert!(!at_least_millis_passed_since(
            Some(now - Duration::from_millis(500)),
            1000
        ));
    }

    #[test]
    fn rotator_should_queue_fixed_action_every_millis() {
        let world = mock_world();
        let now = Instant::now();

        assert!(should_queue_fixed_action(
            &world,
            Some(now - Duration::from_millis(3000)),
            ActionCondition::EveryMillis(2000)
        ));
        assert!(!should_queue_fixed_action(
            &world,
            Some(now - Duration::from_millis(1000)),
            ActionCondition::EveryMillis(2000)
        ));
    }

    #[test]
    fn rotator_should_queue_fixed_action_erda_shower() {
        let mut world = mock_world();
        let now = Instant::now();

        world.skills[SkillKind::ErdaShower].state = Skill::Idle(Point::default(), Vec4b::default());
        assert!(!should_queue_fixed_action(
            &world,
            Some(now - Duration::from_millis(COOLDOWN_BETWEEN_QUEUE_MILLIS as u64 - 1000)),
            ActionCondition::ErdaShowerOffCooldown
        ));
        assert!(should_queue_fixed_action(
            &world,
            Some(now - Duration::from_millis(COOLDOWN_BETWEEN_QUEUE_MILLIS as u64)),
            ActionCondition::ErdaShowerOffCooldown
        ));

        world.skills[SkillKind::ErdaShower].state = Skill::Detecting;
        assert!(!should_queue_fixed_action(
            &world,
            Some(now - Duration::from_millis(COOLDOWN_BETWEEN_QUEUE_MILLIS as u64)),
            ActionCondition::ErdaShowerOffCooldown
        ));
    }

    #[test]
    fn rotator_build_actions() {
        let mut rotator = DefaultRotator::default();
        let actions = vec![NORMAL_ACTION, NORMAL_ACTION, PRIORITY_ACTION];
        let buffs = vec![(BuffKind::Rune, KeyKind::A); 4];
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: actions,
            character_actions: vec![],
            buffs,
            familiars: Familiars::default(),
            familiars_fusion: FamiliarsFusion::default(),
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::CycleChannel,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: true,
            enable_rune_solving: true,
            enable_transparent_shape_solving: true,
            enable_violetta_solving: true,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: false,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);
        assert_eq!(rotator.priority_actions.len(), 11);
        assert_eq!(rotator.normal_actions.len(), 2);
    }

    #[test]
    fn rotator_build_actions_monster_park_excludes_fixed_actions() {
        const EVERY_MILLIS_ACTION: Action = Action::Move(ActionMove {
            position: Position {
                x: 0,
                x_random_range: 0,
                y: 0,
                allow_adjusting: false,
            },
            condition: ActionCondition::EveryMillis(1000),
            wait_after_move_millis: 0,
        });

        let mut rotator = DefaultRotator::default();
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: vec![EVERY_MILLIS_ACTION, PRIORITY_ACTION],
            character_actions: vec![],
            buffs: vec![(BuffKind::Rune, KeyKind::A); 4],
            familiars: Familiars::default(),
            familiars_fusion: FamiliarsFusion::default(),
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::None,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: false,
            enable_rune_solving: false,
            enable_transparent_shape_solving: false,
            enable_violetta_solving: false,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: true,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);

        // Both the map-level EveryMillis action and the map-level ErdaShowerOffCooldown one are
        // excluded - only the always-present unstuck priority action remains. Since commit
        // 7252ede, exclusion is keyed on list origin (`is_map_action`) rather than whether the
        // action carries a fixed position, so a character-level EveryMillis action - even a
        // position-bound `Action::Move` one - is deliberately *not* covered by this case anymore
        // (see `rotator_build_actions_monster_park_includes_position_less_every_millis` and
        // `rotator_build_actions_monster_park_excludes_position_less_map_every_millis` for the
        // character/map-origin distinction).
        assert_eq!(rotator.priority_actions.len(), 1);
        assert!(
            rotator
                .priority_actions
                .values()
                .all(|action| action.condition_kind.is_none())
        );
    }

    #[test]
    fn rotator_build_actions_monster_park_excludes_familiars_swap() {
        let mut rotator = DefaultRotator::default();
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: vec![],
            character_actions: vec![],
            buffs: vec![],
            familiars: Familiars {
                enable_familiars_swapping: true,
                ..Familiars::default()
            },
            familiars_fusion: FamiliarsFusion::default(),
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::None,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: false,
            enable_rune_solving: false,
            enable_transparent_shape_solving: false,
            enable_violetta_solving: false,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: true,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);

        // Only the always-present unstuck priority action remains - familiar swapping is not
        // worth interrupting Monster Park's sweep-then-portal loop for, same as boosters/buffs.
        assert_eq!(rotator.priority_actions.len(), 1);
    }

    #[test]
    fn rotator_build_actions_monster_park_excludes_familiars_fusion() {
        let mut rotator = DefaultRotator::default();
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: vec![],
            character_actions: vec![],
            buffs: vec![],
            familiars: Familiars::default(),
            familiars_fusion: FamiliarsFusion {
                enable_familiars_fusion: true,
                ..FamiliarsFusion::default()
            },
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::None,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: false,
            enable_rune_solving: false,
            enable_transparent_shape_solving: false,
            enable_violetta_solving: false,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: true,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);

        // Only the always-present unstuck priority action remains - familiar fusion is not
        // worth interrupting Monster Park's sweep-then-portal loop for either, same as swap.
        assert_eq!(rotator.priority_actions.len(), 1);
    }

    #[test]
    fn rotator_build_actions_skips_familiars_swap_when_fusion_enabled() {
        fn args(familiars: Familiars, familiars_fusion: FamiliarsFusion) -> RotatorBuildArgs {
            RotatorBuildArgs {
                mode: RotatorMode::default(),
                character_level: 1,
                map_actions: vec![],
                character_actions: vec![],
                buffs: vec![],
                familiars,
                familiars_fusion,
                familiars_potential: FamiliarsPotential::default(),
                familiar_essence_key: KeyKind::A,
                elite_boss_behavior: EliteBossBehavior::None,
                elite_boss_behavior_key: KeyKind::A,
                hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
                hexa_booster_exchange_amount: 1,
                hexa_booster_exchange_all: false,
                enable_panic_mode: false,
                enable_rune_solving: false,
                enable_transparent_shape_solving: false,
                enable_violetta_solving: false,
                enable_reset_normal_actions_on_erda: false,
                enable_using_generic_booster: false,
                enable_using_hexa_booster: false,
                daily_quest_entries: vec![],
                daily_quest_mobbing_key: MobbingKey::default(),
                daily_quest_character_id: None,
                daily_quest_use_key_while_double_jumping: false,
                monster_park_enabled: false,
                thread_of_fate_ask_enabled: false,
                thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
                thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
                thread_of_fate_ask_character_id: None,
            }
        }

        let mut swap_only = DefaultRotator::default();
        swap_only.build_actions(args(
            Familiars {
                enable_familiars_swapping: true,
                ..Familiars::default()
            },
            FamiliarsFusion::default(),
        ));

        let mut fusion_only = DefaultRotator::default();
        fusion_only.build_actions(args(
            Familiars::default(),
            FamiliarsFusion {
                enable_familiars_fusion: true,
                ..FamiliarsFusion::default()
            },
        ));

        let mut both = DefaultRotator::default();
        both.build_actions(args(
            Familiars {
                enable_familiars_swapping: true,
                ..Familiars::default()
            },
            FamiliarsFusion {
                enable_familiars_fusion: true,
                ..FamiliarsFusion::default()
            },
        ));

        // Swapping alone registers its own standalone priority action, same as fusion alone
        // registers its own - each config contributes exactly one of these two mutually exclusive
        // actions on top of the same baseline (e.g. the always-registered unstuck action).
        assert_eq!(
            swap_only.priority_actions.len(),
            fusion_only.priority_actions.len()
        );
        // With fusion also enabled, the standalone swap action is skipped - fusion's own
        // completion always chains into a swap run regardless of outcome, so a second independent
        // timer for it would only race against that chain instead of adding coverage.
        assert_eq!(
            both.priority_actions.len(),
            fusion_only.priority_actions.len()
        );
    }

    #[test]
    fn rotator_advance_monster_park_portal_wiggle_sequences_left_right_up() {
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::Left))
            .once()
            .return_const(());
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::Right))
            .once()
            .return_const(());
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::Up))
            .once()
            .return_const(());
        let mut resources = Resources::new(Some(keys), None);
        let mut rotator = DefaultRotator {
            monster_park_portal_wiggle: Some((PortalWiggleStep::Left, 0)),
            ..Default::default()
        };

        // Left tap fires immediately (no delay) and hands off to Right, with a randomized delay.
        rotator.advance_monster_park_portal_wiggle(&mut resources);
        assert_matches!(
            rotator.monster_park_portal_wiggle,
            Some((PortalWiggleStep::Right, delay)) if delay > 0
        );

        // A non-zero delay just counts down without sending anything.
        let Some((_, delay)) = rotator.monster_park_portal_wiggle else {
            unreachable!()
        };
        rotator.advance_monster_park_portal_wiggle(&mut resources);
        assert_matches!(
            rotator.monster_park_portal_wiggle,
            Some((PortalWiggleStep::Right, remaining)) if remaining == delay - 1
        );

        // Skip straight to the delay elapsing to fire Right and hand off to Up.
        rotator.monster_park_portal_wiggle = Some((PortalWiggleStep::Right, 0));
        rotator.advance_monster_park_portal_wiggle(&mut resources);
        assert_matches!(
            rotator.monster_park_portal_wiggle,
            Some((PortalWiggleStep::Up, delay)) if delay > 0
        );

        // Up completes the sequence.
        rotator.monster_park_portal_wiggle = Some((PortalWiggleStep::Up, 0));
        rotator.advance_monster_park_portal_wiggle(&mut resources);
        assert_eq!(rotator.monster_park_portal_wiggle, None);
    }

    #[test]
    fn rotator_advance_monster_park_portal_wiggle_biases_toward_correction_key() {
        // With a known correction direction of Left, the physical keys swap: the `Left` step
        // (fired first) presses the away key (Right), and the `Right` step (fired last, right
        // before Up) presses the actual corrective key (Left) - see
        // `monster_park_portal_wiggle_toward_key`'s docs for why the last lateral tap must be the
        // corrective one.
        let mut keys = MockInput::new();
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::Right))
            .once()
            .return_const(());
        keys.expect_send_key()
            .withf(|k| matches!(k, KeyKind::Left))
            .once()
            .return_const(());
        let mut resources = Resources::new(Some(keys), None);
        let mut rotator = DefaultRotator {
            monster_park_portal_wiggle: Some((PortalWiggleStep::Left, 0)),
            monster_park_portal_wiggle_toward_key: Some(KeyKind::Left),
            ..Default::default()
        };

        rotator.advance_monster_park_portal_wiggle(&mut resources);
        rotator.monster_park_portal_wiggle = rotator
            .monster_park_portal_wiggle
            .map(|(step, _)| (step, 0));
        rotator.advance_monster_park_portal_wiggle(&mut resources);
    }

    #[test]
    fn rotator_escalate_monster_park_portal_stuck_wiggles_then_escapes() {
        let mut resources = Resources::new(None, None);
        let mut rotator = DefaultRotator::default();

        // First escalation - tries the wiggle instead of escaping outright.
        rotator.escalate_monster_park_portal_stuck(&mut resources, Some(KeyKind::Right));
        assert!(rotator.monster_park_portal_wiggled);
        assert_matches!(
            rotator.monster_park_portal_wiggle,
            Some((PortalWiggleStep::Left, 0))
        );
        assert_eq!(
            rotator.monster_park_portal_wiggle_toward_key,
            Some(KeyKind::Right)
        );
        assert!(rotator.monster_park_portal_escape.is_none());

        // Second escalation (wiggle already tried) - starts the escape instead, moving away
        // from `toward_key` (the direction that closes the gap to the portal).
        rotator.monster_park_portal_wiggle = None;
        rotator.escalate_monster_park_portal_stuck(&mut resources, Some(KeyKind::Right));
        assert!(!rotator.monster_park_portal_wiggled);
        assert_eq!(rotator.monster_park_portal_wiggle, None);
        assert_eq!(
            rotator.monster_park_portal_escape,
            Some((KeyKind::Left, PORTAL_ESCAPE_HOLD_TICKS))
        );
    }

    #[test]
    fn rotator_build_actions_monster_park_includes_position_less_every_millis() {
        // Unlike `EVERY_MILLIS_ACTION` above (an `Action::Move`, always position-bound), a
        // position-less `Action::Key` isn't tied to any particular map - e.g. a periodic
        // character-level buff/skill - so it should still be registered during Monster Park.
        let position_less_every_millis = Action::Key(ActionKey {
            condition: ActionCondition::EveryMillis(1000),
            ..ActionKey::default()
        });

        let mut rotator = DefaultRotator::default();
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: vec![],
            character_actions: vec![position_less_every_millis],
            buffs: vec![],
            familiars: Familiars::default(),
            familiars_fusion: FamiliarsFusion::default(),
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::None,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: false,
            enable_rune_solving: false,
            enable_transparent_shape_solving: false,
            enable_violetta_solving: false,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: true,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);

        // The position-less EveryMillis action plus the always-present unstuck action.
        assert_eq!(rotator.priority_actions.len(), 2);
        assert!(
            rotator.priority_actions.values().any(|action| matches!(
                action.condition_kind,
                Some(ActionCondition::EveryMillis(_))
            ))
        );
    }

    #[test]
    fn rotator_build_actions_monster_park_excludes_position_less_map_every_millis() {
        // A map-level periodic skill rotation keybind (e.g. a hunting ground's "F"/"R"/"Delete"
        // EveryMillis actions) has no fixed position, but it's still tied to that specific map's
        // setup, not the character - unlike a genuine character-level periodic buff/skill, it
        // should be excluded during Monster Park the same as a position-bound map action.
        let position_less_map_every_millis = Action::Key(ActionKey {
            condition: ActionCondition::EveryMillis(1000),
            ..ActionKey::default()
        });

        let mut rotator = DefaultRotator::default();
        let args = RotatorBuildArgs {
            mode: RotatorMode::default(),
            character_level: 1,
            map_actions: vec![position_less_map_every_millis],
            character_actions: vec![],
            buffs: vec![],
            familiars: Familiars::default(),
            familiars_fusion: FamiliarsFusion::default(),
            familiars_potential: FamiliarsPotential::default(),
            familiar_essence_key: KeyKind::A,
            elite_boss_behavior: EliteBossBehavior::None,
            elite_boss_behavior_key: KeyKind::A,
            hexa_booster_exchange_condition: ExchangeHexaBoosterCondition::None,
            hexa_booster_exchange_amount: 1,
            hexa_booster_exchange_all: false,
            enable_panic_mode: false,
            enable_rune_solving: false,
            enable_transparent_shape_solving: false,
            enable_violetta_solving: false,
            enable_reset_normal_actions_on_erda: false,
            enable_using_generic_booster: false,
            enable_using_hexa_booster: false,
            daily_quest_entries: vec![],
            daily_quest_mobbing_key: MobbingKey::default(),
            daily_quest_character_id: None,
            daily_quest_use_key_while_double_jumping: false,
            monster_park_enabled: true,
            thread_of_fate_ask_enabled: false,
            thread_of_fate_ask_npc: ThreadOfFateNpc::default(),
            thread_of_fate_ask_progress: ThreadOfFateAskProgress::default(),
            thread_of_fate_ask_character_id: None,
        };

        rotator.build_actions(args);

        // Only the always-present unstuck action remains.
        assert_eq!(rotator.priority_actions.len(), 1);
        assert!(
            rotator
                .priority_actions
                .values()
                .all(|action| action.condition_kind.is_none())
        );
    }

    #[test]
    fn rotator_priority_actions_every_millis_suppressed_only_at_monster_park_entry_lobby() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        rotator.monster_park_enabled = true;

        let action_id = 7;
        rotator.priority_actions.insert(
            action_id,
            priority_action(
                RotatorAction::Single(NORMAL_ACTION.into()),
                ActionCondition::EveryMillis(1000),
                false,
                false,
            ),
        );

        // At the entry lobby - the fixed action should not be queued.
        rotator.monster_park_at_entry_lobby = true;
        rotator.rotate_priority_actions(&mut resources, &mut world);
        assert!(!rotator.priority_actions_queue.contains(&action_id));

        // Past the lobby, sweeping a stage - the same fixed action should now queue normally.
        rotator.monster_park_at_entry_lobby = false;
        rotator.rotate_priority_actions(&mut resources, &mut world);
        assert!(rotator.priority_actions_queue.contains(&action_id));
    }

    #[test]
    fn rotator_priority_actions_position_less_every_millis_suppressed_only_while_navigating_daily_quest()
     {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        rotator.daily_quest_entries = vec![DailyQuestEntry::new(
            crate::models::DailyQuestId::VanishingJourney,
        )];
        rotator.daily_quest_index = 0;

        let position_less_id = 11;
        rotator.priority_actions.insert(
            position_less_id,
            priority_action(
                RotatorAction::Single(NORMAL_ACTION.into()),
                ActionCondition::EveryMillis(1000),
                false,
                false,
            ),
        );
        let position_bound_id = 12;
        rotator.priority_actions.insert(
            position_bound_id,
            priority_action(
                RotatorAction::Single(NORMAL_ACTION.into()),
                ActionCondition::EveryMillis(1000),
                false,
                true,
            ),
        );

        // Still navigating to the entry - both should be suppressed.
        rotator.daily_quest_arrived = false;
        rotator.rotate_priority_actions(&mut resources, &mut world);
        assert!(!rotator.priority_actions_queue.contains(&position_less_id));
        assert!(!rotator.priority_actions_queue.contains(&position_bound_id));

        // Arrived and mobbing - the position-less one should queue normally, the position-bound
        // one (map-tied to the regular farming map) should stay suppressed.
        rotator.daily_quest_arrived = true;
        rotator.rotate_priority_actions(&mut resources, &mut world);
        assert!(rotator.priority_actions_queue.contains(&position_less_id));
        assert!(!rotator.priority_actions_queue.contains(&position_bound_id));
    }

    #[test]
    fn rotator_priority_actions_every_millis_not_suppressed_while_monster_park_defers_daily_quest()
    {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        // A pending daily quest entry that has never actually been navigated to
        // (`daily_quest_arrived` stays at its default `false`) because an active Monster Park run
        // is holding it back - mirrors `rotate_action`'s `blocked_by_monster_park_run`.
        rotator.daily_quest_entries = vec![DailyQuestEntry::new(
            crate::models::DailyQuestId::VanishingJourney,
        )];
        rotator.daily_quest_index = 0;
        rotator.monster_park_enabled = true;
        rotator.monster_park_run_in_progress = true;

        let position_less_id = 13;
        rotator.priority_actions.insert(
            position_less_id,
            priority_action(
                RotatorAction::Single(NORMAL_ACTION.into()),
                ActionCondition::EveryMillis(1000),
                false,
                false,
            ),
        );

        // Without knowing Monster Park is the one deferring the entry, this would look
        // indistinguishable from "still navigating to the entry" and stay suppressed forever.
        rotator.rotate_priority_actions(&mut resources, &mut world);
        assert!(rotator.priority_actions_queue.contains(&position_less_id));
    }

    #[test]
    fn rotator_rotate_action_start_to_end_then_reverse() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        rotator.normal_rotate_mode = RotatorMode::StartToEndThenReverse;
        for i in 0..3 {
            rotator
                .normal_actions
                .push((i, RotatorAction::Single(NORMAL_ACTION.into())));
        }

        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(world.player.context.normal_action_id(), Some(0));
        assert!(!rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 1);

        world.player.context.clear_actions_aborted(true);
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(world.player.context.normal_action_id(), Some(1));
        assert!(!rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 2);

        world.player.context.clear_actions_aborted(true);
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(world.player.context.normal_action_id(), Some(2));
        assert!(rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 1);

        world.player.context.clear_actions_aborted(true);
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(world.player.context.normal_action_id(), Some(1));
        assert!(rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 2);

        world.player.context.clear_actions_aborted(true);
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(world.player.context.normal_action_id(), Some(0));
        assert!(!rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 1);
    }

    #[test]
    fn rotator_rotate_action_start_to_end() {
        let mut world = mock_world();
        let mut rotator = DefaultRotator::default();
        let mut resources = Resources::new(None, None);
        rotator.normal_rotate_mode = RotatorMode::StartToEnd;
        for i in 0..2 {
            rotator
                .normal_actions
                .push((i, RotatorAction::Single(NORMAL_ACTION.into())));
        }

        rotator.rotate_action(&mut resources, &mut world);
        assert!(world.player.context.has_normal_action());
        assert!(!rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 1);

        world.player.context.clear_actions_aborted(true);

        rotator.rotate_action(&mut resources, &mut world);
        assert!(world.player.context.has_normal_action());
        assert!(!rotator.normal_actions_backward);
        assert_eq!(rotator.normal_index, 0);
    }

    #[test]
    fn rotator_priority_actions_queue() {
        let mut rotator = DefaultRotator::default();
        let mut minimap = MinimapIdle::default();
        minimap.set_rune(Point::default());
        let mut world = mock_world();
        world.minimap.state = Minimap::Idle(minimap);
        world.buffs[BuffKind::Rune].state = Buff::No;
        rotator.priority_actions.insert(
            55,
            PriorityAction {
                condition: Condition(Box::new(|_, world, _| {
                    if matches!(world.minimap.state, Minimap::Idle(_)) {
                        ConditionResult::Queue
                    } else {
                        ConditionResult::Skip
                    }
                })),
                condition_kind: None,
                inner: RotatorAction::Single(PlayerAction::SolveRune),
                metadata: None,
                queue_to_front: true,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );
        let mut resources = Resources::new(None, None);

        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(rotator.priority_actions_queue.len(), 0);
        assert_eq!(world.player.context.priority_action_id(), Some(55));
    }

    #[test]
    fn rotator_priority_actions_queue_to_front() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        // queue 2 non-front priority actions
        rotator.priority_actions.insert(
            2,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );
        rotator.priority_actions.insert(
            3,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(rotator.priority_actions_queue.len(), 1);
        assert_eq!(world.player.context.priority_action_id(), Some(2));

        // add 1 front priority action
        rotator.priority_actions.insert(
            4,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: true,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        // non-front priority action get replaced
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(
            rotator.priority_actions_queue,
            VecDeque::from_iter([2, 3].into_iter())
        );
        assert_eq!(world.player.context.priority_action_id(), Some(4));

        // add another front priority action
        rotator.priority_actions.insert(
            5,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: true,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        // queued front priority action cannot be replaced
        // by another front priority action
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(
            rotator.priority_actions_queue,
            VecDeque::from_iter([5, 2, 3].into_iter())
        );
        assert_eq!(world.player.context.priority_action_id(), Some(4));
    }

    #[test]
    fn rotator_priority_linked_action() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);
        rotator.priority_actions.insert(
            2,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Linked(LinkedAction {
                    inner: NORMAL_ACTION.into(),
                    next: Some(Box::new(LinkedAction {
                        inner: NORMAL_ACTION.into(),
                        next: None,
                    })),
                }),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        // linked action queued
        rotator.rotate_action(&mut resources, &mut world);
        assert!(rotator.priority_actions_queue.is_empty());
        assert!(rotator.priority_queuing_linked_action.is_some());
        assert_eq!(world.player.context.priority_action_id(), Some(2));

        // linked action cannot be replaced by queue to front
        rotator.priority_actions.insert(
            4,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: None,
                inner: RotatorAction::Single(PlayerAction::SolveRune),
                metadata: None,
                queue_to_front: true,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );
        rotator.rotate_action(&mut resources, &mut world);
        assert_eq!(
            rotator.priority_actions_queue,
            VecDeque::from_iter([4].into_iter())
        );

        world.player.context.clear_actions_aborted(true);
        rotator.rotate_action(&mut resources, &mut world);
        assert!(rotator.priority_queuing_linked_action.is_none());
        assert_eq!(
            rotator.priority_actions_queue,
            VecDeque::from_iter([4].into_iter())
        );
        assert_eq!(world.player.context.priority_action_id(), Some(2));
    }

    #[test]
    fn rotate_ping_pong_direction() {
        let mut player = PlayerContext::default();
        let mut rotator = DefaultRotator::default();
        let mut idle = MinimapIdle::default();
        idle.bbox = Rect::new(0, 0, 100, 100); // x: [0, 100]

        // Closer to right, further than left -> Go left
        player.last_known_pos = Some(Point::new(80, 50));
        rotator.rotate_ping_pong(
            &mut player,
            Minimap::Idle(idle),
            MobbingKey::default(),
            Rect::new(20, 20, 80, 80).into(),
        );

        assert_matches!(
            player.normal_action(),
            Some(PlayerAction::PingPong(PingPong {
                direction: PingPongDirection::Left,
                ..
            }))
        );

        // Closer to left, further than right -> Go right
        player.clear_actions_aborted(true);
        player.last_known_pos = Some(Point::new(10, 50));
        rotator.rotate_ping_pong(
            &mut player,
            Minimap::Idle(idle),
            MobbingKey::default(),
            Rect::new(20, 20, 80, 80).into(),
        );

        assert_matches!(
            player.normal_action(),
            Some(PlayerAction::PingPong(PingPong {
                direction: PingPongDirection::Right,
                ..
            }))
        );
    }

    #[test]
    fn rotator_priority_action_is_ignored_when_executing() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);

        // Insert a priority action with condition_kind = None
        let action_id = 99;
        rotator.priority_actions.insert(
            action_id,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| panic!("should not be called"))),
                condition_kind: None,
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );
        // Simulate the action is currently being executed by the player
        world
            .player
            .context
            .set_priority_action(Some(action_id), NORMAL_ACTION.into());

        // Call rotate_priority_actions
        rotator.rotate_priority_actions(&mut resources, &mut world);

        let action = rotator.priority_actions.get(&action_id).unwrap();

        // Assert the action was marked as ignored
        assert!(action.queue_info.ignoring);
        assert!(action.queue_info.last_queued_time.is_some());

        // It should not be in the queue
        assert!(!rotator.priority_actions_queue.contains(&action_id));
    }

    #[test]
    fn rotator_priority_linked_action_is_ignored_when_executing() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);

        let action_id = 42;
        rotator.priority_actions.insert(
            action_id,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| panic!("should not be called"))),
                condition_kind: Some(ActionCondition::Linked),
                inner: RotatorAction::Linked(LinkedAction {
                    inner: NORMAL_ACTION.into(),
                    next: None,
                }),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        // Simulate action is being executed
        world
            .player
            .context
            .set_priority_action(Some(action_id), NORMAL_ACTION.into());

        rotator.rotate_priority_actions(&mut resources, &mut world);

        let action = rotator.priority_actions.get(&action_id).unwrap();

        assert!(action.queue_info.ignoring);
        assert!(action.queue_info.last_queued_time.is_some());
        assert!(!rotator.priority_actions_queue.contains(&action_id));
    }

    #[test]
    fn rotator_erda_shower_action_ignored_if_another_erda_is_queued() {
        let mut rotator = DefaultRotator::default();
        let mut world = mock_world();
        let mut resources = Resources::new(None, None);

        let first_erda_id = 1;
        let second_erda_id = 2;

        rotator.priority_actions.insert(
            first_erda_id,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| ConditionResult::Queue)),
                condition_kind: Some(ActionCondition::ErdaShowerOffCooldown),
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo {
                    last_queued_time: Some(Instant::now()),
                    ..Default::default()
                },
            },
        );

        rotator.priority_actions.insert(
            second_erda_id,
            PriorityAction {
                condition: Condition(Box::new(|_, _, _| panic!("should not be called"))),
                condition_kind: Some(ActionCondition::ErdaShowerOffCooldown),
                inner: RotatorAction::Single(NORMAL_ACTION.into()),
                metadata: None,
                queue_to_front: false,
                suppress_during_daily_quest: DailyQuestSuppression::Never,
                queue_info: PriorityActionQueueInfo::default(),
            },
        );

        // Queue the first erda action manually
        rotator.priority_actions_queue.push_back(first_erda_id);

        // Run rotate
        rotator.rotate_priority_actions(&mut resources, &mut world);

        let second_erda = rotator.priority_actions.get(&second_erda_id).unwrap();

        assert!(second_erda.queue_info.ignoring);
        assert!(second_erda.queue_info.last_queued_time.is_some());
        assert!(!rotator.priority_actions_queue.contains(&second_erda_id));
    }

    fn mock_detector(f: fn(&mut MockDetector)) -> MockDetector {
        let mut detector = MockDetector::new();

        f(&mut detector);

        detector
    }

    async fn queue_or_timeout(mut f: impl FnMut() -> ConditionResult) {
        timeout(Duration::from_secs(1), async move {
            loop {
                let result = f();
                if matches!(result, ConditionResult::Queue) {
                    break;
                }

                yield_now().await;
            }
        })
        .await
        .expect("queue result");
    }

    #[tokio::test]
    async fn unstuck_priority_action_triggers_when_esc_settings_detected() {
        let mut resources = Resources::new(
            None,
            Some(mock_detector(|detector| {
                detector.expect_detect_esc_settings().returning(|| true);
                detector.expect_detect_player_is_dead().returning(|| false);
            })),
        );
        let world = mock_world();
        let info = PriorityActionQueueInfo::default();
        let mut action = unstuck_priority_action();

        queue_or_timeout(|| (action.condition.0)(&mut resources, &world, &info)).await;
    }

    #[tokio::test]
    async fn elite_boss_use_key_priority_action_triggers_when_elite_present() {
        let detector = mock_detector(|detector| {
            detector.expect_detect_elite_boss_bar().return_const(true);
        });
        let mut resources = Resources::new(None, Some(detector));
        let world = mock_world();

        let mut action = elite_boss_use_key_priority_action(KeyKind::A);
        let info = PriorityActionQueueInfo::default();

        queue_or_timeout(|| (action.condition.0)(&mut resources, &world, &info)).await;
    }

    #[tokio::test]
    async fn panic_priority_action_triggers_when_has_other_players() {
        let mut resources = Resources::new(None, None);
        let mut idle = MinimapIdle::default();
        idle.set_has_any_other_player(true);
        let mut world = mock_world();
        world.minimap.state = Minimap::Idle(idle);

        let mut action = panic_priority_action();
        let info = PriorityActionQueueInfo {
            last_queued_time: Some(Instant::now() - std::time::Duration::from_millis(16000)),
            ..Default::default()
        };

        queue_or_timeout(|| (action.condition.0)(&mut resources, &world, &info)).await;
    }

    // TODO: more tests
}
