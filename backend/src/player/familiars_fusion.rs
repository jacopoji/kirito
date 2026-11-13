use std::{
    fmt::Display,
    sync::Arc,
    time::{Duration, Instant},
};

use log::{debug, info};
use opencv::core::{Mat, Point, Rect};

use super::{
    Player,
    familiars_swap::FamiliarsSwapping,
    timeout::{Lifecycle, Timeout, next_timeout_lifecycle},
};
use crate::{
    array::Array,
    bridge::{KeyKind, MouseKind},
    detect::{BuffKind, FamiliarLevel, FamiliarRank, FusionButton, FusionGaugeRarity},
    ecs::Resources,
    models::{FamiliarRarity, SwappableFamiliars},
    player::{PlayerEntity, next_action},
};

/// Number of familiar slots equipped in the Setup tab.
const FAMILIAR_SLOTS: usize = 3;

/// Maximum number of Select-All-then-Fuse rounds to attempt per target familiar before giving up.
///
/// Sized generously above the ~3 rounds needed for a 150-gauge fusion assuming roughly 50 fodder
/// can be selected per round (the visible 4x4 grid undercounts the actual capacity).
const MAX_FUSE_ROUNDS: u32 = 8;

/// Digit keys typed into the Inventory's "how many do you want to use?" quantity field each time
/// more fodder is pulled in from the bag, i.e. "50".
const INVENTORY_USE_QUANTITY_KEYS: [KeyKind; 2] = [KeyKind::Five, KeyKind::Zero];

/// [`Detector::detect_region_template`] threshold for matching the captured equipped-slot icon
/// against the Fusion tab's selected-preview card - see that function's doc for the live-testing
/// data this was tuned against.
const FUSION_SELECTED_PREVIEW_MATCH_THRESHOLD: f64 = 0.6;

/// [`Detector::detect_region_template`] threshold for matching an equipped slot's icon against
/// [`FamiliarsFusing::protected_icons`] - reuses the same value as
/// [`super::familiars_swap`]'s own `RANKED_UP_ICON_MATCH_THRESHOLD`, matching in the same visual
/// context (a small collection/equipped-slot icon, not yet independently live-tested for this
/// specific check).
const PROTECTED_FAMILIAR_ICON_MATCH_THRESHOLD: f64 = 0.6;

/// [`Detector::detect_region_template`] threshold for matching the captured equipped-slot name
/// (title-bar crop, see [`FamiliarsFusing::names`]) against an Inventory `Use`-tab hover tooltip
/// while scanning for fodder.
///
/// The portrait-icon approach previously used here proved unreliable for at least one familiar
/// with an animated glow/particle art style - live testing found a confirmed-genuine icon match
/// scoring only ~0.566 (barely above ~0.42 for non-matching slots in the same run), because no two
/// captures of an animated glow land on identical pixels. The title bar is a flat, static UI
/// element unaffected by that: live testing after switching to it found a confirmed-genuine match
/// (same familiar species duplicate) scoring ~0.756, reproducibly, against ~0.25-0.35 for
/// non-matching slots across two separate runs - a much wider margin than the old icon approach,
/// as expected. An initial guess of 0.8 (picked before this data existed) turned out higher than
/// the observed genuine score and rejected a real match; lowered to sit roughly midway in this
/// now-measured gap.
const INVENTORY_FODDER_MATCH_THRESHOLD: f64 = 0.6;

/// Offset from the detected `Use`-tab button's top-left corner to the item grid's first slot's
/// top-left corner, measured directly against a real screenshot.
///
/// The Inventory window is a free-floating, user-repositionable window - live testing showed it
/// opening flush against the top-left of the screen rather than at any fixed position, so the
/// grid's absolute screen position cannot be assumed. The `Use` tab button is detected as part of
/// opening the window anyway (see [`update_open_inventory`]), so the grid is instead anchored
/// relative to it.
const INVENTORY_GRID_OFFSET_FROM_USE_TAB_X: i32 = -116;
const INVENTORY_GRID_OFFSET_FROM_USE_TAB_Y: i32 = 65;
/// Distance between the top-left corners of adjacent grid cells (columns and rows share the same
/// pitch), measured directly against real screenshots.
const INVENTORY_GRID_CELL_PITCH: i32 = 46;
/// Width/height of the clickable/hoverable icon area within each cell.
const INVENTORY_GRID_CELL_SIZE: i32 = 44;
/// Number of columns in the Inventory `Use`-tab item grid.
const INVENTORY_GRID_COLUMNS: u32 = 14;
/// Number of inventory slots to hover-scan (top-left to bottom-right) before giving up on
/// topping up fodder from the bag - by convention, fodder is expected to be kept in the first
/// few slots rather than scattered throughout the whole (up to 128-slot) inventory.
const INVENTORY_SCAN_MAX_SLOTS: u32 = 28;
/// Tick within a single Inventory slot hover at which the tooltip is first checked for a match,
/// ahead of the slot's full timeout - mirrors the early-check-then-fallback shape used elsewhere
/// in this file (e.g. `CANDIDATE_LEVEL_CHECK_TICK`), but keeps a full retry at the slot's normal
/// timeout tick rather than giving up outright if this early check misses, since unlike those
/// other checks this one has no equivalent live-tested minimum render time behind it yet - a
/// slower-rendering tooltip still gets the full window it always had.
const FODDER_ITEM_EARLY_CHECK_TICK: u32 = 6;

/// Internal state machine representing the current state of familiar fusion (rank-up).
#[derive(Debug, Clone, Copy)]
enum State {
    /// Opening the familiar menu.
    OpenMenu(Timeout),
    /// Clicking the Setup tab, entered only when re-scanning equipped slots for the next target
    /// (see [`update_next_target`]'s docs) while still on the Fusion tab from processing the
    /// previous one.
    ///
    /// [`State::CheckEquippedSlotLevel`] hovers [`FamiliarsFusing::pending_slots`]' cached
    /// bounding boxes, which are only meaningful while the Setup tab is actually showing - they
    /// were captured from it once, before fusion ever switched to the Fusion tab. Live testing
    /// with the rescan going straight to `CheckEquippedSlotLevel` while still on the Fusion tab
    /// showed it hovering those coordinates over the Fusion tab's own collection-list grid
    /// instead, reading whatever unrelated card's hover state happened to be there and never the
    /// real equipped slot's level - the rescan then always concluded nothing was eligible, even
    /// with a second familiar genuinely ready. Mirrors [`super::familiars_swap`]'s own
    /// `State::SwitchToSetupTab` (used for the same button, in the reverse direction).
    SwitchToSetupTab(Timeout),
    /// Finding the 3 equipped familiar slots.
    FindEquippedSlots,
    /// Hovering the equipped slot at `index` and checking whether it is Level 5 - mirrors swap's
    /// `update_free_slot`'s exact same eligibility check (same hover offset and tick timing),
    /// since a familiar still gaining levels isn't a sensible fusion target yet. A slot must also
    /// pass the existing rarity/[`SwappableFamiliars`] filters and not already be in
    /// [`FamiliarsFusing::attempted`] to become the fusion target. An eligible Rare-rarity slot
    /// stops the scan immediately, but an eligible Epic-rarity slot is only kept as a fallback
    /// ([`FamiliarsFusing::fallback_target`]) while the scan keeps going through the remaining
    /// slots, since ranking up an Epic consumes it as fodder/base and its potential/effect can be
    /// worth more kept than a Rare's - a still-leveling or lower-priority Epic elsewhere shouldn't
    /// be skipped past in favor of consuming one prematurely just because it was found first. Once
    /// a target (Rare, or the deferred Epic if no Rare exists) is settled on, the scan stops there
    /// and moves on rather than continuing to `index + 1` - see [`update_next_target`]'s docs for
    /// why only one target is worked on at a time (re-entering this state from `index = 0`
    /// afterwards) instead of collecting every eligible slot up front.
    CheckEquippedSlotLevel(Timeout, u32),
    /// Switching to (or confirming) the Fusion tab.
    OpenFusionTab(Timeout),
    /// Clicking the collection list's `Level` sort header twice (ascending then descending) so
    /// the highest-level familiars - which the currently-equipped target is likely among -
    /// surface at the top of the list, without needing to scroll to find it.
    ///
    /// Only reached for the first target - see [`FamiliarsFusing::sorted_by_level`]'s docs for why
    /// later targets skip straight to [`State::FindInCollectionList`] instead.
    SortByLevel(Timeout, u32, Option<Rect>),
    /// Hovering the next untried same-rarity candidate card (by index) in the Fusion tab's
    /// collection list, checking it is Level 5 (matching the equipped target, since same-species
    /// duplicates can exist at other levels and look visually identical), clicking it if so, then
    /// verifying it's the target familiar via the resulting selected preview card. `Option<Rect>`
    /// caches the candidate's bbox from when hovering started, so the later click re-uses the
    /// exact same position instead of re-detecting - live testing showed a fresh detection at
    /// click time can resolve to a different, adjacent card than the one just hover-checked as
    /// Level 5 (e.g. because the hover itself shifts the list's detected ordering slightly),
    /// silently clicking the wrong duplicate. The `bool` tracks whether this attempt actually
    /// clicked (Level 5 confirmed) so [`Lifecycle::Ended`] knows whether to verify-and-maybe-cancel
    /// or just move on to the next candidate.
    FindInCollectionList(Timeout, u32, Option<Rect>, bool),
    /// Clicking `Cancel` to deselect a wrongly-guessed candidate before trying the next one.
    CancelCandidate(Timeout, u32),
    /// Clicking `Select All` to pull in-collection duplicates into the fodder grid.
    SelectAllFodder(Timeout),
    /// Checking the `Fuse`/`Rank Up` button and clicking `Fuse` if more fodder was selected.
    Fusing(Timeout),
    /// Confirming the "Do you want to fuse this Familiar?" popup that follows clicking `Fuse`.
    /// The first `bool` carries through whether [`MAX_FUSE_ROUNDS`] was hit on this click, so
    /// [`Lifecycle::Ended`] knows whether to go back through [`State::Fusing`] to re-check the
    /// button (the just-confirmed Fuse can by itself have completed the gauge) or finally give up.
    /// The second `bool` tracks whether the popup's `Confirm` button has already been clicked
    /// once, so it isn't clicked again on a later tick - see its docs on [`update_confirm_fuse`]
    /// for why that matters.
    ConfirmFuse(Timeout, bool, bool),
    /// Opening the Inventory window to top up fodder from the bag.
    OpenInventory(Timeout),
    /// Clicking the `Use` tab if it isn't already the focused tab - the Inventory can open on
    /// whichever tab was last active, not always `Use`. The `bool` tracks whether the click has
    /// already been sent once, so it isn't repeated on later ticks while still checking whether
    /// it took effect - see [`update_focus_use_tab`]'s docs for why repeat clicks are unsafe here.
    FocusUseTab(Timeout, bool),
    /// Hovering the next untried Inventory grid slot (by index, scanning top-left to
    /// bottom-right - fodder is expected to be kept in the first few slots by convention) and
    /// checking whether its tooltip preview matches the target familiar.
    FindFodderItem(Timeout, u32),
    /// Double-clicking a matched bag item to open the "how many do you want to use?" dialog.
    DoubleClickItem(Timeout, Rect),
    /// Clearing and typing the quantity into the dialog's input field.
    TypeQuantity(Timeout),
    /// Clicking the dialog's `Confirm` button.
    ConfirmQuantity(Timeout),
    /// Confirming the "Add this familiar to [Character]'s Familiar List?" popup that follows
    /// confirming the quantity - each "Use" of a bag item creates that many new in-collection
    /// duplicates, which is what actually makes them available as fodder in the Fusion tab. The
    /// `bool` tracks whether `Confirm` has already been clicked once (see
    /// [`update_confirm_fuse`]'s docs for why repeat clicks after success are unsafe).
    ConfirmAddFamiliar(Timeout, bool),
    /// Closing the Inventory window, then either resuming the fuse loop or giving up the target.
    CloseInventory(Timeout, bool),
    /// Clicking the `Rank Up` button once the gauge is filled.
    RankingUp(Timeout),
    /// Confirming the "Do you want to attempt to improve the Familiar's rank?" popup that follows
    /// clicking `Rank Up`. The `bool` tracks whether `Confirm` has already been clicked once (see
    /// [`update_confirm_fuse`]'s docs for why repeat clicks after success are unsafe).
    ConfirmRankUp(Timeout, bool),
    /// Clicking `Cancel` to deselect the just-ranked-up familiar before moving on to the next
    /// target. A successful Rank Up changes the familiar's rarity (e.g. Rare -> Epic) but leaves
    /// it selected as the base card - per [`State::CancelCandidate`]'s docs, the game silently
    /// ignores clicking a different candidate while one is already selected, so without this the
    /// next target's search would find nothing and never know why. The `u32` counts verify-and-
    /// retry attempts (see [`update_deselect_ranked_up`]'s docs) - unlike `CancelCandidate`, this
    /// has no natural retry loop to fall back into if the click doesn't take, so it must verify
    /// and retry itself.
    DeselectRankedUp(Timeout, u32),
    /// Marking the current slot done and re-scanning equipped slots for the next eligible target,
    /// if any - see [`update_next_target`]'s docs for why this re-scans instead of advancing
    /// through a precomputed list.
    NextTarget,
    Completing(Timeout, bool),
}

/// Screen positions of Fusion-tab chrome buttons - the tab buttons and the panel's own
/// Select-All/Cancel buttons - that stay put for as long as the familiar menu remains open,
/// cached after their first successful detection so the (comparatively expensive, full-frame)
/// template match backing each one only ever runs once per [`FamiliarsFusing`] instance instead
/// of on every single tick that clicks them (e.g. every one of up to [`MAX_FUSE_ROUNDS`] rounds
/// per target).
///
/// Deliberately excludes anything whose *visibility*, not just position, is what a call site is
/// actually testing for - e.g. the "Do you want to fuse...?" popup's `Confirm` button
/// ([`update_confirm_fuse`] and friends), which is only sometimes on screen and whose whole
/// detect-and-retry-every-tick loop exists specifically to wait for it to render. Caching that
/// position would skip the render-wait entirely on a later reuse and click wherever it last was
/// immediately, regardless of whether a popup is actually showing there yet. Also excludes
/// anything whose on-screen *content*, not just position, must be read fresh every time (e.g.
/// [`State::Fusing`]'s Fuse/Rank-Up button, whose current label is itself the answer being looked
/// for). The buttons cached here have neither problem - they are ordinary always-rendered panel
/// chrome for as long as their tab is showing, not something that pops in and out.
///
/// Deliberately not carried across into the chained [`FamiliarsSwapping`] on completion - a
/// fresh instance always starts with an empty cache, matching the fact that the menu can in
/// principle be closed and reopened (e.g. a future standalone swap run) between the two, at
/// which point cached coordinates from an old window layout could no longer be trusted.
#[derive(Debug, Clone, Copy, Default)]
struct CachedButtons {
    setup_tab: Option<Rect>,
    fusion_tab: Option<Rect>,
    select_all: Option<Rect>,
    cancel: Option<Rect>,
}

/// Returns `cache`'s already-cached value if present, otherwise runs `detect` once and populates
/// `cache` with the result on success - see [`CachedButtons`]'s docs for why this is safe to do
/// for the handful of buttons it covers.
#[inline]
fn cached_button_bbox(
    cache: &mut Option<Rect>,
    detect: impl FnOnce() -> anyhow::Result<Rect>,
) -> Option<Rect> {
    if let Some(bbox) = *cache {
        return Some(bbox);
    }
    let bbox = detect().ok()?;
    *cache = Some(bbox);
    Some(bbox)
}

/// An eligible Epic-rarity fusion target set aside by [`update_check_equipped_slot_level`] while it
/// keeps scanning the remaining equipped slots for a higher-priority Rare-rarity one - see that
/// function's docs for why Rare is preferred.
#[derive(Debug, Clone)]
struct FallbackTarget {
    slot_index: usize,
    bbox: Rect,
    rarity: FamiliarRarity,
    icon: Arc<Mat>,
    name: Arc<Mat>,
}

/// Struct for storing familiar fusion data.
#[derive(Debug, Clone)]
pub struct FamiliarsFusing {
    /// Current state of the familiar fusion state machine.
    state: State,
    /// See [`CachedButtons`]'s docs.
    buttons: CachedButtons,
    /// Only equipped familiars of these rarities will be considered for fusion (`Rare` for
    /// rare -> epic, `Epic` for epic -> unique).
    fusable_rarities: Array<FamiliarRarity, 2>,
    /// Carried through so fusion can chain into the existing swap-and-save flow on completion.
    swappable_slots: SwappableFamiliars,
    swappable_rarities: Array<FamiliarRarity, 2>,
    /// Icons of familiars [`Player::FamiliarsRevealingPotential`] has revealed a "good" potential
    /// line on (see [`crate::player::PlayerContext::protected_familiar_icons`]) - an equipped slot
    /// matching one of these is skipped as a fusion target entirely, even if otherwise eligible,
    /// so a familiar worth keeping never gets ranked away. Empty (the default from [`Self::new`])
    /// when no potentials have ever been revealed as good, matching prior behavior.
    protected_icons: Vec<Arc<Mat>>,
    /// Raw equipped-slot detection results (bbox, is_free), scanned one at a time via hover+level
    /// check in [`State::CheckEquippedSlotLevel`] before being accepted as a fusion target.
    pending_slots: Array<(Rect, bool), FAMILIAR_SLOTS>,
    /// Slot indices (into [`Self::pending_slots`]) already ranked up or given up on this run.
    ///
    /// [`update_next_target`] re-scans equipped slots from index 0 after finishing each target
    /// instead of walking a precomputed list - a rank-up changes the familiar's level/rarity and
    /// can shift its position in the Fusion tab's level-sorted collection list, so a stale
    /// precomputed target list can no longer be trusted once any target has been processed. This
    /// tracks which physical slots are done so the re-scan does not re-attempt them (which would
    /// otherwise loop forever on a give-up, or on a slot that dropped below Level 5 after ranking
    /// up).
    attempted: [bool; FAMILIAR_SLOTS],
    /// Holds at most one entry: the equipped Setup-tab slot bounding box currently being
    /// processed, found by the latest [`State::CheckEquippedSlotLevel`] scan.
    targets: Array<Rect, FAMILIAR_SLOTS>,
    /// Rarity of the current `targets` entry, used to filter which Fusion-tab collection list
    /// candidates are worth trying for the current target.
    target_rarities: Array<FamiliarRarity, FAMILIAR_SLOTS>,
    /// Ad-hoc icon crop of the current `targets` entry, used to find the target familiar within
    /// the Fusion tab (collection list, selected-preview card) since no precompiled template can
    /// exist for an arbitrary familiar type.
    ///
    /// Not used for Inventory fodder matching (see [`Self::names`]) - a glowing/particle-effect
    /// familiar's portrait render isn't pixel-stable enough between the equipped-slot capture and
    /// an Inventory tooltip render, confirmed via live testing (a genuine match scored only
    /// ~0.566, barely above wrong candidates' ~0.42).
    icons: Vec<Arc<Mat>>,
    /// Ad-hoc title-bar (name text) crop of the current `targets` entry, used to find the target
    /// familiar's duplicates in the Inventory `Use` tab by name rather than portrait art - see
    /// [`Self::icons`]'s docs for why the portrait doesn't work reliably there. The title bar is a
    /// flat, non-animated UI element regardless of the familiar's own art, so it isn't affected by
    /// glow/particle effects the way the portrait is.
    names: Vec<Arc<Mat>>,
    /// Icons of targets that were successfully ranked up this run (a subset of `icons`, across all
    /// targets processed so far).
    ///
    /// A ranked-up familiar changes rarity but keeps its Level 5 -> 1 reset, and stays equipped in
    /// its slot - carried through to the chained [`FamiliarsSwapping`] so its own candidate scan
    /// (which otherwise treats a non-Level-5 hover as "an available duplicate, swap it in") can
    /// recognize and skip these by icon instead of being fooled by the level reset into trying to
    /// swap in a familiar that is already equipped elsewhere.
    ranked_up_icons: Vec<Arc<Mat>>,
    /// Icons of targets that specifically ranked up from Rare to Epic this run (a subset of
    /// [`Self::ranked_up_icons`], further filtered to `original_rarity == Rare`).
    ///
    /// Epic is the rarity [`Player::FamiliarsRevealingPotential`] cares about protecting (an
    /// Epic->Unique rank-up produces a Unique familiar - already the final tier, with nothing
    /// left to protect it from), so a freshly-created Epic here is exactly the trigger
    /// [`Player::FamiliarsSwapping`]'s own chaining docs describe for a targeted potential-reveal
    /// re-check, once the one-time full collection scan is done. Carried through the same
    /// fusion -> swap -> reveal-potential chain as [`Self::ranked_up_icons`].
    newly_epic_icons: Vec<Arc<Mat>>,
    /// Equipped slot indices (into [`Self::pending_slots`]) successfully ranked up this run - a
    /// subset of the physical slots [`Self::attempted`] tracks.
    ///
    /// Carried through to the chained [`FamiliarsSwapping`] alongside [`Self::ranked_up_icons`]
    /// so its own equipped-slot eligibility check (`update_free_slot`) can force-free a
    /// just-ranked-up slot by exact physical position rather than icon match - unlike matching
    /// the ranked-up familiar among the swap collection's cards (what `ranked_up_icons` is for),
    /// freeing its own equipped slot needs no visual identification at all since fusion already
    /// knows precisely which slot it just ranked up.
    force_swap_slots: [bool; FAMILIAR_SLOTS],
    /// Equipped slot indices where fusion actually searched the Inventory's Use tab and confirmed
    /// no matching fodder was there ([`update_find_fodder_item`] exhausting
    /// [`INVENTORY_SCAN_MAX_SLOTS`]), as opposed to giving up on a target for any other reason
    /// (hitting [`MAX_FUSE_ROUNDS`], failing to select the target in the collection list, a
    /// detection hiccup opening the Inventory) where fodder may well still be available.
    ///
    /// Carried through to the chained [`FamiliarsSwapping`] so its own equipped-slot eligibility
    /// check only treats a still-fusable-rarity Level-5 familiar as safe to free when fusion has
    /// positively confirmed there's nothing left to feed it - any other give-up reason leaves it
    /// protected, since fusion may simply pick the same target back up successfully next cycle.
    fodder_exhausted_slots: [bool; FAMILIAR_SLOTS],
    /// Index into `targets`/`icons` currently being processed - always `0` since `targets` holds
    /// at most one entry (see [`Self::targets`]'s docs). Kept as an index (rather than accessing
    /// `targets[0]` directly at every call site) so the many existing read sites do not need to
    /// change.
    target_index: usize,
    /// Slot index (into [`Self::pending_slots`]) of the current `targets` entry - the physical
    /// equipped slot last found eligible by [`State::CheckEquippedSlotLevel`]. Recorded so
    /// [`update_next_target`] knows which entry of [`Self::attempted`] to mark once this target is
    /// done.
    current_slot_index: usize,
    /// Whether the Fusion tab's collection list has already been sorted by level this run.
    ///
    /// The sort persists across Setup/Fusion tab switches and only resets when the familiar menu
    /// closes entirely - which doesn't happen mid-run (see [`State::SwitchToSetupTab`]'s docs on
    /// why the menu stays open across targets). So it only needs doing once per
    /// [`FamiliarsFusing`] instance, on the first target - re-clicking it for later targets is not
    /// just wasted time but risky: [`update_sort_by_level`]'s own docs note the sort button's icon
    /// changes once already sorted, dropping its template match score below the detection
    /// threshold, which is exactly the state a later target's first click attempt would find it in.
    sorted_by_level: bool,
    /// Number of Select-All-then-Fuse rounds attempted for the current target.
    round: u32,
    /// Mouse rest point for other operations.
    mouse_rest: Point,
    /// Whether fusing completed without an unexpected detection failure.
    success: bool,
    /// An eligible Epic-rarity target found earlier in the current equipped-slot scan, held back in
    /// case a Rare-rarity target - preferred, since ranking up Epic consumes it as fodder/base and a
    /// kept Epic's potential/effect may be worth more than the rank-up - turns up in a later slot.
    /// See [`update_check_equipped_slot_level`]'s docs for the full rationale.
    fallback_target: Option<FallbackTarget>,
    /// Next [`Instant`] at which a pending lie detector prompt is (re-)checked for - unlike this
    /// module's other detector calls, [`Detector::detect_lie_detector_shape`] and
    /// [`Detector::detect_lie_detector_violetta`] decode their template from a base64 string fresh
    /// on every call when the character has a custom localization template configured (everything
    /// else in this file matches against a `LazyLock`-cached, decoded-once template) - calling
    /// either on every tick measured at ~90-100ms each, turning a ~2.6s `OpenMenu` step into over
    /// 10s. Throttled to the same 3000ms cadence the equivalent priority-action check in
    /// `rotator.rs` already polls at via its own `update_detection_task`, which is more than
    /// responsive enough given the prompt's own multi-second "preparing" countdown.
    lie_detector_check_next_at: Option<Instant>,
}

impl Display for FamiliarsFusing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.state {
            State::OpenMenu(_) => write!(f, "OpenMenu"),
            State::SwitchToSetupTab(_) => write!(f, "SwitchToSetupTab"),
            State::FindEquippedSlots => write!(f, "FindEquippedSlots"),
            State::CheckEquippedSlotLevel(_, _) => write!(f, "CheckEquippedSlotLevel"),
            State::OpenFusionTab(_) => write!(f, "OpenFusionTab"),
            State::SortByLevel(_, _, _) => write!(f, "SortByLevel"),
            State::FindInCollectionList(_, _, _, _) => write!(f, "FindInCollectionList"),
            State::CancelCandidate(_, _) => write!(f, "CancelCandidate"),
            State::SelectAllFodder(_) => write!(f, "SelectAllFodder"),
            State::Fusing(_) => write!(f, "Fusing"),
            State::ConfirmFuse(_, _, _) => write!(f, "ConfirmFuse"),
            State::OpenInventory(_) => write!(f, "OpenInventory"),
            State::FocusUseTab(_, _) => write!(f, "FocusUseTab"),
            State::FindFodderItem(_, _) => write!(f, "FindFodderItem"),
            State::DoubleClickItem(_, _) => write!(f, "DoubleClickItem"),
            State::TypeQuantity(_) => write!(f, "TypeQuantity"),
            State::ConfirmQuantity(_) => write!(f, "ConfirmQuantity"),
            State::ConfirmAddFamiliar(_, _) => write!(f, "ConfirmAddFamiliar"),
            State::CloseInventory(_, _) => write!(f, "CloseInventory"),
            State::RankingUp(_) => write!(f, "RankingUp"),
            State::ConfirmRankUp(_, _) => write!(f, "ConfirmRankUp"),
            State::DeselectRankedUp(_, _) => write!(f, "DeselectRankedUp"),
            State::NextTarget => write!(f, "NextTarget"),
            State::Completing(_, _) => write!(f, "Completing"),
        }
    }
}

impl FamiliarsFusing {
    /// Whether it's safe to inject a side-channel key tap (idle-time attack while fusing) right
    /// now without it landing in a text field or interrupting the Inventory top-up dialog.
    ///
    /// Excludes the entire Inventory fodder-top-up sub-flow - not just [`State::TypeQuantity`]
    /// itself, which is the only place a literal digit is typed into a text field
    /// ([`INVENTORY_USE_QUANTITY_KEYS`]) - for a comfortable safety margin around that whole
    /// dialog. Everything else (hovering candidates, clicking Select All/Fuse/Rank Up, waiting on
    /// confirm popups) is fair game.
    pub(crate) fn is_safe_for_side_key_press(&self) -> bool {
        !matches!(
            self.state,
            State::OpenInventory(_)
                | State::FocusUseTab(_, _)
                | State::FindFodderItem(_, _)
                | State::DoubleClickItem(_, _)
                | State::TypeQuantity(_)
                | State::ConfirmQuantity(_)
                | State::ConfirmAddFamiliar(_, _)
                | State::CloseInventory(_, _)
        )
    }

    pub fn new(
        fusable_rarities: Array<FamiliarRarity, 2>,
        swappable_slots: SwappableFamiliars,
        swappable_rarities: Array<FamiliarRarity, 2>,
    ) -> Self {
        Self {
            state: State::OpenMenu(Timeout::default()),
            buttons: CachedButtons::default(),
            fusable_rarities,
            swappable_slots,
            swappable_rarities,
            protected_icons: Vec::new(),
            pending_slots: Array::new(),
            attempted: [false; FAMILIAR_SLOTS],
            targets: Array::new(),
            target_rarities: Array::new(),
            icons: Vec::new(),
            names: Vec::new(),
            ranked_up_icons: Vec::new(),
            newly_epic_icons: Vec::new(),
            force_swap_slots: [false; FAMILIAR_SLOTS],
            fodder_exhausted_slots: [false; FAMILIAR_SLOTS],
            target_index: 0,
            current_slot_index: 0,
            sorted_by_level: false,
            round: 0,
            mouse_rest: Point::new(50, 50),
            success: true,
            fallback_target: None,
            lie_detector_check_next_at: None,
        }
    }

    /// See [`Self::protected_icons`]'s docs.
    pub fn with_protected_icons(mut self, protected_icons: Vec<Arc<Mat>>) -> Self {
        self.protected_icons = protected_icons;
        self
    }
}

/// Updates [`Player::FamiliarsFusing`] contextual state.
///
/// Note: This state does not use any [`Task`], so all detections are blocking. But this should be
/// acceptable for this state.
pub fn update_familiars_fusing_state(resources: &mut Resources, player: &mut PlayerEntity) {
    let Player::FamiliarsFusing(fusing) = &player.state else {
        panic!("state is not familiars fusing")
    };
    let mut fusing = fusing.clone();

    let now = Instant::now();
    let should_check_lie_detector = fusing.lie_detector_check_next_at.is_none_or(|at| now >= at);
    if should_check_lie_detector {
        fusing.lie_detector_check_next_at = Some(now + Duration::from_millis(3000));
    }
    if should_check_lie_detector
        && fusing.is_safe_for_side_key_press()
        && (resources.detector().detect_lie_detector_shape().is_ok()
            || resources.detector().detect_lie_detector_violetta().is_ok())
    {
        // A lie detector prompt takes priority over fusion (and the swap run fusion would
        // otherwise unconditionally chain into on completion - see this module's own docs on
        // why every completion chains) - waiting for fusion's own retry loop to run its course
        // first risks missing the prompt's response window entirely. Bails straight to `Idle`
        // rather than through `State::Completing`, since that always hands off into
        // `Player::FamiliarsSwapping` and would just reintroduce the same delay one state later.
        // Gated on `is_safe_for_side_key_press` so this never abandons mid-quantity-typing in the
        // Inventory fodder flow, the same one place a stray abort could leave a text field
        // focused - `rotate_priority_actions` already has the lie detector solve queued and
        // waiting, so it picks up on the very next tick once the state is next `Idle`.
        debug!(target: "backend/player", "aborting familiars fusing: lie detector pending");
        player.context.clear_action_completed();
        player.state = Player::Idle;
        return;
    }

    let familiar_key = match player.context.config.familiar_key {
        Some(val) => val,
        None => {
            info!(target:"backend/player","aborted familiars fusing because familiar menu key is not set");
            player.context.clear_action_completed();
            player.state = Player::Idle;
            return;
        }
    };
    let inventory_key = player.context.config.inventory_key;
    let familiar_buff_key = player.context.config.familiar_buff_key;

    match fusing.state {
        State::OpenMenu(_) => {
            update_open_menu(resources, &mut fusing, familiar_key, familiar_buff_key)
        }
        State::SwitchToSetupTab(_) => update_switch_to_setup_tab(resources, &mut fusing),
        State::FindEquippedSlots => update_find_equipped_slots(resources, &mut fusing),
        State::CheckEquippedSlotLevel(_, _) => {
            update_check_equipped_slot_level(resources, &mut fusing)
        }
        State::OpenFusionTab(_) => update_open_fusion_tab(resources, &mut fusing),
        State::SortByLevel(_, _, _) => update_sort_by_level(resources, &mut fusing),
        State::FindInCollectionList(_, _, _, _) => {
            update_find_in_collection_list(resources, &mut fusing)
        }
        State::CancelCandidate(_, _) => update_cancel_candidate(resources, &mut fusing),
        State::SelectAllFodder(_) => update_select_all_fodder(resources, &mut fusing),
        State::Fusing(_) => update_fusing(resources, &mut fusing),
        State::ConfirmFuse(_, _, _) => update_confirm_fuse(resources, &mut fusing),
        State::OpenInventory(_) => update_open_inventory(resources, &mut fusing, inventory_key),
        State::FocusUseTab(_, _) => update_focus_use_tab(resources, &mut fusing),
        State::FindFodderItem(_, _) => update_find_fodder_item(resources, &mut fusing),
        State::DoubleClickItem(_, _) => update_double_click_item(resources, &mut fusing),
        State::TypeQuantity(_) => update_type_quantity(resources, &mut fusing),
        State::ConfirmQuantity(_) => update_confirm_quantity(resources, &mut fusing),
        State::ConfirmAddFamiliar(_, _) => update_confirm_add_familiar(resources, &mut fusing),
        State::CloseInventory(_, _) => {
            update_close_inventory(resources, &mut fusing, inventory_key)
        }
        State::RankingUp(_) => update_ranking_up(resources, &mut fusing),
        State::ConfirmRankUp(_, _) => update_confirm_rank_up(resources, &mut fusing),
        State::DeselectRankedUp(_, _) => update_deselect_ranked_up(resources, &mut fusing),
        State::NextTarget => update_next_target(&mut fusing),
        State::Completing(timeout, completed) => update_completing(&mut fusing, timeout, completed),
    }

    let is_completing = matches!(fusing.state, State::Completing(_, true));
    let success = fusing.success;
    let swappable_slots = fusing.swappable_slots;
    let swappable_rarities = fusing.swappable_rarities;
    // Only actually taken on the completing tick - this function runs once per game tick, and
    // `fusing` gets reassigned back onto `player.state` on every *non*-completing tick too (see
    // the `else` branch below). Taking unconditionally cleared these mid-run, before the target
    // that just ranked up was even done processing (e.g. still going through
    // `State::DeselectRankedUp`/`State::NextTarget`), so by the time a later target's own
    // completion tick ran the take again, both had already been silently wiped one tick after
    // being pushed. Confirmed via live testing: `newly_epic_icons` was consistently empty by the
    // time swap tried to chain a targeted potential-reveal off it, despite a genuine Rare->Epic
    // rank-up having just happened this same run.
    let (ranked_up_icons, newly_epic_icons) = if is_completing {
        (
            std::mem::take(&mut fusing.ranked_up_icons),
            std::mem::take(&mut fusing.newly_epic_icons),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let force_swap_slots = fusing.force_swap_slots;
    let fusable_rarities = fusing.fusable_rarities;
    let fodder_exhausted_slots = fusing.fodder_exhausted_slots;
    let sorted_by_level = fusing.sorted_by_level;

    let next_state_display = fusing.to_string();
    let next = if is_completing {
        Player::FamiliarsSwapping(
            FamiliarsSwapping::new(
                swappable_slots,
                swappable_rarities,
                ranked_up_icons,
                force_swap_slots,
            )
            .with_fusable_rarities(fusable_rarities, fodder_exhausted_slots)
            // The menu stays open across this handoff (see this function's own docs), so the
            // collection list's `Level` sort persists too - if this run ever reached its own
            // State::SortByLevel, the list is left sorted descending, not the "no sort" a fresh
            // menu open would start from. See FamiliarsSwapping::sort_starts_descending's docs.
            .with_sort_starts_descending(sorted_by_level)
            .with_protected_icons(player.context.protected_familiar_icons.clone())
            .with_newly_epic_icons(newly_epic_icons),
        )
    } else {
        Player::FamiliarsFusing(fusing)
    };

    let action = next_action(&player.context);
    debug!(
        target: "backend/player",
        "familiars fusing tick end: next_fusing_state={next_state_display} has_priority_action={} priority_action_id={:?} next_action_is_some={}",
        player.context.has_priority_action(),
        player.context.priority_action_id(),
        action.is_some()
    );
    match action {
        Some(_) => {
            if is_completing {
                // Deliberately not calling `player.context.clear_action_completed()` here - this
                // handoff is into the chained Player::FamiliarsSwapping, not the end of the
                // overall action (that only happens once swap itself reaches Player::Idle, where
                // update_familiars_swapping_state already clears it correctly). Clearing it this
                // early freed up the priority action mid-swap, which let the rotator's own
                // regularly-scheduled swap check fire *while the chained swap was still running*,
                // forcing player.state back to Idle (per set_priority_action's docs) and
                // restarting the whole swap from scratch right on top of it - i.e. running the
                // swap logic twice for a single fusion completion.
                if success {
                    player.context.clear_familiars_fuse_fail_count();
                } else {
                    player.context.track_familiars_fuse_fail_count();
                }
            }

            player.state = next;
        }
        None => player.state = Player::Idle, // Force cancel if not from action
    }
}

/// Tick at which the `Familiar` buff is re-checked and, if still active, deactivated a second
/// time - see [`update_open_menu`]'s docs for why a single untracked check isn't trusted.
const OPEN_MENU_BUFF_RECHECK_TICK: u32 = 45;
/// Tick at which the menu-open key is pressed when a `Familiar` buff key is configured - kept 15
/// ticks after [`OPEN_MENU_BUFF_RECHECK_TICK`] so a second deactivation press has time to visibly
/// take before the menu opens on top of it. Irrelevant when no buff key is configured: there is
/// nothing to wait out in that case, so [`update_open_menu`] presses the key immediately instead.
const OPEN_MENU_KEY_TICK: u32 = 60;

/// Opens the familiar menu, first deactivating the `Familiar` buff skill if it is currently
/// active - live testing showed a game bug where clicking `Select All` in the Fusion tab silently
/// undoes the selection right after if this buff is active, so it must be off before the menu (and
/// eventually the Fusion tab) is even opened. `familiar_buff_key` is [`None`] when the character
/// has no buff key configured or it is disabled, in which case there is nothing to wait out or
/// press, and the menu-open key is sent right away instead of at [`OPEN_MENU_KEY_TICK`].
///
/// The buff is checked twice, not once: an initial check (and press, if active) happens the
/// instant this state starts, but that single check can race a slow-to-render buff icon or a
/// press that silently doesn't register - in either case fusion would otherwise proceed with the
/// buff still on and only find out indirectly later, via a silently-undone Select All. A second
/// check at [`OPEN_MENU_BUFF_RECHECK_TICK`], well before the menu-open key is pressed at
/// [`OPEN_MENU_KEY_TICK`], catches that case and presses the key again. This is deliberately not
/// a check-every-tick retry loop like some other states in this file use - the buff key is a
/// toggle, so pressing it while it is in fact already off (e.g. because the first press worked but
/// this function's own detection is what raced) would turn it back on, the opposite of the intent.
fn update_open_menu(
    resources: &mut Resources,
    fusing: &mut FamiliarsFusing,
    key: KeyKind,
    familiar_buff_key: Option<KeyKind>,
) {
    let State::OpenMenu(timeout) = fusing.state else {
        panic!("familiars fusing state is not opening menu");
    };

    match next_timeout_lifecycle(timeout, 120) {
        Lifecycle::Started(timeout) => {
            resources
                .input
                .send_mouse(fusing.mouse_rest.x, fusing.mouse_rest.y, MouseKind::Move);
            match familiar_buff_key {
                Some(buff_key) => {
                    if resources.detector().detect_player_buff(BuffKind::Familiar) {
                        debug!(target: "backend/player", "familiar buff active, deactivating before opening menu");
                        resources.input.send_key(buff_key);
                    }
                }
                None => resources.input.send_key(key),
            }
            fusing.state = State::OpenMenu(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_menu_opened() {
                fusing.state = State::FindEquippedSlots;
                return;
            }

            fusing.success = false;
            fusing.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => {
            if let Some(buff_key) = familiar_buff_key {
                if timeout.current == OPEN_MENU_BUFF_RECHECK_TICK
                    && resources.detector().detect_player_buff(BuffKind::Familiar)
                {
                    debug!(target: "backend/player", "familiar buff still active before opening menu, deactivating again");
                    resources.input.send_key(buff_key);
                }
                if timeout.current == OPEN_MENU_KEY_TICK {
                    resources.input.send_key(key);
                }
            }

            let menu_key_pressed =
                familiar_buff_key.is_none() || timeout.current > OPEN_MENU_KEY_TICK;
            if menu_key_pressed && resources.detector().detect_familiar_menu_opened() {
                fusing.state = State::FindEquippedSlots;
                return;
            }

            fusing.state = State::OpenMenu(timeout);
        }
    }
}

/// Clicks the Setup tab button, only reached when re-scanning equipped slots for the next target
/// while still on the Fusion tab (see [`State::SwitchToSetupTab`]'s docs) - Setup is therefore
/// always in its *inactive* state here, hence [`Detector::detect_familiar_setup_tab_inactive_button`]
/// rather than the active-state detector (its own docs explain why that distinction matters - the
/// active-state template was found to systematically miss here). A single click is trusted (not
/// retried every tick) since this is an ordinary tab button, not a hover-animated toggle like the
/// Inventory's Use tab - mirrors [`super::familiars_swap`]'s own `update_switch_to_setup_tab`.
fn update_switch_to_setup_tab(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::SwitchToSetupTab(timeout) = fusing.state else {
        panic!("familiars fusing state is not switching to setup tab");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            let bbox = cached_button_bbox(&mut fusing.buttons.setup_tab, || {
                resources
                    .detector()
                    .detect_familiar_setup_tab_inactive_button()
            });
            debug!(target: "backend/player", "setup tab button: {bbox:?}");
            if let Some(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::SwitchToSetupTab(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::FindEquippedSlots;
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::SwitchToSetupTab(timeout);
        }
    }
}

fn update_find_equipped_slots(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let slots = resources.detector().detect_familiar_slots();
    if slots.len() != FAMILIAR_SLOTS {
        debug!(target: "backend/player", "familiar slots is not 3 but was {}, aborting...", slots.len());
        fusing.success = false;
        fusing.state = State::Completing(Timeout::default(), false);
        return;
    }

    debug!(target: "backend/player", "familiar slots detected: {slots:?}");
    fusing.pending_slots = Array::new();
    for slot in slots {
        fusing.pending_slots.push(slot);
    }
    fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 0);
}

/// Tick within a single equipped-slot hover at which its level is checked - mirrors swap's
/// `FAMILIAR_CHECK_LVL_5_TICK`/`FAMILIAR_FREE_SLOTS_TIMEOUT`.
const SLOT_LEVEL_CHECK_TICK: u32 = 5;
const SLOT_LEVEL_CHECK_TIMEOUT: u32 = 10;

fn update_check_equipped_slot_level(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::CheckEquippedSlotLevel(timeout, index) = fusing.state else {
        panic!("familiars fusing state is not checking equipped slot level");
    };

    if index as usize >= fusing.pending_slots.len() {
        // Reaching here without having found a Rare target (see the Lifecycle::Ended arm below,
        // which leaves straight for State::OpenFusionTab as soon as one is found) means every slot
        // is either ineligible, already attempted this run, or (at best) an Epic target deferred by
        // that same arm while hoping for a Rare one - fall back to that deferred Epic target now
        // that the scan has confirmed no Rare target exists.
        if let Some(fallback) = fusing.fallback_target.take() {
            debug!(
                target: "backend/player",
                "finished checking equipped slots, no Rare target found - falling back to deferred Epic target at slot {}",
                fallback.slot_index
            );
            fusing.current_slot_index = fallback.slot_index;
            fusing.targets.push(fallback.bbox);
            fusing.target_rarities.push(fallback.rarity);
            fusing.icons.push(fallback.icon);
            fusing.names.push(fallback.name);
            fusing.target_index = 0;
            fusing.round = 0;
            fusing.state = State::OpenFusionTab(Timeout::default());
            return;
        }

        debug!(target: "backend/player", "finished checking equipped slots, no fusion target found");
        fusing.state = State::Completing(Timeout::default(), false);
        return;
    }

    let (bbox, is_free) = fusing.pending_slots[index as usize];
    let eligible_slot = match fusing.swappable_slots {
        SwappableFamiliars::All => true,
        SwappableFamiliars::Last => index as usize == FAMILIAR_SLOTS - 1,
        SwappableFamiliars::SecondAndLast => {
            index as usize == FAMILIAR_SLOTS - 1 || index as usize == FAMILIAR_SLOTS - 2
        }
    };
    let already_attempted = fusing.attempted[index as usize];
    if is_free || !eligible_slot || already_attempted {
        debug!(
            target: "backend/player",
            "skipping equipped slot {index} (free={is_free}, eligible_by_swappable_slots={eligible_slot}, already_attempted={already_attempted})"
        );
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), index + 1);
        return;
    }

    match next_timeout_lifecycle(timeout, SLOT_LEVEL_CHECK_TIMEOUT) {
        Lifecycle::Started(timeout) => {
            let x = bbox.x + bbox.width / 2;
            resources.input.send_mouse(x, bbox.y + 20, MouseKind::Move);
            fusing.state = State::CheckEquippedSlotLevel(timeout, index);
        }
        Lifecycle::Updated(timeout) => {
            if timeout.current == SLOT_LEVEL_CHECK_TICK {
                let level = resources.detector().detect_familiar_hover_level();
                debug!(
                    target: "backend/player",
                    "equipped slot {index} hover level: {level:?}"
                );

                // Not yet Level 5 (still gaining exp in the active slot) is not checked here -
                // only a confirmed Level 5 makes it a fusion target, mirroring swap's exact same
                // eligibility gate for freeing a slot.
                if matches!(level, Ok(FamiliarLevel::Level5)) {
                    let rank = resources.detector().detect_familiar_slot_rank(bbox);
                    debug!(target: "backend/player", "equipped slot {index} rank: {rank:?}");

                    if let Ok(rank) = rank {
                        let rarity = match rank {
                            FamiliarRank::Rare => FamiliarRarity::Rare,
                            FamiliarRank::Epic => FamiliarRarity::Epic,
                        };
                        if fusing.fusable_rarities.iter().any(|r| *r == rarity) {
                            let icon = resources
                                .detector()
                                .capture_region(familiar_portrait_region(bbox));
                            let name = resources
                                .detector()
                                .detect_familiar_tooltip_title_bar()
                                .and_then(|bbox| resources.detector().capture_region(bbox));
                            if let (Ok(icon), Ok(name)) = (icon, name) {
                                let is_protected = fusing.protected_icons.iter().any(|protected| {
                                    !resources
                                        .detector()
                                        .detect_region_template(
                                            protected,
                                            bbox,
                                            PROTECTED_FAMILIAR_ICON_MATCH_THRESHOLD,
                                        )
                                        .is_empty()
                                });
                                if is_protected {
                                    debug!(
                                        target: "backend/player",
                                        "equipped slot {index} matches a protected familiar icon, skipping"
                                    );
                                } else {
                                    fusing.targets.push(bbox);
                                    fusing.target_rarities.push(rarity);
                                    fusing.icons.push(Arc::new(icon));
                                    fusing.names.push(Arc::new(name));
                                }
                            } else {
                                debug!(
                                    target: "backend/player",
                                    "equipped slot {index} icon or name capture failed"
                                );
                            }
                        } else {
                            debug!(
                                target: "backend/player",
                                "equipped slot {index} rarity {rarity:?} not in fusable_rarities"
                            );
                        }
                    }
                }
            }
            fusing.state = State::CheckEquippedSlotLevel(timeout, index);
        }
        Lifecycle::Ended => {
            fusing.state = if fusing.targets.is_empty() {
                State::CheckEquippedSlotLevel(Timeout::default(), index + 1)
            } else {
                let rarity = fusing.target_rarities[0];
                let rare_configured = fusing
                    .fusable_rarities
                    .iter()
                    .any(|r| *r == FamiliarRarity::Rare);
                if rarity == FamiliarRarity::Epic && rare_configured {
                    // Rare is preferred over Epic (see this function's docs) and hasn't been ruled
                    // out yet - defer this Epic target and keep scanning the remaining slots for a
                    // Rare one instead of stopping here. Only the first Epic found is kept as the
                    // fallback; a later Epic is simply discarded since it wouldn't change the
                    // eventual choice (still "the first eligible Epic if no Rare turns up").
                    if fusing.fallback_target.is_none() {
                        debug!(
                            target: "backend/player",
                            "equipped slot {index} is an eligible Epic target, deferring in favor of a Rare target"
                        );
                        fusing.fallback_target = Some(FallbackTarget {
                            slot_index: index as usize,
                            bbox: fusing.targets[0],
                            rarity,
                            icon: fusing.icons[0].clone(),
                            name: fusing.names[0].clone(),
                        });
                    }
                    fusing.targets = Array::new();
                    fusing.target_rarities = Array::new();
                    fusing.icons = Vec::new();
                    fusing.names = Vec::new();
                    State::CheckEquippedSlotLevel(Timeout::default(), index + 1)
                } else {
                    // Either a Rare target (the top priority, nothing left to gain by scanning
                    // further) or an Epic target with no Rare rarity configured at all (already the
                    // best possible outcome) - stop scanning here rather than continuing to
                    // index + 1, so only one target is worked on before update_next_target
                    // re-scans from scratch (see its docs for why).
                    fusing.current_slot_index = index as usize;
                    fusing.target_index = 0;
                    fusing.round = 0;
                    fusing.fallback_target = None;
                    State::OpenFusionTab(Timeout::default())
                }
            };
        }
    }
}

/// Picks [`State::SortByLevel`] or skips straight to [`State::FindInCollectionList`], depending on
/// whether the collection list has already been sorted by level this run - see
/// [`FamiliarsFusing::sorted_by_level`]'s docs for why re-sorting for a later target is both
/// unnecessary and risky.
#[inline]
fn next_state_after_fusion_tab_opened(fusing: &FamiliarsFusing) -> State {
    if fusing.sorted_by_level {
        State::FindInCollectionList(Timeout::default(), 0, None, false)
    } else {
        State::SortByLevel(Timeout::default(), 0, None)
    }
}

fn update_open_fusion_tab(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::OpenFusionTab(timeout) = fusing.state else {
        panic!("familiars fusing state is not opening fusion tab");
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            // Always click the Fusion tab button rather than skipping when
            // detect_familiar_fusion_tab_opened() already reports true - that check shares the
            // same visual chrome across all 4 tabs (Setup/Info/Extraction/Fusion) and only needs
            // a marginal false positive (seen in practice: a 0.804 match against this check's
            // 0.75 threshold, while this same menu's Setup-tab-button detector needed a stricter
            // 0.95 threshold to reliably tell tabs apart - see its docs on why) to wrongly report
            // the Fusion tab as already open while still on Setup. That false "already open" then
            // skipped ever clicking into Fusion, so every later step (Select All, Fuse) operated
            // blind against the Setup tab instead - in one observed case, what the collection-list
            // logic believed was a duplicate card was actually an equipped Setup-tab slot, and
            // clicking it triggered the game's "slots already filled" popup, stalling the whole
            // run behind it. Clicking the tab button here is a harmless no-op when already on the
            // Fusion tab, so there is no downside to always doing it and leaving
            // Lifecycle::Ended as the sole judge of whether the tab is actually open.
            let tab_button = cached_button_bbox(&mut fusing.buttons.fusion_tab, || {
                resources.detector().detect_familiar_fusion_tab_button()
            });
            debug!(target: "backend/player", "fusion tab button: {tab_button:?}");
            if let Some(bbox) = tab_button {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::OpenFusionTab(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_fusion_tab_opened() {
                debug!(target: "backend/player", "fusion tab opened for target {}", fusing.current_slot_index);
                fusing.state = next_state_after_fusion_tab_opened(fusing);
                return;
            }

            debug!(target: "backend/player", "failed to detect fusion tab opened, aborting...");
            fusing.success = false;
            fusing.state = State::Completing(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::OpenFusionTab(timeout);
        }
    }
}

/// Clicks the `Level` sort header twice (ascending, then descending) so the highest-level
/// familiars surface at the top of the collection list without needing to scroll to find the
/// target - assumes the target familiar's own level ranks it near the top, since it's actively
/// equipped rather than a low-level unused duplicate.
///
/// The button's position is detected once and cached for the second click rather than
/// re-detected each time - live testing showed the button's icon changes to a sort-direction
/// arrow once clicked, dropping the template match score (e.g. 0.54) below the detection
/// threshold on the second attempt, which silently skipped the second click and left the list
/// only single-sorted (so the target's Level 5 duplicate could end up off the visible top of the
/// list instead of at it).
fn update_sort_by_level(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::SortByLevel(timeout, clicks, cached_bbox) = fusing.state else {
        panic!("familiars fusing state is not sorting by level");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let bbox = cached_bbox.ok_or(()).or_else(|_| {
                resources
                    .detector()
                    .detect_familiar_level_sort_button()
                    .map_err(|_| ())
            });
            debug!(target: "backend/player", "level sort button click {clicks}: {bbox:?}");
            if let Ok(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::SortByLevel(timeout, clicks, bbox.ok());
        }
        Lifecycle::Ended => {
            fusing.state = if clicks + 1 < 2 {
                State::SortByLevel(Timeout::default(), clicks + 1, cached_bbox)
            } else {
                debug!(target: "backend/player", "finished sorting by level, searching collection list");
                fusing.sorted_by_level = true;
                State::FindInCollectionList(Timeout::default(), 0, None, false)
            };
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::SortByLevel(timeout, clicks, cached_bbox);
        }
    }
}

/// Tick within a single candidate attempt at which its hover level is checked (mirrors swap's
/// `SWAPPING_DETECT_LEVEL_TICK`).
const CANDIDATE_LEVEL_CHECK_TICK: u32 = 5;

/// Total ticks per candidate attempt in [`update_find_in_collection_list`].
///
/// Generous relative to [`CANDIDATE_LEVEL_CHECK_TICK`] (~20 ticks after the click) so the
/// selected-preview panel's selection animation has time to finish before the icon-match
/// verification runs - live testing showed a genuinely correct click still failed verification
/// when checked only 5 ticks after clicking, most likely because the panel was still transitioning
/// in.
const CANDIDATE_TIMEOUT: u32 = 25;

/// Finds the `index`-th candidate card currently visible in the collection list, pre-filtered by
/// rarity so an already-Unique (or otherwise wrong-rarity) duplicate is skipped without ever
/// clicking it - mirrors swap's `update_find_cards`, which uses the same
/// [`Detector::detect_familiar_cards`] rank classification for the same reason.
fn find_candidate_at(
    resources: &mut Resources,
    fusing: &FamiliarsFusing,
    index: u32,
) -> Option<Rect> {
    let target_rarity = fusing.target_rarities[fusing.target_index];
    resources
        .detector()
        .detect_familiar_cards()
        .into_iter()
        .filter(|(_, rank)| {
            matches!(
                (rank, target_rarity),
                (FamiliarRank::Rare, FamiliarRarity::Rare)
                    | (FamiliarRank::Epic, FamiliarRarity::Epic)
            )
        })
        .nth(index as usize)
        .map(|(bbox, _)| bbox)
}

/// Same-species duplicates can exist at different levels and look visually identical (per the
/// captured icon, which cannot tell levels apart), while only the actually-equipped instance is
/// Level 5 - mirrors swap's `update_swapping` pattern of hovering a candidate, checking its level
/// via [`Detector::detect_familiar_hover_level`], and only clicking (here: selecting) it if it is
/// Level 5, rather than clicking blindly and risking selecting the wrong duplicate.
fn update_find_in_collection_list(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::FindInCollectionList(timeout, index, cached_bbox, clicked) = fusing.state else {
        panic!("familiars fusing state is not finding in collection list");
    };

    match next_timeout_lifecycle(timeout, CANDIDATE_TIMEOUT) {
        Lifecycle::Started(timeout) => match find_candidate_at(resources, fusing, index) {
            Some(bbox) => {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Move);
                fusing.state = State::FindInCollectionList(timeout, index, Some(bbox), false);
            }
            None => {
                debug!(
                    target: "backend/player",
                    "no candidates for target {}, giving up",
                    fusing.current_slot_index
                );
                fusing.state = State::NextTarget;
            }
        },
        Lifecycle::Updated(timeout) => {
            if timeout.current == CANDIDATE_LEVEL_CHECK_TICK && !clicked {
                let level = resources.detector().detect_familiar_hover_level();
                debug!(target: "backend/player", "collection list candidate {index} hover level: {level:?}");
                let is_level_5 = matches!(level, Ok(FamiliarLevel::Level5));
                if is_level_5 {
                    // Re-using the bbox cached from when hovering started, rather than a fresh
                    // find_candidate_at call - live testing showed a fresh detection here can
                    // resolve to a different, adjacent card than the one just hover-checked (the
                    // hover itself can shift detection results slightly), silently clicking the
                    // wrong duplicate right after confirming the correct one was Level 5.
                    if let Some(bbox) = cached_bbox {
                        let (x, y) = bbox_click_point(bbox);
                        resources.input.send_mouse(x, y, MouseKind::Click);
                        fusing.state =
                            State::FindInCollectionList(timeout, index, cached_bbox, true);
                        return;
                    }
                }

                // Not Level 5 (or detection failed); nothing was clicked, so there is nothing to
                // wait out the rest of CANDIDATE_TIMEOUT for (that remaining time exists solely
                // to let a *click's* selection animation settle before verifying it - see
                // CANDIDATE_TIMEOUT's docs). Move directly to the next candidate instead of
                // idling until Lifecycle::Ended.
                fusing.state =
                    State::FindInCollectionList(Timeout::default(), index + 1, None, false);
                return;
            }
            fusing.state = State::FindInCollectionList(timeout, index, cached_bbox, clicked);
        }
        Lifecycle::Ended => {
            // Only reached once a candidate was actually clicked - the not-clicked case now
            // exits early from Lifecycle::Updated above, straight to the next candidate.
            debug_assert!(clicked);

            // Verify the click actually selected the target familiar by matching the resulting
            // selected-preview card's icon against the one captured from the equipped Setup-tab
            // slot.
            //
            // Icon match alone isn't enough to confirm it's the *right* candidate: since
            // find_candidate_at no longer pre-filters by rarity (see its docs), a same-species
            // duplicate already ranked up to a different tier (e.g. already Unique from an
            // earlier run) renders an identical portrait and would pass the icon match too. The
            // selected card's *current* rarity - read via the Fusion Gauge's diamond icon, the
            // same detector [`update_confirm_rank_up`] uses to confirm a rank-up took - must also
            // match the rarity being searched for.
            //
            // Note: reconstructing a Setup-tab-style slot bbox from the icon match's bbox and
            // calling `detect_familiar_slot_rank` on it (an earlier approach) does not work here -
            // live testing showed it always returns `Err` against the Fusion tab's
            // selected-preview card, which does not render the same colored-border style
            // `detect_familiar_slot_rank` is tuned for.
            let region = full_frame_region(resources);
            let icon = fusing.icons[fusing.target_index].clone();
            let icon_bbox = resources
                .detector()
                .detect_region_template(&icon, region, FUSION_SELECTED_PREVIEW_MATCH_THRESHOLD)
                .into_iter()
                .next();
            let target_rarity = fusing.target_rarities[fusing.target_index];
            let gauge_rarity = resources.detector().detect_familiar_fusion_gauge_rarity();
            let matched = icon_bbox.is_some()
                && matches!(
                    (target_rarity, &gauge_rarity),
                    (FamiliarRarity::Rare, Ok(FusionGaugeRarity::Rare))
                        | (FamiliarRarity::Epic, Ok(FusionGaugeRarity::Epic))
                );
            debug!(
                target: "backend/player",
                "collection list candidate {index} clicked, icon_bbox: {icon_bbox:?}, gauge_rarity: {gauge_rarity:?}, verified as target: {matched}"
            );

            fusing.state = if matched {
                // A familiar can already have a filled gauge from fodder fed outside this run -
                // check for Rank Up being immediately available before defaulting to Select All,
                // so an already-ready familiar gets ranked up directly instead of pointlessly
                // selecting more fodder first.
                let button = resources.detector().detect_familiar_fusion_button();
                debug!(target: "backend/player", "fusion button right after selecting target {}: {button:?}", fusing.current_slot_index);
                if matches!(button, Ok((_, FusionButton::RankUp))) {
                    State::RankingUp(Timeout::default())
                } else if resources.detector().detect_familiar_card_boxes().is_empty() {
                    // No in-collection duplicates to select at all - clicking Select All would
                    // be a pointless no-op that only delays reaching the Inventory, which is the
                    // only place fodder can come from in this case.
                    //
                    // Deliberately checking card *shapes* here rather than
                    // detect_familiar_cards's Rare/Epic-classified cards - live testing showed
                    // fusion fodder is typically Common rarity (fed in bulk to fuse a single
                    // Rare/Epic target), which detect_familiar_cards can never classify since it
                    // only has Rare/Epic reference templates. Checking that instead always read
                    // "no duplicates" and skipped straight to the Inventory even with a full grid
                    // of valid fodder still visible.
                    debug!(
                        target: "backend/player",
                        "no in-collection duplicates for target {}, going straight to inventory",
                        fusing.current_slot_index
                    );
                    State::OpenInventory(Timeout::default())
                } else {
                    State::SelectAllFodder(Timeout::default())
                }
            } else {
                State::CancelCandidate(Timeout::default(), index + 1)
            };
        }
    }
}

fn update_cancel_candidate(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::CancelCandidate(timeout, next_index) = fusing.state else {
        panic!("familiars fusing state is not cancelling candidate");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let bbox = cached_button_bbox(&mut fusing.buttons.cancel, || {
                resources.detector().detect_familiar_fusion_cancel_button()
            });
            debug!(target: "backend/player", "fusion cancel button: {bbox:?}");
            if let Some(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::CancelCandidate(timeout, next_index);
        }
        Lifecycle::Ended => {
            // Verify the cancel actually cleared the selection - live testing showed the same
            // (wrong, e.g. already-Unique) candidate kept getting "re-verified" across several
            // different candidate indices in a row, suggesting Cancel might not always take
            // effect and the game silently ignores clicking a different card while one is still
            // selected.
            let region = full_frame_region(resources);
            let icon = fusing.icons[fusing.target_index].clone();
            let still_selected = resources
                .detector()
                .detect_region_template(&icon, region, FUSION_SELECTED_PREVIEW_MATCH_THRESHOLD)
                .into_iter()
                .next();
            debug!(
                target: "backend/player",
                "after cancel, previous selection still visible: {still_selected:?}"
            );
            fusing.state = State::FindInCollectionList(Timeout::default(), next_index, None, false);
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::CancelCandidate(timeout, next_index);
        }
    }
}

fn update_select_all_fodder(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::SelectAllFodder(timeout) = fusing.state else {
        panic!("familiars fusing state is not selecting all fodder");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let bbox = cached_button_bbox(&mut fusing.buttons.select_all, || {
                resources
                    .detector()
                    .detect_familiar_fusion_select_all_button()
            });
            debug!(target: "backend/player", "fusion select all button: {bbox:?}");
            if let Some(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::SelectAllFodder(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::Fusing(Timeout::default());
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::SelectAllFodder(timeout);
        }
    }
}

fn update_fusing(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::Fusing(timeout) = fusing.state else {
        panic!("familiars fusing state is not fusing");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Ended => {
            let button = resources.detector().detect_familiar_fusion_button();
            debug!(
                target: "backend/player",
                "fusion button for target {} round {}: {button:?}",
                fusing.current_slot_index, fusing.round
            );
            match button {
                Ok((_, FusionButton::RankUp)) => {
                    fusing.state = State::RankingUp(Timeout::default());
                }
                Ok((bbox, FusionButton::Fuse)) => {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    fusing.round += 1;

                    let round_cap_hit = fusing.round >= MAX_FUSE_ROUNDS;
                    if round_cap_hit {
                        debug!(
                            target: "backend/player",
                            "target {} hit MAX_FUSE_ROUNDS while fusing, topping up from inventory",
                            fusing.current_slot_index
                        );
                    }
                    fusing.state = State::ConfirmFuse(Timeout::default(), round_cap_hit, false);
                }
                Ok((_, FusionButton::FuseDisabled)) | Err(_) => {
                    // Nothing selected, meaning in-collection fodder is exhausted (or all
                    // remaining duplicates are locked and cannot be consumed); top up from bag.
                    // The disabled button still renders (greyed out) rather than disappearing, so
                    // it must be excluded explicitly instead of falling into the Err(_) case.
                    fusing.state = State::OpenInventory(Timeout::default());
                }
            }
        }
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            fusing.state = State::Fusing(timeout);
        }
    }
}

/// Confirms the "Do you want to fuse this Familiar?" popup that follows clicking `Fuse` - live
/// testing showed this is a separate confirmation step, not an immediate action.
///
/// The popup does not always render in time for a single check right after entering this state -
/// live testing showed `detect_popup_confirm_button` returning `Err` on the vast majority of
/// attempts (only succeeding when the popup happened to already be up), meaning the click was
/// silently skipped and the fuse was left unconfirmed while the state machine moved on regardless.
/// Detection and clicking is now retried on every tick up to [`Lifecycle::Ended`] instead of once
/// on [`Lifecycle::Started`], so it catches the popup whenever it actually appears within the
/// timeout window - but only until the first successful click, tracked via the `clicked` bool.
/// Once the popup closes, its `Confirm` button's screen position sits directly over the bottom of
/// the selected familiar's portrait - live testing showed re-detecting and clicking there again on
/// a later tick landed on that portrait instead (the button template apparently still matches
/// something close enough there), registering as a double-click that deselected the familiar
/// entirely. Stopping after the first successful click avoids that stray second click.
fn update_confirm_fuse(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::ConfirmFuse(timeout, round_cap_hit, clicked) = fusing.state else {
        panic!("familiars fusing state is not confirming fuse");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            let mut clicked = clicked;
            if !clicked {
                let bbox = resources.detector().detect_popup_confirm_button();
                debug!(target: "backend/player", "fuse confirm button: {bbox:?}");
                if let Ok(bbox) = bbox {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    clicked = true;
                }
            }
            fusing.state = State::ConfirmFuse(timeout, round_cap_hit, clicked);
        }
        Lifecycle::Ended => {
            // Some rank-up transitions (confirmed via live testing: Rare -> Epic) never show
            // a distinct `Rank Up` button at all - the very last `Fuse` click that completes
            // the gauge silently performs the rank-up itself, and its confirmation popup gets
            // clicked through by this same generic ConfirmFuse handling, without ever passing
            // through State::RankingUp/ConfirmRankUp. Left unchecked, the target then looks
            // like a fresh, un-ranked-up familiar of its *new* rarity and just keeps getting
            // fed - it did rank up, but nothing noticed, so it never stopped.
            //
            // This check must run even when `round_cap_hit` - hitting MAX_FUSE_ROUNDS can land
            // on the exact Fuse click that completes the gauge, and blindly giving up in that
            // case (the previous behavior) abandoned a familiar that had in fact just ranked
            // up, without ever deselecting it or recording it in `ranked_up_icons` - leaving
            // the chained swap unable to recognize and free it afterwards.
            let target_rarity = fusing.target_rarities[fusing.target_index];
            let gauge_rarity = resources.detector().detect_familiar_fusion_gauge_rarity();
            let already_ranked_up = matches!(
                (target_rarity, &gauge_rarity),
                (FamiliarRarity::Rare, Ok(FusionGaugeRarity::Epic))
                    | (FamiliarRarity::Epic, Ok(FusionGaugeRarity::Unique))
            );
            debug!(
                target: "backend/player",
                "target {} gauge rarity after fuse: {gauge_rarity:?}, already ranked up: {already_ranked_up}, round_cap_hit: {round_cap_hit}",
                fusing.current_slot_index
            );
            fusing.state = if already_ranked_up {
                let icon = fusing.icons[fusing.target_index].clone();
                if target_rarity == FamiliarRarity::Rare {
                    fusing.newly_epic_icons.push(icon.clone());
                }
                fusing.ranked_up_icons.push(icon);
                fusing.force_swap_slots[fusing.current_slot_index] = true;
                State::DeselectRankedUp(Timeout::default(), 0)
            } else if round_cap_hit {
                // Safety valve against a stuck loop; move on to the next target.
                State::NextTarget
            } else {
                // The just-confirmed Fuse can by itself have completed the gauge, making Rank
                // Up immediately available - go back through Fusing to re-check the button
                // rather than assuming more fodder is needed. Fusing's own Err branch (nothing
                // left selected, since Select All already grabbed every in-collection
                // duplicate before this round) is what actually falls through to
                // State::OpenInventory - not this state blind-jumping there and skipping the
                // Rank Up check entirely.
                State::Fusing(Timeout::default())
            };
        }
    }
}

fn update_open_inventory(
    resources: &mut Resources,
    fusing: &mut FamiliarsFusing,
    key: Option<KeyKind>,
) {
    let State::OpenInventory(timeout) = fusing.state else {
        panic!("familiars fusing state is not opening inventory");
    };

    let Some(key) = key else {
        debug!(target: "backend/player", "no inventory key configured, giving up on target {}", fusing.current_slot_index);
        // No inventory hotkey configured; cannot top up fodder from the bag.
        fusing.state = State::NextTarget;
        return;
    };

    match next_timeout_lifecycle(timeout, 40) {
        Lifecycle::Started(timeout) => {
            fusing.state = State::OpenInventory(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_inventory_opened() {
                debug!(target: "backend/player", "inventory opened");
                fusing.state = State::FocusUseTab(Timeout::default(), false);
                return;
            }

            debug!(target: "backend/player", "inventory failed to open, giving up on target {}", fusing.current_slot_index);
            // Inventory failed to open; give up on this target rather than spin forever.
            fusing.state = State::NextTarget;
        }
        Lifecycle::Updated(timeout) => {
            if timeout.current == 20 {
                resources.input.send_key(key);
            }
            fusing.state = State::OpenInventory(timeout);
        }
    }
}

/// Clicks the `Use` tab if it isn't already focused - the Inventory can open on whichever tab was
/// last active, not always `Use`, and its previous single blind-click attempt only ever matched
/// (and clicked) the already-focused appearance, silently doing nothing when a different tab was
/// active.
///
/// Checks whether it's focused yet on every tick up to [`Lifecycle::Ended`] instead of only once
/// on [`Lifecycle::Ended`] - live testing showed a single click landing before the tab-switch
/// animation/detection caught up was enough to make a sole final check falsely read "still not
/// focused" and give up on the whole target, closing the Inventory right after what was actually
/// a successful click. But the click itself still only happens once (tracked via the `clicked`
/// bool): a follow-up test showed repeatedly re-clicking on every tick instead made things worse -
/// the inactive-tab detection score got stuck in a middling, neither-active-nor-inactive range,
/// apparently because each new click kept interrupting the tab's own switch animation before it
/// could finish. Checking repeatedly is safe (read-only); clicking repeatedly is not.
///
/// The timeout window itself was also too short: offline-testing the real active-tab template
/// against a screenshot taken right after a still-failing live attempt scored ~1.0 (comfortably
/// above the 0.9 threshold) - the click and detection were both fine, the switch animation
/// genuinely just hadn't finished rendering by the time the sole 15-tick check fired. Widened to
/// 30, then later 60, ticks to give it more real time to settle before giving up - live testing
/// with the Fusion tab's own detection load already running in the same tick loop showed 30 ticks
/// alone still wasn't always enough (giving up entirely on an otherwise-correctly-selected fusion
/// target instead of ranking it up).
///
/// The click also moves the mouse straight back to [`FamiliarsFusing::mouse_rest`] afterward -
/// live testing showed the cursor left sitting on the tab it just clicked (`bbox_click_point`
/// lands dead-center of it) can itself occlude/alter the exact pixels
/// [`Detector::detect_inventory_use_tab_button`] scans on the very next tick, making an otherwise
/// successful click read as "still not focused" for the rest of the timeout and give up on the
/// whole target. Every other click site in this file clicks a button whose own appearance isn't
/// what gets checked afterward (a different indicator elsewhere confirms the effect instead), so
/// this self-referential click-then-check-the-same-spot shape is the one place that needed it.
fn update_focus_use_tab(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::FocusUseTab(timeout, clicked) = fusing.state else {
        panic!("familiars fusing state is not focusing use tab");
    };

    match next_timeout_lifecycle(timeout, 60) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            if resources
                .detector()
                .detect_inventory_use_tab_button()
                .is_ok()
            {
                debug!(target: "backend/player", "use tab focused");
                fusing.state = State::FindFodderItem(Timeout::default(), 0);
                return;
            }

            let mut clicked = clicked;
            if !clicked {
                let inactive_bbox = resources
                    .detector()
                    .detect_inventory_use_tab_inactive_button();
                debug!(target: "backend/player", "use tab inactive button: {inactive_bbox:?}");
                if let Ok(bbox) = inactive_bbox {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    resources.input.send_mouse(
                        fusing.mouse_rest.x,
                        fusing.mouse_rest.y,
                        MouseKind::Move,
                    );
                    clicked = true;
                }
            }
            fusing.state = State::FocusUseTab(timeout, clicked);
        }
        Lifecycle::Ended => {
            let focused = resources
                .detector()
                .detect_inventory_use_tab_button()
                .is_ok();
            debug!(target: "backend/player", "use tab focused after click attempt: {focused}");
            fusing.state = if focused {
                State::FindFodderItem(Timeout::default(), 0)
            } else {
                // Could not confirm the Use tab is focused after retrying every tick; give up
                // rather than scan the wrong tab.
                State::CloseInventory(Timeout::default(), false)
            };
        }
    }
}

/// Hovers Inventory `Use`-tab grid slots one at a time (top-left to bottom-right, capped at
/// [`INVENTORY_SCAN_MAX_SLOTS`] - by convention, fodder is expected to be kept in the first few
/// slots) and checks each one's hover-tooltip preview against the target's captured icon.
///
/// The grid's tiny item icons cannot be matched directly - verified against real screenshots,
/// even an exhaustively-searched best-case crop only scored ~0.49 correlation against the
/// captured icon, because the game renders the same familiar with a different art
/// composition/crop in the grid than in the equipped-slot card. The hover tooltip, however,
/// renders the *same* card art as the equipped slot and matches reliably (~0.95), which is why
/// this hovers each candidate rather than matching the grid icons themselves.
fn update_find_fodder_item(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::FindFodderItem(timeout, index) = fusing.state else {
        panic!("familiars fusing state is not finding fodder item");
    };

    if index >= INVENTORY_SCAN_MAX_SLOTS {
        debug!(
            target: "backend/player",
            "exhausted {INVENTORY_SCAN_MAX_SLOTS} inventory slots without finding fodder for target {}",
            fusing.current_slot_index
        );
        // Exhausted the conventional fodder slots; give up topping up this target. This is the
        // one place fusion actually confirms there is no more fodder to be had (as opposed to
        // e.g. hitting MAX_FUSE_ROUNDS or a detection hiccup, where fodder may well still be
        // available) - recorded so the chained swap knows it's safe to replace this familiar
        // instead of protecting it indefinitely. See `fodder_exhausted_slots`' docs.
        fusing.fodder_exhausted_slots[fusing.current_slot_index] = true;
        fusing.state = State::CloseInventory(Timeout::default(), false);
        return;
    }

    let Ok(use_tab_bbox) = resources.detector().detect_inventory_use_tab_button() else {
        debug!(
            target: "backend/player",
            "use tab button not found while scanning inventory, giving up on target {}",
            fusing.current_slot_index
        );
        fusing.state = State::CloseInventory(Timeout::default(), false);
        return;
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            let (x, y) = bbox_click_point(inventory_slot_rect(use_tab_bbox, index));
            resources.input.send_mouse(x, y, MouseKind::Move);
            fusing.state = State::FindFodderItem(timeout, index);
        }
        Lifecycle::Updated(timeout) => {
            if timeout.current == FODDER_ITEM_EARLY_CHECK_TICK
                && fodder_item_matches(resources, fusing)
            {
                debug!(
                    target: "backend/player",
                    "inventory slot {index} matches target fodder early (tick {})",
                    timeout.current
                );
                fusing.state = State::DoubleClickItem(
                    Timeout::default(),
                    inventory_slot_rect(use_tab_bbox, index),
                );
                return;
            }
            fusing.state = State::FindFodderItem(timeout, index);
        }
        Lifecycle::Ended => {
            let matched = fodder_item_matches(resources, fusing);
            debug!(target: "backend/player", "inventory slot {index} matches target fodder: {matched}");

            fusing.state = if matched {
                State::DoubleClickItem(Timeout::default(), inventory_slot_rect(use_tab_bbox, index))
            } else {
                State::FindFodderItem(Timeout::default(), index + 1)
            };
        }
    }
}

/// Checks whether the currently-hovered Inventory slot's tooltip preview matches the current
/// target's captured name - shared by [`update_find_fodder_item`]'s early check
/// ([`FODDER_ITEM_EARLY_CHECK_TICK`]) and its full-timeout fallback check, so a fast-rendering
/// tooltip does not have to wait out the rest of the slot's timeout before the state machine
/// moves on.
///
/// Matches on the title-bar name crop rather than the portrait icon - see
/// [`FamiliarsFusing::icons`]'s docs for why the portrait isn't reliable here.
fn fodder_item_matches(resources: &mut Resources, fusing: &FamiliarsFusing) -> bool {
    let region = full_frame_region(resources);
    let name = fusing.names[fusing.target_index].clone();
    resources
        .detector()
        .detect_region_template(&name, region, INVENTORY_FODDER_MATCH_THRESHOLD)
        .into_iter()
        .next()
        .is_some()
}

fn update_double_click_item(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::DoubleClickItem(timeout, bbox) = fusing.state else {
        panic!("familiars fusing state is not double clicking item");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let (x, y) = bbox_click_point(bbox);
            resources.input.send_mouse(x, y, MouseKind::Click);
            resources.input.send_mouse(x, y, MouseKind::Click);
            fusing.state = State::DoubleClickItem(timeout, bbox);
        }
        Lifecycle::Ended => {
            let dialog_opened = resources
                .detector()
                .detect_inventory_use_quantity_dialog()
                .is_ok();
            debug!(target: "backend/player", "quantity dialog opened after double-click: {dialog_opened}");
            fusing.state = if dialog_opened {
                State::TypeQuantity(Timeout::default())
            } else {
                // Dialog did not appear; give up topping up this target.
                State::CloseInventory(Timeout::default(), false)
            };
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::DoubleClickItem(timeout, bbox);
        }
    }
}

fn update_type_quantity(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::TypeQuantity(timeout) = fusing.state else {
        panic!("familiars fusing state is not typing quantity");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            // The field's pre-filled owned count starts already fully selected (confirmed via
            // screenshots showing it highlighted the moment the dialog opens), so typing directly
            // overwrites it - live testing showed clicking the field first deselected that
            // highlight, after which the Backspace presses no longer reliably cleared it.
            let dialog = resources.detector().detect_inventory_use_quantity_dialog();
            debug!(target: "backend/player", "quantity dialog title bbox: {dialog:?}");
            if dialog.is_ok() {
                for key in INVENTORY_USE_QUANTITY_KEYS {
                    resources.input.send_key(key);
                }
            }
            fusing.state = State::TypeQuantity(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::ConfirmQuantity(Timeout::default());
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::TypeQuantity(timeout);
        }
    }
}

fn update_confirm_quantity(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::ConfirmQuantity(timeout) = fusing.state else {
        panic!("familiars fusing state is not confirming quantity");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            let bbox = resources.detector().detect_popup_confirm_button();
            debug!(target: "backend/player", "quantity confirm button: {bbox:?}");
            if let Ok(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::ConfirmQuantity(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::ConfirmAddFamiliar(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::ConfirmQuantity(timeout);
        }
    }
}

/// Confirms the "Add this familiar to [Character]'s Familiar List?" popup that follows
/// confirming the quantity - live testing showed this is what actually converts the used bag
/// items into new in-collection duplicates, which are what the Fusion tab's `Select All` then
/// picks up as fodder.
///
/// Only clicks once (tracked via the `clicked` bool) rather than on every tick - see
/// [`update_confirm_fuse`]'s docs for why re-clicking after the popup has already closed is
/// unsafe (it can land on and deselect whatever is now underneath the button's old position).
fn update_confirm_add_familiar(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::ConfirmAddFamiliar(timeout, clicked) = fusing.state else {
        panic!("familiars fusing state is not confirming add familiar");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            let mut clicked = clicked;
            if !clicked {
                let bbox = resources.detector().detect_popup_confirm_button();
                debug!(target: "backend/player", "add familiar confirm button: {bbox:?}");
                if let Ok(bbox) = bbox {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    clicked = true;
                }
            }
            fusing.state = State::ConfirmAddFamiliar(timeout, clicked);
        }
        Lifecycle::Ended => {
            fusing.state = State::CloseInventory(Timeout::default(), true);
        }
    }
}

fn update_close_inventory(
    resources: &mut Resources,
    fusing: &mut FamiliarsFusing,
    key: Option<KeyKind>,
) {
    let State::CloseInventory(timeout, resume) = fusing.state else {
        panic!("familiars fusing state is not closing inventory");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            if resources.detector().detect_inventory_opened() {
                resources.input.send_key(key.unwrap_or(KeyKind::Esc));
            }
            fusing.state = State::CloseInventory(timeout, resume);
        }
        Lifecycle::Ended => {
            debug!(target: "backend/player", "inventory closed, resuming fuse loop: {resume}");
            fusing.state = if resume {
                // Fresh fodder was just added, so this is effectively starting a new fill cycle -
                // reset the round budget rather than carrying over rounds already spent before
                // topping up (otherwise MAX_FUSE_ROUNDS could be hit again almost immediately).
                fusing.round = 0;
                State::SelectAllFodder(Timeout::default())
            } else {
                State::NextTarget
            };
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::CloseInventory(timeout, resume);
        }
    }
}

/// Clicks the `Rank Up` button. Live testing showed this opens a "Do you want to attempt to
/// improve the Familiar's rank?" confirmation popup rather than acting immediately, handled by
/// [`update_confirm_rank_up`].
fn update_ranking_up(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::RankingUp(timeout) = fusing.state else {
        panic!("familiars fusing state is not ranking up");
    };

    match next_timeout_lifecycle(timeout, 15) {
        Lifecycle::Started(timeout) => {
            if let Ok((bbox, FusionButton::RankUp)) =
                resources.detector().detect_familiar_fusion_button()
            {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::RankingUp(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::ConfirmRankUp(Timeout::default(), false);
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::RankingUp(timeout);
        }
    }
}

/// "Rank Up Guaranteed" is not actually a 100% guarantee, so a click can fail to consume the
/// gauge. Success is confirmed by the Fusion Gauge's diamond icon switching color to the next
/// rarity tier (blue/Rare -> purple/Epic -> orange/Unique) - verified against real screenshots of
/// all 3 tiers. A single-template absolute threshold is not discriminative enough here (the
/// diamonds share the same shape/outline and mostly differ in color, so e.g. the Epic template
/// still scores ~0.9 against a Rare diamond), so [`Detector::detect_familiar_fusion_gauge_rarity`]
/// itself picks the best-scoring of the 3 candidates rather than thresholding one in isolation.
/// If the reported rarity still matches the tier fusion started from, the click didn't take, and
/// the gauge drops back by whatever the attempt consumed rather than emptying outright - fodder
/// already pulled into the grid beyond what that attempt used (e.g. gauge at 125/150, 50 fodder
/// selected, Fuse consumes 25 to fill it, the other 25 stays selected as leftover) exists to
/// recover. But unlike an ordinary `Fuse` click - which per [`update_confirm_fuse`]'s non-cap-hit
/// path only consumes what's needed for the gauge and leaves the rest selected, so its own retry
/// can go straight back to [`State::Fusing`] and find that leftover still selected - live testing
/// (see log evidence: gauge maxed after a single `Fuse` click leaving an expected ~25 leftover,
/// yet the very next button check after a failed Rank Up read `FuseDisabled`, i.e. nothing
/// selected at all) showed clicking `Rank Up` clears the *entire* current selection regardless of
/// whether the attempt succeeds, leftover included. That leftover fodder is not gone, though -
/// it's still sitting in the collection list, just no longer marked selected. So a failed attempt
/// goes back through [`State::SelectAllFodder`] (re-selecting whatever's still there, a no-op if
/// nothing is) rather than [`State::Fusing`] directly, which would only see the now-empty
/// selection and jump straight to [`State::OpenInventory`], pulling in more from the bag than
/// necessary while that leftover sits unused.
///
/// Only clicks once (tracked via the `clicked` bool) rather than on every tick - see
/// [`update_confirm_fuse`]'s docs for why re-clicking after the popup has already closed is
/// unsafe (it can land on and deselect whatever is now underneath the button's old position).
fn update_confirm_rank_up(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::ConfirmRankUp(timeout, clicked) = fusing.state else {
        panic!("familiars fusing state is not confirming rank up");
    };

    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            let mut clicked = clicked;
            if !clicked {
                let bbox = resources.detector().detect_popup_confirm_button();
                debug!(target: "backend/player", "rank up confirm button: {bbox:?}");
                if let Ok(bbox) = bbox {
                    let (x, y) = bbox_click_point(bbox);
                    resources.input.send_mouse(x, y, MouseKind::Click);
                    clicked = true;
                }
            }
            fusing.state = State::ConfirmRankUp(timeout, clicked);
        }
        Lifecycle::Ended => {
            let original_rarity = fusing.target_rarities[fusing.target_index];
            let new_rarity = resources.detector().detect_familiar_fusion_gauge_rarity();
            let ranked_up = matches!(
                (original_rarity, &new_rarity),
                (FamiliarRarity::Rare, Ok(FusionGaugeRarity::Epic))
                    | (FamiliarRarity::Epic, Ok(FusionGaugeRarity::Unique))
            );
            debug!(
                target: "backend/player",
                "target {} rank up check: original={original_rarity:?} gauge={new_rarity:?} ranked_up={ranked_up}",
                fusing.current_slot_index
            );

            fusing.state = if ranked_up {
                let icon = fusing.icons[fusing.target_index].clone();
                if original_rarity == FamiliarRarity::Rare {
                    fusing.newly_epic_icons.push(icon.clone());
                }
                fusing.ranked_up_icons.push(icon);
                fusing.force_swap_slots[fusing.current_slot_index] = true;
                State::DeselectRankedUp(Timeout::default(), 0)
            } else {
                fusing.round += 1;
                if fusing.round >= MAX_FUSE_ROUNDS {
                    debug!(
                        target: "backend/player",
                        "target {} hit MAX_FUSE_ROUNDS while ranking up, giving up",
                        fusing.current_slot_index
                    );
                    // Safety valve against a stuck loop; move on to the next target.
                    State::NextTarget
                } else {
                    // Rank Up clears the whole selection on click, leftover included (see this
                    // function's docs) - re-select whatever's still in the collection list rather
                    // than assuming the bag is the only source left.
                    State::SelectAllFodder(Timeout::default())
                }
            };
        }
    }
}

/// Number of times [`update_deselect_ranked_up`] retries the `Cancel` click before giving up and
/// moving on to the next target anyway.
const MAX_DESELECT_RANKED_UP_ATTEMPTS: u32 = 3;

/// Clicks `Cancel` to deselect the just-ranked-up familiar - it changed rarity but is still
/// selected as the base card, and per [`update_cancel_candidate`]'s docs the game silently
/// ignores clicking a different candidate while one is already selected. Without this, the next
/// target's search would never find (or select) anything and give no indication why.
///
/// Unlike `CancelCandidate`, which naturally retries via its own "try the next candidate" loop
/// even if a single Cancel click doesn't take, this state has nowhere else to fall into - live
/// testing showed exactly that failure mode: the very next target's collection-list search found
/// zero candidates immediately (the panel was still showing the just-ranked-up familiar's own
/// duplicates, not the general list, because it was in fact still selected). So this verifies the
/// click actually cleared the selection (same icon-match check `CancelCandidate` already uses) and
/// retries clicking `Cancel` again if it didn't, up to [`MAX_DESELECT_RANKED_UP_ATTEMPTS`].
fn update_deselect_ranked_up(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::DeselectRankedUp(timeout, attempt) = fusing.state else {
        panic!("familiars fusing state is not deselecting ranked up familiar");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let bbox = cached_button_bbox(&mut fusing.buttons.cancel, || {
                resources.detector().detect_familiar_fusion_cancel_button()
            });
            debug!(target: "backend/player", "deselect ranked up familiar, cancel button: {bbox:?}");
            if let Some(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::DeselectRankedUp(timeout, attempt);
        }
        Lifecycle::Ended => {
            let region = full_frame_region(resources);
            let icon = fusing.icons[fusing.target_index].clone();
            let still_selected = resources
                .detector()
                .detect_region_template(&icon, region, FUSION_SELECTED_PREVIEW_MATCH_THRESHOLD)
                .into_iter()
                .next();
            debug!(
                target: "backend/player",
                "after deselecting ranked-up familiar (attempt {attempt}), still selected: {still_selected:?}"
            );
            fusing.state =
                if still_selected.is_some() && attempt + 1 < MAX_DESELECT_RANKED_UP_ATTEMPTS {
                    State::DeselectRankedUp(Timeout::default(), attempt + 1)
                } else {
                    State::NextTarget
                };
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::DeselectRankedUp(timeout, attempt);
        }
    }
}

/// Marks the just-processed slot done (ranked up or given up on) and re-scans equipped slots from
/// index 0 for the next target, instead of walking a precomputed list.
///
/// Ranking up a familiar resets its level to 1, which can move it out of (or otherwise reshuffle)
/// the Fusion tab's level-sorted collection list - a target list built once up front, before any
/// fusion happened, cannot account for that. Re-scanning from scratch after each target instead
/// re-derives eligibility (Level 5 + fusable rarity) against the current live state every time, and
/// [`FamiliarsFusing::attempted`] keeps it from re-attempting a slot already finished this run.
///
/// Goes through [`State::SwitchToSetupTab`] rather than straight to
/// [`State::CheckEquippedSlotLevel`] - see that state's docs for why hovering the equipped slots
/// only makes sense once the Setup tab is actually showing again.
fn update_next_target(fusing: &mut FamiliarsFusing) {
    fusing.attempted[fusing.current_slot_index] = true;
    debug!(
        target: "backend/player",
        "finished with equipped slot {}, re-scanning for next target",
        fusing.current_slot_index
    );

    fusing.targets = Array::new();
    fusing.target_rarities = Array::new();
    fusing.icons = Vec::new();
    fusing.names = Vec::new();
    fusing.target_index = 0;
    fusing.round = 0;
    fusing.fallback_target = None;
    fusing.state = State::SwitchToSetupTab(Timeout::default());
}

/// Deliberately does not close the familiar menu, unlike [`familiars_swap`]'s own
/// `update_completing` - fusion's completion always hands off into
/// [`Player::FamiliarsSwapping`] (`is_completing` triggers the handoff regardless of
/// [`FamiliarsFusing::success`]; see this state's caller), which relies on finding the menu still
/// open to take its fast `SwitchToSetupTab` path instead of a full reopen (see
/// [`FamiliarsSwapping`]'s `State::SwitchToSetupTab` docs).
///
/// Live testing showed closing it here anyway (an earlier version of this function did) raced
/// against swapping's very next `OpenMenu` check: Esc was sent, but by the time swapping's
/// `detect_familiar_menu_opened` ran (as soon as the next tick), the menu was still visible
/// mid-closing-animation, so swapping still took the `SwitchToSetupTab` shortcut - clicking a tab
/// button on a menu that was actually in the middle of disappearing, landing on nothing, and then
/// finding zero equipped slots and aborting. Swapping's own `update_completing` already closes the
/// menu once the whole chained fuse-then-swap sequence actually finishes, so closing it here too
/// was both redundant and the direct cause of that race.
///
/// [`familiars_swap`]: super::familiars_swap
#[inline]
fn update_completing(fusing: &mut FamiliarsFusing, timeout: Timeout, completed: bool) {
    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            fusing.state = State::Completing(timeout, completed);
        }
        Lifecycle::Ended => {
            debug!(target: "backend/player", "familiars fusing completed, success={}", fusing.success);
            fusing.state = State::Completing(Timeout::default(), true);
        }
    }
}

#[inline]
fn bbox_click_point(bbox: Rect) -> (i32, i32) {
    let x = bbox.x + bbox.width / 2;
    let y = bbox.y + bbox.height / 2;
    (x, y)
}

/// Portrait-only sub-region of an equipped familiar slot's card art (`slot` is the full
/// [`Detector::detect_familiar_slots`] bbox, always 148x196), trimmed of the card's
/// border/background glow and its level/stat footer.
///
/// Matching the *full* card elsewhere (the small Fusion-tab collection-list icon, or a bag item's
/// icon) does not work: verified against real screenshots, it scores ~0.14-0.55 correlation -
/// below any usable threshold, and not just because of scale. The border/background differs
/// between a list icon and a card, and the footer shows level/stats that differ between the
/// equipped copy and any lower-level duplicate fodder, both of which drag the whole-image score
/// down independent of resizing.
///
/// This trimmed portrait region instead scores ~0.94-1.0 against the *same* familiar shown
/// elsewhere at card scale (the Fusion tab's selected-preview card, an Inventory item's hover
/// tooltip preview) while scoring only ~0.4 against a *different* familiar's card - a safe margin
/// for the same 0.75+ thresholds used throughout this codebase's other template matching.
///
/// Cropped down further to just the center of that trimmed region rather than the whole thing -
/// a large-bodied familiar's art can extend past the card frame (wingtips, tails, etc.), and
/// live testing (via a real screenshot) showed this gets clipped inconsistently between the
/// Setup-tab equipped-slot render this icon is captured from and the Fusion tab's
/// selected-preview render it's later matched against, dragging the correlation score down even
/// for a genuinely correct match. The center of the art is far less likely to be clipped in
/// either context, at the cost of some rarely-needed distinguishing detail near the edges.
#[inline]
fn familiar_portrait_region(slot: Rect) -> Rect {
    Rect::new(slot.x + 47, slot.y + 45, 54, 50)
}

/// Computes the on-screen [`Rect`] of the `index`-th Inventory `Use`-tab grid slot (0-based,
/// row-major, top-left to bottom-right), anchored off the detected `Use` tab button's position
/// since the Inventory window can be anywhere on screen (see
/// [`INVENTORY_GRID_OFFSET_FROM_USE_TAB_X`]).
#[inline]
fn inventory_slot_rect(use_tab_bbox: Rect, index: u32) -> Rect {
    let col = (index % INVENTORY_GRID_COLUMNS) as i32;
    let row = (index / INVENTORY_GRID_COLUMNS) as i32;
    Rect::new(
        use_tab_bbox.x + INVENTORY_GRID_OFFSET_FROM_USE_TAB_X + col * INVENTORY_GRID_CELL_PITCH,
        use_tab_bbox.y + INVENTORY_GRID_OFFSET_FROM_USE_TAB_Y + row * INVENTORY_GRID_CELL_PITCH,
        INVENTORY_GRID_CELL_SIZE,
        INVENTORY_GRID_CELL_SIZE,
    )
}

/// Returns a [`Rect`] covering the entire current frame, used to search for an ad-hoc icon
/// template without needing a precisely measured panel/grid region.
#[inline]
fn full_frame_region(resources: &mut Resources) -> Rect {
    use opencv::core::MatTraitConst;

    let size = resources.detector().mat().size().unwrap();
    Rect::new(0, 0, size.width, size.height)
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use mockall::predicate::{eq, function};

    use super::*;
    use crate::player::PlayerAction;
    use crate::{array::Array, bridge::MockInput, detect::MockDetector, player::PlayerContext};

    fn new_fusing() -> FamiliarsFusing {
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        FamiliarsFusing::new(rarities, SwappableFamiliars::All, Array::new())
    }

    /// A minimal valid (non-empty, typed) `Mat` for use as a fake icon crop wherever a mocked
    /// `detect_region_template` call is exercised - unlike `Mat::default()`, whose `Debug` impl
    /// genuinely errors on an empty/untyped matrix (it calls into OpenCV to look up the type
    /// name, which has none), tripping `mockall`'s per-call argument-description formatting
    /// (`ArgPrinter::debug_string`) regardless of whether the expectation itself is satisfied.
    fn dummy_icon() -> Mat {
        use opencv::core::{CV_8UC3, Scalar};

        Mat::new_rows_cols_with_default(1, 1, CV_8UC3, Scalar::all(0.0)).unwrap()
    }

    #[test]
    fn is_safe_for_side_key_press_true_outside_inventory_quantity_flow() {
        let mut fusing = new_fusing();
        for state in [
            State::OpenMenu(Timeout::default()),
            State::FindEquippedSlots,
            State::CheckEquippedSlotLevel(Timeout::default(), 0),
            State::OpenFusionTab(Timeout::default()),
            State::SortByLevel(Timeout::default(), 0, None),
            State::FindInCollectionList(Timeout::default(), 0, None, false),
            State::CancelCandidate(Timeout::default(), 0),
            State::SelectAllFodder(Timeout::default()),
            State::Fusing(Timeout::default()),
            State::ConfirmFuse(Timeout::default(), false, false),
            State::RankingUp(Timeout::default()),
            State::ConfirmRankUp(Timeout::default(), false),
            State::DeselectRankedUp(Timeout::default(), 0),
            State::NextTarget,
            State::Completing(Timeout::default(), false),
        ] {
            fusing.state = state;
            assert!(
                fusing.is_safe_for_side_key_press(),
                "expected {state:?} to be safe for side key press"
            );
        }
    }

    #[test]
    fn is_safe_for_side_key_press_false_during_inventory_quantity_flow() {
        let mut fusing = new_fusing();
        for state in [
            State::OpenInventory(Timeout::default()),
            State::FocusUseTab(Timeout::default(), false),
            State::FindFodderItem(Timeout::default(), 0),
            State::DoubleClickItem(Timeout::default(), Rect::default()),
            State::TypeQuantity(Timeout::default()),
            State::ConfirmQuantity(Timeout::default()),
            State::ConfirmAddFamiliar(Timeout::default(), false),
            State::CloseInventory(Timeout::default(), false),
        ] {
            fusing.state = state;
            assert!(
                !fusing.is_safe_for_side_key_press(),
                "expected {state:?} to not be safe for side key press"
            );
        }
    }

    #[test]
    fn update_familiars_fusing_state_aborts_to_idle_when_lie_detector_shape_pending() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_lie_detector_shape()
            .once()
            .returning(|| Ok(Rect::default()));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());
        let mut player = PlayerEntity {
            state: Player::FamiliarsFusing(fusing),
            context: PlayerContext::default(),
        };

        update_familiars_fusing_state(&mut resources, &mut player);

        assert_matches!(player.state, Player::Idle);
    }

    #[test]
    fn update_familiars_fusing_state_aborts_to_idle_when_lie_detector_violetta_pending() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_lie_detector_shape()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        detector
            .expect_detect_lie_detector_violetta()
            .once()
            .returning(|| Ok(Rect::default()));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());
        let mut player = PlayerEntity {
            state: Player::FamiliarsFusing(fusing),
            context: PlayerContext::default(),
        };

        update_familiars_fusing_state(&mut resources, &mut player);

        assert_matches!(player.state, Player::Idle);
    }

    #[test]
    fn update_familiars_fusing_state_does_not_abort_when_unsafe_for_side_key_press() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_quantity_dialog()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::TypeQuantity(Timeout::default());
        let mut player = PlayerEntity {
            state: Player::FamiliarsFusing(fusing),
            context: PlayerContext::default(),
        };

        player.context.config.familiar_key = Some(KeyKind::A);
        player
            .context
            .set_normal_action(None, PlayerAction::SolveRune);

        update_familiars_fusing_state(&mut resources, &mut player);

        // Lie detector detection is never even attempted (no `MockDetector` expectation set for
        // it) while unsafe - `update_type_quantity`'s own dispatch still runs, so this only
        // asserts the state didn't jump straight to `Idle`.
        assert!(!matches!(player.state, Player::Idle));
    }

    #[test]
    fn update_familiars_fusing_state_proceeds_normally_when_no_lie_detector_pending() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_lie_detector_shape()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        detector
            .expect_detect_lie_detector_violetta()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        let mut input = MockInput::new();
        input.expect_send_mouse().once().returning(|_, _, _| ());
        input
            .expect_send_key()
            .with(eq(KeyKind::A))
            .once()
            .returning(|_| ());
        let mut resources = Resources::new(Some(input), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());
        let mut player = PlayerEntity {
            state: Player::FamiliarsFusing(fusing),
            context: PlayerContext::default(),
        };
        player.context.config.familiar_key = Some(KeyKind::A);

        player
            .context
            .set_normal_action(None, PlayerAction::SolveRune);
        update_familiars_fusing_state(&mut resources, &mut player);

        assert_matches!(player.state, Player::FamiliarsFusing(_));
    }

    #[test]
    fn update_familiars_fusing_state_preserves_ranked_up_icons_mid_run() {
        // Regression test: ranked_up_icons/newly_epic_icons were being cleared via
        // std::mem::take on *every* tick, not just the completing one - an icon pushed after
        // ranking up one target was silently wiped one tick later (e.g. while still going
        // through State::DeselectRankedUp/State::NextTarget for that same target, well before
        // this run's own eventual completion), so a later target's rank-up in the same run could
        // never hand off more than itself, and a single target's rank-up essentially never
        // survived to the chained swap at all.
        let mut detector = MockDetector::default();
        detector
            .expect_detect_lie_detector_shape()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        detector
            .expect_detect_lie_detector_violetta()
            .once()
            .returning(|| Err(anyhow::anyhow!("no match")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.ranked_up_icons.push(Arc::new(dummy_icon()));
        fusing.newly_epic_icons.push(Arc::new(dummy_icon()));
        // A non-completing transition - update_next_target doesn't touch either list.
        fusing.state = State::NextTarget;
        let mut player = PlayerEntity {
            state: Player::FamiliarsFusing(fusing),
            context: PlayerContext::default(),
        };
        player.context.config.familiar_key = Some(KeyKind::A);

        player
            .context
            .set_normal_action(None, PlayerAction::SolveRune);
        update_familiars_fusing_state(&mut resources, &mut player);

        let Player::FamiliarsFusing(fusing) = &player.state else {
            panic!("expected still fusing, got {:?}", player.state);
        };
        assert_eq!(fusing.ranked_up_icons.len(), 1);
        assert_eq!(fusing.newly_epic_icons.len(), 1);
    }

    #[test]
    fn update_open_menu_transitions_to_find_equipped_slots_when_menu_opened() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_menu_opened()
            .once()
            .returning(|| true);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout {
            current: 120,
            started: true,
            ..Default::default()
        });

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, None);

        assert_matches!(fusing.state, State::FindEquippedSlots);
    }

    #[test]
    fn update_open_menu_aborts_when_menu_not_opened() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_menu_opened()
            .once()
            .returning(|| false);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout {
            current: 120,
            started: true,
            ..Default::default()
        });

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, None);

        assert!(!fusing.success);
        assert_matches!(fusing.state, State::Completing(_, false));
    }

    #[test]
    fn update_open_menu_deactivates_familiar_buff_when_active() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_player_buff()
            .withf(|kind| matches!(kind, BuffKind::Familiar))
            .once()
            .returning(|_| true);
        let mut input = MockInput::new();
        input.expect_send_mouse().once().returning(|_, _, _| ());
        input
            .expect_send_key()
            .with(eq(KeyKind::F))
            .once()
            .returning(|_| ());
        let mut resources = Resources::new(Some(input), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, Some(KeyKind::F));

        assert_matches!(fusing.state, State::OpenMenu(_));
    }

    #[test]
    fn update_open_menu_does_not_press_familiar_buff_key_when_buff_inactive() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_player_buff()
            .withf(|kind| matches!(kind, BuffKind::Familiar))
            .once()
            .returning(|_| false);
        let mut input = MockInput::new();
        input.expect_send_mouse().once().returning(|_, _, _| ());
        input.expect_send_key().never();
        let mut resources = Resources::new(Some(input), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, Some(KeyKind::F));

        assert_matches!(fusing.state, State::OpenMenu(_));
    }

    #[test]
    fn update_open_menu_presses_key_immediately_when_no_buff_key() {
        let mut detector = MockDetector::default();
        detector.expect_detect_player_buff().never();
        let mut input = MockInput::new();
        input.expect_send_mouse().once().returning(|_, _, _| ());
        input
            .expect_send_key()
            .with(eq(KeyKind::A))
            .once()
            .returning(|_| ());
        let mut resources = Resources::new(Some(input), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout::default());

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, None);

        assert_matches!(fusing.state, State::OpenMenu(_));
    }

    #[test]
    fn update_open_menu_detects_menu_opened_early_when_no_buff_key() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_menu_opened()
            .once()
            .returning(|| true);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenMenu(Timeout {
            current: 1,
            started: true,
            ..Default::default()
        });

        update_open_menu(&mut resources, &mut fusing, KeyKind::A, None);

        assert_matches!(fusing.state, State::FindEquippedSlots);
    }

    #[test]
    fn update_find_equipped_slots_transitions_to_check_slot_level() {
        let bbox = Rect::new(0, 0, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_slots()
            .once()
            .returning(move || vec![(bbox, false), (bbox, true), (bbox, false)]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();

        update_find_equipped_slots(&mut resources, &mut fusing);

        assert_eq!(fusing.pending_slots.len(), 3);
        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 0));
    }

    #[test]
    fn update_find_equipped_slots_completes_when_wrong_slot_count() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_slots()
            .once()
            .returning(|| vec![(Rect::default(), true)]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();

        update_find_equipped_slots(&mut resources, &mut fusing);

        assert!(!fusing.success);
        assert_matches!(fusing.state, State::Completing(_, false));
    }

    #[test]
    fn update_check_equipped_slot_level_skips_free_slot_without_hovering() {
        // No MockInput/MockDetector expectations set - panics if this state hovers or detects
        // anything, since a free slot should be skipped immediately.
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.pending_slots.push((Rect::default(), true));
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 0);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 1));
    }

    #[test]
    fn update_check_equipped_slot_level_skips_ineligible_slot_without_hovering() {
        let mut resources = Resources::new(None, None);
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        let mut fusing = FamiliarsFusing::new(rarities, SwappableFamiliars::Last, Array::new());
        // Index 0 (first/leftmost) is excluded by SwappableFamiliars::Last even though occupied.
        fusing.pending_slots.push((Rect::default(), false));
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 0);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 1));
    }

    #[test]
    fn update_check_equipped_slot_level_captures_target_when_level_5() {
        let bbox = Rect::new(0, 0, 10, 10);
        let title_bar_bbox = Rect::new(20, 20, 30, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::Level5));
        detector
            .expect_detect_familiar_slot_rank()
            .with(eq(bbox))
            .returning(|_| Ok(FamiliarRank::Rare));
        detector
            .expect_detect_familiar_tooltip_title_bar()
            .returning(move || Ok(title_bar_bbox));
        detector
            .expect_capture_region()
            .with(eq(familiar_portrait_region(bbox)))
            .returning(|_| Ok(Mat::default()));
        detector
            .expect_capture_region()
            .with(eq(title_bar_bbox))
            .returning(|_| Ok(Mat::default()));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_eq!(fusing.targets.len(), 1);
        assert_eq!(fusing.targets[0], bbox);
        assert_eq!(fusing.icons.len(), 1);
        assert_eq!(fusing.names.len(), 1);
    }

    #[test]
    fn update_check_equipped_slot_level_skips_target_matching_protected_icon() {
        let bbox = Rect::new(0, 0, 10, 10);
        let title_bar_bbox = Rect::new(20, 20, 30, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::Level5));
        detector
            .expect_detect_familiar_slot_rank()
            .with(eq(bbox))
            .returning(|_| Ok(FamiliarRank::Rare));
        detector
            .expect_detect_familiar_tooltip_title_bar()
            .returning(move || Ok(title_bar_bbox));
        detector
            .expect_capture_region()
            .with(eq(familiar_portrait_region(bbox)))
            .returning(|_| Ok(dummy_icon()));
        detector
            .expect_capture_region()
            .with(eq(title_bar_bbox))
            .returning(|_| Ok(dummy_icon()));
        detector
            .expect_detect_region_template()
            .with(
                function(|_| true),
                eq(bbox),
                eq(PROTECTED_FAMILIAR_ICON_MATCH_THRESHOLD),
            )
            .returning(|_, _, _| vec![Rect::default()]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing().with_protected_icons(vec![Arc::new(dummy_icon())]);
        fusing.pending_slots.push((bbox, false));
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert!(fusing.targets.is_empty());
        assert!(fusing.icons.is_empty());
        assert!(fusing.names.is_empty());
    }

    #[test]
    fn update_check_equipped_slot_level_skips_target_when_name_capture_fails() {
        // Both the icon and the title-bar name must be captured for a target to be accepted -
        // a title-bar detection failure (e.g. tooltip not fully rendered yet) must not silently
        // fall back to icon-only, since fodder_item_matches indexes fusing.names by target_index
        // and expects it to stay in lockstep with fusing.targets/icons.
        let bbox = Rect::new(0, 0, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::Level5));
        detector
            .expect_detect_familiar_slot_rank()
            .with(eq(bbox))
            .returning(|_| Ok(FamiliarRank::Rare));
        detector
            .expect_detect_familiar_tooltip_title_bar()
            .returning(|| Err(anyhow::anyhow!("not found")));
        detector
            .expect_capture_region()
            .with(eq(familiar_portrait_region(bbox)))
            .returning(|_| Ok(Mat::default()));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert!(fusing.targets.is_empty());
        assert!(fusing.icons.is_empty());
        assert!(fusing.names.is_empty());
    }

    #[test]
    fn update_check_equipped_slot_level_skips_target_when_not_level_5() {
        let bbox = Rect::new(0, 0, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::LevelOther));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert!(fusing.targets.is_empty());
    }

    #[test]
    fn update_check_equipped_slot_level_completes_when_no_targets_found() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        // pending_slots left empty, so index 0 is already past the end.
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 0);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::Completing(_, false));
    }

    #[test]
    fn update_check_equipped_slot_level_moves_to_fusion_tab_when_target_found_on_current_slot() {
        let bbox = Rect::default();
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.pending_slots.push((bbox, false));
        // Simulates a Rare target having just been captured by this slot's tick-5 level check -
        // Rare is the top priority, so it should be taken immediately rather than deferred.
        fusing.targets.push(bbox);
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TIMEOUT,
                started: true,
                ..Default::default()
            },
            1,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenFusionTab(_));
        assert_eq!(fusing.current_slot_index, 1);
        assert_eq!(fusing.target_index, 0);
        assert!(fusing.fallback_target.is_none());
    }

    #[test]
    fn update_check_equipped_slot_level_takes_epic_immediately_when_rare_not_configured() {
        let bbox = Rect::default();
        let mut resources = Resources::new(None, None);
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Epic);
        let mut fusing = FamiliarsFusing::new(rarities, SwappableFamiliars::All, Array::new());
        fusing.pending_slots.push((bbox, false));
        fusing.pending_slots.push((bbox, false));
        fusing.targets.push(bbox);
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TIMEOUT,
                started: true,
                ..Default::default()
            },
            1,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenFusionTab(_));
        assert_eq!(fusing.current_slot_index, 1);
        assert!(fusing.fallback_target.is_none());
    }

    #[test]
    fn update_check_equipped_slot_level_defers_epic_target_when_rare_configured() {
        let bbox = Rect::new(1, 2, 3, 4);
        let mut resources = Resources::new(None, None);
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        rarities.push(FamiliarRarity::Epic);
        let mut fusing = FamiliarsFusing::new(rarities, SwappableFamiliars::All, Array::new());
        fusing.pending_slots.push((bbox, false));
        fusing.pending_slots.push((bbox, false));
        fusing.pending_slots.push((bbox, false));
        // Simulates an Epic target having just been captured on slot 0 - since Rare fusion is
        // also configured, this must be deferred rather than taken immediately, so the scan can
        // keep looking at the remaining slots for a higher-priority Rare target.
        fusing.targets.push(bbox);
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(Mat::default()));
        fusing.names.push(Arc::new(Mat::default()));
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 1));
        assert!(fusing.targets.is_empty());
        assert!(fusing.icons.is_empty());
        assert!(fusing.names.is_empty());
        let fallback = fusing.fallback_target.expect("epic target deferred");
        assert_eq!(fallback.slot_index, 0);
        assert_eq!(fallback.bbox, bbox);
        assert_eq!(fallback.rarity, FamiliarRarity::Epic);
    }

    #[test]
    fn update_check_equipped_slot_level_falls_back_to_deferred_epic_when_scan_exhausted() {
        let epic_bbox = Rect::new(1, 2, 3, 4);
        let mut resources = Resources::new(None, None);
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        rarities.push(FamiliarRarity::Epic);
        let mut fusing = FamiliarsFusing::new(rarities, SwappableFamiliars::All, Array::new());
        fusing.pending_slots.push((epic_bbox, false));
        fusing.pending_slots.push((epic_bbox, false));
        // No Rare target ever turned up - only the deferred Epic from slot 0 remains.
        fusing.fallback_target = Some(FallbackTarget {
            slot_index: 0,
            bbox: epic_bbox,
            rarity: FamiliarRarity::Epic,
            icon: Arc::new(Mat::default()),
            name: Arc::new(Mat::default()),
        });
        // index (2) is already past pending_slots.len(), simulating the scan having finished.
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 2);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenFusionTab(_));
        assert_eq!(fusing.current_slot_index, 0);
        assert_eq!(fusing.target_index, 0);
        assert_eq!(fusing.targets.len(), 1);
        assert_eq!(fusing.targets[0], epic_bbox);
        assert_eq!(fusing.target_rarities[0], FamiliarRarity::Epic);
        assert!(fusing.fallback_target.is_none());
    }

    #[test]
    fn update_check_equipped_slot_level_continues_scan_when_no_target_found_on_current_slot() {
        let bbox = Rect::default();
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.pending_slots.push((bbox, false));
        // No target pushed - mirrors a slot that failed the level/rarity check.
        fusing.state = State::CheckEquippedSlotLevel(
            Timeout {
                current: SLOT_LEVEL_CHECK_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 1));
    }

    #[test]
    fn update_check_equipped_slot_level_skips_already_attempted_slot() {
        let bbox = Rect::default();
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.pending_slots.push((bbox, false));
        fusing.attempted[0] = true;
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 0);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CheckEquippedSlotLevel(_, 1));
    }

    #[test]
    fn update_open_fusion_tab_clicks_tab_button_on_start_even_if_reported_open() {
        // Regression test: the Started tick must always click the Fusion tab button rather than
        // trusting detect_familiar_fusion_tab_opened() to skip it - that check shares near-
        // identical visual chrome across all 4 tabs and has been observed to false-positive on
        // the Setup tab, which previously left the whole run operating blind against the wrong
        // tab. So even when the detector claims the tab is already open, Started must still
        // attempt the click (harmless no-op if genuinely already on Fusion) - only
        // Lifecycle::Ended decides whether the tab is actually open.
        let bbox = Rect::new(5, 5, 10, 10);
        let mut detector = MockDetector::default();
        detector.expect_detect_familiar_fusion_tab_opened().never();
        detector
            .expect_detect_familiar_fusion_tab_button()
            .once()
            .returning(move || Ok(bbox));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::OpenFusionTab(Timeout::default());

        update_open_fusion_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenFusionTab(_));
    }

    #[test]
    fn update_open_fusion_tab_sorts_when_not_yet_sorted() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_tab_opened()
            .once()
            .returning(|| true);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        assert!(!fusing.sorted_by_level);
        fusing.state = State::OpenFusionTab(Timeout {
            current: 30,
            started: true,
            ..Default::default()
        });

        update_open_fusion_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::SortByLevel(_, 0, None));
    }

    #[test]
    fn update_open_fusion_tab_skips_sort_when_already_sorted() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_tab_opened()
            .once()
            .returning(|| true);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.sorted_by_level = true;
        fusing.state = State::OpenFusionTab(Timeout {
            current: 30,
            started: true,
            ..Default::default()
        });

        update_open_fusion_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindInCollectionList(_, 0, None, false));
    }

    #[test]
    fn update_sort_by_level_sets_sorted_flag_when_done() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        assert!(!fusing.sorted_by_level);
        fusing.state = State::SortByLevel(
            Timeout {
                current: 10,
                started: true,
                ..Default::default()
            },
            1,
            Some(Rect::default()),
        );

        update_sort_by_level(&mut resources, &mut fusing);

        assert!(fusing.sorted_by_level);
        assert_matches!(fusing.state, State::FindInCollectionList(_, 0, None, false));
    }

    #[test]
    fn update_fusing_transitions_to_ranking_up() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Ok((Rect::default(), FusionButton::RankUp)));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::Fusing(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });

        update_fusing(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::RankingUp(_));
    }

    #[test]
    fn update_fusing_clicks_fuse_and_transitions_to_confirm() {
        let bbox = Rect::new(5, 5, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(move || Ok((bbox, FusionButton::Fuse)));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::Fusing(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });

        update_fusing(&mut resources, &mut fusing);

        assert_eq!(fusing.round, 1);
        assert_matches!(fusing.state, State::ConfirmFuse(_, false, false));
    }

    #[test]
    fn update_confirm_fuse_rechecks_button_when_not_round_capped() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Rare));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::ConfirmFuse(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            false,
            true,
        );

        update_confirm_fuse(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::Fusing(_));
    }

    #[test]
    fn update_confirm_fuse_deselects_when_rank_up_happened_silently() {
        // Regression test: some rank-up transitions never show a distinct `Rank Up` button - the
        // very last Fuse click completes the gauge and silently ranks up the familiar, so this
        // must be detected via the gauge's own rarity rather than only via State::RankingUp.
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.icons.push(Arc::new(Mat::default()));
        fusing.state = State::ConfirmFuse(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            false,
            true,
        );

        update_confirm_fuse(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DeselectRankedUp(_, 0));
    }

    #[test]
    fn update_confirm_fuse_advances_to_next_target_when_round_cap_hit() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Rare));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::ConfirmFuse(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            true,
            true,
        );

        update_confirm_fuse(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::NextTarget);
    }

    #[test]
    fn update_confirm_fuse_deselects_when_rank_up_happened_silently_at_round_cap() {
        // Regression test: hitting MAX_FUSE_ROUNDS can coincide with the exact Fuse click that
        // completes the gauge - giving up unconditionally in that case (the previous behavior)
        // abandoned a familiar that had in fact just ranked up, without recording it in
        // ranked_up_icons for the chained swap to recognize and free it afterwards.
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.icons.push(Arc::new(Mat::default()));
        fusing.state = State::ConfirmFuse(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            true,
            true,
        );

        update_confirm_fuse(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DeselectRankedUp(_, 0));
        assert_eq!(fusing.ranked_up_icons.len(), 1);
    }

    #[test]
    fn update_confirm_fuse_clicks_only_once() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_popup_confirm_button()
            .once()
            .returning(|| Ok(Rect::new(5, 5, 10, 10)));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::ConfirmFuse(
            Timeout {
                current: 1,
                started: true,
                ..Default::default()
            },
            false,
            false,
        );

        update_confirm_fuse(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::ConfirmFuse(_, false, true));

        // Already clicked - must not detect or click again on a later tick.
        update_confirm_fuse(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::ConfirmFuse(_, false, true));
    }

    #[test]
    fn update_fusing_opens_inventory_when_nothing_selected() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("disabled")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::Fusing(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });

        update_fusing(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenInventory(_));
    }

    #[test]
    fn update_fusing_opens_inventory_when_fuse_button_disabled() {
        // The greyed-out no-material button still renders (unlike the fully-absent case covered
        // by update_fusing_opens_inventory_when_nothing_selected), so it must be handled as its
        // own FusionButton variant rather than falling through to Err(_).
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Ok((Rect::default(), FusionButton::FuseDisabled)));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::Fusing(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });

        update_fusing(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenInventory(_));
    }

    #[test]
    fn update_next_target_marks_slot_attempted_and_rescans_from_start() {
        let mut fusing = new_fusing();
        fusing.current_slot_index = 1;
        fusing.targets.push(Rect::default());
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.icons.push(Arc::new(Mat::default()));
        fusing.names.push(Arc::new(Mat::default()));
        fusing.round = 3;

        update_next_target(&mut fusing);

        assert!(fusing.attempted[1]);
        assert!(fusing.targets.is_empty());
        assert!(fusing.target_rarities.is_empty());
        assert!(fusing.icons.is_empty());
        assert!(fusing.names.is_empty());
        assert_eq!(fusing.target_index, 0);
        assert_eq!(fusing.round, 0);
        assert_matches!(fusing.state, State::SwitchToSetupTab(_));
    }

    #[test]
    fn update_next_target_completes_when_no_slots_left_to_scan() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_setup_tab_inactive_button()
            .returning(|| Err(anyhow::anyhow!("not found")));
        detector
            .expect_detect_familiar_slots()
            .once()
            .returning(Vec::new);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.current_slot_index = 0;

        update_next_target(&mut fusing);
        assert_matches!(fusing.state, State::SwitchToSetupTab(_));

        fusing.state = State::SwitchToSetupTab(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });
        update_switch_to_setup_tab(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::FindEquippedSlots);

        // detect_familiar_slots returning 0 (not FAMILIAR_SLOTS) slots here stands in for
        // "nothing left eligible" - the real rescan would instead find 3 slots all either
        // ineligible or already attempted, which update_check_equipped_slot_level_* tests already
        // cover on their own.
        update_find_equipped_slots(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::Completing(_, false));
    }

    #[test]
    fn update_switch_to_setup_tab_clicks_button_and_transitions_to_find_equipped_slots() {
        let bbox = Rect::new(513, 161, 102, 26);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_setup_tab_inactive_button()
            .once()
            .returning(move || Ok(bbox));
        let mut input = MockInput::new();
        input
            .expect_send_mouse()
            .with(
                eq(564),
                eq(174),
                function(|kind| matches!(kind, MouseKind::Click)),
            )
            .once()
            .returning(|_, _, _| ());
        let mut resources = Resources::new(Some(input), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::SwitchToSetupTab(Timeout::default());

        update_switch_to_setup_tab(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::SwitchToSetupTab(_));

        fusing.state = State::SwitchToSetupTab(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });
        update_switch_to_setup_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindEquippedSlots);
    }

    #[test]
    fn update_ranking_up_clicks_button_and_transitions_to_confirm() {
        let bbox = Rect::new(5, 5, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(move || Ok((bbox, FusionButton::RankUp)));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::RankingUp(Timeout::default());

        update_ranking_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::RankingUp(_));

        fusing.state = State::RankingUp(Timeout {
            current: 15,
            started: true,
            ..Default::default()
        });
        update_ranking_up(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::ConfirmRankUp(_, false));
    }

    #[test]
    fn update_confirm_rank_up_confirms_success_when_diamond_advances_a_tier() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        let mut resources = Resources::new(None, Some(detector));

        // Fixture's target rarity is Rare, so Epic confirms a successful rare -> epic rank up.
        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.icons.push(Arc::new(Mat::default()));
        fusing.state = State::ConfirmRankUp(
            Timeout {
                current: 20,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_confirm_rank_up(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DeselectRankedUp(_, 0));
    }

    #[test]
    fn update_confirm_rank_up_retries_feeding_when_diamond_still_same_tier() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Rare));
        let mut resources = Resources::new(None, Some(detector));

        // Fixture's target rarity is Rare - still Rare means the click did not take.
        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::ConfirmRankUp(
            Timeout {
                current: 20,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_confirm_rank_up(&mut resources, &mut fusing);

        assert_eq!(fusing.round, 1);
        assert_matches!(fusing.state, State::SelectAllFodder(_));
    }

    #[test]
    fn update_confirm_rank_up_clicks_only_once() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_popup_confirm_button()
            .once()
            .returning(|| Ok(Rect::new(5, 5, 10, 10)));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::ConfirmRankUp(
            Timeout {
                current: 1,
                started: true,
                ..Default::default()
            },
            false,
        );

        update_confirm_rank_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::ConfirmRankUp(_, true));

        // Already clicked - must not detect or click again on a later tick.
        update_confirm_rank_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::ConfirmRankUp(_, true));
    }

    #[test]
    fn update_deselect_ranked_up_clicks_cancel_and_advances_to_next_target() {
        use opencv::boxed_ref::BoxedRef;

        let bbox = Rect::new(4, 4, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_cancel_button()
            .once()
            .returning(move || Ok(bbox));
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(|_, _, _| Vec::new());
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::DeselectRankedUp(Timeout::default(), 0);

        update_deselect_ranked_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::DeselectRankedUp(_, 0));

        fusing.state = State::DeselectRankedUp(
            Timeout {
                current: 10,
                started: true,
                ..Default::default()
            },
            0,
        );
        update_deselect_ranked_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::NextTarget);
    }

    #[test]
    fn update_deselect_ranked_up_retries_when_still_selected() {
        use opencv::boxed_ref::BoxedRef;

        // Regression test: a single Cancel click that doesn't take must not be mistaken for
        // success - the next target's search would find nothing and never know why. Verifying via
        // icon match and retrying is what catches this instead of blindly moving on.
        let icon_bbox = Rect::new(4, 4, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::DeselectRankedUp(
            Timeout {
                current: 10,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_deselect_ranked_up(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DeselectRankedUp(_, 1));
    }

    #[test]
    fn update_deselect_ranked_up_gives_up_after_max_attempts() {
        use opencv::boxed_ref::BoxedRef;

        let icon_bbox = Rect::new(4, 4, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::DeselectRankedUp(
            Timeout {
                current: 10,
                started: true,
                ..Default::default()
            },
            MAX_DESELECT_RANKED_UP_ATTEMPTS - 1,
        );

        update_deselect_ranked_up(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::NextTarget);
    }

    #[test]
    fn update_find_in_collection_list_hovers_candidate_at_index() {
        let first_bbox = Rect::new(1, 1, 10, 10);
        let second_bbox = Rect::new(20, 1, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(move || {
                vec![
                    (first_bbox, FamiliarRank::Rare),
                    (second_bbox, FamiliarRank::Rare),
                ]
            });
        let mut keys = MockInput::default();
        keys.expect_send_mouse()
            .with(
                eq(6),
                eq(6),
                function(|action| matches!(action, MouseKind::Move)),
            )
            .once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::FindInCollectionList(Timeout::default(), 0, None, false);

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindInCollectionList(_, 0, _, false));
    }

    #[test]
    fn update_find_in_collection_list_gives_up_when_no_candidates() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(Vec::new);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::FindInCollectionList(Timeout::default(), 0, None, false);

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::NextTarget);
    }

    #[test]
    fn update_find_in_collection_list_clicks_candidate_when_level_5() {
        let rare_bbox = Rect::new(1, 1, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::Level5));
        let mut keys = MockInput::default();
        keys.expect_send_mouse()
            .with(
                eq(6),
                eq(6),
                function(|action| matches!(action, MouseKind::Click)),
            )
            .once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
            Some(rare_bbox),
            false,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(
            fusing.state,
            State::FindInCollectionList(_, 0, Some(_), true)
        );
    }

    #[test]
    fn update_find_in_collection_list_skips_candidate_when_not_level_5() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::LevelOther));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Rare);
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
            Some(Rect::new(1, 1, 10, 10)),
            false,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindInCollectionList(_, 1, _, false));
    }

    #[test]
    fn update_find_in_collection_list_selects_all_fodder_when_rarity_matches() {
        use opencv::boxed_ref::BoxedRef;

        let icon_bbox = Rect::new(100, 100, 108, 100);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("fusion button not found")));
        detector
            .expect_detect_familiar_card_boxes()
            .once()
            .returning(|| vec![Rect::default()]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
            None,
            true,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::SelectAllFodder(_));
    }

    #[test]
    fn update_find_in_collection_list_opens_inventory_when_no_duplicates_available() {
        use opencv::boxed_ref::BoxedRef;

        let icon_bbox = Rect::new(100, 100, 108, 100);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("fusion button not found")));
        detector
            .expect_detect_familiar_card_boxes()
            .once()
            .returning(Vec::new);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
            None,
            true,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenInventory(_));
    }

    #[test]
    fn update_find_in_collection_list_ranks_up_directly_when_gauge_already_full() {
        use opencv::boxed_ref::BoxedRef;

        let icon_bbox = Rect::new(100, 100, 108, 100);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Ok((Rect::default(), FusionButton::RankUp)));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
            None,
            true,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::RankingUp(_));
    }

    #[test]
    fn update_find_in_collection_list_cancels_when_rarity_does_not_match() {
        use opencv::boxed_ref::BoxedRef;

        // Same-species duplicate already ranked up to Unique: icon matches, but the selected
        // card's current gauge rarity does not match the Epic target being searched for.
        let icon_bbox = Rect::new(100, 100, 108, 100);
        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Unique));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_TIMEOUT,
                started: true,
                ..Default::default()
            },
            0,
            None,
            true,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CancelCandidate(_, 1));
    }

    #[test]
    fn update_find_in_collection_list_advances_to_next_index_when_not_clicked() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_hover_level()
            .once()
            .returning(|| Ok(FamiliarLevel::LevelOther));
        let mut resources = Resources::new(None, Some(detector));
        let mut fusing = new_fusing();
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_LEVEL_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
            None,
            false,
        );

        update_find_in_collection_list(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindInCollectionList(_, 1, _, false));
    }

    #[test]
    fn update_cancel_candidate_clicks_cancel_button() {
        let bbox = Rect::new(4, 4, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_familiar_fusion_cancel_button()
            .once()
            .returning(move || Ok(bbox));
        let mut keys = MockInput::default();
        keys.expect_send_mouse().once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::CancelCandidate(Timeout::default(), 1);

        update_cancel_candidate(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CancelCandidate(_, 1));
    }

    #[test]
    fn update_cancel_candidate_resumes_with_next_index() {
        use opencv::boxed_ref::BoxedRef;

        let mut detector = MockDetector::default();
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(|_, _, _| Vec::new());
        let mut resources = Resources::new(None, Some(detector));
        let mut fusing = new_fusing();
        fusing.icons.push(Arc::new(dummy_icon()));
        fusing.state = State::CancelCandidate(
            Timeout {
                current: 10,
                started: true,
                ..Default::default()
            },
            2,
        );

        update_cancel_candidate(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindInCollectionList(_, 2, _, false));
    }

    #[test]
    fn update_find_fodder_item_hovers_next_candidate_slot() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .returning(|| Ok(Rect::new(154, 33, 74, 16)));
        let mut keys = MockInput::default();
        // Slot 0 rect derives from the use-tab bbox: (154 - 116, 33 + 65, 44, 44), center (60, 120).
        keys.expect_send_mouse()
            .with(
                eq(60),
                eq(120),
                function(|action| matches!(action, MouseKind::Move)),
            )
            .once();
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FindFodderItem(Timeout::default(), 0);

        update_find_fodder_item(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindFodderItem(_, 0));
    }

    #[test]
    fn update_find_fodder_item_gives_up_when_use_tab_button_not_found() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .returning(|| Err(anyhow::anyhow!("not found")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FindFodderItem(Timeout::default(), 0);

        update_find_fodder_item(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CloseInventory(_, false));
    }

    #[test]
    fn update_find_fodder_item_gives_up_after_max_slots_scanned() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.state = State::FindFodderItem(Timeout::default(), INVENTORY_SCAN_MAX_SLOTS);

        update_find_fodder_item(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CloseInventory(_, false));
    }

    #[test]
    fn update_find_fodder_item_matches_name_and_double_clicks() {
        use opencv::boxed_ref::BoxedRef;

        let bbox = Rect::new(1, 1, 10, 10);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .returning(move || Ok(bbox));
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(move |_, _, _| vec![bbox]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.names.push(Arc::new(dummy_icon()));
        // `next_timeout_lifecycle` increments `current` before the early
        // check compares it against `FODDER_ITEM_EARLY_CHECK_TICK` - start
        // one tick early so it lands exactly on the check, matching the
        // same `CANDIDATE_LEVEL_CHECK_TICK - 1` convention used elsewhere
        // in this file for the equivalent early-check pattern.
        fusing.state = State::FindFodderItem(
            Timeout {
                current: FODDER_ITEM_EARLY_CHECK_TICK - 1,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_find_fodder_item(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DoubleClickItem(_, _));
    }

    #[test]
    fn update_find_fodder_item_no_name_match_advances_to_next_slot() {
        use opencv::boxed_ref::BoxedRef;

        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .returning(|| Ok(Rect::new(1, 1, 10, 10)));
        detector
            .expect_mat()
            .returning(|| BoxedRef::from(Mat::default()));
        detector
            .expect_detect_region_template()
            .once()
            .returning(|_, _, _| Vec::new());
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.names.push(Arc::new(dummy_icon()));
        // Index only advances on `Lifecycle::Ended` (the slot's full
        // timeout, not the early check) - `next_timeout_lifecycle` here
        // uses a max of 15, matching `update_find_fodder_item`'s own
        // `next_timeout_lifecycle(timeout, 15)` call.
        fusing.state = State::FindFodderItem(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            0,
        );

        update_find_fodder_item(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindFodderItem(_, 1));
    }

    #[test]
    fn update_open_inventory_gives_up_target_without_key() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.state = State::OpenInventory(Timeout::default());

        update_open_inventory(&mut resources, &mut fusing, None);

        assert_matches!(fusing.state, State::NextTarget);
    }

    #[test]
    fn update_focus_use_tab_skips_click_when_already_focused() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .once()
            .returning(|| Ok(Rect::default()));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FocusUseTab(Timeout::default(), false);

        update_focus_use_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FindFodderItem(_, 0));
    }

    #[test]
    fn update_focus_use_tab_clicks_inactive_tab_when_not_focused() {
        let bbox = Rect::new(668, 147, 74, 16);
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("not focused")));
        detector
            .expect_detect_inventory_use_tab_inactive_button()
            .once()
            .returning(move || Ok(bbox));
        let mut keys = MockInput::default();
        // Once for the click itself, once to move the cursor back off the tab it just clicked -
        // see `update_focus_use_tab`'s docs for why the second move matters.
        keys.expect_send_mouse().times(2);
        let mut resources = Resources::new(Some(keys), Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FocusUseTab(Timeout::default(), false);

        update_focus_use_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::FocusUseTab(_, true));
    }

    #[test]
    fn update_focus_use_tab_rechecks_without_reclicking_on_later_ticks() {
        // Regression test: once the click has been sent, later ticks must keep checking whether
        // it took effect but must NOT click again - live testing showed repeated clicks on every
        // tick interrupted the tab's own switch animation, leaving detection stuck in a
        // middling, inconclusive score range instead of ever resolving to focused. Not mocking
        // detect_inventory_use_tab_inactive_button or send_mouse at all means either being called
        // would panic this test.
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .times(2)
            .returning(|| Err(anyhow::anyhow!("not focused")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FocusUseTab(
            Timeout {
                current: 1,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_focus_use_tab(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::FocusUseTab(_, true));

        update_focus_use_tab(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::FocusUseTab(_, true));
    }

    #[test]
    fn update_focus_use_tab_gives_up_when_still_not_focused_after_click() {
        let mut detector = MockDetector::default();
        detector
            .expect_detect_inventory_use_tab_button()
            .returning(|| Err(anyhow::anyhow!("not focused")));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.state = State::FocusUseTab(
            Timeout {
                current: 60,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_focus_use_tab(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::CloseInventory(_, false));
    }

    #[test]
    fn update_close_inventory_resumes_to_select_all_fodder() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.state = State::CloseInventory(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_close_inventory(&mut resources, &mut fusing, None);

        assert_matches!(fusing.state, State::SelectAllFodder(_));
    }

    #[test]
    fn update_close_inventory_gives_up_target() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.state = State::CloseInventory(
            Timeout {
                current: 15,
                started: true,
                ..Default::default()
            },
            false,
        );

        update_close_inventory(&mut resources, &mut fusing, None);

        assert_matches!(fusing.state, State::NextTarget);
    }
}
