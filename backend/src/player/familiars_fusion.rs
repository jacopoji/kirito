use std::{fmt::Display, sync::Arc};

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
    detect::{FamiliarLevel, FamiliarRank, FusionButton, FusionGaugeRarity},
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

/// Internal state machine representing the current state of familiar fusion (rank-up).
#[derive(Debug, Clone, Copy)]
enum State {
    /// Opening the familiar menu.
    OpenMenu(Timeout),
    /// Finding the 3 equipped familiar slots.
    FindEquippedSlots,
    /// Hovering the equipped slot at `index` and checking whether it is Level 5 - mirrors swap's
    /// `update_free_slot`'s exact same eligibility check (same hover offset and tick timing),
    /// since a familiar still gaining levels isn't a sensible fusion target yet. Only slots that
    /// pass this, plus the existing rarity/[`SwappableFamiliars`] filters, become fusion targets.
    CheckEquippedSlotLevel(Timeout, u32),
    /// Switching to (or confirming) the Fusion tab.
    OpenFusionTab(Timeout),
    /// Clicking the collection list's `Level` sort header twice (ascending then descending) so
    /// the highest-level familiars - which the currently-equipped target is likely among -
    /// surface at the top of the list, without needing to scroll to find it.
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
    /// next target's search would find nothing and never know why.
    DeselectRankedUp(Timeout),
    /// Advancing to the next eligible target familiar, if any.
    NextTarget,
    Completing(Timeout, bool),
}

/// Struct for storing familiar fusion data.
#[derive(Debug, Clone)]
pub struct FamiliarsFusing {
    /// Current state of the familiar fusion state machine.
    state: State,
    /// Only equipped familiars of these rarities will be considered for fusion (`Rare` for
    /// rare -> epic, `Epic` for epic -> unique).
    fusable_rarities: Array<FamiliarRarity, 2>,
    /// Carried through so fusion can chain into the existing swap-and-save flow on completion.
    swappable_slots: SwappableFamiliars,
    swappable_rarities: Array<FamiliarRarity, 2>,
    /// Raw equipped-slot detection results (bbox, is_free), scanned one at a time via hover+level
    /// check in [`State::CheckEquippedSlotLevel`] before being accepted as fusion targets.
    pending_slots: Array<(Rect, bool), FAMILIAR_SLOTS>,
    /// Equipped Setup-tab slot bounding boxes eligible for fusion.
    targets: Array<Rect, FAMILIAR_SLOTS>,
    /// Rarity of each `targets` entry (same order), used to filter which Fusion-tab collection
    /// list candidates are worth trying for a given target.
    target_rarities: Array<FamiliarRarity, FAMILIAR_SLOTS>,
    /// Ad-hoc icon crops (one per `targets` entry, same order) used to find each target familiar
    /// elsewhere (Fusion tab's collection list, Inventory's `Use` tab) since no precompiled
    /// template can exist for an arbitrary familiar type.
    icons: Vec<Arc<Mat>>,
    /// Icons of targets that were successfully ranked up this run (a subset of `icons`).
    ///
    /// A ranked-up familiar changes rarity but keeps its Level 5 -> 1 reset, and stays equipped in
    /// its slot - carried through to the chained [`FamiliarsSwapping`] so its own candidate scan
    /// (which otherwise treats a non-Level-5 hover as "an available duplicate, swap it in") can
    /// recognize and skip these by icon instead of being fooled by the level reset into trying to
    /// swap in a familiar that is already equipped elsewhere.
    ranked_up_icons: Vec<Arc<Mat>>,
    /// Index into `targets`/`icons` currently being processed.
    target_index: usize,
    /// Number of Select-All-then-Fuse rounds attempted for the current target.
    round: u32,
    /// Mouse rest point for other operations.
    mouse_rest: Point,
    /// Whether fusing completed without an unexpected detection failure.
    success: bool,
}

impl Display for FamiliarsFusing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.state {
            State::OpenMenu(_) => write!(f, "OpenMenu"),
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
            State::DeselectRankedUp(_) => write!(f, "DeselectRankedUp"),
            State::NextTarget => write!(f, "NextTarget"),
            State::Completing(_, _) => write!(f, "Completing"),
        }
    }
}

impl FamiliarsFusing {
    pub fn new(
        fusable_rarities: Array<FamiliarRarity, 2>,
        swappable_slots: SwappableFamiliars,
        swappable_rarities: Array<FamiliarRarity, 2>,
    ) -> Self {
        Self {
            state: State::OpenMenu(Timeout::default()),
            fusable_rarities,
            swappable_slots,
            swappable_rarities,
            pending_slots: Array::new(),
            targets: Array::new(),
            target_rarities: Array::new(),
            icons: Vec::new(),
            ranked_up_icons: Vec::new(),
            target_index: 0,
            round: 0,
            mouse_rest: Point::new(50, 50),
            success: true,
        }
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

    match fusing.state {
        State::OpenMenu(_) => update_open_menu(resources, &mut fusing, familiar_key),
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
        State::DeselectRankedUp(_) => update_deselect_ranked_up(resources, &mut fusing),
        State::NextTarget => update_next_target(&mut fusing),
        State::Completing(timeout, completed) => {
            update_completing(resources, &mut fusing, timeout, completed)
        }
    }

    let is_completing = matches!(fusing.state, State::Completing(_, true));
    let success = fusing.success;
    let swappable_slots = fusing.swappable_slots;
    let swappable_rarities = fusing.swappable_rarities;
    let ranked_up_icons = std::mem::take(&mut fusing.ranked_up_icons);

    let next_state_display = fusing.to_string();
    let next = if is_completing {
        Player::FamiliarsSwapping(FamiliarsSwapping::new(
            swappable_slots,
            swappable_rarities,
            ranked_up_icons,
        ))
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

fn update_open_menu(resources: &mut Resources, fusing: &mut FamiliarsFusing, key: KeyKind) {
    let State::OpenMenu(timeout) = fusing.state else {
        panic!("familiars fusing state is not opening menu");
    };

    match next_timeout_lifecycle(timeout, 120) {
        Lifecycle::Started(timeout) => {
            resources
                .input
                .send_mouse(fusing.mouse_rest.x, fusing.mouse_rest.y, MouseKind::Move);
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
            if timeout.current == 60 {
                resources.input.send_key(key);
            }

            fusing.state = State::OpenMenu(timeout);
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
        debug!(
            target: "backend/player",
            "finished checking equipped slots, found {} fusion target(s)",
            fusing.targets.len()
        );
        fusing.state = if fusing.targets.is_empty() {
            State::Completing(Timeout::default(), false)
        } else {
            fusing.target_index = 0;
            fusing.round = 0;
            State::OpenFusionTab(Timeout::default())
        };
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
    if is_free || !eligible_slot {
        debug!(
            target: "backend/player",
            "skipping equipped slot {index} (free={is_free}, eligible_by_swappable_slots={eligible_slot})"
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
                            if let Ok(icon) = resources
                                .detector()
                                .capture_region(familiar_portrait_region(bbox))
                            {
                                fusing.targets.push(bbox);
                                fusing.target_rarities.push(rarity);
                                fusing.icons.push(Arc::new(icon));
                            } else {
                                debug!(
                                    target: "backend/player",
                                    "equipped slot {index} icon capture failed"
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
            fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), index + 1);
        }
    }
}

fn update_open_fusion_tab(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::OpenFusionTab(timeout) = fusing.state else {
        panic!("familiars fusing state is not opening fusion tab");
    };

    match next_timeout_lifecycle(timeout, 30) {
        Lifecycle::Started(timeout) => {
            if resources.detector().detect_familiar_fusion_tab_opened() {
                debug!(target: "backend/player", "fusion tab already opened for target {}", fusing.target_index);
                fusing.state = State::SortByLevel(Timeout::default(), 0, None);
                return;
            }
            let tab_button = resources.detector().detect_familiar_fusion_tab_button();
            debug!(target: "backend/player", "fusion tab button: {tab_button:?}");
            if let Ok(bbox) = tab_button {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::OpenFusionTab(timeout);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_fusion_tab_opened() {
                debug!(target: "backend/player", "fusion tab opened for target {}", fusing.target_index);
                fusing.state = State::SortByLevel(Timeout::default(), 0, None);
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
                    fusing.target_index
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
            }
            fusing.state = State::FindInCollectionList(timeout, index, cached_bbox, clicked);
        }
        Lifecycle::Ended => {
            if !clicked {
                // Not Level 5 (or detection failed); try the next candidate. No need to Cancel
                // first since nothing was ever selected.
                fusing.state =
                    State::FindInCollectionList(Timeout::default(), index + 1, None, false);
                return;
            }

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
                .detect_region_template(&icon, region)
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
                debug!(target: "backend/player", "fusion button right after selecting target {}: {button:?}", fusing.target_index);
                if matches!(button, Ok((_, FusionButton::RankUp))) {
                    State::RankingUp(Timeout::default())
                } else if resources.detector().detect_familiar_cards().is_empty() {
                    // No in-collection duplicates to select at all - clicking Select All would
                    // be a pointless no-op that only delays reaching the Inventory, which is the
                    // only place fodder can come from in this case.
                    debug!(
                        target: "backend/player",
                        "no in-collection duplicates for target {}, going straight to inventory",
                        fusing.target_index
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
            let bbox = resources.detector().detect_familiar_fusion_cancel_button();
            debug!(target: "backend/player", "fusion cancel button: {bbox:?}");
            if let Ok(bbox) = bbox {
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
                .detect_region_template(&icon, region)
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
            let bbox = resources
                .detector()
                .detect_familiar_fusion_select_all_button();
            debug!(target: "backend/player", "fusion select all button: {bbox:?}");
            if let Ok(bbox) = bbox {
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
                fusing.target_index, fusing.round
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
                            fusing.target_index
                        );
                    }
                    fusing.state = State::ConfirmFuse(Timeout::default(), round_cap_hit, false);
                }
                Err(_) => {
                    // Nothing selected, meaning in-collection fodder is exhausted (or all
                    // remaining duplicates are locked and cannot be consumed); top up from bag.
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
            fusing.state = if round_cap_hit {
                State::NextTarget
            } else {
                // Some rank-up transitions (confirmed via live testing: Rare -> Epic) never show
                // a distinct `Rank Up` button at all - the very last `Fuse` click that completes
                // the gauge silently performs the rank-up itself, and its confirmation popup gets
                // clicked through by this same generic ConfirmFuse handling, without ever passing
                // through State::RankingUp/ConfirmRankUp. Left unchecked, the target then looks
                // like a fresh, un-ranked-up familiar of its *new* rarity and just keeps getting
                // fed - it did rank up, but nothing noticed, so it never stopped.
                let target_rarity = fusing.target_rarities[fusing.target_index];
                let gauge_rarity = resources.detector().detect_familiar_fusion_gauge_rarity();
                let already_ranked_up = matches!(
                    (target_rarity, &gauge_rarity),
                    (FamiliarRarity::Rare, Ok(FusionGaugeRarity::Epic))
                        | (FamiliarRarity::Epic, Ok(FusionGaugeRarity::Unique))
                );
                debug!(
                    target: "backend/player",
                    "target {} gauge rarity after fuse: {gauge_rarity:?}, already ranked up: {already_ranked_up}",
                    fusing.target_index
                );
                if already_ranked_up {
                    fusing
                        .ranked_up_icons
                        .push(fusing.icons[fusing.target_index].clone());
                    State::DeselectRankedUp(Timeout::default())
                } else {
                    // The just-confirmed Fuse can by itself have completed the gauge, making Rank
                    // Up immediately available - go back through Fusing to re-check the button
                    // rather than assuming more fodder is needed. Fusing's own Err branch (nothing
                    // left selected, since Select All already grabbed every in-collection
                    // duplicate before this round) is what actually falls through to
                    // State::OpenInventory - not this state blind-jumping there and skipping the
                    // Rank Up check entirely.
                    State::Fusing(Timeout::default())
                }
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
        debug!(target: "backend/player", "no inventory key configured, giving up on target {}", fusing.target_index);
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

            debug!(target: "backend/player", "inventory failed to open, giving up on target {}", fusing.target_index);
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
/// 30 ticks to give it more real time to settle before giving up.
fn update_focus_use_tab(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::FocusUseTab(timeout, clicked) = fusing.state else {
        panic!("familiars fusing state is not focusing use tab");
    };

    match next_timeout_lifecycle(timeout, 30) {
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
            fusing.target_index
        );
        // Exhausted the conventional fodder slots; give up topping up this target.
        fusing.state = State::CloseInventory(Timeout::default(), false);
        return;
    }

    let Ok(use_tab_bbox) = resources.detector().detect_inventory_use_tab_button() else {
        debug!(
            target: "backend/player",
            "use tab button not found while scanning inventory, giving up on target {}",
            fusing.target_index
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
        Lifecycle::Ended => {
            let region = full_frame_region(resources);
            let icon = fusing.icons[fusing.target_index].clone();
            let matched = resources
                .detector()
                .detect_region_template(&icon, region)
                .into_iter()
                .next()
                .is_some();
            debug!(target: "backend/player", "inventory slot {index} matches target fodder: {matched}");

            fusing.state = if matched {
                State::DoubleClickItem(Timeout::default(), inventory_slot_rect(use_tab_bbox, index))
            } else {
                State::FindFodderItem(Timeout::default(), index + 1)
            };
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::FindFodderItem(timeout, index);
        }
    }
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
/// If the reported rarity still matches the tier fusion started from, the click didn't take and
/// reset the gauge to empty, so fodder must be fed again from scratch - since `Select All` already
/// grabbed every in-collection duplicate before the failed attempt, that source is exhausted and
/// the only way to refill is via [`State::OpenInventory`].
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
                fusing.target_index
            );

            fusing.state = if ranked_up {
                fusing
                    .ranked_up_icons
                    .push(fusing.icons[fusing.target_index].clone());
                State::DeselectRankedUp(Timeout::default())
            } else {
                fusing.round += 1;
                if fusing.round >= MAX_FUSE_ROUNDS {
                    debug!(
                        target: "backend/player",
                        "target {} hit MAX_FUSE_ROUNDS while ranking up, giving up",
                        fusing.target_index
                    );
                    // Safety valve against a stuck loop; move on to the next target.
                    State::NextTarget
                } else {
                    State::OpenInventory(Timeout::default())
                }
            };
        }
    }
}

/// Clicks `Cancel` to deselect the just-ranked-up familiar - it changed rarity but is still
/// selected as the base card, and per [`update_cancel_candidate`]'s docs the game silently
/// ignores clicking a different candidate while one is already selected. Without this, the next
/// target's search would never find (or select) anything and give no indication why.
fn update_deselect_ranked_up(resources: &mut Resources, fusing: &mut FamiliarsFusing) {
    let State::DeselectRankedUp(timeout) = fusing.state else {
        panic!("familiars fusing state is not deselecting ranked up familiar");
    };

    match next_timeout_lifecycle(timeout, 10) {
        Lifecycle::Started(timeout) => {
            let bbox = resources.detector().detect_familiar_fusion_cancel_button();
            debug!(target: "backend/player", "deselect ranked up familiar, cancel button: {bbox:?}");
            if let Ok(bbox) = bbox {
                let (x, y) = bbox_click_point(bbox);
                resources.input.send_mouse(x, y, MouseKind::Click);
            }
            fusing.state = State::DeselectRankedUp(timeout);
        }
        Lifecycle::Ended => {
            fusing.state = State::NextTarget;
        }
        Lifecycle::Updated(timeout) => {
            fusing.state = State::DeselectRankedUp(timeout);
        }
    }
}

fn update_next_target(fusing: &mut FamiliarsFusing) {
    fusing.target_index += 1;
    fusing.round = 0;
    fusing.state = if fusing.target_index < fusing.targets.len() {
        State::OpenFusionTab(Timeout::default())
    } else {
        debug!(target: "backend/player", "all fusion targets processed, completing");
        State::Completing(Timeout::default(), false)
    };
    debug!(
        target: "backend/player",
        "advancing to fusion target {} of {}",
        fusing.target_index,
        fusing.targets.len()
    );
}

#[inline]
fn update_completing(
    resources: &mut Resources,
    fusing: &mut FamiliarsFusing,
    timeout: Timeout,
    completed: bool,
) {
    match next_timeout_lifecycle(timeout, 20) {
        Lifecycle::Started(timeout) | Lifecycle::Updated(timeout) => {
            fusing.state = State::Completing(timeout, completed);
        }
        Lifecycle::Ended => {
            if resources.detector().detect_familiar_menu_opened() {
                resources.input.send_key(KeyKind::Esc);
            }
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
    use crate::{array::Array, bridge::MockInput, detect::MockDetector};

    fn new_fusing() -> FamiliarsFusing {
        let mut rarities = Array::new();
        rarities.push(FamiliarRarity::Rare);
        FamiliarsFusing::new(rarities, SwappableFamiliars::All, Array::new())
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

        update_open_menu(&mut resources, &mut fusing, KeyKind::A);

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

        update_open_menu(&mut resources, &mut fusing, KeyKind::A);

        assert!(!fusing.success);
        assert_matches!(fusing.state, State::Completing(_, false));
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

        assert_eq!(fusing.targets.len(), 1);
        assert_eq!(fusing.targets[0], bbox);
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
    fn update_check_equipped_slot_level_moves_to_fusion_tab_when_targets_found() {
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.targets.push(Rect::default());
        fusing.pending_slots.push((Rect::default(), false));
        fusing.state = State::CheckEquippedSlotLevel(Timeout::default(), 1);

        update_check_equipped_slot_level(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::OpenFusionTab(_));
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

        assert_matches!(fusing.state, State::DeselectRankedUp(_));
    }

    #[test]
    fn update_confirm_fuse_advances_to_next_target_when_round_cap_hit() {
        let mut resources = Resources::new(None, None);

        let mut fusing = new_fusing();
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
    fn update_next_target_advances_or_completes() {
        let mut fusing = new_fusing();
        fusing.targets.push(Rect::default());
        fusing.targets.push(Rect::default());
        fusing.target_index = 0;

        update_next_target(&mut fusing);
        assert_matches!(fusing.state, State::OpenFusionTab(_));
        assert_eq!(fusing.target_index, 1);

        update_next_target(&mut fusing);
        assert_matches!(fusing.state, State::Completing(_, false));
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
        fusing.state = State::ConfirmRankUp(
            Timeout {
                current: 20,
                started: true,
                ..Default::default()
            },
            true,
        );

        update_confirm_rank_up(&mut resources, &mut fusing);

        assert_matches!(fusing.state, State::DeselectRankedUp(_));
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
        assert_matches!(fusing.state, State::OpenInventory(_));
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
        fusing.state = State::DeselectRankedUp(Timeout::default());

        update_deselect_ranked_up(&mut resources, &mut fusing);
        assert_matches!(fusing.state, State::DeselectRankedUp(_));

        fusing.state = State::DeselectRankedUp(Timeout {
            current: 10,
            started: true,
            ..Default::default()
        });
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

        assert_matches!(fusing.state, State::FindInCollectionList(_, 0, _, false));
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
            .returning(move |_, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("fusion button not found")));
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(|| vec![(Rect::default(), FamiliarRank::Epic)]);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(Mat::default()));
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
            .returning(move |_, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Epic));
        detector
            .expect_detect_familiar_fusion_button()
            .once()
            .returning(|| Err(anyhow::anyhow!("fusion button not found")));
        detector
            .expect_detect_familiar_cards()
            .once()
            .returning(Vec::new);
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(Mat::default()));
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
            .returning(move |_, _| vec![icon_bbox]);
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
        fusing.icons.push(Arc::new(Mat::default()));
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
            .returning(move |_, _| vec![icon_bbox]);
        detector
            .expect_detect_familiar_fusion_gauge_rarity()
            .once()
            .returning(|| Ok(FusionGaugeRarity::Unique));
        let mut resources = Resources::new(None, Some(detector));

        let mut fusing = new_fusing();
        fusing.target_rarities.push(FamiliarRarity::Epic);
        fusing.icons.push(Arc::new(Mat::default()));
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
        let mut resources = Resources::new(None, None);
        let mut fusing = new_fusing();
        fusing.state = State::FindInCollectionList(
            Timeout {
                current: CANDIDATE_TIMEOUT,
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
            .returning(|_, _| Vec::new());
        let mut resources = Resources::new(None, Some(detector));
        let mut fusing = new_fusing();
        fusing.icons.push(Arc::new(Mat::default()));
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
        keys.expect_send_mouse().once();
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
                current: 30,
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
