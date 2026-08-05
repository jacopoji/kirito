use std::{
    collections::HashMap,
    env,
    fmt::Debug,
    sync::{Arc, LazyLock, Mutex},
};

use anyhow::{Result, anyhow, bail};
use base64::{Engine, prelude::BASE64_STANDARD};
use log::{debug, error, info};
#[cfg(test)]
use mockall::automock;
use opencv::{
    boxed_ref::BoxedRef,
    core::{
        BORDER_CONSTANT, CMP_EQ, CMP_GT, CV_8U, CV_32FC3, CV_32S, Mat, MatExprTraitConst, MatTrait,
        MatTraitConst, MatTraitConstManual, ModifyInplace, Point, Range, Rect, Scalar, Size,
        ToInputArray, Vec3b, Vector, add, add_weighted_def, bitwise_and_def, compare,
        copy_make_border, divide2_def, extract_channel, find_non_zero, min_max_loc, no_array,
        subtract_def, transpose_nd,
    },
    imgcodecs::{self, IMREAD_COLOR, IMREAD_GRAYSCALE, imdecode, imencode_def},
    imgproc::{
        CC_STAT_AREA, CC_STAT_HEIGHT, CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
        CHAIN_APPROX_SIMPLE, COLOR_BGR2HSV_FULL, COLOR_BGR2RGB, COLOR_BGRA2BGR, COLOR_BGRA2GRAY,
        INTER_CUBIC, INTER_LINEAR, MORPH_RECT, RETR_EXTERNAL, THRESH_BINARY, TM_CCOEFF_NORMED,
        TM_SQDIFF_NORMED, bounding_rect, connected_components_with_stats, contour_area,
        cvt_color_def, dilate_def, find_contours_def, get_structuring_element_def, match_template,
        min_area_rect, min_enclosing_triangle, resize, threshold,
    },
};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, SessionInputValue, SessionOutputs},
    value::TensorRef,
};
use strsim::jaro_winkler;

use crate::mat::OwnedMat;
use crate::{bridge::KeyKind, models::Localization};

#[derive(Debug, Copy, Clone)]
pub struct Arrow {
    pub key: KeyKind,
    pub region: Rect,
}

/// Struct for storing information about the spinning arrows.
#[derive(Debug, Copy, Clone)]
pub struct SpinArrow {
    /// The centroid of the spinning arrow relative to the whole image.
    pub centroid: Point,
    /// The region of the spinning arrow relative to the whole image.
    pub region: Rect,
    /// The last arrow head relative to the centroid.
    pub last_arrow_head: Option<Point>,
    /// Final result of spinning arrow.
    pub final_arrow: Option<KeyKind>,

    #[cfg(debug_assertions)]
    pub last_last_arrow_head: Option<Point>,
}

#[derive(Clone, Copy, Debug)]
pub enum OtherPlayerKind {
    Guildie,
    Stranger,
    Friend,
}

#[derive(Debug)]
pub enum FamiliarLevel {
    Level5,
    LevelOther,
}

#[derive(Debug)]
pub enum FamiliarRank {
    Rare,
    Epic,
}

#[derive(Debug)]
pub enum FusionButton {
    Fuse,
    RankUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionGaugeRarity {
    Rare,
    Epic,
    Unique,
}

#[derive(Debug)]
pub enum BuffKind {
    Rune,
    Familiar,
    SayramElixir,
    AureliaElixir,
    ExpCouponX2,
    ExpCouponX3,
    ExpCouponX4,
    BonusExpCoupon,
    MvpBonusExpCoupon,
    LegionWealth,
    LegionLuck,
    WealthAcquisitionPotion,
    ExpAccumulationPotion,
    SmallWealthAcquisitionPotion,
    SmallExpAccumulationPotion,
    ForTheGuild,
    HardHitter,
    ExtremeRedPotion,
    ExtremeBluePotion,
    ExtremeGreenPotion,
    ExtremeGoldPotion,
}

#[derive(Debug)]
pub enum QuickSlotsHexaBooster {
    Available,
    Unavailable,
}

#[derive(Debug)]
pub enum SolErda {
    Full,
    AtLeastOne,
    Empty,
}

/// A trait for detecting objects from provided frame.
#[cfg_attr(test, automock)]
pub trait Detector: Debug + Send + Sync {
    /// Gets the original [`OwnedMat`] as a [`BoxedRef`].
    fn mat(&self) -> BoxedRef<'_, Mat>;

    /// Gets the grayscale version.
    fn grayscale(&self) -> &Mat;

    /// Detects a list of mobs.
    ///
    /// Returns a list of mobs coordinate relative to minimap coordinate.
    fn detect_mobs(
        &self,
        minimap: Rect,
        bound: Rect,
        player: Point,
        player_name: Option<String>,
    ) -> Result<Vec<Point>>;

    /// Detects whether to press ESC for unstucking.
    fn detect_esc_settings(&self) -> bool;

    /// Detects the popup `Confirm` button.
    fn detect_popup_confirm_button(&self) -> Result<Rect>;

    /// Detects the new popup `OK` button.
    fn detect_popup_ok_new_button(&self) -> Result<Rect>;

    /// Detects an NPC dialog's `Next` or `End Chat` button, e.g. the Monster Park completion
    /// reward dialog.
    fn detect_popup_dialog_continue_button(&self) -> Result<Rect>;

    /// Detects whether the current on-screen map is Monster Park's entry lobby, as opposed to
    /// one of the run's actual stages.
    fn detect_monster_park_entry_map(&self) -> bool;

    /// Detects the Monster Park dungeon-select dialog's `MONSTER PARK` title banner.
    ///
    /// Sits in the dialog's header, above where an NPC info popup (e.g. "You cannot enter anymore
    /// today") can be shown on top of the dialog, and stays visible for as long as the dialog
    /// itself is open regardless of what's overlaid on it. Use this to check whether the dialog is
    /// still open at all.
    fn detect_monster_park_dialog(&self) -> bool;

    /// Detects the Monster Park dungeon-select dialog's `Enter` button.
    fn detect_monster_park_enter_button(&self) -> Result<Rect>;

    /// Detects the center-screen `CLEAR` banner shown right as a Monster Park stage's last enemy
    /// dies (stages 1-5). A faster, more direct signal than debouncing the minimap's enemy dots
    /// down to zero.
    fn detect_monster_park_clear_text(&self) -> bool;

    /// Detects the "You've cleared all the stages. Please exit through the portal." banner shown
    /// only on Monster Park's final stage. Distinguishes a genuine run completion from an
    /// ordinary per-stage clear.
    fn detect_monster_park_run_complete_text(&self) -> bool;

    /// Detects the `Monster Park Shuttle` NPC in the entry lobby, used to leave once no free
    /// entries are left for the day.
    ///
    /// Falls back through half-of-the-vehicle templates (mirroring [`Self::detect_player`]'s
    /// approach) since the shuttle can be partially scrolled off-screen or have its lower half
    /// occluded by other players/NPCs standing in front of it.
    fn detect_monster_park_shuttle(&self) -> Result<Rect>;

    /// Detects the shuttle's NPC dialog `YES` button (leave for town).
    fn detect_monster_park_shuttle_yes_button(&self) -> Result<Rect>;

    /// Detects whether there is an elite boss bar.
    fn detect_elite_boss_bar(&self) -> bool;

    /// Detects the minimap.
    ///
    /// The `border_threshold` determines the "whiteness" (grayscale value from 0..255) of
    /// the minimap's white border.
    fn detect_minimap(&self, border_threshold: u8) -> Result<Rect>;

    /// Detects the portals from the given `minimap` rectangle.
    ///
    /// Returns `Rect` relative to `minimap` coordinate.
    fn detect_minimap_portals(&self, minimap: Rect) -> Vec<Rect>;

    /// Detects Monster Park enemies (red dots) from the given `minimap` rectangle.
    ///
    /// Returns `Rect`s relative to `minimap` coordinate.
    fn detect_minimap_monster_park_enemies(&self, minimap: Rect) -> Vec<Rect>;

    /// Detects the rune from the given `minimap` rectangle.
    ///
    /// Returns `Rect` relative to `minimap` coordinate.
    fn detect_minimap_rune(&self, minimap: Rect) -> Result<Rect>;

    /// Detects the player in the provided `minimap` rectangle.
    ///
    /// Returns `Rect` relative to `minimap` coordinate.
    fn detect_player(&self, minimap: Rect) -> Result<Rect>;

    /// Detects the player's name.
    ///
    /// Only English name is currently supported.
    fn detect_player_name(&self) -> Result<String>;

    /// Detects whether a player of `kind` is in the minimap.
    fn detect_player_kind(&self, minimap: Rect, kind: OtherPlayerKind) -> bool;

    /// Detects whether the player is dead.
    fn detect_player_is_dead(&self) -> bool;

    /// Detects whether the player is in cash shop.
    fn detect_player_in_cash_shop(&self) -> bool;

    /// Detects the player health bar.
    ///
    /// This is the biggest red health bar below the name.
    fn detect_player_health_bar(&self) -> Result<Rect>;

    /// Detects the player current and max health bars.
    ///
    /// These are the two smaller bars extracted from `health_bar`.
    fn detect_player_current_max_health_bars(&self, health_bar: Rect) -> Result<(Rect, Rect)>;

    /// Detects the player current health and max health.
    fn detect_player_health(&self, current_bar: Rect, max_bar: Rect) -> Result<(u32, u32)>;

    /// Detects whether the player has a buff specified by `kind`.
    fn detect_player_buff(&self, kind: BuffKind) -> bool;

    /// Detects arrows from the given RGBA `Mat` image.
    fn detect_rune_arrows(&self, ignore: Vec<Rect>) -> Vec<Arrow>;

    fn detect_rune_initial_spin_arrows(&self) -> Vec<SpinArrow>;

    fn detect_rune_spin_arrow(&self, arrow: SpinArrow) -> SpinArrow;

    /// Detects the Erda Shower skill from the given BGRA `Mat` image.
    fn detect_erda_shower(&self) -> Result<Rect>;

    /// Detects familiar menu save button.
    fn detect_familiar_save_button(&self) -> Result<Rect>;

    /// Detects familiar menu level button.
    fn detect_familiar_level_button(&self) -> Result<Rect>;

    /// Detects the familiar slots assuming the familiar menu opened.
    ///
    /// Returns a pair of `(Rect, bool)` with `bool` of `true` indicating the slot is free.
    fn detect_familiar_slots(&self) -> Vec<(Rect, bool)>;

    /// Detects whether the familiar slot is free.
    fn detect_familiar_slot_is_free(&self, slot: Rect) -> bool;

    /// Detects the currently mouse hovering familiar level.
    fn detect_familiar_hover_level(&self) -> Result<FamiliarLevel>;

    /// Detects all the familiar cards assuming the familiar menu opened.
    fn detect_familiar_cards(&self) -> Vec<(Rect, FamiliarRank)>;

    /// Detects all the familiar card *shapes* assuming the familiar menu opened, without
    /// requiring a specific rarity classification.
    ///
    /// Unlike [`Self::detect_familiar_cards`], this also finds Common-rarity cards - the
    /// rarity-classifying templates only cover Rare/Epic, so Common fusion material (e.g. the
    /// grid of same-species duplicates fed as fodder, which is typically Common rather than the
    /// Rare/Epic tier being fused) would otherwise never be detected at all.
    fn detect_familiar_card_boxes(&self) -> Vec<Rect>;

    /// Detects familiar menu setup's tab scrollbar assuming familiar menu opened.
    fn detect_familiar_scrollbar(&self) -> Result<Rect>;

    /// Detects whether the familiar menu is opened.
    fn detect_familiar_menu_opened(&self) -> bool;

    /// Detects whether the familiar essence depleted assuming already buffed.
    fn detect_familiar_essence_depleted(&self) -> bool;

    /// Detects the rarity of a familiar occupying the given equipped Setup-tab slot (or the
    /// Fusion tab's selected-preview card, which renders identically) via its colored border
    /// (blue/Rare, purple/Epic, orange/Unique), rather than card art matching.
    ///
    /// The card-art-based rarity classification used for small list-view cards does not work
    /// here - verified against real screenshots, it fails entirely (`Err`) against this larger
    /// card style. `Err` also covers an already-Unique familiar, which isn't a valid fusion
    /// source.
    fn detect_familiar_slot_rank(&self, bbox: Rect) -> Result<FamiliarRank>;

    /// Detects whether a familiar occupying the given equipped Setup-tab slot is Unique rarity -
    /// the counterpart to [`Self::detect_familiar_slot_rank`]'s `Err` case, which discards this
    /// distinction since [`FamiliarRank`] has no `Unique` variant (nothing fuses *from* Unique).
    /// Used by familiar swapping instead, to free a Unique slot regardless of level.
    fn detect_familiar_slot_is_unique(&self, bbox: Rect) -> bool;

    /// Detects familiar menu's Fusion tab button.
    fn detect_familiar_fusion_tab_button(&self) -> Result<Rect>;

    /// Detects familiar menu's Setup tab button in its *inactive*-state appearance - used to
    /// switch back to the Setup tab when the familiar menu is already open on a different tab
    /// (e.g. left on the Fusion tab by a chained familiar-fusion run), instead of closing and
    /// re-opening the whole menu. Setup is necessarily inactive in that situation (some other tab
    /// is the active one), which is the only context this is ever used in.
    ///
    /// Live testing showed the tab's *active*-state appearance (tried first, since that template
    /// already existed) scores only ~0.915 against the inactive-state rendering actually on
    /// screen here - just under the 0.95 threshold needed to tell Setup apart from the
    /// visually-similar Info/Extraction/Fusion tab buttons - so every click meant to switch onto
    /// Setup from another tab was silently missing. A template captured from the inactive
    /// rendering itself is used instead.
    fn detect_familiar_setup_tab_inactive_button(&self) -> Result<Rect>;

    /// Detects whether the familiar menu's Fusion tab is currently opened.
    fn detect_familiar_fusion_tab_opened(&self) -> bool;

    /// Detects the Fusion tab's `Select All` button.
    fn detect_familiar_fusion_select_all_button(&self) -> Result<Rect>;

    /// Detects the familiar collection list's `Level` column sort header button (shared by the
    /// Setup tab's swap-candidate list and the Fusion tab's collection list).
    fn detect_familiar_level_sort_button(&self) -> Result<Rect>;

    /// Detects the Fusion tab's `Cancel` button.
    fn detect_familiar_fusion_cancel_button(&self) -> Result<Rect>;

    /// Detects the Fusion Gauge's diamond icon rarity color (blue = Rare, purple = Epic,
    /// orange/gold = Unique), assuming a familiar is currently selected in the Fusion tab.
    ///
    /// Used to confirm whether clicking `Rank Up` actually succeeded - the diamond's color
    /// switches to the new rarity tier immediately - since "guaranteed" rank-up isn't actually a
    /// 100% guarantee. `Err` when no familiar is selected (no diamond shown at all, e.g. gauge
    /// reads 0/0).
    fn detect_familiar_fusion_gauge_rarity(&self) -> Result<FusionGaugeRarity>;

    /// Detects the Fusion tab's `Fuse`/`Rank Up` button bbox and which state it is in.
    fn detect_familiar_fusion_button(&self) -> Result<(Rect, FusionButton)>;

    /// Detects whether the Inventory window is opened.
    fn detect_inventory_opened(&self) -> bool;

    /// Detects the Inventory window's `Use` tab button while it is the currently focused tab
    /// (highlighted).
    fn detect_inventory_use_tab_button(&self) -> Result<Rect>;

    /// Detects the Inventory window's `Use` tab button while it is *not* the currently focused
    /// tab, so it can be clicked to focus it. The Inventory can open on whichever tab was last
    /// active - live testing showed it isn't always `Use` - so this and
    /// [`Detector::detect_inventory_use_tab_button`] together let the click-to-focus flow tell
    /// "not focused yet" apart from "already focused."
    fn detect_inventory_use_tab_inactive_button(&self) -> Result<Rect>;

    /// Detects the "how many do you want to use?" quantity dialog's text input field, assuming
    /// the Inventory window is opened.
    fn detect_inventory_use_quantity_dialog(&self) -> Result<Rect>;

    /// Captures a BGR crop of the given region of the current frame for later use as an ad-hoc
    /// template with [`Detector::detect_region_template`].
    ///
    /// This is used to find a specific familiar's icon across different menus/windows (e.g. the
    /// Fusion tab's collection list or the Inventory's `Use` tab) where no fixed, precompiled
    /// template can exist because the icon differs per familiar type.
    fn capture_region(&self, region: Rect) -> Result<Mat>;

    /// Finds occurrences of an ad-hoc `template` (from [`Detector::capture_region`]) within
    /// `region` of the current frame.
    fn detect_region_template(&self, template: &Mat, region: Rect) -> Vec<Rect>;

    /// Detects whether the change channel menu is opened.
    fn detect_change_channel_menu_opened(&self) -> bool;

    /// Detects whether the admin image is visible inside the currently opened popup/dialog.
    fn detect_admin_visible(&self) -> bool;

    /// Detects whether there is a timer (e.g. from using booster).
    fn detect_timer_visible(&self) -> bool;

    /// Detects the transparent shape's lie detector popup.
    fn detect_lie_detector_shape(&self) -> Result<Rect>;

    /// Detects whether transparent shape's lie detector is preparing.
    fn detect_lie_detector_shape_preparing(&self) -> bool;

    /// Detects the violetta's lie detector popup.
    fn detect_lie_detector_violetta(&self) -> Result<Rect>;

    /// Detects whether violetta's lie detector is preparing.
    fn detect_lie_detector_violetta_preparing(&self) -> bool;

    /// Detects the state for HEXA Booster in the quick slots.
    fn detect_quick_slots_hexa_booster(&self) -> Result<QuickSlotsHexaBooster>;

    /// Detects the HEXA icon in quick menu.
    fn detect_hexa_quick_menu(&self) -> Result<Rect>;

    /// Detects the `Erda conversion` button in HEXA matrix.
    fn detect_hexa_erda_conversion_button(&self) -> Result<Rect>;

    /// Detects the `HEXA Booster` button in `Erda conversion` menu.
    fn detect_hexa_booster_button(&self) -> Result<Rect>;

    /// Detects the `MAX` button in `Erda conversion` menu.
    fn detect_hexa_max_button(&self) -> Result<Rect>;

    /// Detects the `Convert` button in `Erda conversion` menu.
    fn detect_hexa_convert_button(&self) -> Result<Rect>;

    /// Detects the Sol Erda state from the tracker menu.
    fn detect_hexa_sol_erda(&self) -> Result<SolErda>;

    /// Detects a list of transparent shapes during lie detector event.
    ///
    /// The returned [`Rect`]s have coordinates relative to `region`.
    fn detect_transparent_shapes(&self, region: Rect) -> Vec<(Rect, f32)>;

    /// Detects a list of mushrooms during Violetta lie detector event.
    ///
    /// The returned [`Rect`]s have coordinates relative to `region`.
    fn detect_violetta_mushrooms(&self, region: Rect) -> Vec<(Rect, f32)>;

    /// Detects violetta's face.
    fn detect_violetta_face(&self, region: Rect) -> Result<Rect>;

    /// Detects violetta's number boxes.
    fn detect_violetta_numbers(&self, region: Rect) -> Vec<Rect>;

    /// Detects the world map UI's `WORLD MAP` title.
    ///
    /// Used both to check whether the world map is currently open and as a fixed anchor to
    /// locate the rest of its elements (region/area dropdowns, map content area) via offsets.
    fn detect_world_map_title(&self) -> Result<Rect>;

    /// Detects the "Click to open the World Map" button in the minimap's title bar.
    ///
    /// Used by [`crate::player::navigate_to_hunting_ground`] as a fallback for opening the world
    /// map when [`crate::player::state::PlayerConfiguration::world_map_key`] is not configured or
    /// pressing it didn't actually open the world map - unlike the key, this button always exists
    /// regardless of user keybind configuration.
    fn detect_minimap_world_map_button(&self) -> Result<Rect>;

    /// Detects a text label matching `text` (fuzzy, case-insensitive) within `roi`.
    ///
    /// Used both for world map dropdown options (region/area) and for a hunting ground's
    /// clickable node label on the map itself.
    fn detect_world_map_label(&self, roi: Rect, text: &str) -> Result<Rect>;

    /// Detects daily quest kill-count progress popups (e.g. `"Cernium Region Mob 28 / 100"`)
    /// currently on screen, returning each as `(current, target)`.
    ///
    /// Cheaply gated on a template match of the constant `"Region Mob"` substring rather than
    /// scanning a generous screen-space region with a full OCR pass every time - see the free
    /// function of the same name for why. Only `target`s of `100`, `300` or `500` are returned -
    /// the only values a daily quest's kill count can be - to reject unrelated OCR noise. Since
    /// multiple valid popups can be present at once (one per active daily quest, stacked
    /// vertically), callers should match the returned `target` against the specific
    /// [`crate::models::DailyQuestEntry::kill_target`] they are tracking.
    fn detect_daily_quest_progress_popup(&self) -> Vec<(u32, u32)>;

    /// Detects the "Quest complete!" toast banner shown briefly at the bottom of the screen when
    /// any quest (not necessarily a daily) finishes.
    ///
    /// Regular, non-daily quests raise the exact same banner, so callers must not treat a match
    /// alone as a daily quest completing - combine with [`Self::detect_world_map_label`] against
    /// `"[Daily Quest]"` on a region below the returned [`Rect`] (where the toast's second line,
    /// naming the specific quest, renders) to confirm it actually is one.
    fn detect_quest_complete_popup(&self) -> Result<Rect>;
}

type MatFn = Box<dyn FnOnce() -> Mat + Send>;

/// A detector that lazily transform `Mat`.
#[derive(Debug)]
pub struct DefaultDetector {
    bgra: Arc<OwnedMat>,
    bgr: LazyLock<Mat, MatFn>,
    grayscale: LazyLock<Mat, MatFn>,
    localization: Arc<Localization>,
}

impl DefaultDetector {
    /// Creates a default implementation of [`Detector`] from the given BGRA `mat`.
    pub fn new(mat: OwnedMat, localization: Arc<Localization>) -> Self {
        let bgra = Arc::new(mat);

        let cloned = bgra.clone();
        let bgr = LazyLock::<Mat, MatFn>::new(Box::new(move || to_bgr(&cloned.as_mat())));

        let cloned = bgra.clone();
        let grayscale =
            LazyLock::<Mat, MatFn>::new(Box::new(move || to_grayscale(&cloned.as_mat(), true)));

        Self {
            bgra,
            bgr,
            grayscale,
            localization,
        }
    }

    fn bgra(&self) -> BoxedRef<'_, Mat> {
        self.bgra.as_mat()
    }

    fn bgr(&self) -> &Mat {
        &self.bgr
    }
}

impl Detector for DefaultDetector {
    fn mat(&self) -> BoxedRef<'_, Mat> {
        self.bgra()
    }

    fn grayscale(&self) -> &Mat {
        &self.grayscale
    }

    fn detect_mobs(
        &self,
        minimap: Rect,
        bound: Rect,
        player: Point,
        player_name: Option<String>,
    ) -> Result<Vec<Point>> {
        detect_mobs(self.bgr(), minimap, bound, player, player_name)
    }

    fn detect_esc_settings(&self) -> bool {
        detect_esc_settings(self.bgr(), self.grayscale(), &self.localization)
    }

    fn detect_popup_confirm_button(&self) -> Result<Rect> {
        detect_popup_confirm_button(self.grayscale(), &self.localization)
    }

    fn detect_popup_ok_new_button(&self) -> Result<Rect> {
        detect_popup_ok_new_button(self.grayscale(), &self.localization)
    }

    fn detect_popup_dialog_continue_button(&self) -> Result<Rect> {
        detect_popup_next_button(self.grayscale(), &self.localization)
            .or_else(|_| detect_popup_end_chat_button(self.grayscale(), &self.localization))
    }

    fn detect_monster_park_entry_map(&self) -> bool {
        detect_monster_park_entry_map(self.bgr())
    }

    fn detect_monster_park_dialog(&self) -> bool {
        detect_monster_park_dialog(self.grayscale())
    }

    fn detect_monster_park_enter_button(&self) -> Result<Rect> {
        detect_monster_park_enter_button(self.grayscale())
    }

    fn detect_monster_park_clear_text(&self) -> bool {
        detect_monster_park_clear_text(self.grayscale())
    }

    fn detect_monster_park_run_complete_text(&self) -> bool {
        detect_monster_park_run_complete_text(self.grayscale())
    }

    fn detect_monster_park_shuttle(&self) -> Result<Rect> {
        detect_monster_park_shuttle(self.bgr())
    }

    fn detect_monster_park_shuttle_yes_button(&self) -> Result<Rect> {
        detect_monster_park_shuttle_yes_button(self.grayscale())
    }

    fn detect_elite_boss_bar(&self) -> bool {
        detect_elite_boss_bar(self.grayscale())
    }

    fn detect_minimap(&self, border_threshold: u8) -> Result<Rect> {
        detect_minimap(self.bgr(), border_threshold)
    }

    fn detect_minimap_portals(&self, minimap: Rect) -> Vec<Rect> {
        detect_minimap_portals(self.bgr().roi(minimap).unwrap())
    }

    fn detect_minimap_monster_park_enemies(&self, minimap: Rect) -> Vec<Rect> {
        detect_minimap_monster_park_enemies(self.bgr().roi(minimap).unwrap())
    }

    fn detect_minimap_rune(&self, minimap: Rect) -> Result<Rect> {
        detect_minimap_rune(&self.bgr().roi(minimap).unwrap())
    }

    fn detect_player(&self, minimap: Rect) -> Result<Rect> {
        detect_player(&self.bgr().roi(minimap).unwrap())
    }

    fn detect_player_name(&self) -> Result<String> {
        detect_player_name(self.bgr(), self.grayscale())
    }

    fn detect_player_kind(&self, minimap: Rect, kind: OtherPlayerKind) -> bool {
        detect_player_kind(&self.bgr().roi(minimap).unwrap(), kind)
    }

    fn detect_player_is_dead(&self) -> bool {
        detect_player_is_dead(self.grayscale())
    }

    fn detect_player_in_cash_shop(&self) -> bool {
        detect_player_in_cash_shop(self.grayscale(), &self.localization)
    }

    fn detect_player_health_bar(&self) -> Result<Rect> {
        detect_player_health_bar(self.grayscale())
    }

    fn detect_player_current_max_health_bars(&self, health_bar: Rect) -> Result<(Rect, Rect)> {
        detect_player_current_max_health_bars(self.bgr(), self.grayscale(), health_bar)
    }

    fn detect_player_health(&self, current_bar: Rect, max_bar: Rect) -> Result<(u32, u32)> {
        detect_player_health(self.bgr(), current_bar, max_bar)
    }

    fn detect_player_buff(&self, kind: BuffKind) -> bool {
        let mat = match kind {
            BuffKind::Rune
            | BuffKind::Familiar
            | BuffKind::SayramElixir
            | BuffKind::AureliaElixir
            | BuffKind::ExpCouponX2
            | BuffKind::ExpCouponX3
            | BuffKind::ExpCouponX4
            | BuffKind::BonusExpCoupon
            | BuffKind::MvpBonusExpCoupon
            | BuffKind::ForTheGuild
            | BuffKind::HardHitter => &to_buffs_region(self.grayscale()),
            BuffKind::LegionWealth
            | BuffKind::LegionLuck
            | BuffKind::WealthAcquisitionPotion
            | BuffKind::ExpAccumulationPotion
            | BuffKind::SmallWealthAcquisitionPotion
            | BuffKind::SmallExpAccumulationPotion
            | BuffKind::ExtremeRedPotion
            | BuffKind::ExtremeBluePotion
            | BuffKind::ExtremeGreenPotion
            | BuffKind::ExtremeGoldPotion => &to_buffs_region(self.bgr()),
        };
        detect_player_buff(mat, kind)
    }

    fn detect_rune_arrows(&self, ignore: Vec<Rect>) -> Vec<Arrow> {
        detect_rune_arrows(self.bgr(), ignore).unwrap_or_default()
    }

    fn detect_rune_initial_spin_arrows(&self) -> Vec<SpinArrow> {
        detect_rune_initial_spin_arrows(self.bgr())
    }

    fn detect_rune_spin_arrow(&self, mut arrow: SpinArrow) -> SpinArrow {
        let _ = detect_rune_spin_arrow(self.bgr(), &mut arrow);
        arrow
    }

    fn detect_erda_shower(&self) -> Result<Rect> {
        detect_erda_shower(self.grayscale())
    }

    fn detect_familiar_save_button(&self) -> Result<Rect> {
        detect_familiar_save_button(self.bgr(), &self.localization)
    }

    fn detect_familiar_level_button(&self) -> Result<Rect> {
        detect_familiar_level_button(self.bgr(), &self.localization)
    }

    fn detect_familiar_slots(&self) -> Vec<(Rect, bool)> {
        detect_familiar_slots(self.bgr())
    }

    fn detect_familiar_slot_is_free(&self, slot: Rect) -> bool {
        detect_familiar_slot_is_free(&self.bgr().roi(slot).unwrap())
    }

    fn detect_familiar_hover_level(&self) -> Result<FamiliarLevel> {
        detect_familiar_hover_level(self.bgr())
    }

    fn detect_familiar_cards(&self) -> Vec<(Rect, FamiliarRank)> {
        detect_familiar_cards(self.bgr())
    }

    fn detect_familiar_card_boxes(&self) -> Vec<Rect> {
        detect_familiar_card_boxes(self.bgr())
    }

    fn detect_familiar_scrollbar(&self) -> Result<Rect> {
        detect_familiar_scrollbar(&to_grayscale(&self.bgra(), false))
    }

    fn detect_familiar_menu_opened(&self) -> bool {
        detect_familiar_menu_opened(self.grayscale())
    }

    fn detect_familiar_essence_depleted(&self) -> bool {
        detect_familiar_essence_depleted(&to_buffs_region(self.grayscale()))
    }

    fn detect_familiar_slot_rank(&self, bbox: Rect) -> Result<FamiliarRank> {
        detect_familiar_slot_rank(&self.bgr().roi(bbox)?)
    }

    fn detect_familiar_slot_is_unique(&self, bbox: Rect) -> bool {
        self.bgr()
            .roi(bbox)
            .is_ok_and(|roi| detect_familiar_slot_is_unique(&roi))
    }

    fn detect_familiar_fusion_tab_opened(&self) -> bool {
        detect_familiar_fusion_tab_opened(self.grayscale())
    }

    fn detect_familiar_fusion_select_all_button(&self) -> Result<Rect> {
        detect_familiar_fusion_select_all_button(self.bgr())
    }

    fn detect_familiar_level_sort_button(&self) -> Result<Rect> {
        detect_familiar_level_sort_button(self.bgr())
    }

    fn detect_familiar_fusion_cancel_button(&self) -> Result<Rect> {
        detect_familiar_fusion_cancel_button(self.bgr())
    }

    fn detect_familiar_fusion_gauge_rarity(&self) -> Result<FusionGaugeRarity> {
        detect_familiar_fusion_gauge_rarity(self.bgr())
    }

    fn detect_familiar_fusion_button(&self) -> Result<(Rect, FusionButton)> {
        detect_familiar_fusion_button(self.bgr())
    }

    fn detect_inventory_opened(&self) -> bool {
        detect_inventory_opened(self.grayscale())
    }

    fn detect_inventory_use_tab_button(&self) -> Result<Rect> {
        detect_inventory_use_tab_button(self.bgr())
    }

    fn detect_inventory_use_tab_inactive_button(&self) -> Result<Rect> {
        detect_inventory_use_tab_inactive_button(self.bgr())
    }

        bgr,
        &*FAMILIAR_CARD_RARE_TEMPLATE,
        &*FAMILIAR_CARD_MASK_TEMPLATE,
        Point::default(),
        64,
        0.75,
    )
    .into_iter()
    .filter_map(|result| result.ok().map(|(bbox, _)| bbox))
    .collect::<Vec<_>>();

    cards.sort_by_key(|bbox| (bbox.y, bbox.x));
    cards
}

fn detect_familiar_cards<T: MatTraitConst + ToInputArray>(bgr: &T) -> Vec<(Rect, FamiliarRank)> {
    let cards = detect_familiar_card_boxes(bgr);

    let mut filtered = vec![];
    if cards.is_empty() {
        return filtered;
    }

    for card in cards {
        let roi = bgr.roi(card).unwrap();
        if let Ok(rank) = familiar_rank_of(&roi) {
            filtered.push((card, rank));
        }
    }
    filtered.sort_by_key(|(bbox, _)| (bbox.y, bbox.x));

    filtered
}

/// Minimum score for any single rarity border template to count as a real match - shared by
/// [`detect_familiar_slot_rank`] and [`detect_familiar_slot_is_unique`], which both classify the
/// same equipped-slot border via [`familiar_slot_rank_scores`].
const FAMILIAR_SLOT_RANK_MIN_SCORE: f64 = 0.5;

/// Rare/Epic/Unique border-color match scores for an equipped familiar slot ROI, shared by
/// [`detect_familiar_slot_rank`] and [`detect_familiar_slot_is_unique`] so both only need the 3
/// templates loaded/matched once each rather than duplicating the `LazyLock`s.
///
/// The border is a solid color band framing the whole card, so searching for a small pure-color
/// sample anywhere within the full slot/card ROI (rather than requiring an exact pre-aligned
/// position) is both simpler and more robust to minor position/size variance. Cross-rarity scores
/// are still fairly close (e.g. Epic's border template can score ~0.93 against a genuinely Rare
/// border), so - same as the fusion gauge diamond - callers pick whichever of the 3 candidates
/// scores best rather than thresholding one in isolation.
fn familiar_slot_rank_scores(roi: &impl ToInputArray) -> (f64, f64, f64) {
    static TEMPLATE_RARE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_SLOT_BORDER_RARE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });
    static TEMPLATE_EPIC: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_SLOT_BORDER_EPIC_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });
    static TEMPLATE_UNIQUE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_SLOT_BORDER_UNIQUE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    (
        template_match_score(roi, &*TEMPLATE_RARE),
        template_match_score(roi, &*TEMPLATE_EPIC),
        template_match_score(roi, &*TEMPLATE_UNIQUE),
    )
}

fn detect_familiar_slot_rank(roi: &impl ToInputArray) -> Result<FamiliarRank> {
    let (score_rare, score_epic, score_unique) = familiar_slot_rank_scores(roi);

    let best = [
        (Some(FamiliarRank::Rare), score_rare),
        (Some(FamiliarRank::Epic), score_epic),
        (None, score_unique),
    ]
    .into_iter()
    .max_by(|(_, a), (_, b)| a.total_cmp(b));

    match best {
        Some((Some(rank), score)) if score >= FAMILIAR_SLOT_RANK_MIN_SCORE => Ok(rank),
        _ => Err(anyhow!("familiar slot rank not found or already unique")),
    }
}

/// Detects whether an equipped familiar slot's border is Unique rarity.
///
/// Used by familiar swapping to free the slot regardless of the familiar's level - Unique is the
/// final fusion tier, so once reached there is nothing left to gain from keeping it equipped
/// (leveling further, per [`FamiliarLevel::Level5`]'s usual eligibility gate, no longer matters),
/// and leaving it in place only blocks a Rare/Epic candidate from taking its slot and receiving
/// further fusion instead.
fn detect_familiar_slot_is_unique(roi: &impl ToInputArray) -> bool {
    let (score_rare, score_epic, score_unique) = familiar_slot_rank_scores(roi);
    score_unique >= FAMILIAR_SLOT_RANK_MIN_SCORE
        && score_unique >= score_rare
        && score_unique >= score_epic
}

fn detect_familiar_fusion_tab_opened(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_TAB_OPENED_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    let result = detect_template_single(grayscale, &*TEMPLATE, no_array(), Point::default(), 0.75);
    debug!(target: "backend/detect", "familiar fusion tab opened match: {result:?}");
    result.is_ok()
}

fn detect_familiar_fusion_select_all_button(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_SELECT_ALL_BUTTON_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.75)
}

fn detect_familiar_level_sort_button(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_LEVEL_SORT_BUTTON_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.75)
}

fn detect_familiar_fusion_cancel_button(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_CANCEL_BUTTON_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.75)
}

fn detect_familiar_fusion_gauge_rarity(bgr: &impl ToInputArray) -> Result<FusionGaugeRarity> {
    static TEMPLATE_RARE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_GAUGE_DIAMOND_RARE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });
    static TEMPLATE_EPIC: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_GAUGE_DIAMOND_EPIC_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });
    static TEMPLATE_UNIQUE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_GAUGE_DIAMOND_UNIQUE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    // The 3 diamonds share the same shape/outline and differ mainly in color, so cross-matching
    // scores are all fairly high (e.g. the Epic template still scores ~0.9 against a Rare
    // diamond) - an absolute threshold on a single template is not discriminative enough here.
    // Instead, score against all 3 and pick whichever matches best, only bailing out (via the
    // floor) when no diamond is present at all (e.g. no familiar selected).
    const MIN_SCORE: f64 = 0.5;

    let score_rare = template_match_score(bgr, &*TEMPLATE_RARE);
    let score_epic = template_match_score(bgr, &*TEMPLATE_EPIC);
    let score_unique = template_match_score(bgr, &*TEMPLATE_UNIQUE);

    let best = [
        (FusionGaugeRarity::Rare, score_rare),
        (FusionGaugeRarity::Epic, score_epic),
        (FusionGaugeRarity::Unique, score_unique),
    ]
    .into_iter()
    .max_by(|(_, a), (_, b)| a.total_cmp(b));

    match best {
        Some((rarity, score)) if score >= MIN_SCORE => Ok(rarity),
        _ => Err(anyhow!("fusion gauge diamond not found")),
    }
}

#[inline]
fn template_match_score(mat: &impl ToInputArray, template: &impl ToInputArray) -> f64 {
    let mut result = Mat::default();
    let mut score = 0f64;
    match_template(mat, template, &mut result, TM_CCOEFF_NORMED, &no_array()).unwrap();
    min_max_loc(&result, None, Some(&mut score), None, None, &no_array()).unwrap();
    score
}

fn detect_familiar_fusion_button(bgr: &impl ToInputArray) -> Result<(Rect, FusionButton)> {
    static TEMPLATE_FUSE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_BUTTON_FUSE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });
    static TEMPLATE_RANK_UP: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_FUSION_BUTTON_RANK_UP_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    // Same region hosts either label depending on state; whichever template matches identifies
    // it. Trying Fuse first since it is the more commonly active state while the gauge fills up.
    if let Ok(bbox) = detect_template(bgr, &*TEMPLATE_FUSE, Point::default(), 0.75) {
        return Ok((bbox, FusionButton::Fuse));
    }
    if let Ok(bbox) = detect_template(bgr, &*TEMPLATE_RANK_UP, Point::default(), 0.75) {
        return Ok((bbox, FusionButton::RankUp));
    }

    Err(anyhow!("fusion button not found"))
}

fn detect_inventory_opened(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("INVENTORY_WINDOW_OPENED_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).is_ok()
}

fn detect_inventory_use_tab_button(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("INVENTORY_USE_TAB_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    // The active/inactive templates differ only in background tint (same "Use" text/shape), and
    // cross-match each other at ~0.86 - well above the usual 0.75 threshold used elsewhere in this
    // codebase - so a much stricter threshold is needed here to reliably tell the two states apart
    // (self-match is ~1.0 for both).
    detect_template(bgr, &*TEMPLATE, Point::default(), 0.9)
}

fn detect_inventory_use_tab_inactive_button(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("INVENTORY_USE_TAB_INACTIVE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.9)
}

fn detect_inventory_use_quantity_dialog(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("INVENTORY_USE_QUANTITY_DIALOG_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.75)
}

/// Finds occurrences of an ad-hoc `template` (captured at runtime via
/// [`Detector::capture_region`], since no familiar/item icon can be known ahead of time) within
/// `region` of the current frame.
fn detect_region_template(
    bgr: &(impl MatTraitConst + ToInputArray),
    template: &Mat,
    region: Rect,
) -> Vec<Rect> {
    let Ok(roi) = bgr.roi(region) else {
        return vec![];
    };

    let score = template_match_score(&roi, template);
    debug!(target: "backend/detect", "region template match score: {score}");

    // Live testing against the Fusion tab's selected-preview card showed a genuinely correct
    // match still only scoring ~0.81 (reproducible across runs) - likely a "ready to rank up"
    // glow/highlight the plain equipped-slot capture doesn't have - while a wrong candidate
    // scored ~0.38, a wide enough margin that 0.75 stayed safely discriminative for that
    // familiar. A different familiar with a more pronounced selection glow/particle effect
    // (confirmed via a real screenshot to be a genuinely correct match) scored only ~0.63-0.66,
    // against ~0.44-0.58 for wrong candidates in the same session - the margin is much tighter
    // here, so the threshold is lowered to 0.6 to accommodate it. This may need further per-
    // familiar tuning if a future case's genuine match falls even lower.
    detect_template_multiple(&roi, template, no_array(), region.tl(), 64, 0.6)
        .into_iter()
        .filter_map(|result| result.ok().map(|(bbox, _)| bbox))
        .collect()
}

fn detect_familiar_scrollbar(grayscale: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_SCROLLBAR_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.6)
}

fn detect_familiar_menu_opened(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_MENU_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).is_ok()
}

fn detect_familiar_essence_depleted(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("FAMILIAR_ESSENCE_DEPLETE_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.8).is_ok()
}

pub static CHANGE_CHANNEL_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("CHANGE_CHANNEL_MENU_TEMPLATE")),
        IMREAD_GRAYSCALE,
    )
    .unwrap()
});

fn detect_change_channel_menu_opened(
    grayscale: &impl ToInputArray,
    localization: &Localization,
) -> bool {
    let template = localization
        .change_channel_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, true).ok());

    detect_template(
        grayscale,
        template.as_ref().unwrap_or(&*CHANGE_CHANNEL_TEMPLATE),
        Point::default(),
        0.75,
    )
    .is_ok()
}

fn detect_admin_visible(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(include_bytes!(env!("ADMIN_TEMPLATE")), IMREAD_GRAYSCALE).unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).is_ok()
}

pub static TIMER_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(include_bytes!(env!("TIMER_TEMPLATE")), IMREAD_GRAYSCALE).unwrap()
});

fn detect_timer_visible(grayscale: &impl ToInputArray, localization: &Localization) -> bool {
    let template = localization
        .timer_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, true).ok());

    detect_template(
        grayscale,
        template.as_ref().unwrap_or(&*TIMER_TEMPLATE),
        Point::default(),
        0.75,
    )
    .is_ok()
}

pub static LIE_DETECTOR_TRANSPARENT_SHAPE_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("LIE_DETECTOR_NEW_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_lie_detector_shape(bgr: &impl ToInputArray, localization: &Localization) -> Result<Rect> {
    let template = localization
        .lie_detector_new_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template
            .as_ref()
            .unwrap_or(&*LIE_DETECTOR_TRANSPARENT_SHAPE_TEMPLATE),
        Point::default(),
        0.6,
    )
}

fn detect_lie_detector_shape_preparing(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("LIE_DETECTOR_SHAPE_PREPARE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.6)
}

pub static LIE_DETECTOR_VIOLETTA_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("LIE_DETECTOR_OLD_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_lie_detector_violetta(
    bgr: &impl ToInputArray,
    localization: &Localization,
) -> Result<Rect> {
    let template = localization
        .lie_detector_old_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template
            .as_ref()
            .unwrap_or(&*LIE_DETECTOR_VIOLETTA_TEMPLATE),
        Point::default(),
        0.6,
    )
}

#[allow(unused)]
fn detect_lie_detector_violetta_preparing(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("LIE_DETECTOR_VIOLETTA_PREPARE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.6)
}

fn detect_quick_slots_hexa_booster<T: MatTraitConst + ToInputArray>(
    grayscale: &T,
) -> Result<QuickSlotsHexaBooster> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_BOOSTER_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static TEMPLATE_NUMBER: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_BOOSTER_NUMBER_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static TEMPLATE_NUMBER_MASK: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_BOOSTER_NUMBER_MASK_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    let pad_height = TEMPLATE_NUMBER.size().unwrap().height;
    let booster_bbox =
        detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).map(|bbox| {
            let br = bbox.br();

            let x1 = bbox.x - 1;
            let x2 = br.x + 1;

            let y1 = bbox.y;
            let y2 = br.y + pad_height;

            Rect::new(x1, y1, x2 - x1, y2 - y1)
        })?;
    let booster = grayscale.roi(booster_bbox).expect("can extract roi");
    let has_booster = detect_template_single(
        &booster,
        &*TEMPLATE_NUMBER,
        &*TEMPLATE_NUMBER_MASK,
        Point::default(),
        0.8,
    )
    .is_err();

    if has_booster {
        Ok(QuickSlotsHexaBooster::Available)
    } else {
        Ok(QuickSlotsHexaBooster::Unavailable)
    }
}

fn detect_hexa_menu(grayscale: &impl ToInputArray) -> bool {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(include_bytes!(env!("HEXA_MENU_TEMPLATE")), IMREAD_GRAYSCALE).unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).is_ok()
}

fn detect_hexa_quick_menu(grayscale: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_QUICK_MENU_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75)
}

pub static HEXA_ERDA_CONVERSION_BUTTON_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("HEXA_BUTTON_ERDA_CONVERSION_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_hexa_erda_conversion_button(
    bgr: &impl ToInputArray,
    localization: &Localization,
) -> Result<Rect> {
    let template = localization
        .hexa_erda_conversion_button_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template
            .as_ref()
            .unwrap_or(&*HEXA_ERDA_CONVERSION_BUTTON_TEMPLATE),
        Point::default(),
        0.75,
    )
}

pub static HEXA_BOOSTER_BUTTON_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("HEXA_BUTTON_HEXA_BOOSTER_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_hexa_booster_button(
    bgr: &impl ToInputArray,
    localization: &Localization,
) -> Result<Rect> {
    let template = localization
        .hexa_booster_button_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template.as_ref().unwrap_or(&*HEXA_BOOSTER_BUTTON_TEMPLATE),
        Point::default(),
        0.75,
    )
}

pub static HEXA_MAX_BUTTON_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("HEXA_BUTTON_MAX_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_hexa_max_button(bgr: &impl ToInputArray, localization: &Localization) -> Result<Rect> {
    let template = localization
        .hexa_max_button_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template.as_ref().unwrap_or(&*HEXA_MAX_BUTTON_TEMPLATE),
        Point::default(),
        0.75,
    )
}

pub static HEXA_CONVERT_BUTTON_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
    imgcodecs::imdecode(
        include_bytes!(env!("HEXA_BUTTON_CONVERT_TEMPLATE")),
        IMREAD_COLOR,
    )
    .unwrap()
});

fn detect_hexa_convert_button(
    bgr: &impl ToInputArray,
    localization: &Localization,
) -> Result<Rect> {
    let template = localization
        .hexa_convert_button_base64
        .as_ref()
        .and_then(|base64| to_mat_from_base64(base64, false).ok());

    detect_template(
        bgr,
        template.as_ref().unwrap_or(&*HEXA_CONVERT_BUTTON_TEMPLATE),
        Point::default(),
        0.75,
    )
}

fn detect_hexa_sol_erda(grayscale: &impl ToInputArray) -> Result<SolErda> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_SOL_ERDA_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static FULL_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_SOL_ERDA_FULL_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static FULL_MASK_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_SOL_ERDA_FULL_MASK_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static EMPTY_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_SOL_ERDA_EMPTY_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });
    static EMPTY_MASK_TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("HEXA_SOL_ERDA_EMPTY_MASK_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    if detect_template_single(
        grayscale,
        &*FULL_TEMPLATE,
        &*FULL_MASK_TEMPLATE,
        Point::default(),
        0.8,
    )
    .is_ok()
    {
        return Ok(SolErda::Full);
    }

    if detect_template_single(
        grayscale,
        &*EMPTY_TEMPLATE,
        &*EMPTY_MASK_TEMPLATE,
        Point::default(),
        0.8,
    )
    .is_ok()
    {
        return Ok(SolErda::Empty);
    }

    if detect_template(grayscale, &*TEMPLATE, Point::default(), 0.75).is_ok() {
        return Ok(SolErda::AtLeastOne);
    }

    bail!("sol erda tracker menu not visible")
}

fn detect_transparent_shapes(bgr: &impl MatTraitConst) -> Vec<(Rect, f32)> {
    // The NMS-baked ONNX export's own confidence cutoff isn't controllable from
    // here (empirically, `model.export(..., conf=X)` doesn't change the graph
    // on the exporter version this was trained with), so low-confidence noise
    // must be filtered explicitly. 0.5 was picked from measuring false-positive
    // rates on a fine-tuned model whose export had no effective floor at all;
    // it clears the shipped model's own confidences with room to spare.
    const CONF_THRESHOLD: f32 = 0.5;

    static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
        Mutex::new(
            build_session(include_bytes!(env!("TRANSPARENT_SHAPE_MODEL")))
                .expect("build transparent shape detection session successfully"),
        )
    });

    let size = bgr.size().unwrap();
    let (mat_in, w_ratio, h_ratio, left, top) = preprocess_for_yolo(bgr);
    let mut model = MODEL.lock().unwrap();
    let result = model.run([to_input_value(&mat_in)]).unwrap();
    let mat_out = from_output_value(&result);

    (0..mat_out.rows())
        // SAFETY: 0..result.rows() is within Mat bounds
        .map(|i| unsafe { mat_out.at_row_unchecked::<f32>(i).unwrap() })
        .filter(|pred| pred[4] >= CONF_THRESHOLD)
        .map(|pred| {
            (
                remap_from_yolo(pred, size, w_ratio, h_ratio, left, top),
                pred[4],
            )
        })
        .collect()
}

fn detect_violetta_mushrooms(bgr: &impl MatTraitConst) -> Vec<(Rect, f32)> {
    static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
        Mutex::new(
            build_session(include_bytes!(env!("VIOLETTA_MODEL")))
                .expect("build violetta detection session successfully"),
        )
    });

    let size = bgr.size().unwrap();
    let (mat_in, w_ratio, h_ratio, left, top) = preprocess_for_yolo(bgr);
    let mut model = MODEL.lock().unwrap();
    let result = model.run([to_input_value(&mat_in)]).unwrap();
    let mat_out = from_output_value(&result);

    (0..mat_out.rows())
        // SAFETY: 0..result.rows() is within Mat bounds
        .map(|i| unsafe { mat_out.at_row_unchecked::<f32>(i).unwrap() })
        .map(|pred| {
            (
                remap_from_yolo(pred, size, w_ratio, h_ratio, left, top),
                pred[4],
            )
        })
        .collect()
}

fn detect_violetta_face(bgr: &impl ToInputArray) -> Result<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("LIE_DETECTOR_VIOLETTA_FACE_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    detect_template(bgr, &*TEMPLATE, Point::default(), 0.6)
}

fn detect_violetta_numbers(bgr: &impl ToInputArray) -> Vec<Rect> {
    static TEMPLATE: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("LIE_DETECTOR_VIOLETTA_NUMBER_TEMPLATE")),
            IMREAD_COLOR,
        )
        .unwrap()
    });

    static TEMPLATE_MASK: LazyLock<Mat> = LazyLock::new(|| {
        imgcodecs::imdecode(
            include_bytes!(env!("LIE_DETECTOR_VIOLETTA_NUMBER_MASK_TEMPLATE")),
            IMREAD_GRAYSCALE,
        )
        .unwrap()
    });

    let mut vec =
        detect_template_multiple(bgr, &*TEMPLATE, &*TEMPLATE_MASK, Point::default(), 4, 0.75)
            .into_iter()
            .filter_map(|result| Some(result.ok()?.0))
            .collect::<Vec<Rect>>();
    if vec.is_empty() || vec.len() != 4 {
        return vec![];
    }

    vec.sort_by_key(|rect| rect.x);
    vec
}

/// Detects a single match from `template` with the given BGR image `Mat`.
#[inline]
fn detect_template<T: ToInputArray + MatTraitConst>(
    mat: &impl ToInputArray,
    template: &T,
    offset: Point,
    threshold: f64,
) -> Result<Rect> {
    detect_template_single(mat, template, no_array(), offset, threshold).map(|(bbox, _)| bbox)
}

/// Detects a single match with `mask` from `template` with the given BGR image `Mat`.
#[inline]
fn detect_template_single<T: ToInputArray + MatTraitConst>(
    mat: &impl ToInputArray,
    template: &T,
    mask: impl ToInputArray,
    offset: Point,
    threshold: f64,
) -> Result<(Rect, f64)> {
    detect_template_multiple(mat, template, mask, offset, 1, threshold)
        .into_iter()
        .next()
        .ok_or(anyhow!("no match"))
        .and_then(|x| x)
}

/// Detects multiple matches from `template` from the given BGR image `Mat` and returns up to
/// `max_matches` best results.
#[inline]
fn detect_template_multiple<T: ToInputArray + MatTraitConst>(
    mat: &impl ToInputArray,
    template: &T,
    mask: impl ToInputArray,
    offset: Point,
    max_matches: usize,
    threshold: f64,
) -> Vec<Result<(Rect, f64)>> {
    let suppress_size = template.size().expect("valid template size");
    detect_template_multiple_with_suppression(
        mat,
        template,
        mask,
        offset,
        max_matches,
        threshold,
        suppress_size,
    )
}

/// Same as [`detect_template_multiple`] but allows overriding the size of the box suppressed
/// around each match to avoid re-matching the same instance, instead of always using the full
/// template size.
///
/// A smaller `suppress_size` is useful when matches can legitimately sit close to or overlap each
/// other (e.g. Monster Park's enemy dots) - suppressing a full template-sized box around each
/// match can erase a second, genuinely distinct match before the search gets a chance to find it.
#[inline]
fn detect_template_multiple_with_suppression<T: ToInputArray + MatTraitConst>(
    mat: &impl ToInputArray,
    template: &T,
    mask: impl ToInputArray,
    offset: Point,
    max_matches: usize,
    threshold: f64,
    suppress_size: Size,
) -> Vec<Result<(Rect, f64)>> {
    #[inline]
    fn clear_result(result: &mut Mat, loc: Point, suppress_size: Size) {
        let cols = result.cols();
        let rows = result.rows();

        let x1 = loc.x.clamp(0, cols);
        let y1 = loc.y.clamp(0, rows);
        let x2 = (loc.x + suppress_size.width).clamp(0, cols);
        let y2 = (loc.y + suppress_size.height).clamp(0, rows);

        if x2 <= x1 || y2 <= y1 {
            return;
        }

        result
            .roi_mut(Rect::new(x1, y1, x2 - x1, y2 - y1))
            .expect("valid ROI")
            .set_scalar(Scalar::all(0.0))
            .expect("failed to set scalar");
    }

    #[inline]
    fn append_result(
        matches: &mut Vec<Result<(Rect, f64)>>,
        score: f64,
        loc: Point,
        offset: Point,
        template_size: Size,
    ) {
        // Weird INFINITY values when match template with mask
        // https://github.com/opencv/opencv/issues/23257
        if score == f64::INFINITY {
            return;
        }

        let tl = Point::new(loc.x + offset.x, loc.y + offset.y);
        let br = tl + Point::new(template_size.width, template_size.height);
        let rect = Rect::from_points(tl, br);
        if matches
            .iter()
            .filter_map(|result| result.as_ref().ok())
            .any(|(match_rect, _)| iou(*match_rect, rect) > 0.9)
        {
            return;
        }

        matches.push(Ok((rect, score)));
    }

    let mut result = Mat::default();
    if let Err(err) = match_template(mat, template, &mut result, TM_CCOEFF_NORMED, &mask) {
        error!(target: "backend/detect", "template detection error {err}");
        return vec![];
    }

    let template_size = template.size().unwrap();
    let max_matches = max_matches.max(1);
    let mut matches = Vec::new();
    while matches.len() < max_matches {
        let mut score = 0f64;
        let mut loc = Point::default();
        min_max_loc(
            &result,
            None,
            Some(&mut score),
            None,
            Some(&mut loc),
            &no_array(),
        )
        .unwrap();
        if score < threshold {
            matches.push(Err(anyhow!("template not found").context(score)));
            break;
        }

        clear_result(&mut result, loc, suppress_size);
        append_result(&mut matches, score, loc, offset, template_size);
    }

    matches
}

/// Extracts texts from the non-preprocessed BGR `Mat` and detected text bounding boxes.
fn extract_texts(mat: &impl MatTraitConst, bboxes: &[Rect]) -> Vec<String> {
    static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
        Mutex::new(
            build_session(include_bytes!(env!("TEXT_RECOGNITION_MODEL")))
                .expect("build text recognition session normally"),
        )
    });

    static ALPHABET: LazyLock<String> = LazyLock::new(|| {
        include_str!(env!("TEXT_RECOGNITION_ALPHABET"))
            .split("\n")
            .collect()
    });

    fn ctc_greedy_decode(mat: Mat) -> String {
        let mut result = String::new();
        let blank_class = 0usize;
        let mut prev_class = blank_class;

        for timestep in 0..mat.rows() {
            let logits = mat.row(timestep).unwrap();
            let class = logits
                .iter::<f32>()
                .unwrap()
                .enumerate()
                .max_by(|(_, (_, first)), (_, (_, second))| first.partial_cmp(second).unwrap())
                .map(|(class, _)| class)
                .unwrap();

            if class != blank_class && class != prev_class {
                result.push(ALPHABET.chars().nth(class - 1).unwrap());
            }

            prev_class = class;
        }

        result
    }

    let mut model = MODEL.lock().unwrap();

    bboxes
        .iter()
        .filter_map(|&bbox| {
            let roi = mat.roi(bbox).ok()?;
            let input = preprocess_for_text(&roi);
            Some(to_input_value(&input))
        })
        .map(|input| {
            let result = model.run([input]).unwrap();
            let mat = from_output_value_with_batch_index(&result, 1);
            ctc_greedy_decode(mat)
        })
        .collect::<Vec<_>>()
}

/// Extracts text bounding boxes from the preprocessed [`Mat`].
///
/// This function is adapted from
/// https://github.com/clovaai/CRAFT-pytorch/blob/master/craft_utils.py#L19 with minor changes
fn extract_text_bboxes(
    mat_in: &impl MatTraitConst,
    w_ratio: f32,
    h_ratio: f32,
    x_offset: i32,
    y_offset: i32,
) -> Vec<Rect> {
    const TEXT_SCORE_THRESHOLD: f64 = 0.7;
    const LINK_SCORE_THRESHOLD: f64 = 0.4;
    static TEXT_DETECTION_MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
        Mutex::new(
            build_session(include_bytes!(env!("TEXT_DETECTION_MODEL")))
                .expect("build text detection session normally"),
        )
    });

    let mut model = TEXT_DETECTION_MODEL.lock().unwrap();
    let result = model.run([to_input_value(mat_in)]).unwrap();
    let mat = from_output_value(&result);
    let text_score = mat
        .ranges(&Vector::from_iter([
            Range::all().unwrap(),
            Range::all().unwrap(),
            Range::new(0, 1).unwrap(),
        ]))
        .unwrap()
        .clone_pointee();
    // remove last channel (not sure what other way to do it without clone_pointee first)
    let text_score = text_score
        .reshape_nd(1, &text_score.mat_size()[..2])
        .unwrap();

    let mut text_low_score = Mat::default();
    threshold(
        &text_score,
        &mut text_low_score,
        LINK_SCORE_THRESHOLD,
        1.0,
        THRESH_BINARY,
    )
    .unwrap();

    let mut link_score = mat
        .ranges(&Vector::from_iter([
            Range::all().unwrap(),
            Range::all().unwrap(),
            Range::new(1, 2).unwrap(),
        ]))
        .unwrap()
        .clone_pointee();
    // remove last channel (not sure what other way to do it without clone_pointee first)
    let link_score_size = link_score.mat_size().to_vec();
    let mut link_score = link_score.reshape_nd_mut(1, &link_score_size[..2]).unwrap();
    // SAFETY: can be modified in place
    unsafe {
        link_score.modify_inplace(|mat, mat_mut| {
            threshold(mat, mat_mut, LINK_SCORE_THRESHOLD, 1.0, THRESH_BINARY).unwrap();
        });
    }

    let mut combined_score = Mat::default();
    let mut gt_one_mask = Mat::default();
    add(
        &text_low_score,
        &link_score,
        &mut combined_score,
        &no_array(),
        CV_8U,
    )
    .unwrap();
    compare(&combined_score, &Scalar::all(1.0), &mut gt_one_mask, CMP_GT).unwrap();
    combined_score
        .set_to(&Scalar::all(1.0), &gt_one_mask)
        .unwrap();

    let mut bboxes = Vec::<Rect>::new();
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let labels_count = connected_components_with_stats(
        &combined_score,
        &mut labels,
        &mut stats,
        &mut Mat::default(),
        4,
        CV_32S,
    )
    .unwrap();
    for i in 1..labels_count {
        let area = *stats.at_2d::<i32>(i, CC_STAT_AREA).unwrap();
        if area < 10 {
            continue;
        }
        let mut mask = Mat::default();
        let mut max_score = 0.0f64;
        compare(&labels, &Scalar::all(i as f64), &mut mask, CMP_EQ).unwrap();
        min_max_loc(&text_score, None, Some(&mut max_score), None, None, &mask).unwrap();
        if max_score < TEXT_SCORE_THRESHOLD {
            continue;
        }

        let shape = mask.size().unwrap();
        // SAFETY: The position (row, col) is guaranteed by OpenCV
        let x = unsafe { *stats.at_2d_unchecked::<i32>(i, CC_STAT_LEFT).unwrap() };
        let y = unsafe { *stats.at_2d_unchecked::<i32>(i, CC_STAT_TOP).unwrap() };
        let w = unsafe { *stats.at_2d_unchecked::<i32>(i, CC_STAT_WIDTH).unwrap() };
        let h = unsafe { *stats.at_2d_unchecked::<i32>(i, CC_STAT_HEIGHT).unwrap() };
        let size = area as f64 * w.min(h) as f64 / (w as f64 * h as f64);
        let size = ((size).sqrt() * 2.0) as i32;
        let sx = (x - size + 1).max(0);
        let sy = (y - size + 1).max(0);
        let ex = (x + w + size + 1).min(shape.width);
        let ey = (y + h + size + 1).min(shape.height);
        let kernel =
            get_structuring_element_def(MORPH_RECT, Size::new(size + 1, size + 1)).unwrap();

        let mut link_mask = Mat::default();
        let mut text_mask = Mat::default();
        let mut and_mask = Mat::default();
        let mut seg_map = Mat::zeros(shape.height, shape.width, CV_8U)
            .unwrap()
            .to_mat()
            .unwrap();
        compare(&link_score, &Scalar::all(1.0), &mut link_mask, CMP_EQ).unwrap();
        compare(&text_score, &Scalar::all(0.0), &mut text_mask, CMP_EQ).unwrap();
        bitwise_and_def(&link_mask, &text_mask, &mut and_mask).unwrap();
        seg_map.set_to(&Scalar::all(255.0), &mask).unwrap();
        seg_map.set_to(&Scalar::all(0.0), &and_mask).unwrap();

        let mut seg_contours = Vector::<Point>::new();
        let mut seg_roi = seg_map
            .roi_mut(Rect::from_points(Point::new(sx, sy), Point::new(ex, ey)))
            .unwrap();
        // SAFETY: all of the functions below can be called in place.
        unsafe {
            seg_roi.modify_inplace(|mat, mat_mut| {
                dilate_def(mat, mat_mut, &kernel).unwrap();
                mat.copy_to(mat_mut).unwrap();
            });
        }
        find_non_zero(&seg_map, &mut seg_contours).unwrap();

        let contour = min_area_rect(&seg_contours)
            .unwrap()
            .bounding_rect2f()
            .unwrap();
        let tl = contour.tl();
        let tl = Point::new(
            (tl.x * w_ratio * 2.0) as i32 + x_offset,
            (tl.y * h_ratio * 2.0) as i32 + y_offset,
        );
        let br = contour.br();
        let br = Point::new(
            (br.x * w_ratio * 2.0) as i32 + x_offset,
            (br.y * h_ratio * 2.0) as i32 + y_offset,
        );
        bboxes.push(Rect::from_points(tl, br));
    }
    bboxes
}

#[inline]
fn remap_from_yolo(
    pred: &[f32],
    size: Size,
    w_ratio: f32,
    h_ratio: f32,
    left: i32,
    top: i32,
) -> Rect {
    let tl_x = ((pred[0] - left as f32) / w_ratio)
        .max(0.0)
        .min(size.width as f32);
    let tl_y = ((pred[1] - top as f32) / h_ratio)
        .max(0.0)
        .min(size.height as f32);
    let br_x = ((pred[2] - left as f32) / w_ratio)
        .max(0.0)
        .min(size.width as f32);
    let br_y = ((pred[3] - top as f32) / h_ratio)
        .max(0.0)
        .min(size.height as f32);
    Rect::from_points(
        Point::new(tl_x as i32, tl_y as i32),
        Point::new(br_x as i32, br_y as i32),
    )
}

/// Preprocesses a BGR `Mat` image to a normalized and resized RGB `Mat` image with type `f32`
/// for YOLO detection.
///
/// Returns a triplet of `(Mat, width_ratio, height_ratio, left, top)`
#[inline]
fn preprocess_for_yolo(mat: &impl MatTraitConst) -> (Mat, f32, f32, i32, i32) {
    // https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py
    let mut mat = mat.try_clone().unwrap();

    let size = mat.size().unwrap();
    let (w_ratio, h_ratio) = (640.0 / size.width as f32, 640.0 / size.height as f32);
    let min_ratio = w_ratio.min(h_ratio);

    let w = (size.width as f32 * min_ratio).round();
    let h = (size.height as f32 * min_ratio).round();

    let pad_w = (640.0 - w) / 2.0;
    let pad_h = (640.0 - h) / 2.0;

    let top = (pad_h - 0.1).round() as i32;
    let bottom = (pad_h + 0.1).round() as i32;
    let left = (pad_w - 0.1).round() as i32;
    let right = (pad_w + 0.1).round() as i32;

    // SAFETY: all of the functions below can be called in place.
    unsafe {
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGR2RGB).unwrap();
            resize(
                mat,
                mat_mut,
                Size::new(w as i32, h as i32),
                0.0,
                0.0,
                INTER_LINEAR,
            )
            .unwrap();
            copy_make_border(
                mat,
                mat_mut,
                top,
                bottom,
                left,
                right,
                BORDER_CONSTANT,
                Scalar::all(114.0),
            )
            .unwrap();
            mat.convert_to(mat_mut, CV_32FC3, 1.0 / 255.0, 0.0).unwrap();
        });
    }
    (mat, min_ratio, min_ratio, left, top)
}

/// Preprocesses a BGR `Mat` image for text bounding boxes detection with `5.0` magnification ratio.
/// This function is meant for small images.
#[inline]
fn preprocess_for_text_bboxes(bgr: &impl MatTraitConst) -> (Mat, f32, f32) {
    preprocess_for_text_bboxes_magnified(bgr, 5.0)
}

/// Preprocesses a BGR `Mat` image to a normalized and resized RGB `Mat` image with type `f32`
/// for text bounding boxes detection.
///
/// The preprocess is adapted from: https://github.com/clovaai/CRAFT-pytorch/blob/master/imgproc.py
///
/// Returns a `(Mat, width_ratio, height_ratio)`.
fn preprocess_for_text_bboxes_magnified(
    bgr: &impl MatTraitConst,
    magnification_ratio: f32,
) -> (Mat, f32, f32) {
    let mut mat = bgr.try_clone().unwrap();
    let size = mat.size().unwrap();
    let size_w = size.width as f32;
    let size_h = size.height as f32;
    let size_max = size_w.max(size_h);
    let resize_size = magnification_ratio * size_max;
    let resize_ratio = resize_size / size_max;

    let resize_w = (resize_ratio * size_w) as i32;
    let resize_w = (resize_w + 31) & !31; // rounds to multiple of 32
    let resize_w_ratio = size_w / resize_w as f32;

    let resize_h = (resize_ratio * size_h) as i32;
    let resize_h = (resize_h + 31) & !31;
    let resize_h_ratio = size_h / resize_h as f32;
    // SAFETY: all of the below functions can be called in place
    unsafe {
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGR2RGB).unwrap();
            resize(
                mat,
                mat_mut,
                Size::new(resize_w, resize_h),
                0.0,
                0.0,
                INTER_CUBIC,
            )
            .unwrap();
            mat.convert_to(mat_mut, CV_32FC3, 1.0, 0.0).unwrap();
            // these values are pre-multiplied from the above link in normalizeMeanVariance
            subtract_def(mat, &Scalar::new(123.675, 116.28, 103.53, 0.0), mat_mut).unwrap();
            divide2_def(&mat, &Scalar::new(58.395, 57.12, 57.375, 1.0), mat_mut).unwrap();
        });
    }
    (mat, resize_w_ratio, resize_h_ratio)
}

/// Preprocesses a BGR `Mat` image to a normalized and resized RGB `Mat` image with type `f32`
/// for CRNN+CTC text recognition.
fn preprocess_for_text(bgr: &impl MatTraitConst) -> Mat {
    let mut mat = bgr.try_clone().unwrap();

    unsafe {
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGR2RGB).unwrap();
            resize(mat, mat_mut, Size::new(100, 32), 0.0, 0.0, INTER_LINEAR).unwrap();
            mat.convert_to(mat_mut, CV_32FC3, 1.0, 0.0).unwrap();
            subtract_def(&mat, &Scalar::all(127.5), mat_mut).unwrap();
            divide2_def(&mat, &Scalar::all(127.5), mat_mut).unwrap();
        });
    }

    mat
}

/// Expands `bbox` in all the direction by `count` pixel(s) and clamps to `size` if provided.
#[inline]
/// Expands `bbox` by `count` pixels on every side, optionally clamped to stay inside `size`.
///
/// The margin actually applied on one axis is the *same* on both sides - whichever side has less
/// room within `size` - rather than clamping each side independently. Clamping independently would
/// let a box near one edge expand fully on the far side while getting clipped on the near side,
/// silently shifting the box's center away from the edge instead of just shrinking the margin.
/// For a detected portal near the edge of the minimap, that shifted center then becomes the
/// player's movement target and "have I arrived" check alike - both self-consistent, but both
/// wrong relative to the portal's actual in-game position, so the player can stand exactly on the
/// computed (wrong) point, be told it has arrived, and have Up do nothing, forever.
fn expand_bbox(size: Option<Size>, bbox: Rect, count: i32) -> Rect {
    let br = bbox.br();
    let (x1, x2) = symmetric_expand(bbox.x, br.x, count, size.map(|size| size.width));
    let (y1, y2) = symmetric_expand(bbox.y, br.y, count, size.map(|size| size.height));

    Rect::new(x1, y1, x2 - x1, y2 - y1)
}

/// Expands the `[lo, hi]` range by `count` on both ends, clamped to `[0, bound]` if given - using
/// whichever side has less room as the margin for *both* sides, to keep the range's center fixed.
#[inline]
fn symmetric_expand(lo: i32, hi: i32, count: i32, bound: Option<i32>) -> (i32, i32) {
    let margin = match bound {
        Some(bound) => count.min(lo).min(bound - hi).max(0),
        None => count,
    };
    (lo - margin, hi + margin)
}

/// Computes the intersection over union ratio.
#[inline]
fn iou(first: Rect, second: Rect) -> f32 {
    let intersection = (first & second).area() as f32;
    let union = (first | second).area() as f32;
    intersection / union
}

/// Crops `mat` to the buffs region.
#[inline]
fn to_buffs_region(mat: &impl MatTraitConst) -> BoxedRef<'_, Mat> {
    let size = mat.size().unwrap();
    // Crop to top right of the image for buffs region
    let crop_x = size.width / 3;
    let crop_y = size.height / 4;
    let crop_bbox = Rect::new(size.width - crop_x, 0, crop_x, crop_y);
    mat.roi(crop_bbox).unwrap()
}

/// Crops `mat` to the bottom right of the image for quick slots region.
#[inline]
fn to_quick_slots_region(mat: &impl MatTraitConst) -> (BoxedRef<'_, Mat>, Rect) {
    let size = mat.size().unwrap();
    let crop_x = size.width / 2;
    let crop_y = size.height / 5;
    let crop_bbox = Rect::new(size.width - crop_x, size.height - crop_y, crop_x, crop_y);
    let crop_roi = mat.roi(crop_bbox).unwrap();
    (crop_roi, crop_bbox)
}

/// Converts a BGR `Mat` image to HSV.
#[inline]
fn to_hsv(mat: &impl MatTraitConst) -> Mat {
    let mut mat = mat.try_clone().unwrap();
    unsafe {
        // SAFETY: can be modified inplace
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGR2HSV_FULL).unwrap();
        });
    }
    mat
}

/// Converts a BGRA `Mat` image to BGR.
#[inline]
fn to_bgr(mat: &impl MatTraitConst) -> Mat {
    let mut mat = mat.try_clone().unwrap();
    unsafe {
        // SAFETY: can be modified inplace
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGRA2BGR).unwrap();
        });
    }
    mat
}

/// Converts a BGRA `Mat` image to grayscale.
///
/// `add_contrast` can be set to `true` in order to increase contrast by a fixed amount
/// used for template matching.
#[inline]
fn to_grayscale(mat: &impl MatTraitConst, add_contrast: bool) -> Mat {
    let mut mat = mat.try_clone().unwrap();
    unsafe {
        // SAFETY: all of the functions below can be called in place.
        mat.modify_inplace(|mat, mat_mut| {
            cvt_color_def(mat, mat_mut, COLOR_BGRA2GRAY).unwrap();
            if add_contrast {
                // TODO: is this needed?
                add_weighted_def(mat, 1.5, mat, 0.0, -80.0, mat_mut).unwrap();
            }
        });
    }
    mat
}

/// Converts `base64` to a [`Mat`].
///
/// If `grayscale` is `true`, `base64` will be read with [`IMREAD_GRAYSCALE`]. Otherwise, it is
/// read with [`IMREAD_COLOR`].
fn to_mat_from_base64(base64: &str, grayscale: bool) -> Result<Mat> {
    let flag = if grayscale {
        IMREAD_GRAYSCALE
    } else {
        IMREAD_COLOR
    };
    let bytes = BASE64_STANDARD.decode(base64)?;
    let bytes = Vector::<u8>::from_iter(bytes);

    Ok(imdecode(&bytes, flag)?)
}

/// Converts `mat` to a base64 PNG [`String`].
pub fn to_base64_from_mat(mat: &Mat) -> Result<String> {
    let mut bytes = Vector::new();
    imencode_def(".png", mat, &mut bytes)?;
    Ok(BASE64_STANDARD.encode(bytes))
}

/// Extracts a `Mat` from `SessionOutputs` assuming batch index is 0.
///
/// The returned `Mat` has batch dimension removed.
#[inline]
fn from_output_value(result: &SessionOutputs) -> Mat {
    from_output_value_with_batch_index(result, 0)
}

/// Extracts a `Mat` from `SessionOutputs` and `batch_index`.
///
/// The returned `Mat` has batch dimension removed.
#[inline]
fn from_output_value_with_batch_index(result: &SessionOutputs, batch_index: usize) -> Mat {
    let key = result.keys().next().unwrap();
    let (dims, outputs) = result[key].try_extract_tensor::<f32>().unwrap();

    let dims = dims.iter().map(|&dim| dim as i32).collect::<Vec<i32>>();
    let mat = Mat::new_nd_with_data(dims.as_slice(), outputs).unwrap();

    let new_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(i, &dim)| if i == batch_index { None } else { Some(dim) })
        .collect::<Vec<i32>>();
    let mat = mat.reshape_nd(1, &new_dims).unwrap();

    mat.clone_pointee()
}

/// Converts a continuous, normalized `f32` RGB `Mat` image to `SessionInputValue`.
///
/// The input `Mat` is assumed to be continuous, normalized RGB `f32` data type and
/// will panic if not. The `Mat` is reshaped to single channel, tranposed to `[1, 3, H, W]` and
/// converted to `SessionInputValue`.
#[inline]
fn to_input_value(mat: &impl MatTraitConst) -> SessionInputValue<'static> {
    let mat = mat.reshape_nd(1, &[1, mat.rows(), mat.cols(), 3]).unwrap();
    let mut mat_t = Mat::default();
    transpose_nd(&mat, &Vector::from_slice(&[0, 3, 1, 2]), &mut mat_t).unwrap();
    let shape = mat_t.mat_size();
    let input = (shape.to_vec(), mat_t.data_typed::<f32>().unwrap());
    let tensor = TensorRef::from_array_view(input).unwrap();
    SessionInputValue::Owned(tensor.clone().into_dyn())
}

#[inline]
fn build_session(model: &[u8]) -> Result<Session> {
    Ok(Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_memory(model)?)
}

/// Diagnoses whether the CUDA execution provider actually initializes on this machine.
///
/// [`build_session`] registers `CUDAExecutionProvider` with `ort`'s default
/// [`ExecutionProviderDispatch::fail_silently`](ort::execution_providers::ExecutionProviderDispatch::fail_silently)
/// behavior, so a failure to load CUDA (e.g. missing cuDNN/cuBLAS runtime libraries, even with a
/// working GPU driver) silently falls back to CPU - every model then runs unaccelerated with no
/// visible error anywhere. This instead forces registration with
/// [`ExecutionProviderDispatch::error_on_failure`](ort::execution_providers::ExecutionProviderDispatch::error_on_failure)
/// against a real embedded model so a failure surfaces as an actual error message instead of being
/// swallowed.
#[cfg(debug_assertions)]
pub fn diagnose_cuda() -> String {
    use ort::execution_providers::ExecutionProvider;

    let provider = CUDAExecutionProvider::default();
    match provider.is_available() {
        Ok(false) => return "ONNX Runtime was not compiled with CUDA support.".to_string(),
        Err(err) => return format!("Failed to query available execution providers: {err}"),
        Ok(true) => {}
    }

    static MODEL: &[u8] = include_bytes!(env!("MOB_MODEL"));
    let result = Session::builder().and_then(|builder| {
        builder
            .with_execution_providers([CUDAExecutionProvider::default()
                .build()
                .error_on_failure()])?
            .commit_from_memory(MODEL)
    });

    match result {
        Ok(_) => "CUDA execution provider initialized successfully.".to_string(),
        Err(err) => format!("CUDA execution provider failed to initialize: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use opencv::core::{Rect, Size};

    use super::{expand_bbox, parse_daily_quest_progress};

    #[test]
    fn expand_bbox_interior_expands_symmetrically() {
        let bbox = Rect::new(50, 50, 10, 10);
        let expanded = expand_bbox(Some(Size::new(200, 200)), bbox, 5);

        assert_eq!(expanded, Rect::new(45, 45, 20, 20));
        // Center stays exactly the same as the original box.
        assert_eq!(expanded.x + expanded.width / 2, bbox.x + bbox.width / 2);
        assert_eq!(expanded.y + expanded.height / 2, bbox.y + bbox.height / 2);
    }

    #[test]
    fn expand_bbox_near_right_edge_shrinks_margin_on_both_sides() {
        // Only 2px of room to the right of the box within a 100-wide minimap.
        let bbox = Rect::new(90, 50, 8, 10);
        let expanded = expand_bbox(Some(Size::new(100, 200)), bbox, 5);

        // Both sides use the smaller (2px) margin instead of the right side clipping to 2px
        // while the left side still gets the full 5px - which would shift the center left.
        assert_eq!(expanded, Rect::new(88, 45, 12, 20));
        let orig_center_x = bbox.x + bbox.width / 2;
        let expanded_center_x = expanded.x + expanded.width / 2;
        assert_eq!(
            expanded_center_x, orig_center_x,
            "expanded box center must not shift away from the edge"
        );
    }

    #[test]
    fn expand_bbox_without_size_always_expands_fully() {
        let bbox = Rect::new(0, 0, 10, 10);
        let expanded = expand_bbox(None, bbox, 5);

        assert_eq!(expanded, Rect::new(-5, -5, 20, 20));
    }

    struct StdoutLogger;

    impl log::Log for StdoutLogger {
        fn enabled(&self, _metadata: &log::Metadata) -> bool {
            true
        }

        fn log(&self, record: &log::Record) {
            println!("[{} {}] {}", record.level(), record.target(), record.args());
        }

        fn flush(&self) {}
    }

    #[test]
    #[ignore = "manual validation against local raw_resources screenshots, not a CI fixture"]
    fn detect_daily_quest_progress_popup_against_dataset() {
        use std::fs;

        use opencv::imgcodecs::{self, IMREAD_COLOR, IMREAD_GRAYSCALE};

        use super::detect_daily_quest_progress_popup;

        static LOGGER: StdoutLogger = StdoutLogger;
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Debug);

        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../raw_resources/kill_count_pop_up"
        );
        let mut entries = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        entries.sort();

        for path in entries {
            let bgr = imgcodecs::imread(path.to_str().unwrap(), IMREAD_COLOR).unwrap();
            let grayscale = imgcodecs::imread(path.to_str().unwrap(), IMREAD_GRAYSCALE).unwrap();
            let pairs = detect_daily_quest_progress_popup(&bgr, &grayscale);
            println!("=== {:?}: {pairs:?}", path.file_name().unwrap());
        }
    }

    #[test]
    fn parse_daily_quest_progress_ok() {
        assert_eq!(
            parse_daily_quest_progress("Cernium Region Mob 28 / 100"),
            Some((28, 100))
        );
        assert_eq!(
            parse_daily_quest_progress("Labyrinth of Suffering Region Mob 300 / 300"),
            Some((300, 300))
        );
        // No space around the slash
        assert_eq!(
            parse_daily_quest_progress("Cernium Region Mob 5/500"),
            Some((5, 500))
        );
        // OCR noise around the digits
        assert_eq!(
            parse_daily_quest_progress("Cern1um Reg on Mob 51 / 100 !"),
            Some((51, 100))
        );
        // `/` misread as a standalone `I` - observed from real screenshots
        assert_eq!(
            parse_daily_quest_progress("Hotel Arcus Region Mob 19 I 100 I"),
            Some((19, 100))
        );
        assert_eq!(
            parse_daily_quest_progress("Odium Region Mob 27 I 100"),
            Some((27, 100))
        );
        // A noise word landing directly between `current` and the separator, observed verbatim
        // from real screenshots
        assert_eq!(
            parse_daily_quest_progress("Odium Region Mob 25 the I 100 mheat the aas"),
            Some((25, 100))
        );
        assert_eq!(
            parse_daily_quest_progress("Shangri-La Region Mob 22 mae I 300 all the fom"),
            Some((22, 300))
        );
    }

    #[test]
    fn parse_daily_quest_progress_err() {
        assert_eq!(parse_daily_quest_progress(""), None);
        assert_eq!(parse_daily_quest_progress("Cernium Region Mob"), None);
        assert_eq!(parse_daily_quest_progress("no slash here 100"), None);
        // `/` fused into the target's leading digit (e.g. misread as `1`) instead of landing as
        // its own token - unrecoverable, but must not be misparsed as a real, larger `current`
        assert_eq!(
            parse_daily_quest_progress("Hotel Arcus Region Mob 461 100"),
            None
        );
        // A `current` larger than `target` must be rejected outright even when the text parses
        // cleanly otherwise - a kill count can never exceed its own cap, so this can only be OCR
        // corruption
        assert_eq!(
            parse_daily_quest_progress("Cernium Region Mob 150 / 100"),
            None
        );
    }

    #[test]
    fn clamp_rect_to_mat_intersects_bounds() {
        use opencv::core::{CV_8UC3, Mat, Rect, Scalar};

        use super::clamp_rect_to_mat;

        let mat = Mat::new_rows_cols_with_default(200, 300, CV_8UC3, Scalar::all(0.0)).unwrap();

        // Fully inside - unchanged
        assert_eq!(
            clamp_rect_to_mat(&mat, Rect::new(10, 10, 50, 50)),
            Some(Rect::new(10, 10, 50, 50))
        );

        // Partially outside - clamped down to just the overlapping area
        assert_eq!(
            clamp_rect_to_mat(&mat, Rect::new(280, 180, 100, 100)),
            Some(Rect::new(280, 180, 20, 20))
        );

        // Fully outside - no overlap at all
        assert_eq!(clamp_rect_to_mat(&mat, Rect::new(400, 400, 50, 50)), None);

        // Same shape as the real crash this guards against: a lie detector puzzle region (title
        // position + fixed 755x505 offset) that overshoots a frame this small on both axes -
        // `.roi()` would otherwise panic instead of this returning `None`.
        assert_eq!(clamp_rect_to_mat(&mat, Rect::new(905, 377, 755, 505)), None);
    }
}
