use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString};

use super::DailyQuestEntry;

/// The NPCs the Thread of Fate "Ask" automation supports (see [`ThreadOfFateAskProgress`]).
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Debug,
    Serialize,
    Deserialize,
    EnumIter,
    Display,
    EnumString,
)]
pub enum ThreadOfFateNpc {
    #[default]
    Izuna,
    Kurama,
    #[strum(to_string = "Takeno Konoko")]
    TakenoKonoko,
    Tengu,
}

impl ThreadOfFateNpc {
    /// Fixed pixel offset `(x, y)` from the Threads of Fate menu's title anchor's top-left
    /// corner (see [`crate::detect::Detector::detect_thread_of_fate_ui_title`]) to this NPC's
    /// icon center in the relationship grid.
    ///
    /// The grid's layout doesn't move - measured directly against 6 real reference screenshots
    /// (different NPCs selected, different intro-quest states) and found pixel-identical (±1px)
    /// in every one, so a fixed offset is used instead of template-matching each NPC's icon
    /// every tick. Mirrors [`super::DailyQuestId::navigation`]'s exact same reasoning/convention
    /// for hardcoding verified pixel offsets from an anchor rather than re-detecting.
    pub fn icon_offset_from_title(self) -> (i32, i32) {
        match self {
            ThreadOfFateNpc::Izuna => (328, 192),
            ThreadOfFateNpc::Kurama => (231, 234),
            ThreadOfFateNpc::TakenoKonoko => (-173, 242),
            ThreadOfFateNpc::Tengu => (-51, 127),
        }
    }
}

/// A persistent per-character tracker for the Thread of Fate "Ask" automation.
///
/// Unlike [`DailyQuestEntry`], which gates a single once-a-day completion, asking can be repeated
/// up to [`Self::DAILY_LIMIT`] times per day, and each ask has its own ~30 minute wait
/// ([`Self::WAIT_SECS`]) before the resulting quest is ready to turn in.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct ThreadOfFateAskProgress {
    /// Number of asks accepted today (see [`Self::reset_if_new_day`]).
    #[serde(default)]
    pub count_today: u32,
    /// The UTC day index (see [`DailyQuestEntry::today`]) [`Self::count_today`] was last reset
    /// on, if any.
    #[serde(default)]
    pub last_reset_day: Option<u64>,
    /// The NPC an ask quest is currently pending completion for, if any.
    ///
    /// Kept separate from the user's currently *selected* NPC (see
    /// [`super::Character::thread_of_fate_ask_npc`]) - the user can freely change the selected NPC
    /// while a previous ask is still pending without affecting which NPC's quest actually needs
    /// turning in.
    #[serde(default)]
    pub pending_npc: Option<ThreadOfFateNpc>,
    /// Unix timestamp (seconds) the pending ask quest was accepted, if any.
    #[serde(default)]
    pub pending_since_secs: Option<u64>,
}

impl ThreadOfFateAskProgress {
    /// Maximum number of asks that can be accepted per day.
    pub const DAILY_LIMIT: u32 = 5;
    /// How long an accepted ask quest takes before it is ready to turn in.
    pub const WAIT_SECS: u64 = 30 * 60;

    /// Resets [`Self::count_today`] if the day has rolled over since the last reset.
    pub fn reset_if_new_day(&mut self) {
        let today = DailyQuestEntry::today();
        if self.last_reset_day != Some(today) {
            self.count_today = 0;
            self.last_reset_day = Some(today);
        }
    }

    /// Whether a new ask can be accepted - no ask is currently pending and today's limit has not
    /// been reached.
    pub fn can_accept(&self) -> bool {
        self.pending_npc.is_none() && self.count_today < Self::DAILY_LIMIT
    }

    /// Whether the pending ask quest has waited long enough to be ready to turn in.
    pub fn is_ready_to_complete(&self) -> bool {
        let Some(pending_since_secs) = self.pending_since_secs else {
            return false;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(pending_since_secs) >= Self::WAIT_SECS
    }
}
