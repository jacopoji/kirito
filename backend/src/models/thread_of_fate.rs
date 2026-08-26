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
    /// Whether starting a new ask is blocked because the daily reset was seen while the bot was
    /// still running (see [`Self::reset_if_new_day`]).
    ///
    /// Asking right after a reset seen mid-run is known to desync the desired reward from what
    /// actually gets granted - restarting the bot (see [`Self::clear_reset_block`]) picks it back
    /// up cleanly, so new asks are withheld until then. Completing an already-pending ask is
    /// unaffected - only [`Self::can_accept`] checks this.
    #[serde(default)]
    pub blocked_by_reset: bool,
}

impl ThreadOfFateAskProgress {
    /// Maximum number of asks that can be accepted per day.
    pub const DAILY_LIMIT: u32 = 5;
    /// How long an accepted ask quest takes before it is ready to turn in.
    pub const WAIT_SECS: u64 = 30 * 60;

    /// Resets [`Self::count_today`] if the day has rolled over since the last reset.
    ///
    /// Only called while the bot is actively rotating priority actions, which never happens while
    /// [`crate::operation::Operation`] is halting - so a rollover detected here always means the
    /// reset was seen while running, as opposed to the day having simply moved on while the bot
    /// sat stopped. [`Self::last_reset_day`] being `None` is excluded (that's just this tracker's
    /// first-ever initialization, not a live rollover) so a fresh character isn't blocked before
    /// it has asked even once.
    pub fn reset_if_new_day(&mut self) {
        let today = DailyQuestEntry::today();
        if self.last_reset_day != Some(today) {
            if self.last_reset_day.is_some() {
                self.blocked_by_reset = true;
            }
            self.count_today = 0;
            self.last_reset_day = Some(today);
        }
    }

    /// Clears [`Self::blocked_by_reset`] - called once the bot is stopped and started running
    /// again, which is when the desync [`Self::blocked_by_reset`] guards against is known to
    /// clear up.
    pub fn clear_reset_block(&mut self) {
        self.blocked_by_reset = false;
    }

    /// Whether a new ask can be accepted - no ask is currently pending, today's limit has not been
    /// reached, and the bot hasn't run through a daily reset since it last (re)started (see
    /// [`Self::blocked_by_reset`]).
    pub fn can_accept(&self) -> bool {
        !self.blocked_by_reset && self.pending_npc.is_none() && self.count_today < Self::DAILY_LIMIT
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
