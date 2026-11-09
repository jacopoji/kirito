use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString};

/// The cube whose potential window is being rolled.
///
/// Each one uses a different in-game window layout, so a different Reset button template and
/// potential lines location.
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
pub enum CubeKind {
    /// A single result panel inside the Enhance > Potential menu.
    #[default]
    Glowing,
    /// A before/after window where the "AFTER" panel holds the new lines.
    Bright,
}

/// A stat a potential line can roll that cubing can look for.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    Serialize,
    Deserialize,
    EnumIter,
    Display,
    EnumString,
)]
pub enum CubingStat {
    #[strum(to_string = "STR")]
    Str,
    #[strum(to_string = "DEX")]
    Dex,
    #[strum(to_string = "INT")]
    Int,
    #[strum(to_string = "LUK")]
    Luk,
    #[strum(to_string = "All Stats")]
    AllStats,
    #[strum(to_string = "Attack Power")]
    AttackPower,
    #[strum(to_string = "Magic ATT")]
    MagicAttack,
    #[strum(to_string = "Boss Damage")]
    BossDamage,
    #[strum(to_string = "Critical Damage")]
    CriticalDamage,
    #[strum(to_string = "Ignore Defense")]
    IgnoreDefense,
    #[strum(to_string = "Item Drop Rate")]
    ItemDropRate,
    #[strum(to_string = "Meso Obtained")]
    MesoObtained,
    #[strum(to_string = "Skill Cooldown")]
    SkillCooldown,
}

impl CubingStat {
    /// Whether this stat can be selected for [`CubingSettings::threshold_stats`].
    pub fn supports_threshold(self) -> bool {
        matches!(
            self,
            CubingStat::Str
                | CubingStat::Dex
                | CubingStat::Int
                | CubingStat::Luk
                | CubingStat::AllStats
                | CubingStat::AttackPower
                | CubingStat::MagicAttack
        )
    }
}

/// When to stop rolling and keep the current potential.
///
/// Rolling stops as soon as any enabled rule is satisfied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CubingSettings {
    pub cube_kind: CubeKind,
    /// Whether to stop once the highest of [`Self::threshold_stats`] reaches [`Self::threshold`].
    pub threshold_enabled: bool,
    /// The total percentage across all lines to reach. All Stats count towards STR/DEX/INT/LUK.
    pub threshold: u32,
    pub threshold_stats: Vec<CubingStat>,
    /// Whether to stop once [`Self::line_match_count`] lines roll any of
    /// [`Self::line_match_stats`].
    pub line_match_enabled: bool,
    pub line_match_stats: Vec<CubingStat>,
    /// The number of matching lines required, from `1` to `3`.
    pub line_match_count: u32,
}

impl CubingSettings {
    /// Whether at least one rule can ever be satisfied.
    ///
    /// Rolling without one would burn every cube the user has.
    pub fn has_usable_rule(&self) -> bool {
        (self.threshold_enabled && !self.threshold_stats.is_empty())
            || (self.line_match_enabled && !self.line_match_stats.is_empty())
    }
}

impl Default for CubingSettings {
    fn default() -> Self {
        Self {
            cube_kind: CubeKind::default(),
            threshold_enabled: false,
            threshold: 21,
            threshold_stats: vec![],
            line_match_enabled: false,
            line_match_stats: vec![],
            line_match_count: 2,
        }
    }
}

/// Why a cubing run ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Display)]
pub enum CubingStopReason {
    #[strum(to_string = "Potential meets the requirements")]
    RequirementsMet,
    #[strum(to_string = "Stopped by user")]
    User,
    #[strum(to_string = "Same result 5 times in a row - out of cubes or the game is not focused")]
    SameResult,
    #[strum(to_string = "Reset button not found - open the cube window for the selected cube")]
    ResetButtonNotFound,
    #[strum(to_string = "Could not read any potential lines")]
    NoLinesRead,
    #[strum(to_string = "Stopped because the bot started running")]
    BotStarted,
    #[strum(to_string = "Game capture stopped updating")]
    CaptureStalled,
    #[strum(to_string = "Enable a rule and select at least one stat first")]
    NoUsableRule,
    #[strum(to_string = "Stop the bot before cubing")]
    BotRunning,
}

/// What a cubing run is doing.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum CubingStatus {
    #[default]
    Idle,
    Starting,
    /// Seconds left for the user to focus the game before the first roll.
    Countdown(u32),
    Rolling,
    Stopped(CubingStopReason),
}

impl CubingStatus {
    /// Whether a run is in progress.
    pub fn is_running(self) -> bool {
        matches!(
            self,
            CubingStatus::Starting | CubingStatus::Countdown(_) | CubingStatus::Rolling
        )
    }
}

/// Progress of the current or last cubing run.
#[derive(Clone, PartialEq, Default, Debug)]
pub struct CubingState {
    pub status: CubingStatus,
    pub rolls: u32,
    /// Results of the run, most recent first and capped in length.
    pub log: Vec<String>,
}
