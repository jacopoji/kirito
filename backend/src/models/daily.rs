use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString};

use super::{Bound, Platform};

/// A persistent model representing per-character configuration for a [`DailyQuestId`].
///
/// The navigation to reach the hunting ground and its hunting bound are fixed game content, not
/// user data - see [`DailyQuestId::navigation`] and [`DailyQuestId::bound`]. Only these fields
/// are user-editable; the mobbing key used for all daily quests is shared and lives on
/// [`super::Character::daily_quest_mobbing_key`] instead of per-entry.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DailyQuestEntry {
    pub id: DailyQuestId,
    pub enabled: bool,
    /// The UTC day index (see [`Self::today`]) this quest was last completed on, if any.
    ///
    /// Set by the tick loop through [`crate::ecs::CharacterUpdates`] when
    /// [`crate::rotator::DefaultRotator::rotate_daily_quest`] detects the "Quest complete!"
    /// toast, so an already-completed quest is skipped instead of re-run if the bot restarts or
    /// its actions get rebuilt again later the same day.
    #[serde(default)]
    pub last_completed_day: Option<u64>,
}

impl DailyQuestEntry {
    pub fn new(id: DailyQuestId) -> Self {
        Self {
            id,
            enabled: false,
            last_completed_day: None,
        }
    }

    /// The current UTC day index (days since the Unix epoch).
    ///
    /// This is the boundary daily quest completion is tracked against. Not adjusted for the game
    /// server's actual daily reset time, which may fall at a different point than UTC midnight -
    /// so a quest may appear completed for a few hours before/after the real in-game reset.
    pub fn today() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 86400
    }

    /// Whether this entry was already completed today (see [`Self::today`]).
    pub fn is_completed_today(&self) -> bool {
        self.last_completed_day == Some(Self::today())
    }
}

/// The world map's top-level region dropdown options.
#[derive(
    Clone, Copy, PartialEq, Default, Debug, Serialize, Deserialize, EnumIter, Display, EnumString,
)]
pub enum WorldMapRegion {
    #[default]
    #[strum(to_string = "Maple World")]
    MapleWorld,
    Grandis,
    #[strum(to_string = "Arcane River")]
    ArcaneRiver,
    Hielo,
}

/// A fixed catalog of known daily quest hunting grounds.
///
/// Unlike [`super::Map`], these aren't user-defined - the navigation to reach each one through
/// the in-game world map is fixed game content (see [`Self::navigation`]). Only the
/// per-character fields on [`DailyQuestEntry`] (enabled, hunting bound, mobbing key)
/// are user-editable.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Serialize,
    Deserialize,
    EnumIter,
    Display,
    EnumString,
)]
/// Ordered as they should run - the daily quest solver runs entries in this declaration order
/// (via `Ord`), not the order they happen to be stored/added in.
pub enum DailyQuestId {
    #[strum(to_string = "Vanishing Journey")]
    VanishingJourney,
    #[strum(to_string = "Chu Chu Island")]
    ChuChuIsland,
    Lachelein,
    Arcana,
    Morass,
    Esfera,
    Moonbridge,
    #[strum(to_string = "Labyrinth of Suffering")]
    LabyrinthOfSuffering,
    Limina,
    Cernium,
    #[strum(to_string = "Hotel Arcus")]
    HotelArcus,
    Odium,
    #[strum(to_string = "Shangri-La")]
    ShangriLa,
    Arteria,
    Carcion,
    Tallahart,
}

/// Fixed navigation data for a [`DailyQuestId`] - see [`DailyQuestId::navigation`].
#[derive(Clone, Debug)]
pub struct DailyQuestNavigation {
    pub region: WorldMapRegion,
    pub dropdown_path: Vec<String>,
    pub location_label: String,
    pub location_point: (i32, i32),
    pub sub_location_label: Option<String>,
}

impl DailyQuestId {
    /// The fixed world map navigation for this quest, captured empirically from reference
    /// screenshots - each `location_point` is the exact pixel the cursor's fingertip was pointing
    /// at (verified against a fresh screenshot per entry), as an offset from the `WORLD MAP` title
    /// anchor's top-left corner.
    ///
    /// - Pick `region` from the top dropdown, then each entry of `dropdown_path` in order from
    ///   the next dropdown(s) it reveals (e.g. `["Tenebris", "Moonbridge"]` to drill from Arcane
    ///   River down to Tenebris's Moonbridge sub-map). Every dropdown's option list is
    ///   always-visible text, found via OCR.
    /// - Once there, double-click `location_point` directly. Matching the node's banner via OCR
    ///   was tried instead, but scanning the whole map content area for text is slow enough to
    ///   stall the tick loop for seconds at a time (observed multi-second "ticking running late"
    ///   spikes) - a fixed, verified pixel is instant and, unlike the very first unverified
    ///   attempt at hardcoding this, has actually been checked against a real screenshot per
    ///   entry.
    /// - Some nodes (e.g. two hunting grounds sharing one icon) lead to an intermediate view
    ///   instead of an immediate teleport prompt - `sub_location_label`, if set, is
    ///   double-clicked there to reach the actual target.
    pub fn navigation(self) -> DailyQuestNavigation {
        fn nav(
            region: WorldMapRegion,
            dropdown_path: &[&str],
            location_label: &str,
            location_point: (i32, i32),
            sub_location_label: Option<&str>,
        ) -> DailyQuestNavigation {
            DailyQuestNavigation {
                region,
                dropdown_path: dropdown_path.iter().map(|path| path.to_string()).collect(),
                location_label: location_label.to_string(),
                location_point,
                sub_location_label: sub_location_label.map(str::to_string),
            }
        }

        use WorldMapRegion::{ArcaneRiver, Grandis};

        match self {
            DailyQuestId::VanishingJourney => nav(
                ArcaneRiver,
                &["Vanishing Journey"],
                "Extinction Zone : Spirit Zone",
                (525, 109),
                None,
            ),
            DailyQuestId::ChuChuIsland => nav(
                ArcaneRiver,
                &["Chu Chu Island"],
                "Slurpy Forest : Bitty-Bobble Forest 1",
                (423, 433),
                None,
            ),
            DailyQuestId::Lachelein => nav(
                ArcaneRiver,
                &["Lachelein, the Dreaming City"],
                "Lachelein Ballroom : Revelation Place 3",
                (545, 420),
                None,
            ),
            DailyQuestId::Arcana => nav(
                ArcaneRiver,
                &["Arcana, The Mysterious Forest"],
                "Arcana : Cavern Lower Path",
                (468, 422),
                None,
            ),
            DailyQuestId::Morass => nav(
                ArcaneRiver,
                &["Morass, Swamp of Memory"],
                "Morass : Shadowdance Hall 4",
                (416, 420),
                None,
            ),
            DailyQuestId::Esfera => nav(
                ArcaneRiver,
                &["Esfera, The Origin Sea"],
                "Esfera : Mirror-touched Sea 3",
                (490, 470),
                None,
            ),
            DailyQuestId::Moonbridge => nav(
                ArcaneRiver,
                &["Tenebris", "Moonbridge"],
                "Moonbridge : Void Current 3",
                (604, 455),
                None,
            ),
            DailyQuestId::LabyrinthOfSuffering => nav(
                ArcaneRiver,
                &["Tenebris", "Labyrinth of Suffering"],
                "Tenebris : Labyrinth of Suffering Deep Core 1",
                (262, 453),
                None,
            ),
            DailyQuestId::Limina => nav(
                ArcaneRiver,
                &["Tenebris", "Limina"],
                "Limina : End of the World 2-6",
                (593, 381),
                None,
            ),
            DailyQuestId::Cernium => nav(
                Grandis,
                &["Western Grandis", "Cernium"],
                "Cernium : Royal Library Section 1",
                (399, 151),
                None,
            ),
            DailyQuestId::HotelArcus => nav(
                Grandis,
                &["Western Grandis", "Hotel Arcus"],
                "Hotel Arcus : Nostalgic Drive-in Theater 4",
                (501, 320),
                None,
            ),
            DailyQuestId::Odium => nav(
                Grandis,
                &["Western Grandis", "Odium"],
                "Odium : Captured Alley 2",
                (426, 288),
                None,
            ),
            DailyQuestId::ShangriLa => nav(
                Grandis,
                &["Western Grandis", "Shangri-La"],
                "Shangri-La : Blooming Spring 2",
                (226, 475),
                None,
            ),
            DailyQuestId::Arteria => nav(
                Grandis,
                &["Western Grandis", "Arteria"],
                "Empress Road : Southern Outskirts",
                (271, 473),
                None,
            ),
            DailyQuestId::Carcion => nav(
                Grandis,
                &["Western Grandis", "Carcion"],
                "Carcion : Giant Coral Colony 3",
                (198, 176),
                None,
            ),
            DailyQuestId::Tallahart => nav(
                Grandis,
                &["Western Grandis", "Tallahart"],
                "Tallahart : Silent Ashlands 3",
                (192, 468),
                None,
            ),
        }
    }

    /// The fixed hunting bound for this quest's map, as provided by the user.
    pub fn bound(self) -> Bound {
        match self {
            DailyQuestId::VanishingJourney => Bound {
                x: 18,
                y: 11,
                width: 129,
                height: 41,
            },
            DailyQuestId::ChuChuIsland => Bound {
                x: 25,
                y: 7,
                width: 110,
                height: 34,
            },
            DailyQuestId::Lachelein => Bound {
                x: 25,
                y: 10,
                width: 120,
                height: 30,
            },
            DailyQuestId::Arcana => Bound {
                x: 26,
                y: 8,
                width: 115,
                height: 45,
            },
            DailyQuestId::Morass => Bound {
                x: 3,
                y: 15,
                width: 203,
                height: 50,
            },
            DailyQuestId::Esfera => Bound {
                x: 13,
                y: 20,
                width: 142,
                height: 50,
            },
            DailyQuestId::Moonbridge => Bound {
                x: 15,
                y: 18,
                width: 145,
                height: 48,
            },
            DailyQuestId::LabyrinthOfSuffering => Bound {
                x: 40,
                y: 10,
                width: 169,
                height: 61,
            },
            DailyQuestId::Limina => Bound {
                x: 34,
                y: 44,
                width: 100,
                height: 38,
            },
            DailyQuestId::Cernium => Bound {
                x: 12,
                y: 17,
                width: 141,
                height: 44,
            },
            DailyQuestId::HotelArcus => Bound {
                x: 26,
                y: 20,
                width: 145,
                height: 38,
            },
            DailyQuestId::Odium => Bound {
                x: 7,
                y: 18,
                width: 157,
                height: 40,
            },
            DailyQuestId::ShangriLa => Bound {
                x: 4,
                y: 11,
                width: 174,
                height: 49,
            },
            DailyQuestId::Arteria => Bound {
                x: 7,
                y: 31,
                width: 153,
                height: 44,
            },
            DailyQuestId::Carcion => Bound {
                x: 7,
                y: 15,
                width: 167,
                height: 45,
            },
            DailyQuestId::Tallahart => Bound {
                x: 10,
                y: 23,
                width: 166,
                height: 45,
            },
        }
    }

    /// The fixed platforms for this quest's map, used for platform-aware auto-mob pathing.
    ///
    /// Captured from the user's own manually-drawn platforms for each hunting ground (exported as
    /// regular [`super::Map`] JSON and cross-referenced by hand), same reasoning as
    /// [`Self::navigation`]'s pixel coordinates - the daily quest solver has no [`super::Map`] of
    /// its own to draw platforms on, so without this the auto-mobbing here was blind to platform
    /// layout entirely (see [`crate::rotator::DefaultRotator::rotate_daily_quest`]).
    pub fn platforms(self) -> Vec<Platform> {
        fn platform(x_start: i32, x_end: i32, y: i32) -> Platform {
            Platform { x_start, x_end, y }
        }

        match self {
            DailyQuestId::Cernium => vec![
                platform(14, 153, 11),
                platform(19, 62, 27),
                platform(72, 101, 27),
                platform(110, 147, 27),
                platform(129, 154, 43),
                platform(98, 121, 44),
                platform(52, 92, 47),
                platform(15, 46, 44),
            ],
            DailyQuestId::HotelArcus => vec![
                platform(27, 44, 40),
                platform(32, 49, 24),
                platform(49, 67, 35),
                platform(55, 80, 48),
                platform(90, 107, 48),
                platform(118, 152, 48),
                platform(77, 109, 35),
                platform(121, 137, 35),
                platform(147, 156, 35),
                platform(160, 169, 29),
                platform(122, 154, 22),
                platform(94, 111, 24),
                platform(60, 84, 24),
                platform(22, 175, 11),
            ],
            DailyQuestId::Odium => vec![
                platform(1, 172, 9),
                platform(14, 41, 22),
                platform(53, 75, 23),
                platform(83, 105, 22),
                platform(117, 145, 23),
                platform(17, 44, 37),
                platform(58, 98, 37),
                platform(122, 144, 37),
            ],
            DailyQuestId::ShangriLa => vec![
                platform(21, 41, 37),
                platform(56, 98, 49),
                platform(112, 125, 49),
                platform(134, 166, 49),
                platform(147, 172, 36),
                platform(100, 137, 36),
                platform(55, 90, 36),
                platform(14, 51, 24),
                platform(77, 90, 24),
                platform(102, 120, 24),
                platform(132, 168, 24),
                platform(1, 186, 10),
            ],
            DailyQuestId::Arteria => vec![
                platform(2, 165, 10),
                platform(21, 32, 24),
                platform(40, 80, 25),
                platform(103, 121, 25),
                platform(132, 150, 24),
                platform(120, 159, 33),
                platform(126, 144, 46),
                platform(9, 50, 38),
                platform(19, 31, 49),
                platform(40, 60, 50),
            ],
            DailyQuestId::Carcion => vec![
                platform(1, 177, 9),
                platform(13, 64, 21),
                platform(78, 93, 30),
                platform(98, 112, 26),
                platform(118, 169, 22),
                platform(18, 72, 33),
                platform(104, 132, 39),
                platform(139, 166, 34),
                platform(135, 162, 48),
                platform(12, 53, 48),
                platform(58, 99, 44),
            ],
            DailyQuestId::VanishingJourney => vec![
                platform(15, 53, 33),
                platform(64, 102, 45),
                platform(111, 149, 33),
                platform(54, 111, 24),
                platform(15, 149, 8),
            ],
            DailyQuestId::ChuChuIsland => vec![
                platform(23, 68, 29),
                platform(74, 92, 23),
                platform(96, 142, 29),
                platform(65, 100, 10),
                platform(19, 147, 5),
            ],
            DailyQuestId::Lachelein => vec![
                platform(33, 56, 41),
                platform(51, 65, 37),
                platform(70, 128, 34),
                platform(113, 146, 47),
                platform(20, 146, 23),
            ],
            DailyQuestId::Arcana => vec![
                platform(52, 125, 48),
                platform(135, 144, 48),
                platform(24, 95, 33),
                platform(53, 126, 18),
                platform(134, 142, 18),
                platform(33, 40, 18),
            ],
            DailyQuestId::Morass => vec![
                platform(26, 56, 50),
                platform(26, 62, 37),
                platform(26, 55, 25),
                platform(67, 79, 17),
                platform(75, 86, 23),
                platform(81, 93, 28),
                platform(89, 100, 33),
                platform(71, 77, 41),
                platform(94, 131, 45),
                platform(138, 155, 33),
                platform(161, 191, 45),
                platform(159, 190, 22),
                platform(100, 136, 22),
                platform(3, 205, 10),
            ],
            DailyQuestId::Esfera => vec![
                platform(36, 71, 52),
                platform(103, 147, 54),
                platform(72, 107, 39),
                platform(110, 146, 29),
                platform(63, 98, 23),
                platform(20, 54, 29),
                platform(21, 47, 41),
                platform(9, 157, 12),
            ],
            DailyQuestId::Moonbridge => vec![
                platform(9, 77, 38),
                platform(36, 72, 53),
                platform(74, 94, 46),
                platform(95, 132, 53),
                platform(89, 157, 38),
                platform(127, 164, 24),
                platform(49, 117, 24),
                platform(4, 40, 24),
                platform(1, 166, 11),
            ],
            DailyQuestId::LabyrinthOfSuffering => vec![
                platform(39, 70, 60),
                platform(81, 150, 60),
                platform(161, 187, 60),
                platform(48, 183, 41),
                platform(39, 59, 26),
                platform(76, 98, 26),
                platform(110, 122, 26),
                platform(133, 155, 26),
                platform(172, 193, 26),
                platform(39, 192, 11),
            ],
            DailyQuestId::Limina => vec![
                platform(29, 51, 55),
                platform(29, 67, 42),
                platform(29, 62, 29),
                platform(29, 138, 11),
                platform(111, 138, 31),
                platform(106, 137, 45),
            ],
            DailyQuestId::Tallahart => vec![
                platform(15, 37, 45),
                platform(20, 56, 20),
                platform(33, 87, 32),
                platform(72, 116, 46),
                platform(102, 155, 32),
                platform(67, 86, 19),
                platform(102, 122, 19),
                platform(133, 168, 20),
                platform(1, 185, 7),
            ],
        }
    }
}
