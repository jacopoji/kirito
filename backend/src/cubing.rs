//! Reading and judging rolled potential lines.
//!
//! Ported from <https://github.com/jgormigon/aincrad>. [`crate::services::cubing`] drives the
//! actual rolling.

use std::{collections::BTreeMap, sync::LazyLock};

use opencv::core::Rect;
use regex::{Captures, Regex, RegexBuilder};

use crate::models::{CubeKind, CubingSettings, CubingStat};

/// Stats whose lines read as `<name> +<value>%`, in the order a line is tested against them.
///
/// [`CubingStat::MagicAttack`] must stay before [`CubingStat::AttackPower`].
const VALUE_STATS: [CubingStat; 10] = [
    CubingStat::Str,
    CubingStat::Dex,
    CubingStat::Int,
    CubingStat::Luk,
    CubingStat::AllStats,
    CubingStat::MagicAttack,
    CubingStat::AttackPower,
    CubingStat::BossDamage,
    CubingStat::CriticalDamage,
    CubingStat::IgnoreDefense,
];

/// Values a `%` line can roll, used to tell an OCR error from a legitimate number.
const VALID_STAT_VALUES: [u32; 8] = [3, 4, 6, 7, 9, 10, 12, 13];

/// The number of potential lines on an item.
const MAX_LINES: usize = 3;

fn regex(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .expect("valid pattern")
}

/// Extracts the value of a stat from a line if it contains it.
fn stat_regex(stat: CubingStat) -> &'static Regex {
    const VALUE: &str = r"\s*[:.;]?\s*\+?(\d+)%+";

    static PATTERNS: LazyLock<Vec<(CubingStat, Regex)>> = LazyLock::new(|| {
        let pattern = |name: &str, value: &str| regex(&format!("(?:{name}){value}"));
        vec![
            (CubingStat::Str, pattern(r"STR", VALUE)),
            (CubingStat::Dex, pattern(r"DEX", VALUE)),
            (CubingStat::Int, pattern(r"\bI?NT", VALUE)),
            (CubingStat::Luk, pattern(r"LUK", VALUE)),
            (
                CubingStat::AllStats,
                pattern(r"All\s*Stats?|Alistats?|Alstats?|All", VALUE),
            ),
            (
                CubingStat::MagicAttack,
                pattern(r"Magic\s*(?:ATT|A[ti]tack\s*Power)", VALUE),
            ),
            (
                CubingStat::AttackPower,
                pattern(r"ATT|A[ti]tack\s*Power", VALUE),
            ),
            (
                CubingStat::BossDamage,
                pattern(r"B?[BGx]oss\s*Damage", VALUE),
            ),
            (
                CubingStat::CriticalDamage,
                pattern(r"Critical\s*Damage", VALUE),
            ),
            (
                CubingStat::IgnoreDefense,
                regex(&format!(
                    r"Ign[aoe]r[ae]\s*Defense{VALUE}|Attacks\s*ignore\s*(\d+)%\s*Monster"
                )),
            ),
        ]
    });

    PATTERNS
        .iter()
        .find_map(|(candidate, regex)| (*candidate == stat).then_some(regex))
        .expect("only called with value stats")
}

/// Extracts the value of `stat` from a single `line` part, `0` if not found.
fn extract_stat_value(line: &str, stat: CubingStat) -> u32 {
    static MAGIC: LazyLock<Regex> = LazyLock::new(|| regex(r"Magic"));
    static CHANCE_TO_IGNORE: LazyLock<Regex> = LazyLock::new(|| regex(r"chance\s*to\s*ign"));

    let excluded = match stat {
        CubingStat::AttackPower => MAGIC.is_match(line),
        // "X% chance to ignore Y% damage" is damage reduction.
        CubingStat::IgnoreDefense => CHANCE_TO_IGNORE.is_match(line),
        _ => false,
    };
    if excluded {
        return 0;
    }

    stat_regex(stat)
        .captures(line)
        .and_then(|captures| captures.iter().skip(1).flatten().next())
        .and_then(|value| value.as_str().parse().ok())
        .unwrap_or(0)
}

fn split_parts(line: &str) -> impl Iterator<Item = &str> {
    line.split(',').map(str::trim)
}

/// Totals of every value stat across `lines`, with All Stats also added to STR/DEX/INT/LUK.
pub fn stat_totals(lines: &[String]) -> BTreeMap<CubingStat, u32> {
    let mut totals = BTreeMap::new();
    for part in lines.iter().flat_map(|line| split_parts(line)) {
        let found = VALUE_STATS.iter().find_map(|&stat| {
            let value = extract_stat_value(part, stat);
            (value > 0).then_some((stat, value))
        });
        if let Some((stat, value)) = found {
            *totals.entry(stat).or_insert(0) += value;
        }
    }

    if let Some(&all) = totals.get(&CubingStat::AllStats) {
        for stat in [
            CubingStat::Str,
            CubingStat::Dex,
            CubingStat::Int,
            CubingStat::Luk,
        ] {
            *totals.entry(stat).or_insert(0) += all;
        }
    }
    totals
}

/// Whether a single line part rolled `stat`.
fn part_matches_stat(part: &str, stat: CubingStat) -> bool {
    static ITEM_DROP_RATE: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r"(?:Item|tem|Itern|ltem|Itam|ltam)\s*(?:(?:Drop|Dr0p|Drap)\s*(?:Rate|Rat3)|Acquisition\s*Rate)",
        )
    });
    static MESO_OBTAINED: LazyLock<Regex> =
        LazyLock::new(|| regex(r"\b(?:Mesos?|Mes0|Mes\s+o)\s*(?:Obtained|Obtain3d)"));
    static SKILL_COOLDOWN: LazyLock<Regex> =
        LazyLock::new(|| regex(r"sk[il]{3}coo[il]d[oa]wns?[12](?:sec|s)"));

    match stat {
        CubingStat::ItemDropRate => ITEM_DROP_RATE.is_match(part),
        CubingStat::MesoObtained => MESO_OBTAINED.is_match(part),
        CubingStat::SkillCooldown => {
            let compact = part
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>();
            SKILL_COOLDOWN.is_match(&compact)
        }
        _ => extract_stat_value(part, stat) > 0,
    }
}

/// Formats totals for display, e.g. `STR: 9, Boss Damage: 40`.
pub fn format_totals(totals: &BTreeMap<CubingStat, u32>) -> String {
    totals
        .iter()
        .map(|(stat, value)| format!("{stat}: {value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Checks `lines` against the rules in `settings`, returning why they pass if they do.
pub fn evaluate(lines: &[String], settings: &CubingSettings) -> Option<String> {
    passes_threshold(lines, settings).or_else(|| passes_line_match(lines, settings))
}

fn passes_threshold(lines: &[String], settings: &CubingSettings) -> Option<String> {
    if !settings.threshold_enabled {
        return None;
    }

    let totals = stat_totals(lines);
    let highest = settings
        .threshold_stats
        .iter()
        .map(|stat| totals.get(stat).copied().unwrap_or(0))
        .max()?;
    (highest >= settings.threshold)
        .then(|| format!("highest {highest} reaches threshold {}", settings.threshold))
}

fn passes_line_match(lines: &[String], settings: &CubingSettings) -> Option<String> {
    let required = settings.line_match_count as usize;
    if !settings.line_match_enabled
        || settings.line_match_stats.is_empty()
        || !(1..=MAX_LINES).contains(&required)
    {
        return None;
    }

    let matching = lines
        .iter()
        .flat_map(|line| split_parts(line))
        .filter(|part| {
            settings
                .line_match_stats
                .iter()
                .any(|&stat| part_matches_stat(part, stat))
        })
        .count();
    (matching >= required).then(|| format!("{required} line(s) match"))
}

/// Repairs common OCR misreads of a single line.
pub fn normalize_line(line: &str) -> String {
    static PERCENT_AS_5: LazyLock<Regex> = LazyLock::new(|| regex(r"\+(\d+)5(\s|$|,)"));
    static PERCENT_AS_96: LazyLock<Regex> = LazyLock::new(|| regex(r"\+(\d+)96%*(\s|$|,)"));
    static PERCENT_AS_9: LazyLock<Regex> = LazyLock::new(|| regex(r"\+(\d+)9%*(\s|$|,)"));
    static PERCENT_AS_0: LazyLock<Regex> = LazyLock::new(|| regex(r"\+(\d+)0(\s|$|,)"));
    static STRAY_DIGIT: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r"((?:STR|DEX|\bI?NT|LUK|All\s*Stats?|Alistats?|Alstats?|ATT|A[ti]tack\s*Power)\s*[:.;]?\s*\+?)(\d+)[059]%",
        )
    });
    static DOUBLE_PERCENT: LazyLock<Regex> = LazyLock::new(|| regex(r"(\d+)%%+"));
    static MISSING_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r"\b(?:(STR)\s*\+?\s*STR|(DEX)\s*\+?\s*DEX|(INT)\s*\+?\s*INT|(LUK)\s*\+?\s*LUK|(ATT)\s*\+?\s*ATT)\s*%",
        )
    });
    static LEADING_NOISE: LazyLock<Regex> = LazyLock::new(|| regex(r"^[^A-Za-z0-9\s]+\s*"));

    fn prefix(captures: &Captures<'_>) -> Option<u32> {
        captures[1].parse().ok()
    }

    // "+95" is "+9%"; "+15" and anything already followed by "%" are real numbers.
    let line = PERCENT_AS_5.replace_all(line, |captures: &Captures<'_>| {
        if prefix(captures).is_some_and(|value| value >= 2) {
            format!("+{}%{}", &captures[1], &captures[2])
        } else {
            captures[0].to_string()
        }
    });
    let line = PERCENT_AS_96.replace_all(&line, |captures: &Captures<'_>| {
        if prefix(captures).is_some_and(|value| VALID_STAT_VALUES.contains(&value)) {
            format!("+{}%{}", &captures[1], &captures[2])
        } else {
            captures[0].to_string()
        }
    });
    let line = PERCENT_AS_9.replace_all(&line, |captures: &Captures<'_>| {
        if prefix(captures).is_some_and(|value| VALID_STAT_VALUES.contains(&value)) {
            format!("+{}%{}", &captures[1], &captures[2])
        } else {
            captures[0].to_string()
        }
    });
    let line = PERCENT_AS_0.replace_all(&line, |captures: &Captures<'_>| {
        if prefix(captures).is_some_and(|value| VALID_STAT_VALUES.contains(&value)) {
            format!("+{}%{}", &captures[1], &captures[2])
        } else {
            captures[0].to_string()
        }
    });
    let line = DOUBLE_PERCENT.replace_all(&line, "$1%");
    // These stats top out at 13%, so "+30%" is "+3%". Stats that do roll 30% don't match.
    let line = STRAY_DIGIT.replace_all(&line, |captures: &Captures<'_>| {
        if captures[2]
            .parse()
            .is_ok_and(|value: u32| VALID_STAT_VALUES.contains(&value))
        {
            format!("{}{}%", &captures[1], &captures[2])
        } else {
            captures[0].to_string()
        }
    });
    // The number was read as the stat name. 9 is the most common value.
    let line = MISSING_NUMBER.replace_all(&line, |captures: &Captures<'_>| {
        let stat = captures
            .iter()
            .skip(1)
            .flatten()
            .next()
            .map(|stat| stat.as_str().to_uppercase())
            .unwrap_or_default();
        format!("{stat} +9%")
    });

    LEADING_NOISE.replace(line.trim(), "").trim().to_string()
}

/// Whether a line looks like it holds a stat rather than OCR noise or a wrapped second row.
fn is_stat_line(line: &str) -> bool {
    split_parts(line).any(|part| {
        VALUE_STATS
            .iter()
            .chain(&[
                CubingStat::ItemDropRate,
                CubingStat::MesoObtained,
                CubingStat::SkillCooldown,
            ])
            .any(|&stat| part_matches_stat(part, stat))
    })
}

/// Picks up to [`MAX_LINES`] potential lines from the OCR'd `rows`.
pub fn select_lines(rows: &[String]) -> Vec<String> {
    let rows = rows
        .iter()
        .map(|row| normalize_line(row))
        .filter(|row| !row.is_empty())
        .collect::<Vec<_>>();
    if rows.len() <= MAX_LINES {
        return rows;
    }

    // More rows than lines means one of them wrapped or is noise.
    let stat_lines = rows
        .iter()
        .filter(|row| is_stat_line(row))
        .take(MAX_LINES)
        .cloned()
        .collect::<Vec<_>>();
    if stat_lines.is_empty() {
        rows.into_iter().take(MAX_LINES).collect()
    } else {
        stat_lines
    }
}

/// Where the rolled potential lines are relative to the Reset button.
pub fn potential_lines_roi(kind: CubeKind, reset_button: Rect) -> Rect {
    let (offset_x, offset_above, width, height) = match kind {
        CubeKind::Glowing => (-76, 388, 284, 107),
        CubeKind::Bright => (167, 288, 184, 97),
    };
    Rect::new(
        (reset_button.x + offset_x).max(0),
        (reset_button.y - offset_above).max(0),
        width,
        height,
    )
}

/// Tracks recent rolls to notice when rolling has stopped changing the potential.
#[derive(Debug, Default)]
pub struct RollHistory {
    recent: Vec<BTreeMap<CubingStat, u32>>,
}

impl RollHistory {
    const REQUIRED: usize = 5;

    /// Records a roll and returns whether the last [`Self::REQUIRED`] rolls were identical.
    ///
    /// Compares extracted stats instead of text because OCR varies between frames. Rolls without
    /// any recognized stat are not evidence of anything and break the streak.
    pub fn push_is_stuck(&mut self, lines: &[String]) -> bool {
        let totals = stat_totals(lines);
        if self.recent.len() == Self::REQUIRED {
            self.recent.remove(0);
        }
        self.recent.push(totals);

        self.recent.len() == Self::REQUIRED
            && self.recent.iter().all(|totals| !totals.is_empty())
            && self.recent.windows(2).all(|pair| pair[0] == pair[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|line| line.to_string()).collect()
    }

    fn line_match(stats: Vec<CubingStat>, count: u32) -> CubingSettings {
        CubingSettings {
            line_match_enabled: true,
            line_match_stats: stats,
            line_match_count: count,
            ..CubingSettings::default()
        }
    }

    fn threshold(stats: Vec<CubingStat>, threshold: u32) -> CubingSettings {
        CubingSettings {
            threshold_enabled: true,
            threshold_stats: stats,
            threshold,
            ..CubingSettings::default()
        }
    }

    #[test]
    fn normalize_repairs_percent_misreads() {
        assert_eq!(normalize_line("Attack Power +95"), "Attack Power +9%");
        assert_eq!(normalize_line("INT +796"), "INT +7%");
        assert_eq!(normalize_line("All Stats +49%"), "All Stats +4%");
        assert_eq!(normalize_line("[ LUK: +129"), "LUK: +12%");
        assert_eq!(normalize_line("p DEX: +30"), "p DEX: +3%");
        assert_eq!(normalize_line("P LUK: +30%"), "P LUK: +3%");
        assert_eq!(normalize_line("Magic ATT: +120%"), "Magic ATT: +12%");
        assert_eq!(normalize_line("All Stats +60%"), "All Stats +6%");
        assert_eq!(normalize_line("LUK: +12%%"), "LUK: +12%");
        assert_eq!(normalize_line("Luk +Luk%"), "LUK +9%");
        assert_eq!(normalize_line("@ Attack Power +9%"), "Attack Power +9%");
    }

    #[test]
    fn normalize_keeps_real_values() {
        assert_eq!(normalize_line("Boss Damage +35%"), "Boss Damage +35%");
        assert_eq!(normalize_line("Attack Power +15"), "Attack Power +15");
        assert_eq!(normalize_line("DEX: +12"), "DEX: +12");
        assert_eq!(normalize_line("LUK: +10%"), "LUK: +10%");
        assert_eq!(normalize_line("STR: +12%"), "STR: +12%");
        assert_eq!(normalize_line("Boss Damage +30%"), "Boss Damage +30%");
        assert_eq!(normalize_line("Ignore Defense +30%"), "Ignore Defense +30%");
        assert_eq!(normalize_line("Item Drop Rate +20%"), "Item Drop Rate +20%");
        assert_eq!(
            normalize_line("Attacks ignore 30% Monster"),
            "Attacks ignore 30% Monster"
        );
        assert_eq!(normalize_line("Max HP +300"), "Max HP +300");
        assert_eq!(normalize_line("Ignore Defense +40%"), "Ignore Defense +40%");
    }

    #[test]
    fn totals_add_all_stats_to_main_stats() {
        let totals = stat_totals(&lines(&[
            "DEX: +12%",
            "Skill MP Cost -30%",
            "All Stats +6%",
        ]));

        assert_eq!(totals[&CubingStat::Dex], 18);
        assert_eq!(totals[&CubingStat::Str], 6);
        assert_eq!(totals[&CubingStat::AllStats], 6);
        assert!(!totals.contains_key(&CubingStat::AttackPower));
    }

    #[test]
    fn totals_do_not_confuse_magic_attack_with_attack_power() {
        let totals = stat_totals(&lines(&["Magic ATT: +9%", "MagicAttackPower +12%"]));

        assert_eq!(totals[&CubingStat::MagicAttack], 21);
        assert!(!totals.contains_key(&CubingStat::AttackPower));
    }

    #[test]
    fn flat_stats_do_not_count() {
        assert!(stat_totals(&lines(&["Attack Power +32", "DEX: +12"])).is_empty());
    }

    #[test]
    fn threshold_uses_highest_selected_stat() {
        let rolled = lines(&["LUK: +12%", "LUK: +9%", "DEX: +12%"]);

        assert!(evaluate(&rolled, &threshold(vec![CubingStat::Luk], 21)).is_some());
        assert!(evaluate(&rolled, &threshold(vec![CubingStat::Dex], 21)).is_none());
        assert!(
            evaluate(
                &rolled,
                &threshold(vec![CubingStat::Dex, CubingStat::Luk], 21)
            )
            .is_some()
        );
        assert!(evaluate(&rolled, &threshold(vec![], 1)).is_none());
    }

    #[test]
    fn a_stray_digit_does_not_falsely_pass_the_threshold() {
        let rolled = select_lines(&lines(&["E STR: +6%", "P LUK: +30%", "Es INT: +6%"]));

        assert_eq!(stat_totals(&rolled)[&CubingStat::Luk], 3);
        assert!(evaluate(&rolled, &threshold(vec![CubingStat::Luk], 15)).is_none());
    }

    #[test]
    fn totals_read_misread_separator_and_clipped_int() {
        let rolled = select_lines(&lines(&["E STR. +6%", "is NT: +6%", "I LUK: +3%"]));
        let totals = stat_totals(&rolled);

        assert_eq!(totals[&CubingStat::Str], 6);
        assert_eq!(totals[&CubingStat::Int], 6);
        assert_eq!(totals[&CubingStat::Luk], 3);
    }

    #[test]
    fn disabled_rules_never_pass() {
        let rolled = lines(&["LUK: +12%", "LUK: +12%", "LUK: +12%"]);
        let settings = CubingSettings {
            threshold_stats: vec![CubingStat::Luk],
            threshold: 1,
            line_match_stats: vec![CubingStat::Luk],
            ..CubingSettings::default()
        };

        assert!(evaluate(&rolled, &settings).is_none());
    }

    #[test]
    fn line_match_counts_lines_of_any_selected_stat() {
        let rolled = lines(&["Boss Damage +40%", "Ignore Defense +30%", "STR: +9%"]);
        let stats = vec![CubingStat::BossDamage, CubingStat::IgnoreDefense];

        assert!(evaluate(&rolled, &line_match(stats.clone(), 2)).is_some());
        assert!(evaluate(&rolled, &line_match(stats, 3)).is_none());
    }

    #[test]
    fn line_match_handles_non_percent_stats() {
        let rolled = lines(&[
            "Item Drop Rate +20%",
            "Mesos Obtained +20%",
            "Skill Cooldowns -2 sec",
        ]);

        for stat in [
            CubingStat::ItemDropRate,
            CubingStat::MesoObtained,
            CubingStat::SkillCooldown,
        ] {
            assert!(evaluate(&rolled, &line_match(vec![stat], 1)).is_some());
        }
    }

    #[test]
    fn line_match_excludes_lookalike_lines() {
        let rolled = lines(&[
            "10% chance to ignore 40% damage when attacked",
            "Critical Rate +9%",
            "Damage +9%",
        ]);

        for stat in [
            CubingStat::IgnoreDefense,
            CubingStat::CriticalDamage,
            CubingStat::BossDamage,
        ] {
            assert!(evaluate(&rolled, &line_match(vec![stat], 1)).is_none());
        }
    }

    #[test]
    fn select_lines_drops_wrapped_rows_when_too_many() {
        let rows = lines(&[
            "Attacks ignore 35% Monster",
            "Defense",
            "Boss Damage +40%",
            "STR: +9%",
        ]);

        assert_eq!(
            select_lines(&rows),
            lines(&["Attacks ignore 35% Monster", "Boss Damage +40%", "STR: +9%"])
        );
    }

    #[test]
    fn select_lines_keeps_unrecognized_lines_when_they_fit() {
        let rows = lines(&["Skill MP Cost -30%", "", "DEX: +12%"]);

        assert_eq!(
            select_lines(&rows),
            lines(&["Skill MP Cost -30%", "DEX: +12%"])
        );
    }

    #[test]
    fn history_requires_five_identical_recognized_rolls() {
        let rolled = lines(&["STR: +9%"]);
        let mut history = RollHistory::default();

        for _ in 0..4 {
            assert!(!history.push_is_stuck(&rolled));
        }
        assert!(history.push_is_stuck(&rolled));
    }

    #[test]
    fn history_streak_breaks_on_a_different_roll() {
        let mut history = RollHistory::default();

        for _ in 0..4 {
            assert!(!history.push_is_stuck(&lines(&["STR: +9%"])));
        }
        assert!(!history.push_is_stuck(&lines(&["DEX: +9%"])));
        assert!(!history.push_is_stuck(&lines(&["DEX: +9%"])));
    }

    #[test]
    fn history_ignores_rolls_without_stats() {
        let mut history = RollHistory::default();

        for _ in 0..10 {
            assert!(!history.push_is_stuck(&lines(&["Skill MP Cost -30%"])));
        }
    }

    #[test]
    fn roi_is_clamped_to_the_frame_origin() {
        let roi = potential_lines_roi(CubeKind::Glowing, Rect::new(10, 20, 132, 38));

        assert_eq!(roi, Rect::new(0, 0, 284, 107));
    }
}
