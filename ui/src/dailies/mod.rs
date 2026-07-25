use backend::{
    Character, DailyQuestEntry, DailyQuestId, IntoEnumIterator, WorldMapRegion, update_character,
    upsert_character,
};
use dioxus::prelude::*;

use crate::{
    AppState,
    actions::popup::PopupMobbingKeyInputContent,
    components::{
        button::{Button, ButtonStyle},
        checkbox::Checkbox,
        numbers::PrimitiveIntegerInput,
        popup::{PopupContext, PopupTrigger},
        section::Section,
    },
};

#[component]
pub fn DailiesScreen() -> Element {
    let mut character = use_context::<AppState>().character;
    let disabled = use_memo(move || character().is_none());

    let mut popup_open = use_signal(|| false);

    let save_character = move |updated_character: Character| {
        spawn(async move {
            if let Some(saved) = upsert_character(updated_character).await {
                character.set(Some(saved.clone()));
                update_character(Some(saved)).await;
            }
        });
    };

    let entry_for = move |id: DailyQuestId| -> DailyQuestEntry {
        character()
            .and_then(|character| {
                character
                    .daily_quests
                    .into_iter()
                    .find(|entry| entry.id == id)
            })
            .unwrap_or_else(|| DailyQuestEntry::new(id))
    };

    let set_entry = move |entry: DailyQuestEntry| {
        let Some(mut updated_character) = character() else {
            return;
        };
        match updated_character
            .daily_quests
            .iter_mut()
            .find(|existing| existing.id == entry.id)
        {
            Some(existing) => *existing = entry,
            None => updated_character.daily_quests.push(entry),
        }
        save_character(updated_character);
    };

    rsx! {
        PopupContext {
            open: popup_open,
            on_open: move |open: bool| {
                popup_open.set(open);
            },

            div { class: "flex flex-col pb-15 h-full overflow-y-auto",
                Section { title: "Mobbing key",
                    div { class: "flex items-center gap-3",
                        div { class: "text-xs text-secondary-text flex-grow",
                            "Shared by every daily quest below."
                        }
                        PopupTrigger {
                            Button {
                                style: ButtonStyle::Secondary,
                                disabled: disabled(),
                                "Update mobbing key"
                            }
                        }
                    }
                    div { class: "flex items-center gap-3",
                        div { class: "text-xs text-secondary-text flex-grow",
                            "Attacks mid double jump when nothing else (e.g. grappling) is due that moment, instead of always landing first. Applies to every daily quest below."
                        }
                        Checkbox {
                            checked: character()
                                .map(|character| character.daily_quest_use_key_while_double_jumping)
                                .unwrap_or_default(),
                            disabled: disabled(),
                            on_checked: move |daily_quest_use_key_while_double_jumping| {
                                if let Some(character) = character() {
                                    save_character(Character {
                                        daily_quest_use_key_while_double_jumping,
                                        ..character
                                    });
                                }
                            },
                        }
                    }
                }
                Section { title: "Arcane River",
                    div { class: "flex flex-col",
                        DailyQuestListHeader {}
                        for id in DailyQuestId::iter()
                            .filter(|id| id.navigation().region == WorldMapRegion::ArcaneRiver)
                        {
                            DailyQuestRow {
                                key: "{id}",
                                id,
                                entry: entry_for(id),
                                disabled: disabled(),
                                on_toggle: move |enabled| {
                                    set_entry(DailyQuestEntry {
                                        enabled,
                                        ..entry_for(id)
                                    });
                                },
                                on_kill_target: move |kill_target| {
                                    set_entry(DailyQuestEntry {
                                        kill_target,
                                        ..entry_for(id)
                                    });
                                },
                            }
                        }
                    }
                }
                Section { title: "Grandis",
                    div { class: "flex flex-col",
                        DailyQuestListHeader {}
                        for id in DailyQuestId::iter()
                            .filter(|id| id.navigation().region == WorldMapRegion::Grandis)
                        {
                            DailyQuestRow {
                                key: "{id}",
                                id,
                                entry: entry_for(id),
                                disabled: disabled(),
                                on_toggle: move |enabled| {
                                    set_entry(DailyQuestEntry {
                                        enabled,
                                        ..entry_for(id)
                                    });
                                },
                                on_kill_target: move |kill_target| {
                                    set_entry(DailyQuestEntry {
                                        kill_target,
                                        ..entry_for(id)
                                    });
                                },
                            }
                        }
                    }
                }
            }

            PopupMobbingKeyInputContent {
                on_cancel: move |_| {
                    popup_open.set(false);
                },
                on_value: move |daily_quest_mobbing_key| {
                    if let Some(character) = character() {
                        save_character(Character {
                            daily_quest_mobbing_key,
                            ..character
                        });
                    }
                    popup_open.set(false);
                },
                value: character().map(|character| character.daily_quest_mobbing_key).unwrap_or_default(),
            }
        }
    }
}

const ROW_GRID_CLASS: &str = "grid grid-cols-[24px_1fr_90px] items-center gap-3";

#[component]
fn DailyQuestListHeader() -> Element {
    rsx! {
        div { class: "{ROW_GRID_CLASS} h-6 text-xxs text-tertiary-text",
            div {}
            div { "Quest" }
            div { "Kill target" }
        }
    }
}

#[component]
fn DailyQuestRow(
    id: DailyQuestId,
    entry: DailyQuestEntry,
    disabled: bool,
    on_toggle: Callback<bool>,
    on_kill_target: Callback<u32>,
) -> Element {
    let done_today = entry.is_completed_today();
    let name_class = if done_today {
        "text-tertiary-text"
    } else {
        "text-primary-text"
    };

    rsx! {
        div { class: "{ROW_GRID_CLASS} h-8 px-1 border-b border-secondary-border last:border-b-0 hover:bg-secondary-surface",
            Checkbox { checked: entry.enabled, disabled, on_checked: on_toggle }
            div { class: "text-xs {name_class} text-ellipsis overflow-hidden whitespace-nowrap",
                "{id}"
                if done_today {
                    " (done today)"
                }
            }
            PrimitiveIntegerInput {
                value: entry.kill_target,
                on_value: on_kill_target,
                min_value: 1,
                disabled,
            }
        }
    }
}
