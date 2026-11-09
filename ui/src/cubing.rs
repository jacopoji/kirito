use backend::{
    CubeKind, CubingSettings, CubingStat, CubingState, CubingStatus, IntoEnumIterator, Settings,
    cubing_state_receiver, query_settings, start_cubing, stop_cubing, upsert_settings,
};
use dioxus::prelude::*;
use futures_util::StreamExt;

use crate::{
    AppState,
    components::{
        ContentSide,
        button::{Button, ButtonStyle},
        checkbox::Checkbox,
        labeled::Labeled,
        numbers::PrimitiveIntegerInput,
        section::Section,
        select::{Select, SelectOption},
    },
};

#[component]
pub fn CubingScreen() -> Element {
    let mut app_settings = use_context::<AppState>().settings;
    let settings_view = use_memo(move || app_settings().unwrap_or_default());
    let cubing = use_memo(move || settings_view().cubing);
    let mut state = use_signal(CubingState::default);
    let running = use_memo(move || state().status.is_running());

    // Saves go through a single coroutine so that they apply in the order they were made.
    let coroutine = use_coroutine(move |mut rx: UnboundedReceiver<Settings>| async move {
        while let Some(new_settings) = rx.next().await {
            app_settings.set(Some(upsert_settings(new_settings).await));
        }
    });
    let save = use_callback(move |cubing: CubingSettings| {
        coroutine.send(Settings {
            cubing,
            ..settings_view.peek().clone()
        });
    });

    use_future(move || async move {
        if app_settings.peek().is_none() {
            app_settings.set(Some(query_settings().await));
        }
    });

    // A run outlives this tab, so this also picks it back up when returning to the tab.
    use_future(move || async move {
        let mut receiver = cubing_state_receiver().await;
        loop {
            state.set(receiver.borrow_and_update().clone());
            if receiver.changed().await.is_err() {
                break;
            }
        }
    });

    let threshold_stats = CubingStat::iter()
        .filter(|stat| stat.supports_threshold())
        .collect::<Vec<_>>();
    let line_match_stats = CubingStat::iter().collect::<Vec<_>>();

    rsx! {
        div { class: "flex flex-col pb-15 h-full overflow-y-auto",
            Section { title: "Cubing",
                div { class: "flex items-center gap-3",
                    Labeled { label: "Cube", class: "w-32",
                        Select::<CubeKind> {
                            disabled: running(),
                            on_selected: move |cube_kind| {
                                save(CubingSettings {
                                    cube_kind,
                                    ..cubing.peek().clone()
                                });
                            },
                            for kind in CubeKind::iter() {
                                SelectOption::<CubeKind> {
                                    value: kind,
                                    label: kind.to_string(),
                                    selected: cubing().cube_kind == kind,
                                    disabled: running(),
                                }
                            }
                        }
                    }
                    div { class: "flex-grow text-xs text-secondary-text",
                        "Open the cube window on the item first. Keep the game focused while rolling. The toggle actions hotkey also stops it."
                    }
                    if running() {
                        Button {
                            class: "w-20",
                            style: ButtonStyle::Danger,
                            on_click: move |_| async move {
                                stop_cubing().await;
                            },
                            "Stop"
                        }
                    } else {
                        Button {
                            class: "w-20",
                            style: ButtonStyle::Primary,
                            on_click: move |_| async move {
                                let settings = cubing.peek().clone();
                                start_cubing(settings).await;
                            },
                            "Start"
                        }
                    }
                }
                div { class: "text-xs text-primary-text pt-3",
                    "{status_text(state().status)} - {state().rolls} roll(s)"
                }
            }
            Section { title: "Stat threshold",
                div { class: "flex items-center gap-6",
                    Labeled {
                        label: "Enable",
                        tooltip: "Stops once the highest total of the stats below reaches the value, summed across all lines. All Stats also count towards STR, DEX, INT and LUK.",
                        tooltip_side: ContentSide::Right,
                        class: "flex-row items-center gap-3",
                        Checkbox {
                            checked: cubing().threshold_enabled,
                            disabled: running(),
                            on_checked: move |threshold_enabled| {
                                save(CubingSettings {
                                    threshold_enabled,
                                    ..cubing.peek().clone()
                                });
                            },
                        }
                    }
                    Labeled { label: "Total %", class: "w-20",
                        PrimitiveIntegerInput::<u32> {
                            value: cubing().threshold,
                            min_value: 1,
                            max_value: 100,
                            disabled: running(),
                            on_value: move |threshold| {
                                save(CubingSettings {
                                    threshold,
                                    ..cubing.peek().clone()
                                });
                            },
                        }
                    }
                }
                StatGrid {
                    stats: threshold_stats,
                    selected: cubing().threshold_stats,
                    disabled: running(),
                    on_selected: move |threshold_stats| {
                        save(CubingSettings {
                            threshold_stats,
                            ..cubing.peek().clone()
                        });
                    },
                }
            }
            Section { title: "Matching lines",
                div { class: "flex items-center gap-6",
                    Labeled {
                        label: "Enable",
                        tooltip: "Stops once the number of lines below rolled any of the selected stats.",
                        tooltip_side: ContentSide::Right,
                        class: "flex-row items-center gap-3",
                        Checkbox {
                            checked: cubing().line_match_enabled,
                            disabled: running(),
                            on_checked: move |line_match_enabled| {
                                save(CubingSettings {
                                    line_match_enabled,
                                    ..cubing.peek().clone()
                                });
                            },
                        }
                    }
                    Labeled { label: "Lines", class: "w-20",
                        PrimitiveIntegerInput::<u32> {
                            value: cubing().line_match_count,
                            min_value: 1,
                            max_value: 3,
                            disabled: running(),
                            on_value: move |line_match_count| {
                                save(CubingSettings {
                                    line_match_count,
                                    ..cubing.peek().clone()
                                });
                            },
                        }
                    }
                }
                StatGrid {
                    stats: line_match_stats,
                    selected: cubing().line_match_stats,
                    disabled: running(),
                    on_selected: move |line_match_stats| {
                        save(CubingSettings {
                            line_match_stats,
                            ..cubing.peek().clone()
                        });
                    },
                }
            }
            Section { title: "Results",
                div { class: "flex flex-col gap-1 text-xs text-secondary-text",
                    if state().log.is_empty() {
                        "No results yet."
                    }
                    for (index, line) in state().log.into_iter().enumerate() {
                        div { key: "{index}", "{line}" }
                    }
                }
            }
        }
    }
}

#[component]
fn StatGrid(
    stats: Vec<CubingStat>,
    selected: Vec<CubingStat>,
    disabled: bool,
    on_selected: Callback<Vec<CubingStat>>,
) -> Element {
    rsx! {
        div { class: "grid grid-cols-3 gap-2 pt-3",
            for stat in stats {
                Labeled {
                    key: "{stat}",
                    label: stat.to_string(),
                    class: "flex-row-reverse justify-end items-center gap-2",
                    Checkbox {
                        checked: selected.contains(&stat),
                        disabled,
                        on_checked: {
                            let selected = selected.clone();
                            move |checked: bool| {
                                let mut updated = selected.clone();
                                updated.retain(|existing| *existing != stat);
                                if checked {
                                    updated.push(stat);
                                }
                                on_selected(updated);
                            }
                        },
                    }
                }
            }
        }
    }
}

fn status_text(status: CubingStatus) -> String {
    match status {
        CubingStatus::Idle => "Idle".to_string(),
        CubingStatus::Starting => "Checking current potential".to_string(),
        CubingStatus::Countdown(seconds) => format!("Focus the game - starting in {seconds}s"),
        CubingStatus::Rolling => "Rolling".to_string(),
        CubingStatus::Stopped(reason) => reason.to_string(),
    }
}
