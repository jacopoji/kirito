use std::time::{SystemTime, UNIX_EPOCH};

use backend::{
    Character, DebugState, ThreadOfFateAskProgress, TransparentShapeDifficulty,
    annotate_transparent_shape, auto_record_lie_detector, auto_save_rune, check_cuda_available,
    debug_state_receiver, evaluate_transparent_shape, record_video, test_spin_rune,
    test_transparent_shape, test_violetta, update_character, upsert_character,
};
use dioxus::prelude::*;
use tokio::sync::broadcast::error::RecvError;

use crate::{
    AppState,
    components::{
        button::{Button, ButtonStyle},
        section::Section,
        select::{Select, SelectOption},
    },
};

const GROUND_TRUTH_DIFFICULTIES: [(TransparentShapeDifficulty, &str); 6] = [
    (TransparentShapeDifficulty::Normal, "Normal"),
    (TransparentShapeDifficulty::Hard, "Hard"),
    (TransparentShapeDifficulty::Failure, "Failure"),
    (TransparentShapeDifficulty::Failure2, "Failure 2"),
    (TransparentShapeDifficulty::Failure3, "Failure 3"),
    (TransparentShapeDifficulty::Failure4, "Failure 4"),
];

#[component]
pub fn DebugScreen() -> Element {
    let mut state = use_signal(DebugState::default);
    let mut cuda_status = use_signal::<Option<String>>(|| None);
    let mut character = use_context::<AppState>().character;
    let mut ground_truth_difficulty = use_signal(|| TransparentShapeDifficulty::Normal);

    use_future(move || async move {
        let mut rx = debug_state_receiver().await;
        loop {
            let current_state = match rx.recv().await {
                Ok(state) => state,
                Err(RecvError::Closed) => break,
                Err(RecvError::Lagged(_)) => continue,
            };
            if current_state != *state.peek() {
                state.set(current_state);
            }
        }
    });

    rsx! {
        div { class: "flex flex-col h-full overflow-y-auto",
            Section { title: "Debug",
                div { class: "grid grid-cols-2 gap-3",
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_spin_rune().await;
                        },

                        "Test spin rune"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_violetta().await;
                        },

                        "Test Violetta"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Normal).await;
                        },

                        "Test transparent shape normal"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Hard).await;
                        },

                        "Test transparent shape hard"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Failure).await;
                        },

                        "Test transparent shape failure"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Failure2).await;
                        },

                        "Test transparent shape failure 2"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Failure3).await;
                        },

                        "Test transparent shape failure 3"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async {
                            test_transparent_shape(TransparentShapeDifficulty::Failure4).await;
                        },

                        "Test transparent shape failure 4"
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async move {
                            record_video(!state.peek().is_recording).await;
                        },

                        if state().is_recording {
                            "Stop recording"
                        } else {
                            "Start recording"
                        }
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async move {
                            auto_save_rune(!state.peek().is_rune_auto_saving).await;
                        },

                        if state().is_rune_auto_saving {
                            "Stop auto saving rune"
                        } else {
                            "Start auto saving rune"
                        }
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async move {
                            let recording = state.peek().is_lie_detector_auto_recording;
                            auto_record_lie_detector(!recording).await;
                        },

                        if state().is_lie_detector_auto_recording {
                            "Stop auto record lie detector"
                        } else {
                            "Start auto record lie detector"
                        }
                    }
                    Button {
                        style: ButtonStyle::Secondary,
                        on_click: move |_| async move {
                            cuda_status.set(Some("Checking...".to_string()));
                            let message = check_cuda_available().await;
                            cuda_status.set(Some(message));
                        },

                        "Check CUDA"
                    }
                }
                if let Some(status) = cuda_status() {
                    p { class: "text-xs text-primary-text font-mono whitespace-pre-wrap", "{status}" }
                }
            }
            Section { title: "Transparent Shape Ground Truth",
                div { class: "flex flex-col gap-3",
                    Select::<TransparentShapeDifficulty> {
                        class: "w-full",
                        placeholder: "Select a fixture...",
                        on_selected: move |difficulty| {
                            ground_truth_difficulty.set(difficulty);
                        },

                        for (difficulty , label) in GROUND_TRUTH_DIFFICULTIES {
                            SelectOption::<TransparentShapeDifficulty> {
                                value: difficulty,
                                label: label.to_string(),
                                selected: ground_truth_difficulty() == difficulty,
                            }
                        }
                    }
                    div { class: "grid grid-cols-2 gap-3",
                        Button {
                            style: ButtonStyle::Secondary,
                            on_click: move |_| async move {
                                annotate_transparent_shape(ground_truth_difficulty()).await;
                            },

                            "Annotate ground truth"
                        }
                        Button {
                            style: ButtonStyle::Secondary,
                            on_click: move |_| async move {
                                evaluate_transparent_shape(ground_truth_difficulty()).await;
                            },

                            "Evaluate against ground truth"
                        }
                    }
                }
            }
            Section { title: "Thread of Fate (testing only)",
                div { class: "grid grid-cols-2 gap-3",
                    Button {
                        style: ButtonStyle::Secondary,
                        disabled: character()
                            .and_then(|character| character.thread_of_fate_ask_progress.pending_npc)
                            .is_none(),
                        on_click: move |_| async move {
                            let Some(current) = character.peek().clone() else {
                                return;
                            };
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            let pending_since_secs = Some(
                                now.saturating_sub(ThreadOfFateAskProgress::WAIT_SECS),
                            );
                            let updated = Character {
                                thread_of_fate_ask_progress: ThreadOfFateAskProgress {
                                    pending_since_secs,
                                    ..current.thread_of_fate_ask_progress
                                },
                                ..current
                            };
                            if let Some(saved) = upsert_character(updated).await {
                                character.set(Some(saved.clone()));
                                update_character(Some(saved)).await;
                            }
                        },

                        "Fast-forward pending ask to ready-to-complete"
                    }
                }
            }
        }
    }
}
