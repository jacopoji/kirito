#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![feature(variant_count)]
#![feature(map_try_insert)]
#![feature(push_mut)]
#![feature(iter_intersperse)]

use std::{env::current_exe, io::stdout, string::ToString, sync::LazyLock};

use actions::ActionsScreen;
use backend::{Character, DatabaseEvent, Localization, Map, Settings, database_event_receiver};
use characters::CharactersScreen;
use cubing::CubingScreen;
use dailies::DailiesScreen;
#[cfg(debug_assertions)]
use debug::DebugScreen;
use dioxus::{
    desktop::{
        WindowBuilder,
        tao::platform::windows::WindowBuilderExtWindows,
        wry::dpi::{PhysicalSize, Size},
    },
    prelude::*,
};
use fern::Dispatch;
use log::LevelFilter;
use minimap::MinimapScreen;
use rand::distr::{Alphanumeric, SampleString};
use settings::SettingsScreen;
use tokio::sync::broadcast::error::RecvError;

use crate::localization::LocalizationScreen;

mod actions;
mod characters;
mod components;
mod cubing;
mod dailies;
#[cfg(debug_assertions)]
mod debug;
mod localization;
mod minimap;
mod settings;

const TAILWIND_CSS: Asset = asset!("public/tailwind.css");
const AUTO_NUMERIC_JS: Asset = asset!("public/autoNumeric.min.js");
const SORTABLE_JS: Asset = asset!("public/Sortable.min.js");
const TAB_ACTIONS: &str = "Actions";
const TAB_CHARACTERS: &str = "Characters";
const TAB_DAILIES: &str = "Dailies";
const TAB_CUBING: &str = "Cubing";
const TAB_SETTINGS: &str = "Settings";
const TAB_LOCALIZATION: &str = "Localization";
#[cfg(debug_assertions)]
const TAB_DEBUG: &str = "Debug";

static TABS: LazyLock<Vec<String>> = LazyLock::new(|| {
    vec![
        TAB_ACTIONS.to_string(),
        TAB_CHARACTERS.to_string(),
        TAB_DAILIES.to_string(),
        TAB_CUBING.to_string(),
        TAB_SETTINGS.to_string(),
        TAB_LOCALIZATION.to_string(),
        #[cfg(debug_assertions)]
        TAB_DEBUG.to_string(),
    ]
});

fn main() {
    let level = if cfg!(debug_assertions) {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };
    Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339(std::time::SystemTime::now()),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(level)
        .filter(|metadata| {
            let target = metadata.target();
            target.starts_with("backend") || target.starts_with("ui") || target == "panic"
        })
        .chain(stdout())
        .chain(fern::log_file(current_exe().unwrap().parent().unwrap().join("log.txt")).unwrap())
        .apply()
        .unwrap();
    log_panics::init();

    backend::init();
    let window = WindowBuilder::new()
        .with_drag_and_drop(false)
        .with_inner_size(Size::new(PhysicalSize::new(1024, 483)))
        .with_min_inner_size(Size::new(PhysicalSize::new(320, 483)))
        .with_title(Alphanumeric.sample_string(&mut rand::rng(), 16));
    let cfg = dioxus::desktop::Config::default()
        .with_menu(None)
        .with_window(window);
    dioxus::LaunchBuilder::desktop().with_cfg(cfg).launch(App);
}

#[derive(Clone, Copy)]
pub struct AppState {
    map: Signal<Option<Map>>,
    map_preset: Signal<Option<String>>,
    character: Signal<Option<Character>>,
    settings: Signal<Option<Settings>>,
    localization: Signal<Option<Localization>>,
    position: Signal<(i32, i32)>,
}

#[component]
fn App() -> Element {
    let mut selected_tab = use_signal(|| TAB_CHARACTERS.to_string());
    let mut character = use_signal(|| None);

    use_context_provider(|| AppState {
        map: Signal::new(None),
        map_preset: Signal::new(None),
        character,
        settings: Signal::new(None),
        localization: Signal::new(None),
        position: Signal::new((0, 0)),
    });

    // Picks up changes made outside the UI (e.g. the tick loop marking a daily quest as
    // completed for today, see `backend::ecs::CharacterUpdates`), so the currently selected
    // character stays in sync without requiring the user to reselect it.
    use_future(move || async move {
        let mut rx = database_event_receiver();
        loop {
            let event = match rx.recv().await {
                Ok(value) => value,
                Err(RecvError::Closed) => break,
                Err(RecvError::Lagged(_)) => continue,
            };
            if let DatabaseEvent::CharacterUpdated(updated) = event
                && character
                    .peek()
                    .as_ref()
                    .is_some_and(|current| current.id == updated.id)
            {
                character.set(Some(updated));
            }
        }
    });

    rsx! {
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Script { src: AUTO_NUMERIC_JS }
        document::Script { src: SORTABLE_JS }
        div { class: "flex min-w-3xl lg:min-w-5xl min-h-120 h-full",
            MinimapScreen {}
            div { class: "flex-grow flex flex-col lg:flex-row z-1",
                Tabs {
                    tabs: TABS.clone(),
                    on_select_tab: move |tab| {
                        selected_tab.set(tab);
                    },
                    selected_tab: selected_tab(),
                }
                div { class: "relative w-full h-full overflow-x-hidden overflow-y-auto pl-2 lg:pl-0",
                    match selected_tab().as_str() {
                        TAB_ACTIONS => rsx! {
                            ActionsScreen {}
                        },
                        TAB_CHARACTERS => rsx! {
                            CharactersScreen {}
                        },
                        TAB_DAILIES => rsx! {
                            DailiesScreen {}
                        },
                        TAB_CUBING => rsx! {
                            CubingScreen {}
                        },
                        TAB_SETTINGS => rsx! {
                            SettingsScreen {}
                        },
                        TAB_LOCALIZATION => rsx! {
                            LocalizationScreen {}
                        },
                        #[cfg(debug_assertions)]
                        TAB_DEBUG => rsx! {
                            DebugScreen {}
                        },
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}

#[derive(PartialEq, Props, Clone)]
struct TabsProps {
    tabs: Vec<String>,
    on_select_tab: EventHandler<String>,
    selected_tab: String,
}

#[component]
fn Tabs(
    TabsProps {
        tabs,
        on_select_tab,
        selected_tab,
    }: TabsProps,
) -> Element {
    rsx! {
        div { class: "flex flex-row lg:flex-col px-2 gap-3",
            for tab in tabs {
                Tab {
                    name: tab.clone(),
                    selected: selected_tab == tab,
                    on_click: move |_| {
                        on_select_tab(tab.clone());
                    },
                }
            }
        }
    }
}

#[component]
fn Tab(name: String, selected: bool, on_click: EventHandler) -> Element {
    let selected_class = if selected { "bg-secondary-surface" } else { "" };

    rsx! {
        button {
            class: "flex items-center pl-2 w-32 h-10 {selected_class} hover:bg-secondary-surface",
            onclick: move |_| {
                on_click(());
            },
            p { class: "text-primary-text font-medium", {name} }
        }
    }
}
