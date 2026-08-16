use std::{
    mem,
    sync::{Arc, Mutex},
};

#[cfg(debug_assertions)]
use opencv::{
    core::Size,
    videoio::{VideoWriter, VideoWriterTrait},
};

use crate::services::Event;
#[cfg(test)]
use crate::{Settings, bridge::MockInput, detect::MockDetector};
use crate::{
    bridge::Input,
    buff::BuffEntities,
    detect::Detector,
    minimap::MinimapEntity,
    models::{DailyQuestId, ThreadOfFateNpc},
    notification::Notification,
    operation::Operation,
    player::PlayerEntity,
    rng::Rng,
    skill::SkillEntities,
};
#[cfg(debug_assertions)]
use crate::{debug::save_rune_for_training, run::FPS, solvers::SolvedArrow, utils::DatasetDir};

#[derive(Debug)]
#[cfg(debug_assertions)]
pub struct RecordingHandle {
    writer: VideoWriter,
}

#[cfg(debug_assertions)]
impl RecordingHandle {
    pub fn write(&mut self, detector: &dyn Detector) {
        self.writer.write(&detector.mat()).unwrap()
    }
}

#[derive(Debug, Default)]
#[cfg(debug_assertions)]
pub struct Debug {
    pub auto_save_rune: bool,
    pub auto_record_lie_detector: bool,
    last_rune_detector: Option<Arc<dyn Detector>>,
    last_rune_result: Option<[SolvedArrow; 4]>,
}

#[cfg(debug_assertions)]
impl Debug {
    pub fn new_recording(&self, frame_size: Size) -> RecordingHandle {
        use rand::distr::SampleString;
        use rand_distr::Alphanumeric;

        let id = Alphanumeric.sample_string(&mut rand::rng(), 8);
        let file = DatasetDir::Recordings
            .to_folder()
            .join(format!("ld_{id}.mp4"));
        let fourcc = VideoWriter::fourcc('H', 'V', 'C', '1').unwrap();
        let writer =
            VideoWriter::new(file.to_str().unwrap(), fourcc, FPS as f64, frame_size, true).unwrap();

        RecordingHandle { writer }
    }

    pub fn save_last_rune_result(&self) {
        if !self.auto_save_rune {
            return;
        }

        if let Some((detector, result)) =
            self.last_rune_detector.as_ref().zip(self.last_rune_result)
        {
            save_rune_for_training(&detector.mat(), result);
        }
    }

    pub fn set_last_rune_result(&mut self, detector: Arc<dyn Detector>, result: [SolvedArrow; 4]) {
        self.last_rune_detector = Some(detector);
        self.last_rune_result = Some(result);
    }
}

/// A resource for the tick loop to request that a daily quest's completion be persisted.
///
/// The rotator (see `rotator::DefaultRotator::rotate_daily_quest`) only has access to
/// [`Resources`], not [`crate::services::Services`]' `CharacterService`/database access, so it
/// cannot persist a completion by itself. Instead it pushes here, and
/// `Services::poll` drains this every tick (from a point that does have `CharacterService`
/// access) to actually update and re-save the character. Mirrors [`Notification`]'s shape as the
/// existing precedent for a [`Resources`] field usable directly from deep tick code.
///
/// Each entry carries the id of the character the rotator was actually built for (see
/// [`crate::rotator::DefaultRotator::daily_quest_character_id`]) alongside the completed
/// [`DailyQuestId`] - the currently *selected* character in `CharacterService` can differ from
/// that (e.g. the user switched characters in the UI while a previous character's daily quest run
/// was still in flight), and persisting against whichever character happens to be selected at
/// drain time would silently mark the wrong character's entry as completed.
/// A completed daily quest, paired with the id of the character it was completed for.
type CompletedDailyQuest = (Option<i64>, DailyQuestId);

/// A Thread of Fate ask accepted, paired with the id of the character it was accepted for and the
/// unix timestamp (seconds) it was accepted at.
type AcceptedThreadOfFateAsk = (Option<i64>, ThreadOfFateNpc, u64);

/// A Thread of Fate ask completed (turned in), paired with the id of the character it was
/// completed for.
type CompletedThreadOfFateAsk = Option<i64>;

#[derive(Debug, Default, Clone)]
pub struct CharacterUpdates {
    completed_daily_quests: Arc<Mutex<Vec<CompletedDailyQuest>>>,
    accepted_thread_of_fate_asks: Arc<Mutex<Vec<AcceptedThreadOfFateAsk>>>,
    completed_thread_of_fate_asks: Arc<Mutex<Vec<CompletedThreadOfFateAsk>>>,
}

impl CharacterUpdates {
    pub fn mark_daily_quest_completed(&self, character_id: Option<i64>, id: DailyQuestId) {
        self.completed_daily_quests
            .lock()
            .unwrap()
            .push((character_id, id));
    }

    /// Takes and clears all completed daily quest ids marked since the last drain.
    pub fn drain_completed_daily_quests(&self) -> Vec<CompletedDailyQuest> {
        mem::take(&mut *self.completed_daily_quests.lock().unwrap())
    }

    pub fn mark_thread_of_fate_ask_accepted(
        &self,
        character_id: Option<i64>,
        npc: ThreadOfFateNpc,
        accepted_at_secs: u64,
    ) {
        self.accepted_thread_of_fate_asks.lock().unwrap().push((
            character_id,
            npc,
            accepted_at_secs,
        ));
    }

    /// Takes and clears all accepted Thread of Fate asks marked since the last drain.
    pub fn drain_accepted_thread_of_fate_asks(&self) -> Vec<AcceptedThreadOfFateAsk> {
        mem::take(&mut *self.accepted_thread_of_fate_asks.lock().unwrap())
    }

    pub fn mark_thread_of_fate_ask_completed(&self, character_id: Option<i64>) {
        self.completed_thread_of_fate_asks
            .lock()
            .unwrap()
            .push(character_id);
    }

    /// Takes and clears all completed Thread of Fate asks marked since the last drain.
    pub fn drain_completed_thread_of_fate_asks(&self) -> Vec<CompletedThreadOfFateAsk> {
        mem::take(&mut *self.completed_thread_of_fate_asks.lock().unwrap())
    }
}

/// A struct containing shared resources.
///
/// TODO: Reduce field visibilities.
#[derive(Debug)]
pub struct Resources {
    /// A resource to hold debugging information.
    #[cfg(debug_assertions)]
    pub debug: Debug,
    /// A resource to send inputs.
    pub input: Box<dyn Input>,
    /// A resource for generating random values.
    pub rng: Rng,
    /// A resource for sending notifications through web hook.
    pub notification: Notification,
    /// A resource for requesting a daily quest completion be persisted.
    pub character_updates: CharacterUpdates,
    /// A resource to detect game information.
    ///
    /// This is [`None`] when no frame as ever been captured.
    pub detector: Option<Arc<dyn Detector>>,
    /// A resource indicating current operation state.
    pub operation: Operation,
    /// A resource indicating current tick.
    pub tick: u64,
}

impl Resources {
    #[cfg(test)]
    pub fn new(input: Option<MockInput>, detector: Option<MockDetector>) -> Self {
        use std::{cell::RefCell, rc::Rc};

        use crate::operation::{OperationConfiguration, OperationState};

        Self {
            #[cfg(debug_assertions)]
            debug: Debug::default(),
            input: Box::new(input.unwrap_or_default()),
            rng: Rng::new(rand::random(), rand::random()),
            notification: Notification::new(Rc::new(RefCell::new(Settings::default()))),
            character_updates: CharacterUpdates::default(),
            detector: detector.map(|detector| Arc::new(detector) as Arc<dyn Detector>),
            operation: Operation {
                config: OperationConfiguration {
                    run_timer: false,
                    run_timer_millis: 0,
                },
                state: OperationState::Running,
            },
            tick: 0,
        }
    }

    /// Retrieves a reference to a [`Detector`] for the latest captured frame.
    ///
    /// # Panics
    ///
    /// Panics if no frame has ever been captured.
    #[inline]
    pub fn detector(&self) -> &dyn Detector {
        self.detector
            .as_ref()
            .expect("detector is not available because no frame has ever been captured")
            .as_ref()
    }

    /// Same as [`Self::detector`] but cloned.
    #[inline]
    pub fn detector_cloned(&self) -> Arc<dyn Detector> {
        self.detector
            .as_ref()
            .cloned()
            .expect("detector is not available because no frame has ever been captured")
    }
}

/// Different game-related events.
#[derive(Debug, Clone, Copy)]
pub enum WorldEvent {
    RunTimerEnded,
    PlayerDied,
    MinimapChanged,
    CaptureFailed,
    LieDetectorShapeAppeared,
    LieDetectorViolettaAppeared,
    EliteBossAppeared,
    /// The transparent shape lie detector puzzle was confirmed solved: the puzzle genuinely
    /// opened and was played (not just a misfire on the countdown), and the map hasn't changed
    /// (i.e. no jail teleport) within the grace period after it closed. See the `run.rs` tick
    /// loop for how this is derived.
    LieDetectorShapeSolved,
}

impl Event for WorldEvent {}

/// A container for entities.
#[derive(Debug)]
pub struct World {
    pub minimap: MinimapEntity,
    pub player: PlayerEntity,
    pub skills: SkillEntities,
    pub buffs: BuffEntities,
}
