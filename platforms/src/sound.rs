//! Plays a short alert sound, e.g. to notify the user of an important in-game event.

/// Plays `sound_path` (a `.wav` file) if given and playable, otherwise falls back to the OS'
/// default alert sound.
///
/// Does nothing if unsupported on this platform.
pub fn play_alert(sound_path: Option<&str>) {
    #[cfg(windows)]
    {
        crate::windows::play_alert(sound_path);
    }
    #[cfg(not(windows))]
    {
        let _ = sound_path;
    }
}
