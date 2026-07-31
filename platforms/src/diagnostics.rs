//! Process-level resource usage, for diagnosing memory/handle leaks over long-running sessions.

#[cfg(windows)]
pub use crate::windows::ProcessStats;

#[cfg(not(windows))]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessStats {
    pub working_set_mb: f64,
    pub commit_mb: f64,
    pub gdi_objects: u32,
    pub user_objects: u32,
}

/// Gets the current process' memory and GDI/USER handle usage, or `None` if unsupported on this
/// platform or the underlying query failed.
pub fn process_stats() -> Option<ProcessStats> {
    #[cfg(windows)]
    {
        crate::windows::process_stats()
    }
    #[cfg(not(windows))]
    {
        None
    }
}
