use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use windows::{
    Win32::{
        Foundation::{HWND, LPARAM},
        Graphics::Dwm::{DWMWA_CLOAKED, DwmGetWindowAttribute},
        System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::{GR_GDIOBJECTS, GR_USEROBJECTS, GetCurrentProcess, GetGuiResources},
        },
        UI::WindowsAndMessaging::{
            DispatchMessageW, EnumWindows, GWL_EXSTYLE, GWL_STYLE, GetMessageW, GetWindowLongPtrW,
            IsWindowVisible, MSG, TranslateMessage, WS_DISABLED, WS_EX_TOOLWINDOW,
        },
    },
    core::BOOL,
};

mod bitblt;
mod handle;
mod input;
mod wgc;
mod window_box;

pub use {bitblt::*, handle::*, input::*, wgc::*, window_box::*};

use crate::{Error, Result, capture::Frame};

#[derive(Debug)]
pub enum WindowsCapture {
    BitBlt(BitBltCapture),
    BitBltArea(WindowBoxCapture),
    Wgc(WgcCapture),
}

impl WindowsCapture {
    #[inline]
    pub fn grab(&mut self) -> Result<Frame> {
        match self {
            WindowsCapture::BitBlt(capture) => capture.grab(),
            WindowsCapture::BitBltArea(capture) => capture.grab(),
            WindowsCapture::Wgc(capture) => capture.grab(),
        }
    }
}

pub fn init() {
    static INITIALIZED: AtomicBool = AtomicBool::new(false);

    if INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Acquire)
        .is_ok()
    {
        let barrier = Arc::new(Barrier::new(2));
        let keys_barrier = barrier.clone();
        thread::spawn(move || {
            let _hook = input::init();
            let mut msg = MSG::default();
            keys_barrier.wait();
            while unsafe { GetMessageW(&raw mut msg, None, 0, 0) }.as_bool() {
                unsafe {
                    let _ = TranslateMessage(&raw const msg);
                    let _ = DispatchMessageW(&raw const msg);
                }
            }
        });
        barrier.wait();
    }
}

impl Error {
    #[inline]
    pub(crate) fn from_last_win_error() -> Error {
        Error::from(windows::core::Error::from_win32())
    }
}

impl From<windows::core::Error> for Error {
    fn from(error: windows::core::Error) -> Self {
        Error::Win32(error.code().0 as u32, error.message())
    }
}

pub fn query_capture_handles() -> Vec<Handle> {
    unsafe extern "system" fn callback(handle: HWND, params: LPARAM) -> BOOL {
        if !unsafe { IsWindowVisible(handle) }.as_bool() {
            return true.into();
        }

        let mut cloaked = 0u32;
        let _ = unsafe {
            DwmGetWindowAttribute(
                handle,
                DWMWA_CLOAKED,
                (&raw mut cloaked).cast(),
                std::mem::size_of::<u32>() as u32,
            )
        };
        if cloaked != 0 {
            return true.into();
        }

        let style = unsafe { GetWindowLongPtrW(handle, GWL_STYLE) } as u32;
        let ex_style = unsafe { GetWindowLongPtrW(handle, GWL_EXSTYLE) } as u32;
        if style & WS_DISABLED.0 != 0 || ex_style & WS_EX_TOOLWINDOW.0 != 0 {
            return true.into();
        }

        let vec_ptr = params.0 as *mut Vec<Handle>;
        let vec = unsafe { &mut *vec_ptr };
        vec.push(Handle::new(HandleKind::Fixed(handle.into())));

        true.into()
    }

    let mut vec = Vec::new();
    let _ = unsafe { EnumWindows(Some(callback), LPARAM(&raw mut vec as isize)) };
    vec
}

/// Raw process resource usage, for diagnosing memory/handle growth over long runs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessStats {
    /// Physical memory currently in the process' working set, in MB.
    pub working_set_mb: f64,
    /// Committed (paging file backed) memory, in MB. Grows with native (e.g. OpenCV/CUDA)
    /// allocations even before they're touched, so it's usually the earliest leak signal.
    pub commit_mb: f64,
    /// Count of GDI objects (bitmaps, device contexts, ...) owned by the process.
    pub gdi_objects: u32,
    /// Count of USER objects (windows, hooks, ...) owned by the process.
    pub user_objects: u32,
}

pub fn process_stats() -> Option<ProcessStats> {
    let process = unsafe { GetCurrentProcess() };
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    unsafe { GetProcessMemoryInfo(process, &raw mut counters, counters.cb) }.ok()?;

    let gdi_objects = unsafe { GetGuiResources(process, GR_GDIOBJECTS) };
    let user_objects = unsafe { GetGuiResources(process, GR_USEROBJECTS) };

    Some(ProcessStats {
        working_set_mb: counters.WorkingSetSize as f64 / 1024.0 / 1024.0,
        commit_mb: counters.PagefileUsage as f64 / 1024.0 / 1024.0,
        gdi_objects,
        user_objects,
    })
}
