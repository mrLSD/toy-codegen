use libc::{c_uint, size_t};
use llvm_sys::core::LLVMGetVersion;
use std::fmt::Display;
use std::ops::Deref;

/// c_uint wrapper (from C-type)
pub struct CUint(c_uint);

impl From<u32> for CUint {
    fn from(value: u32) -> Self {
        // Force to unwrap c_uint to u32 with expect fail message
        Self(c_uint::try_from(value).expect("c_unit casting fail from u32"))
    }
}

impl From<usize> for CUint {
    fn from(value: usize) -> Self {
        // Force to unwrap c_uint
        Self(c_uint::try_from(value).expect("c_unit casting fail from usize"))
    }
}

impl Deref for CUint {
    type Target = c_uint;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// size_t wrapper (from C-type)
pub struct SizeT(size_t);

impl From<usize> for SizeT {
    fn from(value: usize) -> Self {
        // Force to unwrap size_t
        Self(size_t::try_from(value).expect("size_t casting fail from usize"))
    }
}

impl Deref for SizeT {
    type Target = size_t;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// CString wrapper
pub struct CString(std::ffi::CString);

impl From<&str> for CString {
    fn from(value: &str) -> Self {
        // Force to unwrap CString
        Self(std::ffi::CString::new(value).expect("CString casting fail from str"))
    }
}

impl Deref for CString {
    type Target = std::ffi::CString;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// LLVM version representation
pub struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

impl Version {
    /// Init and return current LLVM version
    pub fn new() -> Self {
        let mut major: c_uint = 0;
        let mut minor: c_uint = 0;
        let mut patch: c_uint = 0;
        unsafe {
            LLVMGetVersion(&mut major, &mut minor, &mut patch);
        }
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Return LLVM version data: (major, minor, patch)
    pub fn get(&self) -> (u32, u32, u32) {
        (self.minor, self.minor, self.patch)
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
