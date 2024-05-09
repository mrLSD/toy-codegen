use llvm_sys::core::LLVMGetVersion;
use std::fmt::Display;
use std::os::raw::c_uint;

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
