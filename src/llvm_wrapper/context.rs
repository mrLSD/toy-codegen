use llvm_sys::core::{LLVMContextCreate, LLVMContextDispose};
use llvm_sys::prelude::LLVMContextRef;
use std::ops::Deref;

/// LLVM Context wrapper
pub struct ContextRef(LLVMContextRef);

impl ContextRef {
    /// Create new LLVM Context
    #[must_use]
    pub fn new() -> Self {
        unsafe { Self(LLVMContextCreate()) }
    }

    /// Get LLVM raw context reference
    #[must_use]
    pub const fn get(&self) -> LLVMContextRef {
        self.0
    }
}

impl Drop for ContextRef {
    /// Dispose  context
    fn drop(&mut self) {
        unsafe {
            LLVMContextDispose(self.0);
        }
    }
}

impl Deref for ContextRef {
    type Target = LLVMContextRef;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
