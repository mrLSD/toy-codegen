use llvm_sys::core::{LLVMContextCreate, LLVMContextDispose};
use llvm_sys::prelude::LLVMContextRef;

/// LLVM Context wrapper
pub struct ContextRef(LLVMContextRef);

impl ContextRef {
    /// Create new LLVM Context
    pub fn new() -> Self {
        unsafe { Self(LLVMContextCreate()) }
    }

    /// Get LLVM raw context reference
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
