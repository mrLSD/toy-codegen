use crate::llvm_wrapper::utils::CString;
use llvm_sys::core::{LLVMDisposeModule, LLVMDumpModule, LLVMModuleCreateWithName};
use llvm_sys::prelude::LLVMModuleRef;
use std::ops::Deref;

/// LLVM Module wrapper
pub struct ModuleRef(LLVMModuleRef);

impl ModuleRef {
    /// Create LLVM module with name
    ///
    /// ## Panics
    /// If LLVM module creation failed function expected to panic
    #[must_use]
    pub fn new(module_name: &str) -> Self {
        unsafe {
            let c_name = CString::from(module_name);
            let module_ref = LLVMModuleCreateWithName(c_name.as_ptr());
            // Force panic as it's unexpected situation
            assert!(!module_ref.is_null(), "Failed to create LLVM module");

            Self(module_ref)
        }
    }

    /// Get raw module reference
    #[must_use]
    pub const fn get(&self) -> LLVMModuleRef {
        self.0
    }

    /// Dump module to stdout
    pub fn dump_module(&self) {
        unsafe {
            LLVMDumpModule(self.0);
        }
    }
}

impl Deref for ModuleRef {
    type Target = LLVMModuleRef;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for ModuleRef {
    /// Dispose module
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.0);
        }
    }
}
