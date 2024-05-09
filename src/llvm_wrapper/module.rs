use llvm_sys::core::{LLVMDisposeModule, LLVMDumpModule, LLVMModuleCreateWithName};
use llvm_sys::prelude::LLVMModuleRef;
use std::ffi::CString;

/// LLVM Module wrapper
pub struct ModuleRef(LLVMModuleRef);

impl ModuleRef {
    /// Create LLVM module with name
    /// TODO: return error
    pub fn new(module_name: &str) -> Self {
        unsafe {
            let module_name = CString::new(module_name).expect("CString::new failed");
            let module_ref = LLVMModuleCreateWithName(module_name.as_ptr());
            if module_ref.is_null() {
                panic!("Failed to create LLVM module");
            }
            Self(module_ref)
        }
    }

    // Get raw module reference
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

impl Drop for ModuleRef {
    /// Dispose module
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.0);
        }
    }
}
