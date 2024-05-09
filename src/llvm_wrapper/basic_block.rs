use super::context::ContextRef;
use super::value::ValueRef;
use llvm_sys::core::LLVMAppendBasicBlockInContext;
use llvm_sys::prelude::LLVMBasicBlockRef;
use std::ffi::CString;

/// LLVM Basic block wrapper
pub struct BasicBlockRef(LLVMBasicBlockRef);

impl BasicBlockRef {
    // Get raw basic block reference
    pub const fn get(&self) -> LLVMBasicBlockRef {
        self.0
    }

    /// Append basic block in context
    /// TODO: return error
    pub fn append_in_context(context: &ContextRef, function: &ValueRef, name: &str) -> Self {
        unsafe {
            let bb_name = CString::new(name).expect("CString::new failed");
            Self(LLVMAppendBasicBlockInContext(
                context.get(),
                function.get(),
                bb_name.as_ptr(),
            ))
        }
    }
}
