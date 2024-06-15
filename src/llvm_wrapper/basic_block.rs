use super::context::ContextRef;
use super::value::ValueRef;
use crate::llvm_wrapper::utils::CString;
use llvm_sys::core::LLVMAppendBasicBlockInContext;
use llvm_sys::prelude::LLVMBasicBlockRef;

/// LLVM Basic block wrapper
pub struct BasicBlockRef(LLVMBasicBlockRef);

impl BasicBlockRef {
    // Get raw basic block reference
    #[must_use]
    pub const fn get(&self) -> LLVMBasicBlockRef {
        self.0
    }

    /// Append basic block in context
    /// TODO: return error
    #[must_use]
    pub fn append_in_context(context: &ContextRef, function: &ValueRef, name: &str) -> Self {
        unsafe {
            let c_name = CString::from(name);
            Self(LLVMAppendBasicBlockInContext(
                **context,
                **function,
                c_name.as_ptr(),
            ))
        }
    }
}
