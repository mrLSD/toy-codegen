use super::module::ModuleRef;
use super::types::TypeRef;
use llvm_sys::core::LLVMAddFunction;
use llvm_sys::prelude::LLVMValueRef;
use std::ffi::CString;

/// LLVM Value wrapper
pub struct ValueRef(LLVMValueRef);

impl ValueRef {
    /// Create Value form raw Value reference
    pub fn create(value_ref: LLVMValueRef) -> Self {
        Self(value_ref)
    }

    /// Get raw value reference
    pub const fn get(&self) -> LLVMValueRef {
        self.0
    }

    /// Set add function value based on Function type
    /// TODO: return error
    pub fn add_function(module: &ModuleRef, fn_name: &str, fn_type: &TypeRef) -> Self {
        unsafe {
            let fn_name = CString::new(fn_name).expect("CString::new failed");
            Self(LLVMAddFunction(
                module.get(),
                fn_name.as_ptr(),
                fn_type.get(),
            ))
        }
    }
}
