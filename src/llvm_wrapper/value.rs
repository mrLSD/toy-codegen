use super::module::ModuleRef;
use super::types::TypeRef;
use crate::llvm_wrapper::utils::{CString, CUint, SizeT};
use llvm_sys::core::{LLVMAddFunction, LLVMGetParam, LLVMSetValueName2};
use llvm_sys::prelude::LLVMValueRef;
use std::ops::Deref;
use std::rc::Rc;

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

    /// Get function parameter by index
    pub fn get_func_param(func_value: Rc<Self>, index: usize) -> Self {
        unsafe { Self(LLVMGetParam(**func_value, *CUint::from(index))) }
    }

    /// Set value name, by default in LLVM values monotonic increased
    pub fn set_value_name(&self, name: &str) {
        unsafe {
            let c_name = CString::from(name);
            LLVMSetValueName2(**self, c_name.as_ptr(), *SizeT::from(name.len()));
        }
    }

    /// Set add function value based on Function type
    /// TODO: return error
    pub fn add_function(module: &ModuleRef, fn_name: &str, fn_type: &TypeRef) -> Self {
        unsafe {
            let c_name = CString::from(fn_name);
            Self(LLVMAddFunction(**module, c_name.as_ptr(), **fn_type))
        }
    }
}

impl Deref for ValueRef {
    type Target = LLVMValueRef;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
