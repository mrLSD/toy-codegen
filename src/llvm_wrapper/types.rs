use super::context::ContextRef;
use llvm_sys::core::{
    LLVMFunctionType, LLVMInt32TypeInContext, LLVMInt64TypeInContext, LLVMInt8TypeInContext,
    LLVMVoidTypeInContext,
};
use llvm_sys::prelude::LLVMTypeRef;
use std::os::raw::c_uint;

/// LLVM Type structure wrapper
pub struct TypeRef(LLVMTypeRef);

impl TypeRef {
    // Get raw type reference
    pub const fn get(&self) -> LLVMTypeRef {
        self.0
    }

    /// Create void type in context
    pub fn void_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMVoidTypeInContext(context.get())) }
    }

    /// Create i8 type in context
    pub fn i8_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt8TypeInContext(context.get())) }
    }

    /// Create i32 type in context
    pub fn i32_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt32TypeInContext(context.get())) }
    }

    /// Create i64 type in context
    pub fn i64_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt64TypeInContext(context.get())) }
    }

    /// Create function type based on argument types array, and function return type
    /// TODO: return error
    pub fn function_type(args_type: &[Self], return_type: &Self) -> Self {
        unsafe {
            let args = if args_type.is_empty() {
                std::ptr::null_mut()
            } else {
                args_type
                    .iter()
                    .map(|v| v.0)
                    .collect::<Vec<_>>()
                    .as_mut_ptr()
            };
            Self(LLVMFunctionType(
                return_type.0,
                args,
                c_uint::try_from(args_type.len()).expect("usize casting fail"),
                0,
            ))
        }
    }
}
