use super::context::ContextRef;
use std::ops::Deref;

use crate::llvm_wrapper::utils::CUint;
use llvm_sys::core::{
    LLVMArrayType2, LLVMDoubleTypeInContext, LLVMFloatTypeInContext, LLVMFunctionType,
    LLVMInt16TypeInContext, LLVMInt1TypeInContext, LLVMInt32TypeInContext, LLVMInt64TypeInContext,
    LLVMInt8TypeInContext, LLVMPointerType, LLVMVoidTypeInContext,
};
use llvm_sys::prelude::LLVMTypeRef;

/// LLVM Type structure wrapper
pub struct TypeRef(LLVMTypeRef);

impl TypeRef {
    // Get raw type reference
    #[must_use]
    pub const fn get(&self) -> LLVMTypeRef {
        self.0
    }

    /// Create Void type in context
    #[must_use]
    pub fn void_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMVoidTypeInContext(**context)) }
    }

    /// Create Ptr type in context
    #[must_use]
    pub fn ptr_type(ptr_raw_type: &Self, address_space: u32) -> Self {
        unsafe { Self(LLVMPointerType(**ptr_raw_type, *CUint::from(address_space))) }
    }

    /// Create f32 type in context
    #[must_use]
    pub fn f32_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMFloatTypeInContext(**context)) }
    }

    /// Create f64 type in context
    #[must_use]
    pub fn f64_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMDoubleTypeInContext(**context)) }
    }

    /// Create bool type in context
    #[must_use]
    pub fn bool_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt1TypeInContext(**context)) }
    }

    /// Create i8 type in context
    #[must_use]
    pub fn i8_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt8TypeInContext(**context)) }
    }

    /// Create i16 type in context
    #[must_use]
    pub fn i16_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt16TypeInContext(**context)) }
    }

    /// Create i32 type in context
    #[must_use]
    pub fn i32_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt32TypeInContext(**context)) }
    }

    /// Create i64 type in context
    #[must_use]
    pub fn i64_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt64TypeInContext(**context)) }
    }

    /// Create array type in context based on Type
    #[must_use]
    pub fn array_type(array_type: &Self, size: u64) -> Self {
        unsafe { Self(LLVMArrayType2(**array_type, size)) }
    }

    /// Create function type based on argument types array, and function return type
    /// TODO: return error
    #[must_use]
    pub fn function_type(args_type: &[Self], return_type: &Self) -> Self {
        unsafe {
            let mut args_type = args_type.iter().map(|v| v.0).collect::<Vec<_>>();
            let args = if args_type.is_empty() {
                std::ptr::null_mut()
            } else {
                args_type.as_mut_ptr()
            };
            Self(LLVMFunctionType(
                return_type.0,
                args,
                *CUint::from(args_type.len()),
                0,
            ))
        }
    }
}

impl Deref for TypeRef {
    type Target = LLVMTypeRef;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
