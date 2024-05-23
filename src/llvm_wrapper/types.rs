use super::context::ContextRef;

use llvm_sys::core::{
    LLVMArrayType2, LLVMDoubleType, LLVMDoubleTypeInContext, LLVMFloatTypeInContext,
    LLVMFunctionType, LLVMInt16TypeInContext, LLVMInt1TypeInContext, LLVMInt32TypeInContext,
    LLVMInt64TypeInContext, LLVMInt8TypeInContext, LLVMPointerType, LLVMVoidTypeInContext,
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

    /// Create Void type in context
    pub fn void_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMVoidTypeInContext(context.get())) }
    }

    /// Create Ptr type in context
    pub fn ptr_type(ptr_raw_type: Self, address_space: u32) -> Self {
        unsafe {
            Self(LLVMPointerType(
                ptr_raw_type.get(),
                c_uint::try_from(address_space).expect("usize casting fail"),
            ))
        }
    }

    /// Create f32 type in context
    pub fn f32_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMFloatTypeInContext(context.get())) }
    }

    /// Create f64 type in context
    pub fn f64_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMDoubleTypeInContext(context.get())) }
    }

    /// Create bool type in context
    pub fn bool_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt1TypeInContext(context.get())) }
    }

    /// Create i8 type in context
    pub fn i8_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt8TypeInContext(context.get())) }
    }

    /// Create i16 type in context
    pub fn i16_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt16TypeInContext(context.get())) }
    }

    /// Create i32 type in context
    pub fn i32_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt32TypeInContext(context.get())) }
    }

    /// Create i64 type in context
    pub fn i64_type(context: &ContextRef) -> Self {
        unsafe { Self(LLVMInt64TypeInContext(context.get())) }
    }

    /// Create array type in context based on Type
    pub fn array_type(array_type: &Self, size: u64) -> Self {
        unsafe { Self(LLVMArrayType2(array_type.get(), size)) }
    }

    /// Create function type based on argument types array, and function return type
    /// TODO: return error
    pub fn function_type(args_type: &[Self], return_type: &Self) -> Self {
        unsafe {
            let args_type = args_type.iter().map(|v| v.0).collect::<Vec<_>>();
            let args = if args_type.is_empty() {
                std::ptr::null_mut()
            } else {
                args_type.as_ptr() as *mut LLVMTypeRef
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
