use super::basic_block::BasicBlockRef;
use super::context::ContextRef;
use super::value::ValueRef;
use llvm_sys::core::{
    LLVMBuildRetVoid, LLVMCreateBuilderInContext, LLVMDisposeBuilder, LLVMPositionBuilderAtEnd,
};
use llvm_sys::prelude::LLVMBuilderRef;

/// LLVM Builder wrapper
pub struct BuilderRef(LLVMBuilderRef);

impl BuilderRef {
    /// Create LLVM module with name
    #[must_use]
    pub fn new(context: &ContextRef) -> Self {
        unsafe { Self(LLVMCreateBuilderInContext(**context)) }
    }

    /// Get raw builder reference
    #[must_use]
    pub const fn get(&self) -> LLVMBuilderRef {
        self.0
    }

    /// Set builder position at end
    pub fn position_at_end(&self, basic_block: &BasicBlockRef) {
        unsafe { LLVMPositionBuilderAtEnd(self.0, basic_block.get()) }
    }

    /// Set and return builder return void value
    #[must_use]
    pub fn build_ret_void(&self) -> ValueRef {
        unsafe { ValueRef::create(LLVMBuildRetVoid(self.0)) }
    }
}

impl Drop for BuilderRef {
    /// Dispose Builder
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.0);
        }
    }
}
