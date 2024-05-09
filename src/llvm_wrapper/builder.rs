use super::basic_block::BasicBlockRef;
use super::context::ContextRef;
use llvm_sys::core::{LLVMCreateBuilderInContext, LLVMDisposeBuilder, LLVMPositionBuilderAtEnd};
use llvm_sys::prelude::LLVMBuilderRef;

/// LLVM Builder wrapper
pub struct BuilderRef(LLVMBuilderRef);

impl BuilderRef {
    /// Create LLVM module with name
    pub fn new(context: &ContextRef) -> Self {
        unsafe { Self(LLVMCreateBuilderInContext(context.get())) }
    }

    /// Get raw builder reference
    pub const fn get(&self) -> LLVMBuilderRef {
        self.0
    }

    /// Set builder position at end
    pub fn position_at_end(&self, basic_block: &BasicBlockRef) {
        unsafe { LLVMPositionBuilderAtEnd(self.0, basic_block.get()) }
    }
    // TODO: fix it
    // /// Set and return builder return void value
    // pub  fn build_ret_void(&self) -> ValueRef {
    //     unsafe { ValueRef(LLVMBuildRetVoid(self.0)) }
    // }
}

impl Drop for BuilderRef {
    /// Dispose Builder
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.0);
        }
    }
}
