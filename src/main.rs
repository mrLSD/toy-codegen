#![allow(clippy::new_without_default)]

pub mod ast;
pub mod compiler;
pub mod llvm_wrapper;
// mod func;

fn main() -> anyhow::Result<()> {
    // Get semantic state
    let semantic_state = ast::semantic_state()?;
    // Compile from semantic state source
    compiler::compile(&semantic_state)
}
