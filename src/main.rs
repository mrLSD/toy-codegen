#![allow(
    clippy::new_without_default,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

pub mod ast;
mod codegen;
pub mod compiler;
mod func;
pub mod llvm_wrapper;

fn main() -> anyhow::Result<()> {
    // Get semantic state
    let semantic_state = ast::semantic_state()?;
    // Compile from semantic state source
    compiler::compile(&semantic_state)
}
