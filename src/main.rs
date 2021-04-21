#![allow(clippy::module_name_repetitions)]
mod ast;
mod compiler;
mod func;

fn main() -> anyhow::Result<()> {
    // Get semantic state
    let semantic_state = ast::semantic_state()?;
    // Compile from semantic state source
    compiler::compile(&semantic_state)
}
