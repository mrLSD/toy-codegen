#![allow(clippy::module_name_repetitions)]
mod ast;
mod compiler;
mod func;

fn main() -> anyhow::Result<()> {
    let semantic_state = ast::semantic_stack()?;
    compiler::compile(&semantic_state)
}
