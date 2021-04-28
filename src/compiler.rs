//! # Compiler
//! Compile from `SemanticStateContext` source with LLVM codegen backend.
//! Apply error flow.
//!
//! Save resultd to `.ll` source and `.o` binary.

use crate::func::FuncCodegen;
use anyhow::{anyhow, bail, ensure};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::semantic::SemanticStackContext;
use thiserror::Error;

const RESULT_LL_FILE: &str = "target/res.ll";
const RESULT_O_FILE: &str = "target/res.o";

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Unexpected semantic context length")]
    UnexpectedSemanticContextLength,
    #[error("Unexpected semantic context for Global length")]
    UnexpectedSemanticContextForGlobalLength,
    #[error("Unexpected semantic global instruction: {0:?}")]
    UnexpectedGlobalInstruction(SemanticStackContext),
    #[error("Failed write result to file: {file}\nwith error: {msg}")]
    WriteResultToFile { file: String, msg: String },
}

#[derive(Debug, Error)]
pub enum TargetError {
    #[error("Failed target initialize native: {0}")]
    TargetInitializeNative(String),
    #[error("Failed get target from triple: {0}")]
    TargetFromTriple(String),
    #[error("Failed create target machine")]
    CreateTargetMachine,
}

/// Apply module ot initialized Target Machine
fn apply_target_to_module(target_machine: &TargetMachine, module: &Module) {
    module.set_triple(&target_machine.get_triple());
    module.set_data_layout(&target_machine.get_target_data().get_data_layout());
}

fn get_native_target_machine() -> anyhow::Result<TargetMachine> {
    Target::initialize_native(&InitializationConfig::default())
        .map_err(TargetError::TargetInitializeNative)?;
    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple)
        .map_err(|v| TargetError::TargetFromTriple(v.to_string()))?;
    target
        .create_target_machine(
            &target_triple,
            &TargetMachine::get_host_cpu_name().to_string(),
            &TargetMachine::get_host_cpu_features().to_string(),
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Small,
        )
        .ok_or_else(|| anyhow!(TargetError::CreateTargetMachine))
}

/// # Compile processing.
/// As a compilation source is `semantic_state` results.
pub fn compile(semantic_state: &State) -> anyhow::Result<()> {
    ensure!(
        semantic_state.context.len() == 1,
        CompileError::UnexpectedSemanticContextLength
    );
    let global_context = semantic_state.global.context.clone().get();
    ensure!(
        global_context.len() == 1,
        CompileError::UnexpectedSemanticContextForGlobalLength
    );
    let ctx = semantic_state.context[0].clone();

    // Init LLVM codegen variables
    let context = Context::create();
    let module = context.create_module("main");
    let builder = context.create_builder();

    // Fetch global context
    for global_ctx in global_context {
        match &global_ctx {
            SemanticStackContext::FunctionDeclaration { fn_decl } => {
                let mut fn_codegen = FuncCodegen::new(&context);
                // Function declaration codegen
                fn_codegen.fn_declaration(&module, fn_decl);
                // Function body codegen
                fn_codegen.func_body(&builder, &ctx, fn_decl)?;
            }
            _ => bail!(CompileError::UnexpectedGlobalInstruction(global_ctx)),
        }
    }

    // Get target machine and apply to LLVM-backend
    let target_machine = get_native_target_machine()?;
    apply_target_to_module(&target_machine, &module);

    // Store result to llvm-ir source file
    module
        .print_to_file(RESULT_LL_FILE)
        .map_err(|err| CompileError::WriteResultToFile {
            file: RESULT_LL_FILE.to_string(),
            msg: err.to_string(),
        })?;

    // Store result to bin file
    Ok(target_machine
        .write_to_file(&module, FileType::Object, RESULT_O_FILE.as_ref())
        .map_err(|err| CompileError::WriteResultToFile {
            file: RESULT_O_FILE.to_string(),
            msg: err.to_string(),
        })?)
}
