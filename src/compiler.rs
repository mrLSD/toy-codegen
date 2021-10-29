//! # Compiler
//! Compile from `SemanticStateContext` source with LLVM codegen backend.
//! Apply error flow.
//!
//! Save result to `.ll` source and `.o` binary.

use crate::ast::{CustomExpression, CustomExpressionInstruction};
use crate::codegen::function::FuncCodegen;
use anyhow::{bail, ensure};
use llvm_lib::builder::BuilderRef;
use llvm_lib::core::context::ContextRef;
use llvm_lib::core::module::ModuleRef;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::semantic::SemanticStackContext;
use std::rc::Rc;
use thiserror::Error;

// const RESULT_LL_FILE: &str = "target/res.ll";
// const RESULT_O_FILE: &str = "target/res.o";

/// Compile stage errors
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Unexpected semantic context length")]
    UnexpectedSemanticContextLength,
    #[error("Unexpected semantic global instruction: {0:?}")]
    UnexpectedGlobalInstruction(SemanticStackContext<CustomExpressionInstruction>),
    #[error("Failed write result to file: {file}\nwith error: {msg}")]
    WriteResultToFile { file: String, msg: String },
}
/*
/// Target init errors
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

/// Get LLVM native target machine
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
*/

/// # Compile processing.
/// As a compilation source is `semantic_state` results.
pub fn compile(
    semantic_state: &State<
        CustomExpression<CustomExpressionInstruction>,
        CustomExpressionInstruction,
    >,
) -> anyhow::Result<()> {
    let global_context = semantic_state.global.context.clone().get();

    let contexts = semantic_state.context.clone();
    ensure!(
        global_context.len() == contexts.len(),
        CompileError::UnexpectedSemanticContextLength
    );
    for ctx in &contexts {
        let b_ctx = ctx.borrow().get_context();
        println!("CTX: {:#?}", b_ctx);
    }

    // Init LLVM codegen variables
    let context = Rc::new(ContextRef::new());
    let module = Rc::new(ModuleRef::new("main"));
    let builder = Rc::new(BuilderRef::new(&context));

    // Fetch global context
    for global_ctx in global_context {
        // println!("\n\nGLOBAL: {global_ctx:#?}");
        match &global_ctx {
            SemanticStackContext::FunctionDeclaration { fn_decl } => {
                let mut fn_codegen =
                    FuncCodegen::new(context.clone(), module.clone(), builder.clone());
                fn_codegen.set();
                // Function declaration codegen
                fn_codegen.func_declaration(fn_decl)?;
                // Function body codegen
                //fn_codegen.function_body(fn_decl)?;
            }
            _ => bail!(CompileError::UnexpectedGlobalInstruction(global_ctx)),
        }
    }
    module.dump_module();

    /*
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
    */
    Ok(())
}
