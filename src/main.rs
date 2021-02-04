use crate::func::FuncCodegen;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;
use semantic_analyzer::ast;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::semantic::SemanticStackContext;

mod func;

fn semantic_stack() -> State {
    let content: ast::Main = vec![ast::MainStatement::Function(ast::FunctionStatement {
        name: ast::FunctionName::new(ast::Ident::new("fn1")),
        result_type: ast::Type::Primitive(ast::PrimitiveTypes::I8),
        parameters: vec![ast::FunctionParameter {
            name: ast::ParameterName::new(ast::Ident::new("x")),
            parameter_type: ast::Type::Primitive(ast::PrimitiveTypes::I8),
        }],
        body: vec![
            ast::BodyStatement::LetBinding(ast::LetBinding {
                name: ast::ValueName::new(ast::Ident::new("y")),
                mutable: true,
                value_type: Some(ast::Type::Primitive(ast::PrimitiveTypes::I8)),
                value: Box::new(ast::Expression {
                    expression_value: ast::ExpressionValue::PrimitiveValue(
                        ast::PrimitiveValue::I8(12),
                    ),
                    operation: Some((
                        ast::ExpressionOperations::Plus,
                        Box::new(ast::Expression {
                            expression_value: ast::ExpressionValue::PrimitiveValue(
                                ast::PrimitiveValue::I8(1),
                            ),
                            operation: None,
                        }),
                    )),
                }),
            }),
            ast::BodyStatement::Binding(ast::Binding {
                name: ast::ValueName::new(ast::Ident::new("y")),
                value: Box::new(ast::Expression {
                    expression_value: ast::ExpressionValue::ValueName(ast::ValueName::new(
                        ast::Ident::new("y"),
                    )),
                    operation: Some((
                        ast::ExpressionOperations::Minus,
                        Box::new(ast::Expression {
                            expression_value: ast::ExpressionValue::PrimitiveValue(
                                ast::PrimitiveValue::I8(20),
                            ),
                            operation: None,
                        }),
                    )),
                }),
            }),
            ast::BodyStatement::Return(ast::Expression {
                expression_value: ast::ExpressionValue::PrimitiveValue(ast::PrimitiveValue::I8(10)),
                operation: None,
            }),
        ],
    })];
    let mut state = State::new();
    state.run(&content);
    assert!(state.errors.is_empty());
    state
}

/// Apply module ot initialized Target Machine
fn apply_target_to_module(target_machine: &TargetMachine, module: &Module) {
    module.set_triple(&target_machine.get_triple());
    module.set_data_layout(&target_machine.get_target_data().get_data_layout());
}

fn get_native_target_machine() -> Result<TargetMachine, String> {
    Target::initialize_native(&InitializationConfig::default())?;
    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple).map_err(|v| v.to_string())?;
    target
        .create_target_machine(
            &target_triple,
            &TargetMachine::get_host_cpu_name().to_string(),
            &TargetMachine::get_host_cpu_features().to_string(),
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Small,
        )
        .ok_or_else(|| String::from("Failed to create target machine"))
}

pub struct Codegen<'a> {
    pub context: &'a Context,
    pub module: Module<'a>,
}

fn compiler(semantic_state: &State) {
    let context = Context::create();
    let module = context.create_module("main");
    let builder = context.create_builder();
    assert_eq!(semantic_state.context.len(), 1);
    let global_context = semantic_state.global.context.clone().get();
    assert_eq!(global_context.len(), 1);
    let ctx = semantic_state.context[0].clone();

    match &global_context[0] {
        SemanticStackContext::FunctionDeclaration { fn_decl } => {
            let mut fnv = FuncCodegen::new(&context);
            fnv.fn_declaration(&module, fn_decl);
            fnv.func_body(&builder, ctx, fn_decl);
        }
        _ => panic!("wrong global_context instruction"),
    }

    let target_machine = get_native_target_machine().unwrap();
    apply_target_to_module(&target_machine, &module);
    module.print_to_file("res.ll").expect("Failed generate");

    target_machine
        .write_to_file(&module, FileType::Object, "res.o".as_ref())
        .unwrap();
}

fn main() {
    let semantic_state = semantic_stack();
    compiler(&semantic_state);
}
