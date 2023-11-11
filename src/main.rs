use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{
    BasicMetadataTypeEnum, BasicType, FloatType, FunctionType, IntType, StringRadix,
};
use inkwell::values::{FunctionValue, PointerValue};
use inkwell::OptimizationLevel;
use semantic_analyzer::ast;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::block_state::BlockState;
use semantic_analyzer::types::expression::ExpressionResultValue;
use semantic_analyzer::types::semantic::SemanticStackContext;
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::{FunctionStatement, PrimitiveValue};
use std::cell::RefCell;
use std::rc::Rc;

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
                        ast::ExpressionOperations::Plus,
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
            RelocMode::PIC,
            CodeModel::Large,
        )
        .ok_or_else(|| String::from("Failed to create target machine"))
}

fn get_fn_type<'a>(
    context: &'a Context,
    ty: &PrimitiveTypes,
    param_types: &[BasicMetadataTypeEnum<'a>],
) -> FunctionType<'a> {
    match ty {
        PrimitiveTypes::I8 => context.i8_type().fn_type(param_types, false),
        PrimitiveTypes::I16 => context.i16_type().fn_type(param_types, false),
        PrimitiveTypes::I32 => context.i32_type().fn_type(param_types, false),
        PrimitiveTypes::I64 => context.i64_type().fn_type(param_types, false),
        PrimitiveTypes::F32 => context.f32_type().fn_type(param_types, false),
        PrimitiveTypes::F64 => context.f64_type().fn_type(param_types, false),
        PrimitiveTypes::None => context.void_type().fn_type(param_types, false),
        _ => panic!("wrong primitive type"),
    }
}

fn convert_meta_primitive_type<'a, T>(context: &'a Context, ty: &PrimitiveTypes) -> T
where
    T: From<IntType<'a>> + From<FloatType<'a>>,
{
    match ty {
        PrimitiveTypes::I8 => context.i8_type().into(),
        PrimitiveTypes::I16 => context.i16_type().into(),
        PrimitiveTypes::I32 => context.i32_type().into(),
        PrimitiveTypes::I64 => context.i64_type().into(),
        PrimitiveTypes::F32 => context.f32_type().into(),
        PrimitiveTypes::F64 => context.f64_type().into(),
        _ => panic!("wrong primitive type"),
    }
}

fn create_entry_block_alloca<'a, T: BasicType<'a>>(
    context: &'a Context,
    fn_value: FunctionValue<'a>,
    alloc_ty: T,
    name: &str,
) -> PointerValue<'a> {
    let builder = context.create_builder();
    let entry = fn_value.get_first_basic_block().unwrap();

    match entry.get_first_instruction() {
        Some(first_instr) => builder.position_before(&first_instr),
        None => builder.position_at_end(entry),
    }

    builder.build_alloca(alloc_ty, name).unwrap()
}

fn fn_init<'a>(
    context: &'a Context,
    module: &Module<'a>,
    fn_decl: &FunctionStatement,
) -> FunctionValue<'a> {
    let args_types = fn_decl
        .parameters
        .iter()
        .map(|p| match &p.parameter_type {
            Type::Primitive(ty) => convert_meta_primitive_type(context, ty),
            _ => panic!("wrong type for fn param"),
        })
        .collect::<Vec<BasicMetadataTypeEnum>>();

    let fn_type = match &fn_decl.result_type {
        Type::Primitive(ty) => get_fn_type(context, ty, &args_types),
        _ => panic!("wrong type"),
    };
    let fn_val = module.add_function(&fn_decl.name.to_string(), fn_type, None);
    for (i, arg) in fn_val.get_param_iter().enumerate() {
        let param_name = fn_decl.parameters[i].to_string();
        match &fn_decl.parameters[i].parameter_type {
            Type::Primitive(ty) => match ty {
                PrimitiveTypes::I8 => arg.into_int_value().set_name(&param_name),
                PrimitiveTypes::I16 => arg.into_int_value().set_name(&param_name),
                PrimitiveTypes::I32 => arg.into_int_value().set_name(&param_name),
                PrimitiveTypes::I64 => arg.into_int_value().set_name(&param_name),
                PrimitiveTypes::F32 => arg.into_float_value().set_name(&param_name),
                PrimitiveTypes::F64 => arg.into_float_value().set_name(&param_name),
                _ => panic!("wrong primitive type"),
            },
            _ => panic!("wrong param type"),
        }
    }
    fn_val
}

fn fn_init_params<'a>(
    context: &'a Context,
    fn_val: FunctionValue<'a>,
    builder: &Builder<'a>,
    fn_decl: &FunctionStatement,
) -> FunctionValue<'a> {
    for (i, arg) in fn_val.get_param_iter().enumerate() {
        let param_name = fn_decl.parameters[i].to_string();
        match &fn_decl.parameters[i].parameter_type {
            Type::Primitive(ty) => match ty {
                PrimitiveTypes::I8 => {
                    let alloca =
                        create_entry_block_alloca(context, fn_val, context.i8_type(), &param_name);
                    builder.build_store(alloca, arg).unwrap();
                }
                _ => panic!("wrong primitive type"),
            },
            _ => panic!("wrong param type"),
        }
    }
    fn_val
}

fn func_body<'a>(
    context: &'a Context,
    _module: &Module<'a>,
    builder: &Builder<'a>,
    _func: FunctionValue<'a>,
    func_body: Rc<RefCell<BlockState>>,
) {
    let ctx = func_body.borrow().context.clone().get();
    match &ctx[0] {
        SemanticStackContext::ExpressionFunctionReturn { expr_result } => {
            let ret = match &expr_result.expr_value {
                ExpressionResultValue::PrimitiveValue(pval) => match pval {
                    PrimitiveValue::I8(val) => context
                        .i8_type()
                        .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                        .unwrap(),
                    PrimitiveValue::I16(val) => context
                        .i16_type()
                        .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                        .unwrap(),
                    _ => panic!("type not supported"),
                },
                _ => panic!("type not supported"),
            };
            builder.build_return(Some(&ret)).expect("return val");
        }
        _ => println!("->"),
    }
}

fn compiler(semantic_state: &State) {
    let context = Context::create();
    let module = context.create_module("main");
    let builder = context.create_builder();
    assert_eq!(semantic_state.context.len(), 1);

    //println!("{semantic_state:#?}");
    let global_context = semantic_state.global.context.clone().get();
    assert_eq!(global_context.len(), 1);
    match &global_context[0] {
        SemanticStackContext::FunctionDeclaration { fn_decl } => {
            let func = fn_init(&context, &module, fn_decl);
            let entry = context.append_basic_block(func, "entry");
            builder.position_at_end(entry);
            fn_init_params(&context, func, &builder, fn_decl);

            func_body(
                &context,
                &module,
                &builder,
                func,
                semantic_state.context[0].clone(),
            );
        }
        _ => panic!("wrong instr"),
    }

    module.print_to_file("res.ll").expect("Failed generate");
    let target_machine = get_native_target_machine().unwrap();
    apply_target_to_module(&target_machine, &module);
    target_machine
        .write_to_file(&module, FileType::Object, "res.o".as_ref())
        .unwrap();
}

fn main() {
    let semantic_state = semantic_stack();
    compiler(&semantic_state);
    //println!("Stack: {:#?}", semantic_state.global);
}
