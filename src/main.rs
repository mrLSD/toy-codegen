use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{
    BasicMetadataTypeEnum, BasicType, FloatType, FunctionType, IntType, StringRadix,
};
use inkwell::values::PointerValue;
use inkwell::values::{ArrayValue, FloatValue, FunctionValue, IntValue};
use inkwell::OptimizationLevel;
use semantic_analyzer::ast;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::block_state::BlockState;
use semantic_analyzer::types::expression::{
    ExpressionOperations, ExpressionResult, ExpressionResultValue,
};
use semantic_analyzer::types::semantic::SemanticStackContext;
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::{FunctionStatement, PrimitiveValue};
use std::cell::RefCell;
use std::collections::HashMap;
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
            RelocMode::Default,
            CodeModel::Small,
        )
        .ok_or_else(|| String::from("Failed to create target machine"))
}

pub struct Codegen<'a> {
    pub context: &'a Context,
    pub module: Module<'a>,
}

#[derive(Clone)]
enum ConstValue<'a> {
    Int(IntValue<'a>),
    Float(FloatValue<'a>),
    Bool(IntValue<'a>),
    Char(IntValue<'a>),
    String(ArrayValue<'a>),
    #[allow(dead_code)]
    Pointer(PointerValue<'a>),
    None,
}

struct FuncCodegen<'ctx> {
    context: &'ctx Context,
    func_val: Option<FunctionValue<'ctx>>,
    entities: HashMap<String, ConstValue<'ctx>>,
}

impl<'ctx> FuncCodegen<'ctx> {
    fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            func_val: None,
            entities: HashMap::new(),
        }
    }

    fn get_func(&self) -> FunctionValue<'ctx> {
        self.func_val.unwrap()
    }

    fn set_func(&mut self, func_val: FunctionValue<'ctx>) {
        self.func_val = Some(func_val);
    }

    fn fn_declaration(&mut self, module: &Module<'ctx>, fn_decl: &FunctionStatement) {
        let args_types = fn_decl
            .parameters
            .iter()
            .map(|p| match &p.parameter_type {
                Type::Primitive(ty) => self.convert_meta_primitive_type(ty),
                _ => panic!("func param type currently can be only Type::Primitive"),
            })
            .collect::<Vec<BasicMetadataTypeEnum>>();

        let fn_type = match &fn_decl.result_type {
            Type::Primitive(ty) => self.get_fn_type(ty, &args_types),
            _ => panic!("func result type currently can be only Type::Primitive"),
        };
        let func_val = module.add_function(&fn_decl.name.to_string(), fn_type, None);
        self.set_func(func_val);
        for (i, arg) in func_val.get_param_iter().enumerate() {
            let param_name = fn_decl.parameters[i].to_string();
            match &fn_decl.parameters[i].parameter_type {
                Type::Primitive(ty) => match ty {
                    PrimitiveTypes::I8
                    | PrimitiveTypes::I16
                    | PrimitiveTypes::I32
                    | PrimitiveTypes::I64
                    | PrimitiveTypes::U8
                    | PrimitiveTypes::U16
                    | PrimitiveTypes::U32
                    | PrimitiveTypes::U64
                    | PrimitiveTypes::Bool
                    | PrimitiveTypes::Char => arg.into_int_value().set_name(&param_name),
                    PrimitiveTypes::F32 | PrimitiveTypes::F64 => {
                        arg.into_float_value().set_name(&param_name)
                    }
                    PrimitiveTypes::String => arg.into_struct_value().set_name(&param_name),
                    PrimitiveTypes::Ptr => arg.into_pointer_value().set_name(&param_name),
                    PrimitiveTypes::None => panic!("None: func parameter not supported"),
                },
                _ => panic!("func param type currently can be only Type::Primitive"),
            }
        }
    }

    fn get_fn_type(
        &self,
        ty: &PrimitiveTypes,
        param_types: &[BasicMetadataTypeEnum<'ctx>],
    ) -> FunctionType<'ctx> {
        match ty {
            PrimitiveTypes::I8 | PrimitiveTypes::U8 => {
                self.context.i8_type().fn_type(param_types, false)
            }
            PrimitiveTypes::I16 | PrimitiveTypes::U16 => {
                self.context.i16_type().fn_type(param_types, false)
            }
            PrimitiveTypes::I32 | PrimitiveTypes::U32 | PrimitiveTypes::Char => {
                self.context.i32_type().fn_type(param_types, false)
            }
            PrimitiveTypes::I64 | PrimitiveTypes::U64 => {
                self.context.i64_type().fn_type(param_types, false)
            }
            PrimitiveTypes::F32 => self.context.f32_type().fn_type(param_types, false),
            PrimitiveTypes::F64 => self.context.f64_type().fn_type(param_types, false),
            PrimitiveTypes::Bool => self.context.bool_type().fn_type(param_types, false),
            PrimitiveTypes::String => self.context.metadata_type().fn_type(param_types, false),
            PrimitiveTypes::None => self.context.void_type().fn_type(param_types, false),
            PrimitiveTypes::Ptr => panic!("function type Ptr not resolved"),
        }
    }

    fn convert_meta_primitive_type<T>(&self, ty: &PrimitiveTypes) -> T
    where
        T: From<IntType<'ctx>> + From<FloatType<'ctx>>,
    {
        match ty {
            PrimitiveTypes::I8 => self.context.i8_type().into(),
            PrimitiveTypes::I16 => self.context.i16_type().into(),
            PrimitiveTypes::I32 => self.context.i32_type().into(),
            PrimitiveTypes::I64 => self.context.i64_type().into(),
            PrimitiveTypes::F32 => self.context.f32_type().into(),
            PrimitiveTypes::F64 => self.context.f64_type().into(),
            _ => panic!("wrong primitive type"),
        }
    }

    fn create_entry_block_alloca<T: BasicType<'ctx>>(
        &self,
        alloc_ty: T,
        name: &str,
    ) -> PointerValue<'ctx> {
        let func_val = self.get_func();
        let builder = self.context.create_builder();
        let entry = func_val.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(alloc_ty, name).unwrap()
    }

    fn fn_init_params(&self, builder: &Builder<'ctx>, fn_decl: &FunctionStatement) {
        let func_val = self.get_func();
        for (i, arg) in func_val.get_param_iter().enumerate() {
            let param_name = fn_decl.parameters[i].to_string();
            match &fn_decl.parameters[i].parameter_type {
                Type::Primitive(ty) => match ty {
                    PrimitiveTypes::I8 => {
                        let alloca =
                            self.create_entry_block_alloca(self.context.i8_type(), &param_name);
                        builder.build_store(alloca, arg).unwrap();
                    }
                    _ => panic!("wrong primitive type"),
                },
                _ => panic!("wrong param type"),
            }
        }
    }

    fn expr_primitive_value(&self, pval: &PrimitiveValue) -> ConstValue<'ctx> {
        match pval {
            PrimitiveValue::I8(val) => ConstValue::Int(
                self.context
                    .i8_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::I16(val) => ConstValue::Int(
                self.context
                    .i16_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::I32(val) => ConstValue::Int(
                self.context
                    .i32_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::I64(val) => ConstValue::Int(
                self.context
                    .i64_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::U8(val) => ConstValue::Int(
                self.context
                    .i8_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::U16(val) => ConstValue::Int(
                self.context
                    .i16_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::U32(val) => ConstValue::Int(
                self.context
                    .i32_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::U64(val) => ConstValue::Int(
                self.context
                    .i64_type()
                    .const_int_from_string(&format!("{val:?}"), StringRadix::Decimal)
                    .unwrap(),
            ),
            PrimitiveValue::F32(val) => ConstValue::Float(
                self.context
                    .f32_type()
                    .const_float_from_string(&format!("{val:?}")),
            ),
            PrimitiveValue::F64(val) => ConstValue::Float(
                self.context
                    .f64_type()
                    .const_float_from_string(&format!("{val:?}")),
            ),
            PrimitiveValue::Bool(val) => {
                ConstValue::Bool(self.context.bool_type().const_int((*val).into(), false))
            }
            PrimitiveValue::Char(val) => {
                ConstValue::Char(self.context.i32_type().const_int((*val).into(), false))
            }
            PrimitiveValue::String(val) => {
                ConstValue::String(self.context.const_string(val.as_bytes(), false))
            }
            PrimitiveValue::Ptr => panic!("Pointer value not supported"),
            PrimitiveValue::None => ConstValue::None,
        }
    }

    fn expr_value_operation(
        &self,
        _builder: &Builder<'ctx>,
        value: &ExpressionResult,
    ) -> ConstValue<'ctx> {
        match &value.expr_value {
            ExpressionResultValue::PrimitiveValue(pv) => self.expr_primitive_value(pv),
            ExpressionResultValue::Register(reg) => {
                self.entities
                    .get(&reg.to_string())
                    .expect("failed get entity")
                    .clone()
                // let ty: BasicTypeEnum = match &value.expr_type {
                //     Type::Primitive(pty) => self.convert_meta_primitive_type(pty),
                //     _ => panic!("operation type currently can be only Type::Primitive"),
                // };
                // let pval = builder.build_load(ty, *val, "left_val").unwrap();
                // match &value.expr_type {
                //     Type::Primitive(pty) => match pty {
                //         PrimitiveTypes::I8
                //         | PrimitiveTypes::I16
                //         | PrimitiveTypes::I32
                //         | PrimitiveTypes::I64 => ConstValue::Int(pval.into_int_value()),
                //         PrimitiveTypes::F32 | PrimitiveTypes::F64 => {
                //             ConstValue::Float(pval.into_float_value())
                //         }
                //         _ => panic!("operation type currently can be only Type::Primitive"),
                //     },
                //     _ => panic!("operation type currently can be only Type::Primitive"),
                // }
            }
        }
    }

    fn expr_operation(
        &mut self,
        builder: &Builder<'ctx>,
        operation: &ExpressionOperations,
        left_value: &ExpressionResult,
        right_value: &ExpressionResult,
        register_number: u64,
    ) {
        let const_left_value = self.expr_value_operation(builder, left_value);
        let const_right_value = self.expr_value_operation(builder, right_value);
        let res = match operation {
            ExpressionOperations::Plus => {
                if let (ConstValue::Int(lhs), ConstValue::Int(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Int(builder.build_int_add(*lhs, *rhs, "tmp_add").unwrap())
                } else if let (ConstValue::Float(lhs), ConstValue::Float(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Float(builder.build_float_add(*lhs, *rhs, "tmp_add").unwrap())
                } else {
                    panic!("unsupported type for operation ");
                }
            }
            ExpressionOperations::Minus => {
                if let (ConstValue::Int(lhs), ConstValue::Int(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Int(builder.build_int_sub(*lhs, *rhs, "tmp_sub").unwrap())
                } else if let (ConstValue::Float(lhs), ConstValue::Float(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Float(builder.build_float_sub(*lhs, *rhs, "tmp_sub").unwrap())
                } else {
                    panic!("unsupported type for operation ");
                }
            }
            ExpressionOperations::Multiply => {
                if let (ConstValue::Int(lhs), ConstValue::Int(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Int(builder.build_int_mul(*lhs, *rhs, "tmp_mul").unwrap())
                } else if let (ConstValue::Float(lhs), ConstValue::Float(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Float(builder.build_float_mul(*lhs, *rhs, "tmp_mul").unwrap())
                } else {
                    panic!("unsupported type for operation ");
                }
            }
            ExpressionOperations::Divide => {
                if let (ConstValue::Int(lhs), ConstValue::Int(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Int(builder.build_int_signed_div(*lhs, *rhs, "tmp_div").unwrap())
                } else if let (ConstValue::Float(lhs), ConstValue::Float(rhs)) =
                    (&const_left_value, &const_right_value)
                {
                    ConstValue::Float(builder.build_float_div(*lhs, *rhs, "tmp_div").unwrap())
                } else {
                    panic!("unsupported type for operation ");
                }
            }
            _ => panic!("only restricted kind of operations"),
        };
        self.entities.insert(register_number.to_string(), res);
    }

    fn func_body(
        &mut self,
        builder: &Builder<'ctx>,
        func_body: Rc<RefCell<BlockState>>,
        fn_decl: &FunctionStatement,
    ) {
        let func_val = self.get_func();
        let entry = self.context.append_basic_block(func_val, "entry");
        builder.position_at_end(entry);
        self.fn_init_params(builder, fn_decl);
        builder.position_at_end(entry);

        let ctxs = func_body.borrow().context.clone().get();
        for ctx in ctxs {
            println!("{:?}", ctx);
            match &ctx {
                SemanticStackContext::ExpressionOperation {
                    operation,
                    left_value,
                    right_value,
                    register_number,
                } => {
                    self.expr_operation(
                        builder,
                        operation,
                        left_value,
                        right_value,
                        *register_number,
                    );
                }
                SemanticStackContext::ExpressionFunctionReturn { expr_result } => {
                    let ret = match &expr_result.expr_value {
                        ExpressionResultValue::PrimitiveValue(pval) => {
                            self.expr_primitive_value(pval)
                        }
                        _ => panic!("type not supported"),
                    };
                    match ret {
                        ConstValue::Int(v) => builder.build_return(Some(&v)).expect("return val"),
                        ConstValue::Float(v) => builder.build_return(Some(&v)).expect("return val"),
                        ConstValue::Bool(v) => builder.build_return(Some(&v)).expect("return val"),
                        ConstValue::Char(v) => builder.build_return(Some(&v)).expect("return val"),
                        ConstValue::String(v) => {
                            builder.build_return(Some(&v)).expect("return val")
                        }
                        ConstValue::Pointer(p) => {
                            builder.build_return(Some(&p)).expect("return val")
                        }
                        ConstValue::None => builder.build_return(None).expect("return val"),
                    };
                }
                _ => println!("->"),
            }
        }
    }
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
        .write_to_file(&module, FileType::Object, "res.S".as_ref())
        .unwrap();
}

fn main() {
    let semantic_state = semantic_stack();
    compiler(&semantic_state);
}
