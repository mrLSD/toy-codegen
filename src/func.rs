use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{
    BasicMetadataTypeEnum, BasicTypeEnum, FloatType, FunctionType, IntType, StringRadix,
};
use inkwell::values::PointerValue;
use inkwell::values::{ArrayValue, FloatValue, FunctionValue, IntValue};
use semantic_analyzer::types::block_state::BlockState;
use semantic_analyzer::types::expression::{
    ExpressionOperations, ExpressionResult, ExpressionResultValue,
};
use semantic_analyzer::types::semantic::SemanticStackContext;
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::{FunctionParameter, FunctionStatement, PrimitiveValue, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub enum ConstValue<'a> {
    Int(IntValue<'a>),
    Float(FloatValue<'a>),
    Bool(IntValue<'a>),
    Char(IntValue<'a>),
    String(ArrayValue<'a>),
    #[allow(dead_code)]
    Pointer(PointerValue<'a>),
    None,
}

pub struct FuncCodegen<'ctx> {
    pub context: &'ctx Context,
    pub func_val: Option<FunctionValue<'ctx>>,
    pub entities: HashMap<String, ConstValue<'ctx>>,
}

impl<'ctx> FuncCodegen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
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

    pub fn fn_declaration(&mut self, module: &Module<'ctx>, fn_decl: &FunctionStatement) {
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

    fn create_entry_block_alloca(
        &self,
        alloc_ty: BasicMetadataTypeEnum<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let func_val = self.get_func();
        let builder = self.context.create_builder();
        let entry = func_val.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        if alloc_ty.is_int_type() {
            builder
                .build_alloca(alloc_ty.into_int_type(), name)
                .unwrap()
        } else if alloc_ty.is_float_type() {
            builder
                .build_alloca(alloc_ty.into_float_type(), name)
                .unwrap()
        } else if alloc_ty.is_array_type() {
            builder
                .build_alloca(alloc_ty.into_array_type(), name)
                .unwrap()
        } else if alloc_ty.is_pointer_type() {
            builder
                .build_alloca(alloc_ty.into_pointer_type(), name)
                .unwrap()
        } else {
            builder
                .build_alloca(alloc_ty.into_struct_type(), name)
                .unwrap()
        }
    }

    fn fn_init_params(&self, _builder: &Builder<'ctx>, fn_decl: &FunctionStatement) {
        let func_val = self.get_func();
        for (i, _arg) in func_val.get_param_iter().enumerate() {
            let param_name = fn_decl.parameters[i].to_string();
            let _alloca = match &fn_decl.parameters[i].parameter_type {
                Type::Primitive(ty) => {
                    let ty_val: BasicMetadataTypeEnum = self.convert_meta_primitive_type(ty);
                    self.create_entry_block_alloca(ty_val, &param_name)
                }
                _ => panic!("wrong param type"),
            };
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

    fn expr_value_operation(&self, value: &ExpressionResult) -> ConstValue<'ctx> {
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
        let const_left_value = self.expr_value_operation(left_value);
        let const_right_value = self.expr_value_operation(right_value);
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

    fn expression_function_return(&self, builder: &Builder<'ctx>, expr_result: &ExpressionResult) {
        let ret = self.expr_value_operation(expr_result);
        match ret {
            ConstValue::Int(v) => builder.build_return(Some(&v)).expect("return val"),
            ConstValue::Float(v) => builder.build_return(Some(&v)).expect("return val"),
            ConstValue::Bool(v) => builder.build_return(Some(&v)).expect("return val"),
            ConstValue::Char(v) => builder.build_return(Some(&v)).expect("return val"),
            ConstValue::String(v) => builder.build_return(Some(&v)).expect("return val"),
            ConstValue::Pointer(p) => builder.build_return(Some(&p)).expect("return val"),
            ConstValue::None => builder.build_return(None).expect("return val"),
        };
    }

    fn let_binding(
        &mut self,
        builder: &Builder<'ctx>,
        let_decl: &Value,
        expr_result: &ExpressionResult,
    ) {
        let name = let_decl.inner_name.to_string();
        let alloca = match &let_decl.inner_type {
            Type::Primitive(ty) => {
                let ty_val: BasicMetadataTypeEnum = self.convert_meta_primitive_type(ty);
                self.create_entry_block_alloca(ty_val, &name)
            }
            _ => panic!("wrong param type"),
        };
        // Set position for next instr
        let entry = self.get_func().get_first_basic_block().unwrap();
        builder.position_at_end(entry);
        let res_val = self.expr_value_operation(expr_result);
        match res_val {
            ConstValue::Int(val) => builder.build_store(alloca, val).unwrap(),
            ConstValue::Float(val) => builder.build_store(alloca, val).unwrap(),
            ConstValue::Bool(val) => builder.build_store(alloca, val).unwrap(),
            ConstValue::Char(val) => builder.build_store(alloca, val).unwrap(),
            ConstValue::String(val) => builder.build_store(alloca, val).unwrap(),
            ConstValue::Pointer(val) => builder.build_store(alloca, val).unwrap(),
            // For None just store ZERO
            ConstValue::None => builder
                .build_store(alloca, self.context.i8_type().const_zero())
                .unwrap(),
        };
        // Store to entities
        self.entities.insert(name, ConstValue::Pointer(alloca));
    }

    fn binding(&self, _builder: &Builder<'ctx>, _val: &Value, _expr_result: &ExpressionResult) {
        //STORE
    }

    fn expr_value(&mut self, builder: &Builder<'ctx>, expression: &Value, register_number: u64) {
        let name = expression.inner_name.to_string();
        // Get from entities
        let entity = self.entities.get(&name).unwrap();
        let pval = if let ConstValue::Pointer(pval) = entity {
            pval
        } else {
            panic!("value not pointer")
        };
        let ty: BasicTypeEnum = match &expression.inner_type {
            Type::Primitive(pty) => self.convert_meta_primitive_type(pty),
            _ => panic!("operation type currently can be only Type::Primitive"),
        };
        let load_res = builder.build_load(ty, *pval, &name).unwrap();
        let res = match &expression.inner_type {
            Type::Primitive(pty) => match pty {
                PrimitiveTypes::I8
                | PrimitiveTypes::I16
                | PrimitiveTypes::I32
                | PrimitiveTypes::I64
                | PrimitiveTypes::U8
                | PrimitiveTypes::U16
                | PrimitiveTypes::U32
                | PrimitiveTypes::U64
                | PrimitiveTypes::Bool
                | PrimitiveTypes::Char => ConstValue::Int(load_res.into_int_value()),
                PrimitiveTypes::F32 | PrimitiveTypes::F64 => {
                    ConstValue::Float(load_res.into_float_value())
                }
                PrimitiveTypes::String => ConstValue::String(load_res.into_array_value()),
                PrimitiveTypes::Ptr => ConstValue::Pointer(load_res.into_pointer_value()),
                PrimitiveTypes::None => ConstValue::None,
            },
            _ => panic!("load type currently can be only Type::Primitive"),
        };
        self.entities.insert(register_number.to_string(), res);
    }

    fn func_arg(&self, _builder: &Builder<'ctx>, _value: &Value, _func_arg: &FunctionParameter) {}

    pub fn func_body(
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

        let ctxs = func_body.borrow().get_context().get();
        for ctx in ctxs {
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
                SemanticStackContext::LetBinding {
                    let_decl,
                    expr_result,
                } => self.let_binding(builder, let_decl, expr_result),
                SemanticStackContext::Binding { val, expr_result } => {
                    self.binding(builder, val, expr_result)
                }
                SemanticStackContext::ExpressionFunctionReturn { expr_result } => {
                    self.expression_function_return(builder, expr_result)
                }
                SemanticStackContext::ExpressionValue {
                    expression,
                    register_number,
                } => self.expr_value(builder, expression, *register_number),
                SemanticStackContext::FunctionArg { value, func_arg } => {
                    self.func_arg(builder, value, func_arg)
                }
                _ => println!("-> {ctx:?}"),
            }
        }
    }
}
