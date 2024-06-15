use crate::llvm_wrapper::basic_block::BasicBlockRef;
use crate::llvm_wrapper::types::TypeRef;
use crate::llvm_wrapper::value::ValueRef;
use crate::llvm_wrapper::{builder::BuilderRef, context::ContextRef, module::ModuleRef};
use anyhow::bail;
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::FunctionStatement;
use std::rc::Rc;

/// # Function codegen
/// Contains:
/// - `context` - LLVM context
/// - `func_val` - LLVM function value as basic entity for function codegen
/// - `entities` - map of function entities based on `ConstValue`
pub struct FuncCodegen {
    context: Rc<ContextRef>,
    module: Rc<ModuleRef>,
    builder: Rc<BuilderRef>,
    func_value: Rc<ValueRef>,
    //pub entities: HashMap<String, ConstValue<'ctx>>,
}

impl FuncCodegen {
    /// Create function codegen
    pub fn new(context: Rc<ContextRef>, module: Rc<ModuleRef>, builder: Rc<BuilderRef>) -> Self {
        Self {
            context,
            module,
            builder,
            func_value: Rc::new(ValueRef::create(std::ptr::null_mut())),
            // entities: HashMap::new(),
        }
    }

    /// Convert types form Semantic types to llvm-wrapper
    fn convert_to_type(&self, ty: &PrimitiveTypes) -> TypeRef {
        match ty {
            PrimitiveTypes::I8 | PrimitiveTypes::U8 => TypeRef::i8_type(&self.context),
            PrimitiveTypes::I16 | PrimitiveTypes::U16 => TypeRef::i16_type(&self.context),
            PrimitiveTypes::I32 | PrimitiveTypes::U32 | PrimitiveTypes::Char => {
                TypeRef::i32_type(&self.context)
            }
            PrimitiveTypes::I64 | PrimitiveTypes::U64 => TypeRef::i64_type(&self.context),
            PrimitiveTypes::F32 => TypeRef::f32_type(&self.context),
            PrimitiveTypes::F64 => TypeRef::f64_type(&self.context),
            PrimitiveTypes::Bool => TypeRef::bool_type(&self.context),
            PrimitiveTypes::String => {
                let array_type = TypeRef::i8_type(&self.context);
                TypeRef::array_type(&array_type, 0)
            }
            PrimitiveTypes::None => TypeRef::void_type(&self.context),
            PrimitiveTypes::Ptr => {
                let ptr_raw_type = TypeRef::i8_type(&self.context);
                TypeRef::ptr_type(&ptr_raw_type, 0)
            }
        }
    }

    /// Generate function type
    fn get_func_type(&self, return_type: &PrimitiveTypes, arg_types: &[TypeRef]) -> TypeRef {
        let fn_return_type = self.convert_to_type(return_type);
        TypeRef::function_type(arg_types, &fn_return_type)
    }

    /// Set function value
    fn set_func_value(&mut self, func_val: Rc<ValueRef>) {
        self.func_value = func_val;
    }

    /// Function declaration
    pub fn func_declaration(&mut self, fn_decl: &FunctionStatement) -> anyhow::Result<()> {
        // Prepare function argument types. For function-declaration
        // we need only types
        let mut args_types: Vec<TypeRef> = vec![];

        for param in &fn_decl.parameters {
            let res = match &param.parameter_type {
                Type::Primitive(ty) => self.convert_to_type(ty),
                _ => bail!(error::FuncCodegenError::IncompatibleTypeForFuncParam(
                    param.parameter_type.clone()
                )),
            };
            args_types.push(res);
        }

        // Get function declaration type
        let fn_type = match &fn_decl.result_type {
            Type::Primitive(ty) => self.get_func_type(ty, &args_types),
            _ => bail!(error::FuncCodegenError::WrongFuncReturnType(
                fn_decl.result_type.clone()
            )),
        };
        // Generate and set function-value - basic codegen entity for function
        let func_val = Rc::new(ValueRef::add_function(
            &self.module,
            &fn_decl.name.to_string(),
            &fn_type,
        ));
        self.set_func_value(func_val.clone());
        // Set function arguments name
        for (i, param) in fn_decl.parameters.iter().enumerate() {
            ValueRef::get_func_param(&func_val, i).set_value_name(&param.to_string());
        }

        Ok(())
    }

    pub fn set(&self) {
        let void_ty = TypeRef::void_type(&self.context);
        let fn_type = TypeRef::function_type(&[], &void_ty);
        let function = ValueRef::add_function(&self.module, "main", &fn_type);
        let bb = BasicBlockRef::append_in_context(&self.context, &function, "entry");
        self.builder.position_at_end(&bb);
        let _ = self.builder.build_ret_void();
    }
}

pub mod error {
    use semantic_analyzer::types::types::{PrimitiveTypes, Type};
    use thiserror::Error;

    /// `FuncCodegen` errors coverage
    #[derive(Debug, Error)]
    // TODO: clippy
    #[allow(dead_code)]
    pub enum FuncCodegenError {
        #[error("FunctionValue not exist")]
        FuncValueNotExist,
        #[error("Failed convert type to LLVM: {0:?}")]
        FailedConvertType(PrimitiveTypes),
        #[error("Incompatible type for function parameter: {0:?}")]
        IncompatibleTypeForFuncParam(Type),
        #[error("Incompatible type for LetBinding: {0:?}")]
        IncompatibleTypeForLetBinding(Type),
        #[error("Wrong function return type: {0:?}")]
        WrongFuncReturnType(Type),
        #[error("Function parameter type None is deprecated")]
        FuncParameterNoneTypeDeprecated,
        #[error("Failed create function basic block")]
        FailedCreateBasicBlock,
        #[error("Failed generate build alloc: {0:?}")]
        FailedBuildAlloc(String),
        #[error("Failed convert IntValue for: {0}")]
        FailedConvertIntVal(String),
        #[error("Failed get entity for register: {0:?}")]
        FailedGetEntityForRegister(u64),
        #[error("Unsupported expression operation value type")]
        UnsupportedExpressionOperationValueType,
        #[error("Unsupported expression operation kind")]
        UnsupportedExpressionOperationKind,
        #[error("Failed generate  Build return: {0:?}")]
        FailedBuildReturn(String),
        #[error("Binding value not found: {0}")]
        BindingValueNotFound(String),
    }
}
