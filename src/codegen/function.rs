use crate::llvm_wrapper::basic_block::BasicBlockRef;
use crate::llvm_wrapper::types::TypeRef;
use crate::llvm_wrapper::value::ValueRef;
use crate::llvm_wrapper::{builder::BuilderRef, context::ContextRef, module::ModuleRef};
use anyhow::bail;
use llvm_sys::prelude::LLVMBasicBlockRef;
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::{FunctionStatement, PrimitiveValue};

/// # Function codegen
/// Contains:
/// - `context` - LLVM context
/// - `func_val` - LLVM function value as basic entity for function codegen
/// - `entities` - map of functiob entities based on `ConstValue`
pub struct FuncCodegen<'a> {
    context: &'a ContextRef,
    module: &'a ModuleRef,
    builder: &'a BuilderRef,
    func_value: ValueRef,
    //pub entities: HashMap<String, ConstValue<'ctx>>,
}

impl<'a> FuncCodegen<'a> {
    /// Create function codegen
    pub fn new(context: &'a ContextRef, module: &'a ModuleRef, builder: &'a BuilderRef) -> Self {
        Self {
            context,
            module,
            builder,
            func_value: ValueRef::create(std::ptr::null_mut()),
            // entities: HashMap::new(),
        }
    }

    /// Convert types form Semantic types to llvm-wrapper
    fn convert_to_type(&self, ty: &PrimitiveTypes) -> anyhow::Result<TypeRef> {
        match ty {
            PrimitiveTypes::I8 | PrimitiveTypes::U8 => Ok(TypeRef::i8_type(self.context)),
            PrimitiveTypes::I16 | PrimitiveTypes::U16 => Ok(TypeRef::i16_type(self.context)),
            PrimitiveTypes::I32 | PrimitiveTypes::U32 | PrimitiveTypes::Char => {
                Ok(TypeRef::i32_type(self.context))
            }
            PrimitiveTypes::I64 | PrimitiveTypes::U64 => Ok(TypeRef::i64_type(self.context)),
            PrimitiveTypes::F32 => Ok(TypeRef::f32_type(self.context)),
            PrimitiveTypes::F64 => Ok(TypeRef::f32_type(self.context)),
            PrimitiveTypes::Bool => Ok(TypeRef::bool_type(self.context)),
            PrimitiveTypes::String => {
                let array_type = TypeRef::i8_type(self.context);
                Ok(TypeRef::array_type(&array_type, 0))
            }
            PrimitiveTypes::None => Ok(TypeRef::void_type(self.context)),
            PrimitiveTypes::Ptr => {
                let ptr_raw_type = TypeRef::i8_type(self.context);
                Ok(TypeRef::ptr_type(ptr_raw_type, 0))
            }
        }
    }

    /// Generate function type
    fn get_func_type(
        &self,
        return_type: &PrimitiveTypes,
        arg_types: &[TypeRef],
    ) -> anyhow::Result<TypeRef> {
        let fn_return_type = self.convert_to_type(return_type)?;
        Ok(TypeRef::function_type(arg_types, &fn_return_type))
    }

    /// Set function value
    fn set_func_value(&mut self, func_val: ValueRef) {
        self.func_value = func_val;
    }

    /// Function declaration
    pub fn func_declaration(&mut self, fn_decl: &FunctionStatement) -> anyhow::Result<()> {
        // Prepare function argument types. For function-declaration
        // we need only types
        let mut args_types: Vec<TypeRef> = vec![];

        for param in &fn_decl.parameters {
            let res = match &param.parameter_type {
                Type::Primitive(ty) => self.convert_to_type(ty)?,
                _ => bail!(error::FuncCodegenError::IncompatibleTypeForFuncParam(
                    param.parameter_type.clone()
                )),
            };
            args_types.push(res);
        }

        // Get function declaration type
        let fn_type = match &fn_decl.result_type {
            Type::Primitive(ty) => self.get_func_type(ty, &args_types)?,
            _ => bail!(error::FuncCodegenError::WrongFuncReturnType(
                fn_decl.result_type.clone()
            )),
        };
        // Generate and set function-value - basic codegen entity for function
        let func_val = ValueRef::add_function(self.module, &fn_decl.name.to_string(), &fn_type);
        self.set_func_value(func_val);

        /*
        // Attach function parameters
        for (i, arg) in func_val.get_param_iter().enumerate() {
            let param_name = fn_decl.parameters[i].to_string();
            let param_type = &fn_decl.parameters[i].parameter_type;
            match param_type {
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
                        arg.into_float_value().set_name(&param_name);
                    }
                    PrimitiveTypes::String => arg.into_array_value().set_name(&param_name),
                    PrimitiveTypes::Ptr => arg.into_pointer_value().set_name(&param_name),
                    PrimitiveTypes::None => {
                        bail!(FuncCodegenError::FuncParameterNoneTypeDeprecated)
                    }
                },
                _ => bail!(FuncCodegenError::IncompatibleTypeForFuncParam(
                    param_type.clone()
                )),
            }
        }
        */
        Ok(())
    }

    pub fn set(&self) {
        let void_ty = TypeRef::void_type(self.context);
        let fn_type = TypeRef::function_type(&[], &void_ty);
        let function = ValueRef::add_function(self.module, "main", &fn_type);
        let bb = BasicBlockRef::append_in_context(self.context, &function, "entry");
        self.builder.position_at_end(&bb);
        self.builder.build_ret_void();
    }
}

pub mod error {
    use semantic_analyzer::types::types::{PrimitiveTypes, Type};
    use thiserror::Error;

    /// `FuncCodegen` errors coverage
    #[derive(Debug, Error)]
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
