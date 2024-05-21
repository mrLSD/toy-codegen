//! # Semantic State
//! Semantic state generation based on semantic `AST`.

use anyhow::bail;
use semantic_analyzer::ast::{self, Ident};
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::block_state::BlockState;
use semantic_analyzer::types::error::StateErrorResult;
use semantic_analyzer::types::expression::{ExpressionResult, ExpressionResultValue};
use semantic_analyzer::types::semantic::{ExtendedExpression, SemanticContextInstruction};
use semantic_analyzer::types::types::{PrimitiveTypes, Type};
use semantic_analyzer::types::PrimitiveValue;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use thiserror::Error;

/// Semantic state processing errors
#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("Semantic analyzer errors count: {:?}", .0.len())]
    SemanticAnalyze(Vec<StateErrorResult>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomExpression<I: SemanticContextInstruction> {
    _marker: PhantomData<I>,
}

impl<I: SemanticContextInstruction> ExtendedExpression<I> for CustomExpression<I> {
    fn expression(
        &self,
        _state: &mut State<Self, I>,
        _block_state: &Rc<RefCell<BlockState<I>>>,
    ) -> ExpressionResult {
        ExpressionResult {
            expr_type: Type::Primitive(PrimitiveTypes::None),
            expr_value: ExpressionResultValue::PrimitiveValue(PrimitiveValue::None),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomExpressionInstruction;

impl SemanticContextInstruction for CustomExpressionInstruction {}

pub struct CustomSemantic<I: SemanticContextInstruction> {
    _state: State<CustomExpression<I>, I>,
}

impl Default for CustomSemantic<CustomExpressionInstruction> {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomSemantic<CustomExpressionInstruction> {
    pub fn new() -> Self {
        Self {
            _state: State::new(),
        }
    }
}

/// Init Semantic state. As source used custom semantic `AST`.
/// And applied semantic-analyzer processing with result of `SemanticState`.
pub fn semantic_state(
) -> anyhow::Result<State<CustomExpression<CustomExpressionInstruction>, CustomExpressionInstruction>>
{
    let fs = ast::FunctionStatement::new(
        ast::FunctionName::new(Ident::new("calculation")),
        vec![ast::FunctionParameter {
            name: ast::ParameterName::new(Ident::new("x")),
            parameter_type: ast::Type::Primitive(ast::PrimitiveTypes::I32),
        }],
        ast::Type::Primitive(ast::PrimitiveTypes::I32),
        vec![
            ast::BodyStatement::LetBinding(ast::LetBinding {
                name: ast::ValueName::new(Ident::new("y")),
                mutable: true,
                value_type: Some(ast::Type::Primitive(ast::PrimitiveTypes::I32)),
                value: Box::new(ast::Expression {
                    expression_value: ast::ExpressionValue::PrimitiveValue(
                        ast::PrimitiveValue::I32(12),
                    ),
                    operation: Some((
                        ast::ExpressionOperations::Plus,
                        Box::new(ast::Expression {
                            expression_value: ast::ExpressionValue::ValueName(ast::ValueName::new(
                                Ident::new("x"),
                            )),
                            operation: Some((
                                ast::ExpressionOperations::Multiply,
                                Box::new(ast::Expression {
                                    expression_value: ast::ExpressionValue::PrimitiveValue(
                                        ast::PrimitiveValue::I32(2),
                                    ),
                                    operation: None,
                                }),
                            )),
                        }),
                    )),
                }),
            }),
            ast::BodyStatement::Binding(ast::Binding {
                name: ast::ValueName::new(Ident::new("y")),
                value: Box::new(ast::Expression {
                    expression_value: ast::ExpressionValue::ValueName(ast::ValueName::new(
                        Ident::new("y"),
                    )),
                    operation: Some((
                        ast::ExpressionOperations::Plus,
                        Box::new(ast::Expression {
                            expression_value: ast::ExpressionValue::ValueName(ast::ValueName::new(
                                Ident::new("x"),
                            )),
                            operation: None,
                        }),
                    )),
                }),
            }),
            ast::BodyStatement::Return(ast::Expression {
                expression_value: ast::ExpressionValue::ValueName(ast::ValueName::new(Ident::new(
                    "y",
                ))),
                operation: None,
            }),
        ],
    );
    let content: ast::Main<
        CustomExpressionInstruction,
        CustomExpression<CustomExpressionInstruction>,
    > = vec![ast::MainStatement::Function(fs)];
    let mut state: State<
        CustomExpression<CustomExpressionInstruction>,
        CustomExpressionInstruction,
    > = State::default();
    state.run(&content);
    if !state.errors.is_empty() {
        println!("{:#?}", state.errors);
        bail!(SemanticError::SemanticAnalyze(state.errors));
    }
    Ok(state)
}
