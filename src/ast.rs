//! # Semantic State
//! Semantic state generation based on semantic `AST`.

use anyhow::bail;
use semantic_analyzer::ast::{self, Ident};
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::error::StateErrorResult;
use thiserror::Error;

/// Semantic state processing errors
#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("Semantic analyzer errors count: {:?}", .0.len())]
    SemanticAnalyze(Vec<StateErrorResult>),
}

/// Init Semantic state. As source used custom semantic `AST`.
/// And applied semantic-analyzer processing with result of `SemanticState`.
pub fn semantic_state() -> anyhow::Result<State> {
    let content: ast::Main = vec![ast::MainStatement::Function(ast::FunctionStatement {
        name: ast::FunctionName::new(Ident::new("calculation")),
        result_type: ast::Type::Primitive(ast::PrimitiveTypes::I32),
        parameters: vec![ast::FunctionParameter {
            name: ast::ParameterName::new(Ident::new("x")),
            parameter_type: ast::Type::Primitive(ast::PrimitiveTypes::I32),
        }],
        body: vec![
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
                        ast::ExpressionOperations::Minus,
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
    })];
    let mut state = State::new();
    state.run(&content);
    if !state.errors.is_empty() {
        println!("{:#?}", state.errors);
        bail!(SemanticError::SemanticAnalyze(state.errors));
    }
    Ok(state)
}
