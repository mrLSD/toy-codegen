//! # Semantic State
//! Semantic state generation based on semantic `AST`.

use anyhow::bail;
use semantic_analyzer::ast;
use semantic_analyzer::semantic::State;
use semantic_analyzer::types::error::StateErrorResult;
use thiserror::Error;

/// Semantic state processing errors
#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("Errors count: {:?}", .0.len())]
    SemanticAnalyze(Vec<StateErrorResult>),
}

/// Init Semantic state. As source used custom semantic `AST`.
/// And applied semantic-analyzer processing with result of `SemanticState`.
pub fn semantic_state() -> anyhow::Result<State> {
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
    if !state.errors.is_empty() {
        bail!(SemanticError::SemanticAnalyze(state.errors));
    }
    Ok(state)
}