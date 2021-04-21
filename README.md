# Toy Codegen

Toy Codegen is the example compiler for [semantic-analyzer-rs](https://github.com/mrLSD).

As Codegen backend used LLVM, with [inkwell](https://github.com/TheDan64/inkwell).

## Basic design

There is a basic implementation of `code-generation`. The `Semantic AST` structure is 
used as a basic data source. This means that any subset of `programming languages` 
can be represented, the main criterion for which is representability in a 
finite AST structure of Semantic Analyzer, which is used in the `semantic-analyzer` 
library.

### Codegen implementation

The codege is focused only on those instructions that are presented as 
an AST sample. And it is not intended to cover all possible complex 
cases of code generation based on a complete set of `SemanticStateContext` 
instructions. However, the end result is a fully functional binary code.

The basic parts is:
- [x] Function declarations
  - [x] Functions parameters initializations
- [ ] Types declarations
- [ ] Constants declarations
- [x] Function body implementation
  - [x] Let-binding
    - [x] Codegen variables state
  - [ ] Binding
  - [x] Expressions
    - [x] Codegen type conversions
  - [ ] If conditions flow
  - [ ] Loop conditions flow
  - [ ] Function return flow
- [x] Target triple initialization (LLVM backend)
- [x] LLVM IR source code generation (and store to file)
- [x] Binary code generation (and store to file)
- [x] Code generation error flow with custom error types

## LICENCE: MIT
