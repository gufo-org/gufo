# Reasoning budget follows the constraint grammar

Sampling (`sampling.hpp`, `json_constraint.hpp`): with a constraint, a
`sampling::ReasoningBudget` also closes when the grammar leaves its
`JsonConstraint::WithReasoning` phase, as on a close spelled in ordinary text
tokens or a native call header written during reasoning. A spent budget no
longer forces its end into the answer after that. `JsonConstraint` gains
private members recording where the reasoning phase's rules and lexemes start.

Source compatible; extensions that embed `JsonConstraint` rebuild.
