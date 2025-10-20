# -*- coding: utf-8 -*-
# alu_rpn_calculator_vN.py
"""
RPN_ALU Calculator Module
====================
Reverse Polish Notation calculator

This module provides functionality to:
- Parse mathematical expressions into tokens
- Convert infix notation to RPN using the Shunting Yard algorithm
- Evaluate RPN expressions
- Handle lists of expressions with isolated error handling
- Support basic arithmetic, trigonometric functions, and mathematical constants

"""

import logging
import traceback
from math import sin, cos, tan, radians, sqrt, pi, e
# from collections.abc import Sequence

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


##############################
# CONSTANTS AND CONFIGURATION
##############################

# Operator registry with precedence, arity, and handler functions
OPERATOR_REGISTRY = {
    "+": {
        "precedence": 1,
        "arity": 2,
        "associativity": "left",
        "handler": lambda a, b: a + b,
    },
    "-": {
        "precedence": 1,
        "arity": 2,
        "associativity": "left",
        "handler": lambda a, b: a - b,
    },
    "*": {
        "precedence": 2,
        "arity": 2,
        "associativity": "left",
        "handler": lambda a, b: a * b,
    },
    "/": {
        "precedence": 2,
        "arity": 2,
        "associativity": "left",
        "handler": lambda a, b: a / b if b != 0 else None,  # None signals error
    },
    "**": {
        "precedence": 4,
        "arity": 2,
        "associativity": "right",
        "handler": lambda a, b: a**b,
    },
    "u-": {  # Unary minus
        "precedence": 3,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: -a,
    },
    "sin": {
        "precedence": 5,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: sin(radians(a)),
    },
    "cos": {
        "precedence": 5,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: cos(radians(a)),
    },
    "tan": {
        "precedence": 5,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: tan(radians(a)),
    },
    "sqrt": {
        "precedence": 5,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: sqrt(a) if a >= 0 else None,  # None signals error
    },
    "abs": {
        "precedence": 5,
        "arity": 1,
        "associativity": "right",
        "handler": lambda a: abs(a),
    },
}

# Mathematical constants - resolved during tokenization
MATH_CONSTANTS = {"pi": pi, "e": e}

# Token types for explicit type system
TOKEN_TYPE_NUMBER = "NUMBER"
TOKEN_TYPE_OPERATOR = "OPERATOR"
TOKEN_TYPE_FUNCTION = "FUNCTION"
TOKEN_TYPE_LPAREN = "LPAREN"
TOKEN_TYPE_RPAREN = "RPAREN"
TOKEN_TYPE_CONSTANT = "CONSTANT"


##############################
# SANITIZATION AND VALIDATION
##############################


def sanitize_expression(raw_expression):
    """
    Sanitize and normalize the input expression.

    This function replaces various mathematical symbols with their standard
    ASCII equivalents and removes unnecessary characters.

    Args:
        raw_expression (str): The raw mathematical expression to sanitize

    Returns:
        str: The sanitized expression

    Raises:
        None - returns original if error occurs
    """
    try:
        logger.debug(f"Sanitizing expression: {raw_expression}")

        # Replace special mathematical symbols with standard operators
        expression = raw_expression.replace("×", "*")
        expression = expression.replace("−", "-")
        expression = expression.replace("^", "**")
        expression = expression.replace("=", "")
        expression = expression.replace(
            "°", ""
        )  # Remove degree symbol - handled in trig functions

        # Remove whitespace
        expression = expression.replace(" ", "")
        expression = expression.replace("\n", "")
        expression = expression.replace("\t", "")

        logger.debug(f"Sanitized expression: {expression}")
        return expression

    except Exception as error:
        logger.error(f"Error in sanitize_expression: {traceback.format_exc()}")
        return raw_expression  # Return original if sanitization fails


def validate_parentheses(tokens):
    """
    Validate that parentheses are balanced in the token list.

    Args:
        tokens (list): List of tokens to validate

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    logger.debug("Validating parentheses balance")

    open_count = 0

    for index, token in enumerate(tokens):
        if token == "(":
            open_count += 1
        elif token == ")":
            open_count -= 1
            if open_count < 0:
                error_msg = f"Unmatched closing parenthesis at position {index}"
                logger.warning(error_msg)
                return False, error_msg

    if open_count > 0:
        error_msg = f"Unclosed parentheses: {open_count} left open"
        logger.warning(error_msg)
        return False, error_msg

    return True, None


###############
# TOKENIZATION
###############


def tokenize_expression(expression):
    """
    Convert a mathematical expression string into a list of tokens.

    This function parses the expression character by character, identifying:
    - Numbers (integers and floats)
    - Operators (binary and unary)
    - Functions (sin, cos, tan, sqrt, abs)
    - Parentheses
    - Mathematical constants (pi, e)

    Args:
        expression (str): The mathematical expression to tokenize

    Returns:
        tuple: (success: bool, tokens: list or None, error: dict or None)
    """
    logger.info(f"Starting tokenization of: {expression}")

    tokens = []
    position = 0
    expression_length = len(expression)

    try:
        while position < expression_length:
            current_char = expression[position]
            logger.debug(f"Position {position}: examining '{current_char}'")

            # Skip whitespace (should be removed by sanitization, but safety check)
            if current_char in " \t\n":
                position += 1
                continue

            # Check for multi-character operators and functions
            # Check for exponentiation '**'
            if expression[position : position + 2] == "**":
                tokens.append("**")
                logger.debug(f"Found exponentiation operator at position {position}")
                position += 2
                continue

            # Check for functions
            if expression[position : position + 4] == "sqrt":
                tokens.append("sqrt")
                logger.debug(f"Found sqrt function at position {position}")
                position += 4
                continue

            if expression[position : position + 3] in ["sin", "cos", "tan", "abs"]:
                function_name = expression[position : position + 3]
                tokens.append(function_name)
                logger.debug(f"Found {function_name} function at position {position}")
                position += 3
                continue

            # Check for constants
            if expression[position : position + 2] == "pi":
                tokens.append(
                    ("pi", MATH_CONSTANTS["pi"])
                )  # Store as tuple (name, value)
                logger.debug(f"Found constant pi at position {position}")
                position += 2
                continue

            if current_char == "e" and (
                position + 1 >= expression_length
                or expression[position + 1] not in "0123456789."
            ):
                tokens.append(
                    ("e", MATH_CONSTANTS["e"])
                )  # Store as tuple (name, value)
                logger.debug(f"Found constant e at position {position}")
                position += 1
                continue

            # Check for numbers (including negative numbers)
            if current_char.isdigit() or current_char == ".":
                # Parse the full number
                number_start = position
                has_decimal = current_char == "."
                position += 1

                while position < expression_length:
                    if expression[position].isdigit():
                        position += 1
                    elif expression[position] == "." and not has_decimal:
                        has_decimal = True
                        position += 1
                    else:
                        break

                number_str = expression[number_start:position]
                try:
                    number_value = float(number_str)
                    tokens.append(number_value)
                    logger.debug(
                        f"Found number {number_value} at position {number_start}"
                    )
                except ValueError:
                    error_msg = f"Invalid number format '{number_str}' at position {number_start}"
                    logger.warning(error_msg)
                    return False, None, {"std_err": error_msg}
                continue

            # Handle minus sign (could be unary or binary)
            if current_char == "-":
                # Determine if this is unary or binary minus
                # Unary if: at start, after operator, after opening parenthesis
                is_unary = (
                    position == 0
                    or tokens
                    and (tokens[-1] in ["(", "+", "-", "*", "/", "**", "u-"])
                )

                if is_unary:
                    tokens.append("u-")
                    logger.debug(f"Found unary minus at position {position}")
                else:
                    tokens.append("-")
                    logger.debug(f"Found binary minus at position {position}")
                position += 1
                continue

            # Handle other single-character operators and parentheses
            if current_char in "+-*/()":
                tokens.append(current_char)
                logger.debug(
                    f"Found operator/parenthesis '{current_char}' at position {position}"
                )
                position += 1
                continue

            # Unknown character
            error_msg = f"Unknown character '{current_char}' at position {position}"
            logger.warning(error_msg)
            return False, None, {"std_err": error_msg}

        logger.info(f"Tokenization successful: {len(tokens)} tokens")
        return True, tokens, None

    except Exception as error:
        error_trace = traceback.format_exc()
        logger.error(f"Exception in tokenize_expression: {error_trace}")
        return False, None, {"std_err": f"Tokenization exception: {error_trace}"}


##########################
# SHUNTING YARD ALGORITHM
##########################


def shunting_yard_algorithm(tokens):
    """
    Convert infix notation tokens to Reverse Polish Notation using Dijkstra's Shunting Yard algorithm.

    This function implements the classic algorithm to convert from infix to postfix notation,
    respecting operator precedence and associativity.

    Args:
        tokens (list): List of tokens in infix notation

    Returns:
        tuple: (success: bool, rpn: list or None, error: dict or None)
    """
    logger.info("Starting Shunting Yard algorithm")

    output_queue = []
    operator_stack = []

    try:
        for index, token in enumerate(tokens):
            logger.debug(
                f"Processing token {index}: {token}, Stack: {operator_stack}, Queue: {output_queue}"
            )

            # Handle numbers and resolved constants
            if isinstance(token, (int, float)):
                output_queue.append(token)
                continue

            # Handle constants (stored as tuples)
            if isinstance(token, tuple) and len(token) == 2:
                constant_name, constant_value = token
                output_queue.append(constant_value)
                logger.debug(
                    f"Added constant {constant_name}={constant_value} to output"
                )
                continue

            # Handle operators and functions
            if token in OPERATOR_REGISTRY:
                operator_info = OPERATOR_REGISTRY[token]

                # Pop operators with higher or equal precedence (for left-associative)
                while operator_stack and operator_stack[-1] != "(":
                    if operator_stack[-1] in OPERATOR_REGISTRY:
                        stack_top_info = OPERATOR_REGISTRY[operator_stack[-1]]

                        # Check precedence and associativity
                        if stack_top_info["precedence"] > operator_info[
                            "precedence"
                        ] or (
                            stack_top_info["precedence"] == operator_info["precedence"]
                            and operator_info["associativity"] == "left"
                        ):
                            output_queue.append(operator_stack.pop())
                        else:
                            break
                    else:
                        break

                operator_stack.append(token)
                continue

            # Handle left parenthesis
            if token == "(":
                operator_stack.append(token)
                continue

            # Handle right parenthesis
            if token == ")":
                # Pop operators until we find the matching left parenthesis
                found_left_paren = False
                while operator_stack:
                    if operator_stack[-1] == "(":
                        operator_stack.pop()  # Remove the left parenthesis
                        found_left_paren = True
                        break
                    output_queue.append(operator_stack.pop())

                if not found_left_paren:
                    error_msg = f"Unmatched closing parenthesis at token index {index}"
                    logger.warning(error_msg)
                    return False, None, {"std_err": error_msg}
                continue

            # Unknown token type
            error_msg = f"Unknown token type: {token} at index {index}"
            logger.warning(error_msg)
            return False, None, {"std_err": error_msg}

        # Pop remaining operators
        while operator_stack:
            if operator_stack[-1] in "()":
                error_msg = "Unmatched parentheses in expression"
                logger.warning(error_msg)
                return False, None, {"std_err": error_msg}
            output_queue.append(operator_stack.pop())

        logger.info(f"Shunting Yard successful: {len(output_queue)} tokens in RPN")
        return True, output_queue, None

    except Exception as error:
        error_trace = traceback.format_exc()
        logger.error(f"Exception in shunting_yard_algorithm: {error_trace}")
        return False, None, {"std_err": f"Shunting Yard exception: {error_trace}"}


#################
# RPN EVALUATION
#################


def evaluate_rpn(rpn_tokens):
    """
    Evaluate a list of tokens in Reverse Polish Notation.

    This function uses a stack-based approach to evaluate the RPN expression.

    Args:
        rpn_tokens (list): List of tokens in RPN order

    Returns:
        tuple: (success: bool, result: float or None, recipe: list, error: dict or None)
    """
    logger.info("Starting RPN evaluation")

    evaluation_stack = []
    recipe = []  # Track operations for debugging

    try:
        for token_index, token in enumerate(rpn_tokens):
            logger.debug(
                f"Evaluating token {token_index}: {token}, Stack: {evaluation_stack}"
            )

            # Handle numbers
            if isinstance(token, (int, float)):
                evaluation_stack.append(token)
                recipe.append(("PUSH", token))
                continue

            # Handle operators
            if token in OPERATOR_REGISTRY:
                operator_info = OPERATOR_REGISTRY[token]
                required_operands = operator_info["arity"]

                # Check for sufficient operands
                if len(evaluation_stack) < required_operands:
                    error_msg = f"Insufficient operands for operator '{token}': need {required_operands}, have {len(evaluation_stack)}"
                    logger.warning(error_msg)
                    return False, None, recipe, {"std_err": error_msg}

                # Pop operands
                operands = []
                for _ in range(required_operands):
                    operands.append(evaluation_stack.pop())
                operands.reverse()  # Reverse to get correct order

                # Apply operator
                handler = operator_info["handler"]
                result = handler(*operands)

                # Check for operation errors (like division by zero)
                if result is None:
                    if token == "/":
                        error_msg = f"Division by zero error"
                    elif token == "sqrt":
                        error_msg = f"Cannot calculate square root of negative number: {operands[0]}"
                    else:
                        error_msg = (
                            f"Operation '{token}' failed with operands {operands}"
                        )
                    logger.warning(error_msg)
                    return False, None, recipe, {"std_err": error_msg}

                evaluation_stack.append(result)
                recipe.append(("OPERATOR", token))
                continue

            # Unknown token (shouldn't happen if tokenization was successful)
            error_msg = f"Unknown token in RPN: {token}"
            logger.warning(error_msg)
            return False, None, recipe, {"std_err": error_msg}

        # Check final stack state
        if len(evaluation_stack) != 1:
            error_msg = f"Invalid expression: evaluation stack has {len(evaluation_stack)} values instead of 1"
            logger.warning(error_msg)
            return False, None, recipe, {"std_err": error_msg}

        final_result = evaluation_stack[0]
        logger.info(f"RPN evaluation successful: result = {final_result}")
        return True, final_result, recipe, None

    except Exception as error:
        error_trace = traceback.format_exc()
        logger.error(f"Exception in evaluate_rpn: {error_trace}")
        return False, None, recipe, {"std_err": f"Evaluation exception: {error_trace}"}


############################
# MAIN CALCULATOR FUNCTIONS
############################


def rpn_calculator(raw_expression):
    """
    Calculate the result of a mathematical expression.

    This is the main entry point for single expression calculation.
    Processes the expression through sanitization, tokenization,
    Shunting Yard conversion, and RPN evaluation.

    Args:
        raw_expression (str): The mathematical expression to evaluate

    Returns:
        tuple: (original_expression, recipe, std_err, result)
            - original_expression: The input expression
            - recipe: List of operations performed
            - std_err: Error dictionary if any, None otherwise
            - result: The calculated result or None if error
    """
    logger.info(f"Processing expression: {raw_expression}")

    try:
        # Step 1: Sanitize the expression
        sanitized_expression = sanitize_expression(raw_expression)

        # Step 2: Tokenize the expression
        tokenize_success, tokens, tokenize_error = tokenize_expression(
            sanitized_expression
        )

        if not tokenize_success:
            logger.warning(f"Tokenization failed for: {raw_expression}")
            return (raw_expression, [], tokenize_error, None)

        # Step 3: Validate parentheses
        parentheses_valid, parentheses_error = validate_parentheses(tokens)

        if not parentheses_valid:
            logger.warning(f"Parentheses validation failed: {parentheses_error}")
            return (raw_expression, [], {"std_err": parentheses_error}, None)

        # Step 4: Convert to RPN using Shunting Yard
        shunting_success, rpn_tokens, shunting_error = shunting_yard_algorithm(tokens)

        if not shunting_success:
            logger.warning(f"Shunting Yard failed for: {raw_expression}")
            return (raw_expression, [], shunting_error, None)

        # Step 5: Evaluate the RPN expression
        eval_success, result, recipe, eval_error = evaluate_rpn(rpn_tokens)

        if not eval_success:
            logger.warning(f"Evaluation failed for: {raw_expression}")
            return (raw_expression, recipe, eval_error, None)

        # Success!
        logger.info(f"Successfully calculated: {raw_expression} = {result}")
        return (raw_expression, recipe, None, result)

    except Exception as error:
        # Unexpected exception - this is a true malfunction
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected exception in rpn_calculator: {error_trace}")
        return (
            raw_expression,
            [],
            {"std_err": f"Calculator exception: {error_trace}"},
            None,
        )


def alu_list_rpn_calculator(expression_list):
    """
    Process a list of mathematical expressions.

    Each expression is processed independently, so errors in one expression
    do not affect the processing of others. This provides fault isolation
    for batch processing.

    Args:
        expression_list (Sequence[str]): List of mathematical expressions

    Returns:
        list: List of tuples, each in format (expression, recipe, std_err, result)
    """
    logger.info(f"Processing list of {len(expression_list)} expressions")

    results = []

    for index, expression in enumerate(expression_list):
        logger.debug(
            f"Processing expression {index + 1}/{len(expression_list)}: {expression}"
        )

        try:
            result_tuple = rpn_calculator(expression)
            results.append(result_tuple)

            # Log the result for monitoring
            expression_str, recipe, error, result = result_tuple
            if error:
                logger.info(
                    f"Expression {index + 1} had error: {error.get('std_err', 'Unknown error')}"
                )
            else:
                logger.info(f"Expression {index + 1} successful: {result}")

        except Exception as error:
            # Even if rpn_calculator fails catastrophically, continue with other expressions
            error_trace = traceback.format_exc()
            logger.error(f"Failed to process expression {index + 1}: {error_trace}")
            results.append(
                (
                    expression,
                    [],
                    {"std_err": f"Processing exception: {error_trace}"},
                    None,
                )
            )

    logger.info(f"Completed processing {len(expression_list)} expressions")
    return results


def alu_find_equivalent_expressions(calculation_results):
    """
    Find expressions that evaluate to the same result.

    Groups expressions by their calculated values, identifying mathematical
    equivalences.

    Args:
        calculation_results (list): List of tuples from alu_list_rpn_calculator

    Returns:
        list: List of tuples (expressions_list, common_value) for equivalent expressions
    """
    logger.info("Finding equivalent expressions")

    try:
        # Build equivalence groups
        equivalence_groups = {}

        for expression, recipe, error, result in calculation_results:
            # Skip expressions with errors
            if error is not None or result is None:
                logger.debug(f"Skipping expression with error: {expression}")
                continue

            # Group by result value
            if result not in equivalence_groups:
                equivalence_groups[result] = []
            equivalence_groups[result].append(expression)

        # Filter to only groups with multiple expressions
        equivalent_sets = [
            (expressions, value)
            for value, expressions in equivalence_groups.items()
            if len(expressions) > 1
        ]

        logger.info(f"Found {len(equivalent_sets)} groups of equivalent expressions")
        return equivalent_sets

    except Exception as error:
        error_trace = traceback.format_exc()
        logger.error(f"Exception in alu_find_equivalent_expressions: {error_trace}")
        return []


def mathlist_run(math_list_string):
    # print("math_list_string", math_list_string)
    math_list = math_list_string.split(",")

    result_list_of_lists = []

    for i in math_list:
        result = alu_list_rpn_calculator([i])

        # inspection
        print(result)

        result_list_of_lists.append(result)

    return result_list_of_lists


#################
# TEST EXECUTION
#################


def run_tests():
    """
    Run tests on the calculator system.

    Tests various mathematical expressions to ensure correct functionality.
    """
    logger.info("Starting test suite")

    # Test 1: Basic expression
    print("\n=== Test 1: Basic Expressions ===")
    result = rpn_calculator("(7 - 3) - (5 - 10)")
    print(f"Result: {result}")

    # Test 2: Simple addition
    print("\n=== Test 2: Simple Addition ===")
    results = alu_list_rpn_calculator(["1+1"])
    for r in results:
        print(f"Result: {r}")

    # Test 3: Trigonometric functions
    print("\n=== Test 3: Trigonometric Functions ===")
    results = alu_list_rpn_calculator(["sin(30)", "cos(45)", "tan(60)"])
    for r in results:
        print(f"Result: {r}")

    # Test 4: Error handling
    print("\n=== Test 4: Error Handling ===")
    results = alu_list_rpn_calculator(
        [
            "1/0",  # Division by zero
            "sqrt(-1)",  # Square root of negative
            "2++2",  # Invalid syntax
            "sin(30",  # Unbalanced parentheses
        ]
    )
    for r in results:
        print(f"Result: {r}")

    # Test 5: Complex expressions
    print("\n=== Test 5: Complex Expressions ===")
    math_list_string = """3 + 4 * 2 / ( 1 - 5 ),-5 - (-6),
    −4×(−9),−4*(−9),
    5 + (-3), 10 + (-7), -4 + 11, -2 + 9"""

    math_list = [expr.strip() for expr in math_list_string.split(",")]
    results = alu_list_rpn_calculator(math_list)
    for r in results:
        print(f"Result: {r}")

    # Test 6: Constants
    print("\n=== Test 6: Mathematical Constants ===")
    results = alu_list_rpn_calculator(["2*pi", "e**2", "sqrt(pi)"])
    for r in results:
        print(f"Result: {r}")

    # Test 7: Find equivalences
    print("\n=== Test 7: Finding Equivalent Expressions ===")
    test_expressions = ["2+2", "2*2", "4", "1+3", "8/2", "2**2"]
    results = alu_list_rpn_calculator(test_expressions)
    equivalences = alu_find_equivalent_expressions(results)
    for expressions, value in equivalences:
        print(f"Equivalent expressions with value {value}: {expressions}")

    logger.info("Test suite completed")

    ###########
    # Tests 8
    ###########
    print("\n=== Test 8 ===")

    i = "(7 - 3) - (5 - 10)"
    # # i = "(7) - (10)"
    result = alu_list_rpn_calculator([i])
    print(result)

    result = alu_list_rpn_calculator(["1+1"])
    print(result)

    result = alu_list_rpn_calculator(["sin(30°)"])
    print(result)

    result = alu_list_rpn_calculator(["sin 30"])
    print(result)

    result = alu_list_rpn_calculator(["abs(-3)"])
    print(result)

    math_list_string = """3 + 4 * 2 / ( 1 - 5 ),-5 - (-6),
    −4×(−9),−4*(−9),
    5 + (-3), 10 + (-7), -4 + 11, -2 + 9,
    (-5) + (-3), (-10) - (-7), (-15) + (-8), (-4) - 11,
    5 - (-3), 10 - (-7), 15 - (-8), 4 - (-11),
    5 * (-3), 10 * (-7), -4 * 11, -2 * 9,
    5 / (-3), 10 / (-7), -4 / 11, -2 / 9,
    abs(5), abs(-3), abs(10), abs(-7),
    5**2, (-3)**2, 10**2, (-7)**2,
    5**3, 3**4, 2**7, 6**2,
    0.5**3, 0.3**4, 0.2**7, 0.6**2,
    2**3, 3**4, 4**2, sin(30°) ,cos(30°) , tan(30°)"""

    print(mathlist_run(math_list_string))

    ##############
    # Test Set 9
    ##############
    print("\n=== Tests 9 ===")
    # Basic Arithmetic Operations: Test all basic arithmetic operations (addition, subtraction, multiplication, division) with positive, negative, and zero values.
    math_list = ["1 + 2", "-3 - 4", "5 * -6", "7 / -8", "0 * 3", "-5 + 0"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # alu_list_rpn_calculator(["5 * -6"])

    # Unary Operations: Check the handling of unary minus in various contexts.
    math_list = ["-(3 + 4)", "-3 + 4", "3 + -4", "--5", "-(-3 - -4)"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    result = alu_list_rpn_calculator(["-(-3 - -4)"])
    print(result)

    # Floating Point Numbers: Test with floating-point numbers to ensure accurate handling of decimal values.
    math_list = ["3.5 + 2.1", "-4.7 * 0.5"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Parentheses and Order of Operations: Test expressions with nested parentheses and mixed operations to check if the order of operations is respected.
    math_list = ["(2 + 3) * 4", "2 * (3 + 4)", "(5 - (6 / 2)) * 3"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Functions and Advanced Operations: Include trigonometric functions, exponentiation, square roots, and absolute values.
    math_list = ["sin(30)", "cos(-45)", "tan(60)", "2 ** 3", "sqrt(16)", "abs(-9)"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Edge Cases: Test cases that often result in errors like division by zero, very large and small numbers, and invalid inputs.
    math_list = [
        "1 / 0",
        "99999999 * 9999999",
        "1 / 0.0000001",
        "sin(90) / cos(90)",
        "sqrt(-1)",
    ]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Complex Expressions: Combine multiple operations in a single expression to test complex parsing.
    math_list = ["3 + 4 * 2 / (1 - 5)", "5 - -3 + 2 * 4"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Whitespace and Formatting Variations: Ensure the parser can handle expressions with varying whitespace and formatting.
    math_list = [" 3+5 ", "\n4*5", " 7\t- 2 "]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Special Mathematical Constants and Variables: If your calculator supports it, include tests with special constants like π, e, etc.
    math_list = ["2 * pi", "e ** 2"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    # Invalid Expressions: Test with invalid or malformed expressions to ensure appropriate error handling.
    math_list = ["2 ++ 2", "sin(30 + 60", "2 */ 3"]
    result = alu_list_rpn_calculator(math_list)
    print(result)

    print(
        "\n\nnote: python-logging prints at the top of notebook output (which can be confusing)"
    )


# Run tests if executed directly
if __name__ == "__main__":
    print("Running Tests...")
    run_tests()
