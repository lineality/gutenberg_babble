# -*- coding: utf-8 -*-
# math_expression_generator_v.py
# flat version, can be modules, to generate math expressions in words & numbers
"""
This may be used along with RPN-ALU calculator code.
"""

"""
number_to_words.py - Convert numbers to English words representation.

This module provides functionality to convert integers and decimal numbers
into their English word equivalents. Designed for mathematical expression
verbalization.

"""

import traceback
from decimal import Decimal
import random


def integer_to_words(number: int) -> str:
    """
    Convert an integer to its English word representation.

    Handles integers from -999,999 to 999,999.

    Args:
        number: Integer to convert

    Returns:
        String representation in English words

    Examples:
        >>> integer_to_words(0)
        'zero'
        >>> integer_to_words(-42)
        'negative forty two'
        >>> integer_to_words(1234)
        'one thousand two hundred thirty four'
    """
    try:
        # Handle zero case
        if number == 0:
            return "zero"

        # Handle negative numbers
        if number < 0:
            return f"negative {integer_to_words(abs(number))}"

        # Word mappings for numbers
        ones_words = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        tens_words = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        # Handle numbers less than 20
        if number < 20:
            return ones_words[number]

        # Handle numbers less than 100
        if number < 100:
            tens = number // 10
            ones = number % 10
            if ones == 0:
                return tens_words[tens]
            else:
                return f"{tens_words[tens]} {ones_words[ones]}"

        # Handle numbers less than 1000
        if number < 1000:
            hundreds = number // 100
            remainder = number % 100
            result = f"{ones_words[hundreds]} hundred"
            if remainder > 0:
                result += f" {integer_to_words(remainder)}"
            return result

        # Handle numbers less than 1,000,000
        if number < 1000000:
            thousands = number // 1000
            remainder = number % 1000
            result = f"{integer_to_words(thousands)} thousand"
            if remainder > 0:
                result += f" {integer_to_words(remainder)}"
            return result

        # Numbers beyond our range
        raise ValueError(
            f"Number {number} is outside supported range (-999999 to 999999)"
        )

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error converting integer {number} to words: {str(e)}")


def decimal_to_words(number: float, max_decimal_places: int = 4) -> str:
    """
    Convert a decimal number to its English word representation.

    Uses digit-by-digit verbalization for decimal portion to avoid ambiguity.

    Args:
        number: Float number to convert
        max_decimal_places: Maximum decimal places to verbalize (default 4)

    Returns:
        String representation in English words

    Examples:
        >>> decimal_to_words(3.14)
        'three point one four'
        >>> decimal_to_words(-2.5)
        'negative two point five'
    """
    try:
        # Convert to string to handle precision properly
        number_str = str(number)

        # Handle negative sign
        is_negative = number < 0
        if is_negative:
            number_str = number_str[1:]  # Remove negative sign
            number = abs(number)

        # Split into integer and decimal parts
        if "." in number_str:
            integer_part, decimal_part = number_str.split(".")

            # Limit decimal places
            if len(decimal_part) > max_decimal_places:
                decimal_part = decimal_part[:max_decimal_places]

            # Convert integer part
            integer_value = int(integer_part)
            result = integer_to_words(integer_value)

            # Add decimal part digit by digit
            result += " point"

            digit_words = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]

            for digit in decimal_part:
                result += f" {digit_words[int(digit)]}"

            # Add negative prefix if needed
            if is_negative:
                result = f"negative {result}"

            return result
        else:
            # No decimal part, just convert as integer
            return integer_to_words(int(number))

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error converting decimal {number} to words: {str(e)}")


def constant_to_words(constant: str) -> str:
    """
    Convert mathematical constants to their English representation.

    Args:
        constant: String name of constant ('pi' or 'e')

    Returns:
        English verbalization of the constant
    """
    try:
        constant_variations = {
            "pi": ["pi", "the constant pi"],
            "e": ["e", "the constant e", "Euler's number"],
        }

        if constant.lower() in constant_variations:
            # Randomly select a variation if configured
            return random.choice(constant_variations[constant.lower()])
        else:
            return constant

    except Exception as e:
        traceback.print_exc()


"""
expression_ast.py - Abstract Syntax Tree node definitions for mathematical expressions.

This module defines the core data structures for representing mathematical
expressions as tree structures, enabling reliable generation and serialization.

"""

import traceback
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random


@dataclass
class ASTNode(ABC):
    """
    Abstract base class for all AST nodes in mathematical expressions.
    """

    @abstractmethod
    def to_symbolic(self) -> str:
        """Convert node to symbolic mathematical notation."""
        pass

    @abstractmethod
    def to_english(self) -> str:
        """Convert node to English words representation."""
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """Get the maximum depth of this subtree."""
        pass

    @abstractmethod
    def get_operator_count(self) -> int:
        """Count total operators in this subtree."""
        pass


@dataclass
class NumberNode(ASTNode):
    """
    Represents a numeric value in the expression tree.

    Attributes:
        value: The numeric value (int or float)
        is_constant: Whether this is a mathematical constant (pi, e)
        constant_name: Name of the constant if applicable
    """

    value: float
    is_constant: bool = False
    constant_name: str = None

    def to_symbolic(self) -> str:
        """Convert number to symbolic notation."""
        try:
            if self.is_constant:
                return self.constant_name
            elif isinstance(self.value, int) or self.value.is_integer():
                return str(int(self.value))
            else:
                return str(self.value)
        except Exception as e:
            traceback.print_exc()
            return str(self.value)

    def to_english(self) -> str:
        """Convert number to English words."""
        try:
            if self.is_constant:
                # from number_to_words import constant_to_words
                return constant_to_words(self.constant_name)
            elif isinstance(self.value, int) or self.value.is_integer():
                # from number_to_words import integer_to_words
                return integer_to_words(int(self.value))
            else:
                # from number_to_words import decimal_to_words
                return decimal_to_words(self.value)
        except Exception as e:
            traceback.print_exc()
            return str(self.value)  # Fallback

    def get_depth(self) -> int:
        """Leaf node has depth 0."""
        return 0

    def get_operator_count(self) -> int:
        """Number nodes have no operators."""
        return 0


@dataclass
class UnaryOpNode(ASTNode):
    """
    Represents a unary operation (negation, sqrt, abs, trig functions).

    Attributes:
        operator: The unary operator symbol/name
        operand: The operand node
    """

    operator: str
    operand: ASTNode

    def to_symbolic(self) -> str:
        """Convert unary operation to symbolic notation."""
        try:
            operand_str = self.operand.to_symbolic()

            if self.operator == "neg":  # Unary negation
                # Check if operand needs parentheses
                if isinstance(self.operand, BinaryOpNode):
                    return f"-({operand_str})"
                else:
                    return f"-{operand_str}"
            elif self.operator in ["sqrt", "abs", "sin", "cos", "tan"]:
                return f"{self.operator}({operand_str})"
            else:
                return f"{self.operator}({operand_str})"

        except Exception as e:
            traceback.print_exc()
            return f"{self.operator}({self.operand.to_symbolic()})"

    def to_english(self) -> str:
        """Convert unary operation to English words."""
        try:
            operand_english = self.operand.to_english()

            operator_words = {
                "neg": "negative",
                "sqrt": "square root of",
                "abs": "absolute value of",
                "sin": "sine of",
                "cos": "cosine of",
                "tan": "tangent of",
            }

            operator_word = operator_words.get(self.operator, self.operator)

            if self.operator == "neg":
                # Special handling for negation
                if isinstance(self.operand, BinaryOpNode):
                    return (
                        f"negative open parenthesis {operand_english} close parenthesis"
                    )
                else:
                    return f"negative {operand_english}"
            else:
                return f"{operator_word} {operand_english}"

        except Exception as e:
            traceback.print_exc()
            return f"{self.operator} of {self.operand.to_english()}"

    def get_depth(self) -> int:
        """Depth is 1 + operand depth."""
        return 1 + self.operand.get_depth()

    def get_operator_count(self) -> int:
        """Count this operator plus operand operators."""
        return 1 + self.operand.get_operator_count()


@dataclass
class BinaryOpNode(ASTNode):
    """
    Represents a binary operation (addition, subtraction, multiplication, division, exponentiation).

    Attributes:
        operator: The binary operator symbol
        left: Left operand node
        right: Right operand node
        needs_parentheses: Whether this node needs parentheses when serialized
    """

    operator: str
    left: ASTNode
    right: ASTNode
    needs_parentheses: bool = False

    def to_symbolic(self) -> str:
        """Convert binary operation to symbolic notation."""
        try:
            left_str = self.left.to_symbolic()
            right_str = self.right.to_symbolic()

            # Map operator to symbol
            operator_symbols = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "pow": "**",
            }

            op_symbol = operator_symbols.get(self.operator, self.operator)

            result = f"{left_str} {op_symbol} {right_str}"

            if self.needs_parentheses:
                result = f"({result})"

            return result

        except Exception as e:
            traceback.print_exc()
            return f"({self.left.to_symbolic()} {self.operator} {self.right.to_symbolic()})"

    def to_english(self) -> str:
        """Convert binary operation to English words."""
        try:
            left_english = self.left.to_english()
            right_english = self.right.to_english()

            # Operator word mappings with variations
            operator_words_simple = {
                "add": "plus",
                "sub": "minus",
                "mul": "times",
                "div": "divided by",
                "pow": "to the power of",
            }

            operator_word = operator_words_simple.get(self.operator, self.operator)

            result = f"{left_english} {operator_word} {right_english}"

            # Add explicit parentheses if needed
            if self.needs_parentheses:
                result = f"open parenthesis {result} close parenthesis"

            return result

        except Exception as e:
            traceback.print_exc()
            return f"{self.left.to_english()} {self.operator} {self.right.to_english()}"

    def get_depth(self) -> int:
        """Depth is 1 + max of children depths."""
        return 1 + max(self.left.get_depth(), self.right.get_depth())

    def get_operator_count(self) -> int:
        """Count this operator plus children operators."""
        return 1 + self.left.get_operator_count() + self.right.get_operator_count()


@dataclass
class ParenthesesNode(ASTNode):
    """
    Explicitly represents parentheses grouping.

    Attributes:
        expression: The expression inside the parentheses
    """

    expression: ASTNode

    def to_symbolic(self) -> str:
        """Convert to symbolic with parentheses."""
        return f"({self.expression.to_symbolic()})"

    def to_english(self) -> str:
        """Convert to English with explicit parentheses."""
        return f"open parenthesis {self.expression.to_english()} close parenthesis"

    def get_depth(self) -> int:
        """Depth is same as inner expression."""
        return self.expression.get_depth()

    def get_operator_count(self) -> int:
        """Operator count is same as inner expression."""
        return self.expression.get_operator_count()


"""
expression_builder.py - Random mathematical expression tree builder.

This module provides functionality to generate random mathematical expression
trees based on configurable parameters and constraints.

"""

import random
import traceback
# from expression_ast import (
#     ASTNode, NumberNode, UnaryOpNode, BinaryOpNode, ParenthesesNode
# )


def generate_random_number(
    positive_range: int,
    negative_range: int,
    allow_negative: bool,
    allow_decimals: bool,
    max_decimal_places: int,
    allow_zero: bool = True,
) -> NumberNode:
    """
    Generate a random number node based on constraints.

    Args:
        positive_range: Maximum positive value
        negative_range: Minimum negative value (should be negative)
        allow_negative: Whether to allow negative numbers
        allow_decimals: Whether to generate decimal numbers
        max_decimal_places: Maximum decimal places if decimals allowed
        allow_zero: Whether zero is allowed

    Returns:
        NumberNode with generated value
    """
    try:
        # Determine range based on constraints
        min_val = negative_range if allow_negative else 1 if not allow_zero else 0
        max_val = positive_range

        # Exclude zero if not allowed
        if not allow_zero and min_val <= 0 <= max_val:
            # Generate positive or negative, but not zero
            if allow_negative:
                if random.random() < 0.5:
                    value = random.randint(1, max_val)
                else:
                    value = random.randint(negative_range, -1)
            else:
                value = random.randint(1, max_val)
        else:
            value = random.randint(min_val, max_val)

        # Convert to decimal if requested
        if allow_decimals and random.random() < 0.3:  # 30% chance of decimal
            decimal_places = random.randint(1, max_decimal_places)
            decimal_part = random.random()
            value = value + round(decimal_part, decimal_places)

        return NumberNode(value=float(value))

    except Exception as e:
        traceback.print_exc()
        # Fallback to simple positive integer
        return NumberNode(value=1.0)


def generate_random_constant() -> NumberNode:
    """
    Generate a random mathematical constant (pi or e).

    Returns:
        NumberNode representing the constant
    """
    try:
        import math

        if random.random() < 0.5:
            return NumberNode(value=math.pi, is_constant=True, constant_name="pi")
        else:
            return NumberNode(value=math.e, is_constant=True, constant_name="e")

    except Exception as e:
        traceback.print_exc()
        return NumberNode(value=3.14159, is_constant=True, constant_name="pi")


def generate_expression_tree(
    current_depth: int = 0,
    max_depth: int = 3,
    operator_count: int = 0,
    max_operators: int = 10,
    config: dict = None,
) -> ASTNode:
    """
    Recursively generate a random expression tree.

    Args:
        current_depth: Current recursion depth
        max_depth: Maximum allowed depth
        operator_count: Current operator count
        max_operators: Maximum allowed operators
        config: Configuration dictionary with all parameters

    Returns:
        Root node of generated expression tree
    """
    try:
        # Default config if not provided
        if config is None:
            config = {
                "positive_value_range": 100,
                "negative_value_range": -100,
                "allow_negative_values": True,
                "max_decimal_places_in_input": 2,
                "include_addition": True,
                "include_subtraction": True,
                "include_multiplication": True,
                "include_division": False,
                "include_parentheses": True,
                "include_constants": False,
                "allow_decimals": False,
                "allow_zero": True,
                "include_exponentiation": False,
                "include_sqrt": False,
                "include_abs": False,
                "include_trig": False,
            }

        # Base case: generate a number if at max depth or max operators
        if current_depth >= max_depth or operator_count >= max_operators:
            # Small chance to generate a constant if enabled
            if config.get("include_constants", False) and random.random() < 0.1:
                return generate_random_constant()
            else:
                return generate_random_number(
                    positive_range=config["positive_value_range"],
                    negative_range=config["negative_value_range"],
                    allow_negative=config["allow_negative_values"],
                    allow_decimals=config.get("allow_decimals", False),
                    max_decimal_places=config["max_decimal_places_in_input"],
                    allow_zero=config.get("allow_zero", True),
                )

        # Decide node type based on depth and randomness
        # Higher chance of operators at lower depths
        operator_chance = 0.7 * (1 - current_depth / max_depth)

        if random.random() < operator_chance and operator_count < max_operators:
            # Generate an operator node
            available_operators = []

            # Binary operators
            if config.get("include_addition", True):
                available_operators.append("add")
            if config.get("include_subtraction", True):
                available_operators.append("sub")
            if config.get("include_multiplication", True):
                available_operators.append("mul")
            if config.get("include_division", False):
                available_operators.append("div")
            if config.get("include_exponentiation", False):
                available_operators.append("pow")

            # Unary operators
            unary_operators = []
            if config.get("allow_negative_values", True) and random.random() < 0.2:
                unary_operators.append("neg")
            if config.get("include_sqrt", False):
                unary_operators.append("sqrt")
            if config.get("include_abs", False):
                unary_operators.append("abs")
            if config.get("include_trig", False):
                unary_operators.extend(["sin", "cos", "tan"])

            # Decide between unary and binary
            if unary_operators and random.random() < 0.2:  # 20% chance for unary
                operator = random.choice(unary_operators)
                operand = generate_expression_tree(
                    current_depth + 1,
                    max_depth,
                    operator_count + 1,
                    max_operators,
                    config,
                )
                node = UnaryOpNode(operator=operator, operand=operand)
            else:
                # Binary operator
                if not available_operators:
                    available_operators = ["add"]  # Fallback

                operator = random.choice(available_operators)

                # Generate left and right operands
                left = generate_expression_tree(
                    current_depth + 1,
                    max_depth,
                    operator_count + 1,
                    max_operators,
                    config,
                )

                # For division in valid mode, ensure denominator is not zero
                if operator == "div" and not config.get("allow_divide_by_zero", True):
                    # Force non-zero right operand
                    config_copy = config.copy()
                    config_copy["allow_zero"] = False
                    right = generate_expression_tree(
                        current_depth + 1,
                        max_depth,
                        operator_count + 1,
                        max_operators,
                        config_copy,
                    )
                else:
                    right = generate_expression_tree(
                        current_depth + 1,
                        max_depth,
                        operator_count + 1,
                        max_operators,
                        config,
                    )

                # Determine if parentheses needed based on operator precedence
                needs_parens = False
                if config.get("include_parentheses", False) and random.random() < 0.3:
                    needs_parens = True

                node = BinaryOpNode(
                    operator=operator,
                    left=left,
                    right=right,
                    needs_parentheses=needs_parens,
                )

            return node
        else:
            # Generate a number node
            if config.get("include_constants", False) and random.random() < 0.1:
                return generate_random_constant()
            else:
                return generate_random_number(
                    positive_range=config["positive_value_range"],
                    negative_range=config["negative_value_range"],
                    allow_negative=config["allow_negative_values"],
                    allow_decimals=config.get("allow_decimals", False),
                    max_decimal_places=config["max_decimal_places_in_input"],
                    allow_zero=config.get("allow_zero", True),
                )

    except Exception as e:
        traceback.print_exc()
        # Fallback to simple number
        return NumberNode(value=1.0)


def build_expression_with_config(config: dict) -> ASTNode:
    """
    Build an expression tree using the provided configuration.

    Args:
        config: Configuration dictionary with all generation parameters

    Returns:
        Root node of the generated expression tree
    """
    try:
        # Extract structural parameters
        max_depth = config.get("max_nesting_depth", 3)
        max_operators = config.get("max_number_of_operators", 10)

        # Apply valid-only mode restrictions if enabled
        if config.get("valid_expressions_only_mode", False):
            # Ultra-conservative settings
            config = config.copy()  # Don't modify original
            config["allow_zero"] = False
            config["include_division"] = False
            config["include_exponentiation"] = False
            config["include_sqrt"] = False
            config["allow_negative_values"] = False
            config["positive_value_range"] = min(100, config["positive_value_range"])
            config["allow_decimals"] = False

        # Generate the expression tree
        tree = generate_expression_tree(
            current_depth=0,
            max_depth=max_depth,
            operator_count=0,
            max_operators=max_operators,
            config=config,
        )

        return tree

    except Exception as e:
        traceback.print_exc()
        # Fallback to simple expression
        return BinaryOpNode(
            operator="add",
            left=NumberNode(value=1.0),
            right=NumberNode(value=1.0),
            needs_parentheses=False,
        )


"""
generator.py - Main API for mathematical expression generation.


This module provides the primary interface for generating pairs of
mathematical expressions in English words and symbolic notation.

"""

import traceback
import random
# from expression_builder import build_expression_with_config

"""
Additional helper functions for expression validation and filtering.
"""


def count_operators_in_symbolic(symbolic_expression: str) -> int:
    """
    Count the number of operators in a symbolic expression.

    This is a simple heuristic counter that identifies operators by their symbols.

    Args:
        symbolic_expression: The symbolic math expression string

    Returns:
        Count of operators found

    Examples:
        >>> count_operators_in_symbolic("4")
        0
        >>> count_operators_in_symbolic("2 + 3")
        1
        >>> count_operators_in_symbolic("(5 - 3) * 2")
        2
    """
    try:
        # Operators to count (matching ALU system)
        operators = ["+", "-", "*", "/", "**"]

        # Functions to count
        functions = ["sin", "cos", "tan", "sqrt", "abs"]

        count = 0

        # Count symbol operators
        for op in operators:
            # Count occurrences, but be careful with ** vs * and -- vs -
            if op == "**":
                count += symbolic_expression.count("**")
            elif op == "*":
                # Count * but not ** (already counted)
                temp_expr = symbolic_expression.replace("**", "")
                count += temp_expr.count("*")
            elif op == "-":
                # This is tricky - we want to count binary minus, not unary negative
                # Simple heuristic: count standalone minus not at start or after (
                parts = symbolic_expression.split()
                for i, part in enumerate(parts):
                    if part == "-" and i > 0:
                        count += 1
            else:
                count += symbolic_expression.count(op)

        # Count function calls
        for func in functions:
            count += symbolic_expression.count(f"{func}(")

        return count

    except Exception as e:
        traceback.print_exc()
        return 0


def count_operands_in_symbolic(symbolic_expression: str) -> int:
    """
    Count the number of operands (numbers/constants) in a symbolic expression.

    Args:
        symbolic_expression: The symbolic math expression string

    Returns:
        Count of operands found

    Examples:
        >>> count_operands_in_symbolic("4")
        1
        >>> count_operands_in_symbolic("2 + 3")
        2
        >>> count_operands_in_symbolic("pi * 2")
        2
    """
    try:
        import re

        # Remove operators and parentheses, then count remaining tokens
        # This is a simple heuristic

        # Pattern to match numbers (including decimals) and constants
        pattern = r"\b\d+\.?\d*\b|\b(?:pi|e)\b"

        matches = re.findall(pattern, symbolic_expression)

        return len(matches)

    except Exception as e:
        traceback.print_exc()
        return 0


def expression_meets_minimum_complexity(
    english: str, symbolic: str, min_operators: int = 1, min_operands: int = 2
) -> bool:
    """
    Check if an expression meets minimum complexity requirements.

    Args:
        english: English word version of expression
        symbolic: Symbolic notation version of expression
        min_operators: Minimum number of operators required
        min_operands: Minimum number of operands required

    Returns:
        True if expression meets requirements, False otherwise

    Examples:
        >>> expression_meets_minimum_complexity("four", "4")
        False
        >>> expression_meets_minimum_complexity("two plus three", "2 + 3")
        True
    """
    try:
        operator_count = count_operators_in_symbolic(symbolic)
        operand_count = count_operands_in_symbolic(symbolic)

        return operator_count >= min_operators and operand_count >= min_operands

    except Exception as e:
        traceback.print_exc()
        return False


def math_expression_generator(
    n_expressions: int = 1,
    min_operators: int = 1,
    min_operands: int = 2,
    max_attempts_per_expression: int = 100,
    random_seed=None,
    **kwargs,
) -> list[tuple[str, str]]:
    """
    Generate n mathematical expression pairs with minimum complexity requirements.

    This is a wrapper around generate_n_expressions_raw_no_seed() that filters out
    expressions that don't meet minimum complexity requirements and
    regenerates until the desired count is reached.

    Args:
        n_expressions: Number of expression pairs to generate
        min_operators: Minimum number of operators required (default: 1)
        min_operands: Minimum number of operands required (default: 2)
        max_attempts_per_expression: Maximum generation attempts per expression
        **kwargs: All other arguments passed to generate_n_expressions_raw_no_seed()

    Returns:
        List of tuples (english_words, symbolic_notation) meeting complexity requirements

    Raises:
        RuntimeError: If unable to generate expressions meeting requirements

    Examples:
        >>> results = generate_n_expressions_raw_no_seed_with_min_operands(
        ...     n_expressions=5,
        ...     min_operators=1,
        ...     min_operands=2,
        ...     positive_value_range=10
        ... )
        >>> len(results)
        5
        >>> all(count_operators_in_symbolic(sym) >= 1 for _, sym in results)
        True
    """
    try:
        # Set the random seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        valid_results = []
        total_attempts = 0
        max_total_attempts = n_expressions * max_attempts_per_expression

        while len(valid_results) < n_expressions:
            # Safety check to prevent infinite loops
            if total_attempts >= max_total_attempts:
                raise RuntimeError(
                    f"Could not generate {n_expressions} expressions meeting "
                    f"minimum complexity requirements (min_operators={min_operators}, "
                    f"min_operands={min_operands}) after {total_attempts} attempts. "
                    f"Try adjusting your configuration parameters (e.g., increase "
                    f"max_nesting_depth or max_number_of_operators)."
                )

            # Generate one candidate expression
            candidates = generate_n_expressions_raw_no_seed(n_expressions=1, **kwargs)
            total_attempts += 1

            # Check if it meets minimum complexity
            for english, symbolic in candidates:
                if expression_meets_minimum_complexity(
                    english, symbolic, min_operators, min_operands
                ):
                    valid_results.append((english, symbolic))
                    break

            # Progress indicator for large generation runs
            if total_attempts % 50 == 0 and len(valid_results) < n_expressions:
                print(
                    f"Progress: {len(valid_results)}/{n_expressions} valid expressions "
                    f"generated after {total_attempts} attempts..."
                )

        return valid_results[:n_expressions]  # Ensure exact count

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Error generating expressions with minimum complexity: {str(e)}"
        )


def generate_n_expressions_raw_no_seed(
    n_expressions: int = 1,
    positive_value_range: int = 999999,
    negative_value_range: int = -999999,
    allow_negative_values: bool = True,
    always_include_random_word_variations: bool = True,
    max_decimal_places_in_input: int = 4,
    include_addition: bool = True,
    include_subtraction: bool = True,
    include_multiplication: bool = True,
    include_division: bool = True,
    valid_expressions_only_mode: bool = False,
    allow_divide_by_zero: bool = True,
    max_nesting_depth: int = 4,
    max_number_of_operators: int = 100,
    include_sqrt: bool = False,
    include_trig: bool = False,
    include_parentheses: bool = False,
    include_abs: bool = False,
    include_exponentiation: bool = False,
    include_constants: bool = False,
    allow_decimals: bool = True,
) -> list[tuple[str, str]]:
    """
    Generate n mathematical expression pairs (English, symbolic).

    This is the main API function that generates mathematical expressions
    according to the specified configuration parameters.

    NOTE! For random seed, use the wrapper -> math_expression_generator()
    (or modified the function)

    Args:
        n_expressions: Number of expression pairs to generate
        positive_value_range: Maximum positive value for numbers
        negative_value_range: Minimum negative value for numbers
        allow_negative_values: Whether to allow negative numbers
        always_include_random_word_variations: Use word variations in English
        max_decimal_places_in_input: Maximum decimal places for decimals
        include_addition: Include addition operator
        include_subtraction: Include subtraction operator
        include_multiplication: Include multiplication operator
        include_division: Include division operator
        valid_expressions_only_mode: Ultra-conservative valid-only mode
        allow_divide_by_zero: Whether to allow division by zero
        max_nesting_depth: Maximum depth of expression tree
        max_number_of_operators: Maximum operators in an expression
        include_sqrt: Include square root function
        include_trig: Include trigonometric functions
        include_parentheses: Include explicit parentheses
        include_abs: Include absolute value function
        include_exponentiation: Include exponentiation operator
        include_constants: Include mathematical constants (pi, e)
        allow_decimals: Whether to generate decimal numbers

    Returns:
        List of tuples (english_words, symbolic_notation)

    Examples:
        >>> generate_n_expressions_raw_no_seed(1, include_division=False)
        [('two plus three', '2 + 3')]

        >>> generate_n_expressions_raw_no_seed(1, valid_expressions_only_mode=True)
        [('five times seven', '5 * 7')]
    """
    try:
        # Validate inputs
        if n_expressions < 1:
            raise ValueError(f"n_expressions must be at least 1, got {n_expressions}")

        if positive_value_range < 1:
            raise ValueError(
                f"positive_value_range must be at least 1, got {positive_value_range}"
            )

        if negative_value_range > -1:
            raise ValueError(
                f"negative_value_range must be negative, got {negative_value_range}"
            )

        # Build configuration dictionary
        config = {
            "positive_value_range": positive_value_range,
            "negative_value_range": negative_value_range,
            "allow_negative_values": allow_negative_values,
            "always_include_random_word_variations": always_include_random_word_variations,
            "max_decimal_places_in_input": max_decimal_places_in_input,
            "include_addition": include_addition,
            "include_subtraction": include_subtraction,
            "include_multiplication": include_multiplication,
            "include_division": include_division,
            "valid_expressions_only_mode": valid_expressions_only_mode,
            "allow_divide_by_zero": allow_divide_by_zero,
            "max_nesting_depth": max_nesting_depth,
            "max_number_of_operators": max_number_of_operators,
            "include_sqrt": include_sqrt,
            "include_trig": include_trig,
            "include_parentheses": include_parentheses,
            "include_abs": include_abs,
            "include_exponentiation": include_exponentiation,
            "include_constants": include_constants,
            "allow_decimals": allow_decimals,
            "allow_zero": not valid_expressions_only_mode,
        }

        # Generate expressions
        result_pairs = []

        for i in range(n_expressions):
            try:
                # Build expression tree
                expression_tree = build_expression_with_config(config)

                # Convert to both formats
                english_version = expression_tree.to_english()
                symbolic_version = expression_tree.to_symbolic()

                # Clean up whitespace
                english_version = " ".join(english_version.split())
                symbolic_version = " ".join(symbolic_version.split())

                result_pairs.append((english_version, symbolic_version))

            except Exception as e:
                # Log error but continue generating
                print(f"Error generating expression {i + 1}: {str(e)}")
                traceback.print_exc()

                # Add a simple fallback expression
                result_pairs.append(("one plus one", "1 + 1"))

        return result_pairs

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error in generate_n_expressions_raw_no_seed: {str(e)}")


# Testing function
def test_generator():
    """
    Test the expression generator with various configurations.
    """
    try:
        print("=" * 60)
        print("EXPRESSION GENERATOR TESTS")
        print("=" * 60)

        # Test 1: Simple expressions
        print("\n1. Simple expressions (no division, no negatives):")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=6,
            positive_value_range=10,
            allow_negative_values=False,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=False,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # Test 2: Valid-only mode
        print("\n2. Valid-only mode (ultra-conservative):")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=6,
            valid_expressions_only_mode=True,
            max_nesting_depth=2,
            max_number_of_operators=2,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # Test 3: With negatives and parentheses
        print("\n3. With negatives and parentheses:")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=6,
            positive_value_range=20,
            negative_value_range=-20,
            allow_negative_values=True,
            include_parentheses=True,
            max_nesting_depth=3,
            max_number_of_operators=3,
            allow_decimals=False,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # Test 4: With decimals
        print("\n4. With decimal numbers:")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=6,
            positive_value_range=10,
            allow_negative_values=False,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=True,
            max_decimal_places_in_input=2,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # Test 5: With constants
        print("\n5. With mathematical constants:")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=6,
            positive_value_range=10,
            allow_negative_values=False,
            include_constants=True,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=False,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # Test 6: minimum two operands
        print("\n6. minimum two operands")
        results = generate_n_expressions_raw_no_seed(
            n_expressions=10,
            positive_value_range=10,
            allow_negative_values=False,
            include_constants=True,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=False,
        )
        for english, symbolic in results:
            print(f"  '{english}' → '{symbolic}'")

        # raw tests
        print(
            "\nRaw generate w/ generate_n_expressions_raw_no_seed() -> maybe stubs with no operators"
        )
        results = generate_n_expressions_raw_no_seed(
            n_expressions=10,
            positive_value_range=10,
            allow_negative_values=False,
            include_constants=True,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=False,
        )
        for i in results:
            print(i)

        print(
            "\nRaw generate w/ math_expression_generator() -> should be min one operator"
        )
        results = math_expression_generator(
            n_expressions=10,
            positive_value_range=10,
            allow_negative_values=False,
            include_constants=True,
            include_division=False,
            max_nesting_depth=2,
            max_number_of_operators=2,
            allow_decimals=False,
            random_seed=42,
        )
        for i in results:
            print(i)

        print("\n" + "=" * 60)
        print("All tests performed...")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    test_generator()
