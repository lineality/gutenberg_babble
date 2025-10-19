"""
TOML Dataset Generator for Mathematical Expressions

This module generates N unique mathematical expression datasets,
each containing an expression (in English and symbolic form) and
its evaluation steps via RPN calculator. Each dataset is saved as
a numbered TOML file.

The module handles:
- Duplicate filtering (exact symbolic string matching)
- Error resilience (continues until N valid files produced)
- Batch generation with buffer strategy
- Versioned output directory creation
- Comprehensive error logging

Author: Generated Assistant
Date: 2024
"""

import os
import traceback
from pathlib import Path
from datetime import datetime
from collections.abc import Iterator


def create_versioned_directory(base_dir_name: str) -> Path:
    """
    Create output directory with version suffix if it already exists.

    If base_dir_name exists, tries base_dir_name_v2, base_dir_name_v3, etc.
    until finding an available name.

    Args:
        base_dir_name: Base directory name (e.g., 'toml_production_output_100')

    Returns:
        Path object of created directory

    Raises:
        OSError: If directory creation fails after finding available name
        PermissionError: If lacking permissions to create directory
    """
    try:
        # Try base name first
        output_dir = Path.cwd() / base_dir_name

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
            print(f"Created output directory: {output_dir}")
            return output_dir

        # Base exists, find next available version
        version = 2
        while True:
            versioned_name = f"{base_dir_name}_v{version}"
            output_dir = Path.cwd() / versioned_name

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=False)
                print(f"Created versioned output directory: {output_dir}")
                return output_dir

            version += 1

            # Safety check to prevent infinite loop
            if version > 10000:
                error_msg = f"Could not create directory after checking 10000 versions of {base_dir_name}"
                raise RuntimeError(error_msg)

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied creating directory {base_dir_name}: {permission_error}"
        )
        print(f"ERROR: {error_msg}")
        raise
    except OSError as os_error:
        error_msg = f"OS error creating directory {base_dir_name}: {os_error}"
        print(f"ERROR: {error_msg}")
        raise
    except Exception as unexpected_error:
        error_msg = (
            f"Unexpected error creating directory {base_dir_name}: {unexpected_error}"
        )
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        raise


def append_to_error_log(error_message: str, context: dict | None = None) -> None:
    """
    Append error message with timestamp and context to error log file.

    Creates error_log.txt in current working directory if it doesn't exist.
    Each error entry includes timestamp, context information, and full error details.

    Args:
        error_message: The error message or traceback to log
        context: Optional dict with contextual information (e.g., expression, step)

    Returns:
        None

    Side Effects:
        Appends to error_log.txt in current working directory
    """
    try:
        error_log_path = Path.cwd() / "error_log.txt"
        timestamp = datetime.now().isoformat()

        # Build log entry
        log_entry_parts = [
            "\n" + "=" * 80,
            f"TIMESTAMP: {timestamp}",
        ]

        # Add context if provided
        if context:
            log_entry_parts.append("CONTEXT:")
            for key, value in context.items():
                log_entry_parts.append(f"  {key}: {value}")

        log_entry_parts.extend(
            [
                "ERROR:",
                error_message,
                "=" * 80 + "\n",
            ]
        )

        log_entry = "\n".join(log_entry_parts)

        # Append to file
        with open(error_log_path, "a", encoding="utf-8") as error_file:
            error_file.write(log_entry)

    except Exception as logging_error:
        # If we can't log, at least print to console
        print(f"CRITICAL: Failed to write to error log: {logging_error}")
        print(f"Original error: {error_message}")
        traceback.print_exc()


def write_toml_file(
    output_dir: Path,
    file_number: int,
    english_text: str,
    symbolic_expr: str,
    rpn_steps: list,
    answer: float | int,
) -> bool:
    """
    Write a single TOML dataset file with expression and evaluation data.

    Creates a file named {file_number}.toml containing:
    - Expression section: English text and symbolic notation
    - Evaluation section: RPN steps and final answer

    Args:
        output_dir: Directory where file should be created
        file_number: File number (used as filename)
        english_text: Natural language expression
        symbolic_expr: Mathematical symbolic notation
        rpn_steps: List of RPN calculation steps
        answer: Final calculated answer

    Returns:
        True if file written successfully, False otherwise

    Side Effects:
        Creates .toml file in output_dir
        Logs errors to error_log.txt on failure
    """
    try:
        file_path = output_dir / f"{file_number}.toml"

        # Build TOML content (quasi-TOML custom format as specified)
        toml_content_lines = [
            "[[expression section]]",
            "",
            "[English]",
            english_text,
            "",
            "[symbolic]",
            symbolic_expr,
            "",
            "",
            "[[evaluation section]]",
            "",
            "[rpn_steps]",
            str(rpn_steps),
            "",
            "[answer]",
            f"|||{answer}|||",
        ]

        toml_content = "\n".join(toml_content_lines)

        # Write to file atomically (write to temp, then rename)
        temp_file_path = file_path.with_suffix(".toml.tmp")

        with open(temp_file_path, "w", encoding="utf-8") as toml_file:
            toml_file.write(toml_content)

        # Atomic rename
        temp_file_path.rename(file_path)

        return True

    except OSError as os_error:
        error_context = {
            "file_number": file_number,
            "output_dir": str(output_dir),
            "operation": "write_toml_file",
        }
        error_message = (
            f"OS error writing TOML file: {os_error}\n{traceback.format_exc()}"
        )
        append_to_error_log(error_message, error_context)
        return False

    except Exception as unexpected_error:
        error_context = {
            "file_number": file_number,
            "output_dir": str(output_dir),
            "operation": "write_toml_file",
        }
        error_message = f"Unexpected error writing TOML file: {unexpected_error}\n{traceback.format_exc()}"
        append_to_error_log(error_message, error_context)
        return False


def process_expression_with_alu(
    english_text: str,
    symbolic_expr: str,
    alu_calculator_function,
) -> tuple[list | None, float | None]:
    """
    Process a symbolic expression through the ALU calculator.

    Attempts to calculate RPN steps and final answer for the given expression.
    Handles all potential errors and returns None values on failure.

    Args:
        english_text: Natural language version (for error context)
        symbolic_expr: Mathematical symbolic expression to evaluate
        alu_calculator_function: The ALU calculator function to use

    Returns:
        Tuple of (rpn_steps_list, answer_value)
        Returns (None, None) if any error occurs during calculation

    Side Effects:
        Logs errors to error_log.txt on failure
    """
    try:
        # Call ALU calculator
        # Expected return: (input_expr, rpn_steps_list, std_err_or_none, solution_float)
        result = alu_calculator_function(symbolic_expr)

        print("result = alu_calculator_function(symbolic_expr)")
        print(result)

        if result is None:
            error_context = {
                "english": english_text,
                "symbolic": symbolic_expr,
                "operation": "alu_calculation",
            }
            error_message = "ALU calculator returned None"
            append_to_error_log(error_message, error_context)
            return None, None

        # Unpack result
        input_expr, rpn_steps, std_err, solution = result

        # Check for standard error
        if std_err is not None:
            error_context = {
                "english": english_text,
                "symbolic": symbolic_expr,
                "operation": "alu_calculation",
            }
            error_message = f"ALU calculation returned standard error: {std_err}"
            append_to_error_log(error_message, error_context)
            return None, None

        # Validate we got valid data
        if rpn_steps is None or solution is None:
            error_context = {
                "english": english_text,
                "symbolic": symbolic_expr,
                "operation": "alu_calculation",
            }
            error_message = f"ALU returned None for steps or solution: steps={rpn_steps}, solution={solution}"
            append_to_error_log(error_message, error_context)
            return None, None

        return rpn_steps, solution

    except Exception as unexpected_error:
        error_context = {
            "english": english_text,
            "symbolic": symbolic_expr,
            "operation": "alu_calculation",
        }
        error_message = f"Exception during ALU calculation: {unexpected_error}\n{traceback.format_exc()}"
        append_to_error_log(error_message, error_context)
        return None, None


def generate_toml_dataset(
    n_required: int,
    base_output_dir_name: str = "toml_production_output_{N}",
    batch_buffer: int = 50,
    random_seed: int | None = None,
    generator_kwargs: dict | None = None,
) -> dict[str, int]:
    """
    Generate N unique mathematical expression datasets as TOML files.

    This function orchestrates the entire pipeline:
    1. Creates versioned output directory
    2. Generates expression batches with buffer
    3. Filters duplicates (exact symbolic string matching)
    4. Processes expressions through ALU calculator
    5. Writes successful results as numbered TOML files
    6. Continues until N valid files are created
    7. Logs all errors and returns statistics

    The function is resilient to errors - if expression generation or
    ALU calculation fails for any individual expression, it logs the
    error and continues until N valid files are produced.

    Args:
        n_required: Number of unique valid TOML files to generate
        base_output_dir_name: Base name for output directory
        batch_buffer: Extra expressions to generate per batch (helps with duplicates)
        random_seed: Optional random seed for reproducibility
        generator_kwargs: Optional kwargs to pass to expression generator
                         (defaults to generator's built-in defaults if None)

    Returns:
        Dictionary with statistics:
            'files_created': Number of TOML files successfully created
            'total_expressions_generated': Total expressions generated (all batches)
            'duplicates_filtered': Number of duplicate expressions filtered out
            'generation_errors': Number of expression generation errors
            'alu_errors': Number of ALU calculation errors
            'write_errors': Number of file write errors

    Raises:
        ImportError: If required modules cannot be imported
        RuntimeError: If directory creation fails

    Side Effects:
        - Creates output directory with TOML files
        - Appends errors to error_log.txt
        - Prints progress information to console
    """
    # Import required modules
    try:
        from alu_rpn_calculator_v12 import rpn_calculator
        from math_expression_generator_v5 import math_expression_generator
    except ImportError as import_error:
        error_message = f"Failed to import required modules: {import_error}\n{traceback.format_exc()}"
        print(f"CRITICAL ERROR: {error_message}")
        append_to_error_log(error_message, {"operation": "module_import"})
        raise

    # Initialize statistics tracking
    stats = {
        "files_created": 0,
        "total_expressions_generated": 0,
        "duplicates_filtered": 0,
        "generation_errors": 0,
        "alu_errors": 0,
        "write_errors": 0,
    }

    # Initialize duplicate tracking set (tracks symbolic expressions)
    seen_symbolic_expressions = set()

    # Create output directory
    try:
        output_dir = create_versioned_directory(base_output_dir_name)
    except Exception as dir_error:
        error_message = f"Failed to create output directory: {dir_error}"
        print(f"CRITICAL ERROR: {error_message}")
        append_to_error_log(error_message, {"operation": "create_directory"})
        raise RuntimeError(error_message) from dir_error

    print(f"\nStarting generation of {n_required} unique expression datasets...")
    print(f"Output directory: {output_dir}")
    print(f"Batch buffer size: {batch_buffer}")

    # Main generation loop
    file_counter = 1  # Start at 1 as specified
    batch_number = 1

    while stats["files_created"] < n_required:
        # Calculate how many more we need
        remaining_needed = n_required - stats["files_created"]
        batch_size = remaining_needed + batch_buffer

        print(f"\n--- Batch {batch_number} ---")
        print(
            f"Generating {batch_size} expressions ({remaining_needed} needed + {batch_buffer} buffer)..."
        )

        # Generate batch of expressions
        try:
            # Prepare generator kwargs
            if generator_kwargs is None:
                gen_kwargs = {}
            else:
                gen_kwargs = generator_kwargs.copy()

            # Add random seed if provided
            if random_seed is not None:
                gen_kwargs["random_seed"] = random_seed

            # Generate expressions
            expression_batch = math_expression_generator(
                n_expressions=batch_size, **gen_kwargs
            )

            stats["total_expressions_generated"] += batch_size
            print(f"Generated {len(expression_batch)} expressions")

        except Exception as generation_error:
            stats["generation_errors"] += 1
            error_context = {
                "operation": "expression_generation",
                "batch_number": batch_number,
                "batch_size": batch_size,
            }
            error_message = f"Error generating expression batch: {generation_error}\n{traceback.format_exc()}"
            append_to_error_log(error_message, error_context)
            print(f"ERROR: Failed to generate batch {batch_number}, retrying...")
            batch_number += 1
            continue

        # Process each expression in the batch
        expressions_processed_in_batch = 0

        for english_text, symbolic_expr in expression_batch:
            # Check if we've already reached our goal
            if stats["files_created"] >= n_required:
                break

            # Check for duplicate (exact symbolic string match)
            if symbolic_expr in seen_symbolic_expressions:
                stats["duplicates_filtered"] += 1
                continue

            # Mark as seen
            seen_symbolic_expressions.add(symbolic_expr)

            # Process through ALU
            rpn_steps, answer = process_expression_with_alu(
                english_text, symbolic_expr, rpn_calculator
            )

            # Check if ALU processing succeeded
            if rpn_steps is None or answer is None:
                stats["alu_errors"] += 1
                continue

            # Write TOML file
            write_success = write_toml_file(
                output_dir=output_dir,
                file_number=file_counter,
                english_text=english_text,
                symbolic_expr=symbolic_expr,
                rpn_steps=rpn_steps,
                answer=answer,
            )

            if write_success:
                stats["files_created"] += 1
                file_counter += 1
                expressions_processed_in_batch += 1

                # Progress update every 10 files
                if stats["files_created"] % 10 == 0:
                    print(
                        f"  Progress: {stats['files_created']}/{n_required} files created"
                    )
            else:
                stats["write_errors"] += 1

        print(
            f"Batch {batch_number} complete: {expressions_processed_in_batch} valid files added"
        )
        print(f"Total progress: {stats['files_created']}/{n_required} files created")

        batch_number += 1

        # Safety check to prevent infinite loop
        if batch_number > 1000:
            error_message = (
                "Exceeded 1000 batches - possible infinite loop or excessive errors"
            )
            print(f"CRITICAL ERROR: {error_message}")
            append_to_error_log(error_message, {"stats": stats})
            raise RuntimeError(error_message)

    # Final summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Files created: {stats['files_created']}")
    print(f"Total expressions generated: {stats['total_expressions_generated']}")
    print(f"Duplicates filtered: {stats['duplicates_filtered']}")
    print(f"Generation errors: {stats['generation_errors']}")
    print(f"ALU calculation errors: {stats['alu_errors']}")
    print(f"File write errors: {stats['write_errors']}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    return stats


def main(make_n_files=10) -> None:
    """
    Main entry point for TOML dataset generation.

    Provides a simple CLI interface for generating datasets.
    Can be customized or replaced with argparse for more options.

    Returns:
        None

    Side Effects:
        Calls generate_toml_dataset and prints results
    """
    try:
        # Example usage - generate 10 unique expression datasets
        n_files = make_n_files

        print("TOML Mathematical Expression Dataset Generator")
        print("=" * 80)

        stats = generate_toml_dataset(
            n_required=n_files,
            base_output_dir_name=f"toml_production_output_{n_files}",
            batch_buffer=50,
            random_seed=None,  # Set to integer for reproducibility
            generator_kwargs=None,  # Uses generator defaults
        )

        print(stats)
        print("\nGeneration completed successfully!")

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user (Ctrl+C)")
        print("Partial results may exist in output directory")

    except Exception as unexpected_error:
        error_message = (
            f"Unexpected error in main: {unexpected_error}\n{traceback.format_exc()}"
        )
        print(f"\nCRITICAL ERROR: {error_message}")
        append_to_error_log(error_message, {"operation": "main"})
        raise


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an integer input.")
    parser.add_argument("value", type=int, help="An integer value to be processed")
    args = parser.parse_args()
    main(args.value)
