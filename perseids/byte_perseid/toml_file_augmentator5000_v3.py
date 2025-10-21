# toml_file_augmentator5000_vN.py
"""
File augmentation utility to introduce whitespace and newline variation

This module provides functionality to:
- with reproducible random seed
- create augmented versions of text files
- randomly add leading spaces
- randomly modify newline patterns
- preservings text content of each line
"""

import random
import traceback
from pathlib import Path
from collections.abc import Sequence


def augment_dir_files(
    input_directory: str,
    n_versions_of_each_file: int = 1,
    random_seed: int = 42,
    pct_chance_of_leading_space: int = 10,
    max_extra_leading_spaces_added: int = 2,
    pct_chance_of_extra_newline: int = 10,
    max_extra_newlines_added: int = 2,
    pct_chance_reduce_serial_newlines: int = 50,
    min_serial_newlines: int = 1,
    name_prefix: str = "_aug_",
    files_suffix_to_include_list: list[str] | None = None,
) -> None:
    """
    Augment all text files in a directory with controlled whitespace variations.

    This function processes each file in the specified directory and creates N
    augmented versions with random whitespace modifications. The augmentations
    include adding leading spaces to lines and modifying newline patterns while
    preserving the actual text content.

    Args:
        input_directory: Path to the directory containing files to augment.
        n_versions_of_each_file: Number of augmented versions to create per file.
        random_seed: Seed for random number generator to ensure reproducibility.
        pct_chance_of_leading_space: Percentage chance (0-100) to add leading spaces to a line.
        max_extra_leading_spaces_added: Maximum number of spaces that can be added to line start.
        pct_chance_of_extra_newline: Percentage chance (0-100) to add extra newlines.
        max_extra_newlines_added: Maximum number of newlines that can be added.
        pct_chance_reduce_serial_newlines: Percentage chance (0-100) to reduce consecutive newlines.
        min_serial_newlines: Minimum number of consecutive newlines to maintain when reducing.
        name_prefix: Prefix template for augmented filenames (version number prepended).
        files_suffix_to_include_list: List of file extensions to process (e.g., ['txt', 'toml']).
                                      If None, defaults to ['txt', 'toml'].

    Returns:
        None. Creates augmented files in the same directory as source files.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        PermissionError: If there are permission issues reading or writing files.
        Exception: Any other errors during processing are caught and logged.

    Example:
        >>> augment_dir_files(
        ...     input_directory="/path/to/files",
        ...     n_versions_of_each_file=3,
        ...     random_seed=42,
        ...     pct_chance_of_leading_space=15,
        ...     max_extra_leading_spaces_added=3
        ... )
    """
    try:
        # Initialize random generator ONCE for entire augmentation process
        random_generator = random.Random(random_seed)

        # Set default file suffixes if none provided
        if files_suffix_to_include_list is None:
            files_suffix_to_include_list = ["txt", "toml"]

        # Validate that input directory exists
        input_path = Path(input_directory)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_directory}"
            )

        if not input_path.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: {input_directory}"
            )

        # Get list of files to process (non-recursive, flat directory only)
        files_to_process = _get_files_to_process(
            directory_path=input_path, file_suffixes=files_suffix_to_include_list
        )

        print(f"Found {len(files_to_process)} files to process in {input_directory}")

        # Process each file
        for source_file_path in files_to_process:
            try:
                _process_single_file(
                    random_generator=random_generator,
                    source_file_path=source_file_path,
                    n_versions=n_versions_of_each_file,
                    pct_chance_of_leading_space=pct_chance_of_leading_space,
                    max_extra_leading_spaces_added=max_extra_leading_spaces_added,
                    pct_chance_of_extra_newline=pct_chance_of_extra_newline,
                    max_extra_newlines_added=max_extra_newlines_added,
                    pct_chance_reduce_serial_newlines=pct_chance_reduce_serial_newlines,
                    min_serial_newlines=min_serial_newlines,
                    name_prefix=name_prefix,
                )
                print(f"Successfully processed: {source_file_path.name}")
            except Exception as exception_during_file_processing:
                print(
                    f"Error processing file {source_file_path.name}: {exception_during_file_processing}"
                )
                traceback.print_exc()
                # Continue processing other files even if one fails
                continue

        print(f"Augmentation complete. Processed {len(files_to_process)} files.")

    except Exception as exception_during_augmentation:
        print(f"Error during directory augmentation: {exception_during_augmentation}")
        traceback.print_exc()
        raise


def _get_files_to_process(
    directory_path: Path, file_suffixes: Sequence[str]
) -> list[Path]:
    """
    Get list of files in directory that match the specified suffixes.

    This function scans a directory (non-recursively) and returns paths to all
    files with extensions matching the provided suffixes.

    Args:
        directory_path: Path object representing the directory to scan.
        file_suffixes: Sequence of file extensions to include (without dots).

    Returns:
        List of Path objects for files matching the specified suffixes.

    Raises:
        PermissionError: If directory cannot be read due to permissions.
        Exception: Any other errors during directory scanning.
    """
    try:
        files_to_process = []

        # Normalize suffixes to lowercase for case-insensitive matching
        normalized_suffixes = [suffix.lower().lstrip(".") for suffix in file_suffixes]

        # Iterate through directory contents (non-recursive)
        for item_path in directory_path.iterdir():
            # Only process files, not directories
            if item_path.is_file():
                # Get file extension without the dot
                file_extension = item_path.suffix.lstrip(".").lower()

                # Check if file extension matches our criteria
                if file_extension in normalized_suffixes:
                    files_to_process.append(item_path)

        return files_to_process

    except Exception as exception_during_file_listing:
        print(f"Error listing files in directory: {exception_during_file_listing}")
        traceback.print_exc()
        raise


# def _process_single_file(
#     source_file_path: Path,
#     n_versions: int,
#     random_seed: int,
#     pct_chance_of_leading_space: int,
#     max_extra_leading_spaces_added: int,
#     pct_chance_of_extra_newline: int,
#     max_extra_newlines_added: int,
#     pct_chance_reduce_serial_newlines: int,
#     min_serial_newlines: int,
#     name_prefix: str,
# ) -> None:
#     """
#     Process a single file to create N augmented versions.

#     This function reads a source file and creates multiple augmented versions
#     with random whitespace modifications. Each version is written to a new file
#     with a numbered prefix.

#     Args:
#         source_file_path: Path to the source file to augment.
#         n_versions: Number of augmented versions to create.
#         random_seed: Base seed for random number generator.
#         pct_chance_of_leading_space: Percentage chance to add leading spaces.
#         max_extra_leading_spaces_added: Maximum spaces to add at line start.
#         pct_chance_of_extra_newline: Percentage chance to add newlines.
#         max_extra_newlines_added: Maximum newlines to add.
#         pct_chance_reduce_serial_newlines: Percentage chance to reduce serial newlines.
#         min_serial_newlines: Minimum serial newlines to maintain.
#         name_prefix: Prefix template for output filenames.

#     Returns:
#         None. Creates augmented files in the same directory.

#     Raises:
#         IOError: If file cannot be read or written.
#         Exception: Any other errors during file processing.
#     """
#     try:
#         # Read the entire source file content
#         with open(source_file_path, "r", encoding="utf-8") as source_file:
#             original_content = source_file.read()

#         # Create N augmented versions
#         for version_number in range(1, n_versions + 1):
#             try:
#                 # Create augmented content for this version
#                 augmented_content = _augment_file_content(
#                     original_content=original_content,
#                     random_seed=random_seed,
#                     version_number=version_number,
#                     pct_chance_of_leading_space=pct_chance_of_leading_space,
#                     max_extra_leading_spaces_added=max_extra_leading_spaces_added,
#                     pct_chance_of_extra_newline=pct_chance_of_extra_newline,
#                     max_extra_newlines_added=max_extra_newlines_added,
#                     pct_chance_reduce_serial_newlines=pct_chance_reduce_serial_newlines,
#                     min_serial_newlines=min_serial_newlines,
#                 )

#                 # Generate output filename with version prefix
#                 output_filename = (
#                     f"{version_number}{name_prefix}{source_file_path.name}"
#                 )
#                 output_file_path = source_file_path.parent / output_filename

#                 # Write augmented content to new file
#                 with open(output_file_path, "w", encoding="utf-8") as output_file:
#                     _ = output_file.write(augmented_content)

#             except Exception as exception_during_version_creation:
#                 print(
#                     f"Error creating version {version_number} of {source_file_path.name}: {exception_during_version_creation}"
#                 )
#                 traceback.print_exc()
#                 # Continue with next version even if this one fails
#                 continue

#     except Exception as exception_during_file_processing:
#         print(
#             f"Error processing file {source_file_path}: {exception_during_file_processing}"
#         )
#         traceback.print_exc()
#         raise


def _process_single_file(
    random_generator: random.Random,
    source_file_path: Path,
    n_versions: int,
    pct_chance_of_leading_space: int,
    max_extra_leading_spaces_added: int,
    pct_chance_of_extra_newline: int,
    max_extra_newlines_added: int,
    pct_chance_reduce_serial_newlines: int,
    min_serial_newlines: int,
    name_prefix: str,
) -> None:
    """
    Process a single file to create N augmented versions.

    This function reads a source file and creates multiple augmented versions
    with random whitespace modifications. Each version is written to a new file
    with a numbered prefix. All versions share the same random seed, drawing
    from the same random sequence sequentially to ensure different outputs.

    Args:
        source_file_path: Path to the source file to augment.
        n_versions: Number of augmented versions to create.
        random_seed: Seed for random number generator (initialized once for all versions).
        pct_chance_of_leading_space: Percentage chance to add leading spaces.
        max_extra_leading_spaces_added: Maximum spaces to add at line start.
        pct_chance_of_extra_newline: Percentage chance to add newlines.
        max_extra_newlines_added: Maximum newlines to add.
        pct_chance_reduce_serial_newlines: Percentage chance to reduce serial newlines.
        min_serial_newlines: Minimum serial newlines to maintain.
        name_prefix: Prefix template for output filenames.

    Returns:
        None. Creates augmented files in the same directory.

    Raises:
        IOError: If file cannot be read or written.
        Exception: Any other errors during file processing.
    """
    try:
        # Read the entire source file content
        with open(source_file_path, "r", encoding="utf-8") as source_file:
            original_content = source_file.read()

        # Create N augmented versions
        for version_number in range(1, n_versions + 1):
            try:
                # Create augmented content for this version
                # Pass the shared random_generator instance
                augmented_content = _augment_file_content(
                    original_content=original_content,
                    random_generator=random_generator,
                    pct_chance_of_leading_space=pct_chance_of_leading_space,
                    max_extra_leading_spaces_added=max_extra_leading_spaces_added,
                    pct_chance_of_extra_newline=pct_chance_of_extra_newline,
                    max_extra_newlines_added=max_extra_newlines_added,
                    pct_chance_reduce_serial_newlines=pct_chance_reduce_serial_newlines,
                    min_serial_newlines=min_serial_newlines,
                )

                # Generate output filename with version prefix
                output_filename = (
                    f"{version_number}{name_prefix}{source_file_path.name}"
                )
                output_file_path = source_file_path.parent / output_filename

                # Write augmented content to new file
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(augmented_content)

            except Exception as exception_during_version_creation:
                print(
                    f"Error creating version {version_number} of {source_file_path.name}: {exception_during_version_creation}"
                )
                traceback.print_exc()
                # Continue with next version even if this one fails
                continue

    except Exception as exception_during_file_processing:
        print(
            f"Error processing file {source_file_path}: {exception_during_file_processing}"
        )
        traceback.print_exc()
        raise


def _augment_file_content(
    original_content: str,
    random_generator: random.Random,
    pct_chance_of_leading_space: int,
    max_extra_leading_spaces_added: int,
    pct_chance_of_extra_newline: int,
    max_extra_newlines_added: int,
    pct_chance_reduce_serial_newlines: int,
    min_serial_newlines: int,
) -> str:
    """
    Create an augmented version of file content with whitespace variations.

    This function processes the original content character by character, making
    random decisions about adding leading spaces and modifying newline patterns
    according to the specified probabilities and constraints.

    Processing order per line (as specified):
    - Line 1: Add leading spaces (if chance triggers), proceed to newline
    - Line 2+: Check serial newlines (reduce if present, else add if chance triggers),
              then add leading spaces (if chance triggers), proceed to newline
    - Stop at last newline

    Args:
        original_content: The original file content as a string.
        random_generator: Initialized random number generator (shared across versions).
        pct_chance_of_leading_space: Percentage chance (0-100) to add leading spaces.
        max_extra_leading_spaces_added: Maximum spaces to add at line start.
        pct_chance_of_extra_newline: Percentage chance (0-100) to add newlines.
        max_extra_newlines_added: Maximum newlines to add.
        pct_chance_reduce_serial_newlines: Percentage chance (0-100) to reduce serial newlines.
        min_serial_newlines: Minimum serial newlines to maintain when reducing.

    Returns:
        Augmented content as a string.

    Raises:
        Exception: Any errors during content augmentation.
    """
    try:
        # Remove these lines - no longer needed:
        # version_specific_seed = random_seed + version_number
        # random_generator = random.Random(version_specific_seed)

        # Build the augmented content character by character
        augmented_content = []

        # Track position in original content
        current_position = 0
        content_length = len(original_content)

        # Track whether we're on line 1 (special case - no newline processing before it)
        is_first_line = True

        # Process until we've consumed all content
        while current_position < content_length:
            # If this is NOT the first line, we're at the start of a new line
            # and need to check for serial newlines or add newlines
            if not is_first_line:
                # Look back to see if we just processed newline(s)
                # Count consecutive newlines we've already added to augmented content
                serial_newlines_count = _count_trailing_newlines(augmented_content)

                # Serial newlines exist if count >= 2
                if serial_newlines_count >= 2:
                    # Chance to reduce serial newlines
                    if _should_trigger_event(
                        random_generator, pct_chance_reduce_serial_newlines
                    ):
                        # Reduce the serial newlines
                        augmented_content = _reduce_serial_newlines(
                            content_list=augmented_content,
                            current_serial_count=serial_newlines_count,
                            min_serial_newlines=min_serial_newlines,
                            random_generator=random_generator,
                        )
                else:
                    # No serial newlines, so chance to add extra newlines
                    if _should_trigger_event(
                        random_generator, pct_chance_of_extra_newline
                    ):
                        # Add random number of newlines (1 to max)
                        num_newlines_to_add = random_generator.randint(
                            1, max_extra_newlines_added
                        )
                        augmented_content.extend(["\n"] * num_newlines_to_add)

            # Now handle leading spaces for this line
            if _should_trigger_event(random_generator, pct_chance_of_leading_space):
                # Add random number of leading spaces (1 to max)
                num_spaces_to_add = random_generator.randint(
                    1, max_extra_leading_spaces_added
                )
                augmented_content.extend([" "] * num_spaces_to_add)

            # Now copy characters from original content until we hit a newline
            # (or reach end of file)
            while current_position < content_length:
                current_char = original_content[current_position]
                augmented_content.append(current_char)
                current_position += 1

                # If we hit a newline, this line is complete
                if current_char == "\n":
                    # Check if this is the last newline in the file
                    if _is_last_newline(original_content, current_position):
                        # Stop processing as instructed
                        # Copy any remaining content after last newline as-is
                        if current_position < content_length:
                            remaining_content = original_content[current_position:]
                            augmented_content.append(remaining_content)
                        # Exit both loops
                        return "".join(augmented_content)

                    # Mark that we're no longer on the first line
                    is_first_line = False
                    # Break inner loop to start processing next line
                    break

        # Return the augmented content as a single string
        return "".join(augmented_content)

    except Exception as exception_during_augmentation:
        print(f"Error during content augmentation: {exception_during_augmentation}")
        traceback.print_exc()
        raise


def _count_trailing_newlines(content_list: list[str]) -> int:
    """
    Count the number of consecutive newline characters at the end of content list.

    This function examines the content list from the end backwards to count
    how many consecutive newline characters exist.

    Args:
        content_list: List of characters representing the content so far.

    Returns:
        Count of consecutive newlines at the end of the list.
    """
    try:
        newline_count = 0

        # Traverse backwards from end of list
        for char in reversed(content_list):
            if char == "\n":
                newline_count += 1
            else:
                # Stop at first non-newline character
                break

        return newline_count

    except Exception as exception_during_counting:
        print(f"Error counting trailing newlines: {exception_during_counting}")
        traceback.print_exc()
        return 0


def _reduce_serial_newlines(
    content_list: list[str],
    current_serial_count: int,
    min_serial_newlines: int,
    random_generator: random.Random,
) -> list[str]:
    """
    Reduce the number of consecutive newlines at the end of content list.

    This function removes newline characters from the end of the content list
    while respecting the minimum serial newlines constraint.

    Args:
        content_list: List of characters representing the content so far.
        current_serial_count: Current number of consecutive newlines.
        min_serial_newlines: Minimum number of newlines that must remain.
        random_generator: Random number generator for consistent randomization.

    Returns:
        Modified content list with reduced newlines.
    """
    try:
        # Calculate how many newlines we could potentially remove
        max_removable = current_serial_count - min_serial_newlines

        # If we can't remove any, return unchanged
        if max_removable <= 0:
            return content_list

        # Choose random number of newlines to remove (1 to max_removable)
        num_to_remove = random_generator.randint(1, max_removable)

        # Remove that many newlines from the end
        # Create a new list without the trailing newlines we want to remove
        removal_count = 0
        new_content_list = []

        # Process list in reverse to efficiently remove trailing newlines
        for char in reversed(content_list):
            if char == "\n" and removal_count < num_to_remove:
                # Skip this newline (remove it)
                removal_count += 1
            else:
                # Keep this character
                new_content_list.append(char)

        # Reverse back to original order
        new_content_list.reverse()

        return new_content_list

    except Exception as exception_during_reduction:
        print(f"Error reducing serial newlines: {exception_during_reduction}")
        traceback.print_exc()
        return content_list


def _is_last_newline(content: str, position_after_newline: int) -> bool:
    """
    Check if the newline we just processed is the last newline in the file.

    This function looks ahead from the current position to see if there are
    any more newline characters in the remaining content.

    Args:
        content: The complete original content string.
        position_after_newline: Position in content immediately after the newline we just processed.

    Returns:
        True if no more newlines exist in the remaining content, False otherwise.
    """
    try:
        # Check if there are any more newlines in the remaining content
        remaining_content = content[position_after_newline:]
        has_more_newlines = "\n" in remaining_content

        # If no more newlines exist, this was the last one
        return not has_more_newlines

    except Exception as exception_during_check:
        print(f"Error checking for last newline: {exception_during_check}")
        traceback.print_exc()
        # Conservative approach: assume it's not the last newline if error occurs
        return False


def _should_trigger_event(
    random_generator: random.Random, percentage_chance: int
) -> bool:
    """
    Determine if a random event should trigger based on percentage chance.

    This function uses the random generator to decide if an event with the
    given percentage chance should occur.

    Args:
        random_generator: Random number generator for consistent randomization.
        percentage_chance: Chance of event occurring (0-100).

    Returns:
        True if event should trigger, False otherwise.
    """
    try:
        # Generate random number between 0 and 99
        random_value = random_generator.randint(0, 99)

        # Event triggers if random value is less than the percentage
        # e.g., 10% chance means values 0-9 trigger (10 out of 100 values)
        return random_value < percentage_chance

    except Exception as exception_during_check:
        print(f"Error checking if event should trigger: {exception_during_check}")
        traceback.print_exc()
        # Conservative approach: don't trigger event if error occurs
        return False


import argparse

if __name__ == "__main__":
    # user specifies N: how many to make
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("value", type=str, help="An string value to be processed")
    args = parser.parse_args()
    this_inputdirectory: str = str(args.value)
    """
    Example usage of the augmentation utility.

    This demonstrates how to call the augment_dir_files function with
    various parameters to create augmented versions of text files.
    """
    try:
        # Example: Augment all .txt and .toml files in a directory
        augment_dir_files(
            input_directory=this_inputdirectory,
            n_versions_of_each_file=2,
            random_seed=42,
            pct_chance_of_leading_space=10,
            max_extra_leading_spaces_added=3,
            pct_chance_of_extra_newline=10,
            max_extra_newlines_added=3,
            pct_chance_reduce_serial_newlines=10,
            min_serial_newlines=1,
            name_prefix="_aug_",
            files_suffix_to_include_list=["txt", "toml"],
        )
    except Exception as main_exception:
        print(f"Error in main execution: {main_exception}")
        traceback.print_exc()
