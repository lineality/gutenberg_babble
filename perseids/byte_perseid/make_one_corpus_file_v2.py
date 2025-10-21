import os
"""
Make a combined file
from all files
in flat directory
that have the specified
EXTENSION
"""

# EXTENSION = ".txt"
EXTENSION = ".toml"

def make_combined_file(output_filename=f"xyz{EXTENSION}"):
    # Get all .txt files in the current working directory
    # txt_files = [f for f in os.listdir() if f.endswith('.txt') and f != output_filename]
    txt_files = [f for f in os.listdir() if f.endswith(EXTENSION) and f != output_filename]

    # Sort the files alphabetically for consistent order
    # txt_files.sort()

    """
    shorter training files may be simpler tests...
    sequence of learning?
    """
    # Sort the files by size (ascending order)
    txt_files.sort(key=lambda f: os.path.getsize(f))

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(f"=== Content from {txt_file} ===\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"All {EXTENSION} files combined into {output_filename}")

if __name__ == "__main__":
    make_combined_file()

