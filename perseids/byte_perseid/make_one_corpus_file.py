import os

def combine_txt_files(output_filename="xyz.txt"):
    # Get all .txt files in the current working directory
    txt_files = [f for f in os.listdir() if f.endswith('.txt') and f != output_filename]

    # Sort the files alphabetically for consistent order
    txt_files.sort()

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(f"=== Content from {txt_file} ===\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"All .txt files combined into {output_filename}")

if __name__ == "__main__":
    combine_txt_files()

