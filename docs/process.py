import argparse
from pathlib import Path

from convert_to_mkdocstrings import MkDocstringsParser


def process_files(input_dir):
    parser = MkDocstringsParser()
    for file in Path(input_dir).glob("*.md"):
        output_file = file.with_suffix(".mdx").name
        print(f"Processing {file} -> {output_file}")
        parser.process_file(str(file), str(Path(input_dir) / "mintlify" / output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown files")
    parser.add_argument("input_dir", type=str, help="Input directory")
    args = parser.parse_args()

    process_files(args.input_dir)
