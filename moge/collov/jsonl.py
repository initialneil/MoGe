import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional


def write(
    data: List[Dict[str, Any]], base_filename: str, max_lines_per_file: int = None
) -> None:
    """
    Write data to jsonl files, splitting into multiple files if max_lines_per_file is specified.

    Args:
        data: List of dictionaries to write as JSON lines
        base_filename: Base filename for output (e.g., "data.jsonl")
        max_lines_per_file: Maximum lines per file. If None, writes to a single file.

    Raises:
        ValueError: If max_lines_per_file is <= 0
    """
    if not data:
        return

    if max_lines_per_file is None:
        max_lines_per_file = len(data)

    if max_lines_per_file <= 0:
        raise ValueError("max_lines_per_file must be greater than 0")

    total_items = len(data)
    if total_items <= max_lines_per_file:
        # Single file case
        with open(base_filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    # Multi-file case
    file_count = (
        total_items + max_lines_per_file - 1
    ) // max_lines_per_file  # Ceiling division
    suffix_digits = len(str(file_count - 1))  # Number of digits needed

    for i in range(file_count):
        start = i * max_lines_per_file
        end = start + max_lines_per_file
        chunk = data[start:end]

        filename = f"{base_filename}.{i:0{suffix_digits}d}" if i > 0 else base_filename
        with open(filename, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _get_numbered_files(base_name):
    split_files = glob.glob(base_filename + ".*")
    return [f for f in split_files if f.split(".")[-1].isdigit()]


def read(base_filename: str) -> List[Dict[str, Any]]:
    """
    Read data from jsonl files, handling both single file and split file cases.

    Args:
        base_filename: Base filename used when writing (e.g., "data.jsonl")

    Returns:
        List of dictionaries containing the loaded data

    Raises:
        FileNotFoundError: If no matching files are found
    """
    # Check for split files pattern first
    split_files = glob.glob(base_filename + ".*")
    numbered_files = [f for f in split_files if f.split(".")[-1].isdigit()]

    if numbered_files:
        # Read split files case
        data = []

        # First read the base file (contains first chunk)
        try:
            with open(base_filename, "r", encoding="utf-8") as f:
                data.extend([json.loads(line) for line in f])
        except FileNotFoundError:
            pass  # Some implementations might not write to base_filename

        # Then read all numbered files in order
        numbered_files.sort(key=lambda x: int(x.split(".")[-1]))
        for file in numbered_files:
            with open(file, "r", encoding="utf-8") as f:
                data.extend([json.loads(line) for line in f])

        return data

    # Single file case
    try:
        with open(base_filename, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No files found matching pattern: {base_filename}"
        ) from e


def read_iter(base_filename: str) -> Iterator[Dict[str, Any]]:
    """
    Lazily read data from jsnnl files, handling both single file and split file cases.

    Args:
        base_filename: Base filename used when writing (e.g., "data.jsonl")

    Yields:
        Dictionaries containing the loaded data

    Raises:
        FileNotFoundError: If no matching files are found
    """
    # Check for split files pattern first
    split_files = glob.glob(base_filename + ".*")
    numbered_files = [f for f in split_files if f.split(".")[-1].isdigit()]
    files_to_read = []

    if numbered_files:
        # First try to add the base file (contains first chunk)
        try:
            files_to_read.append(base_filename)
        except FileNotFoundError:
            pass

        # Add all numbered files in order
        numbered_files.sort(key=lambda x: int(x.split(".")[-1]))
        files_to_read.extend(numbered_files)
    else:
        # Single file case
        files_to_read.append(base_filename)

    # Verify at least one file exists
    if not files_to_read or not any(glob.glob(files_to_read[0])):
        raise FileNotFoundError(f"No files found matching pattern: {base_filename}")

    # Lazy reading of all files
    for file in files_to_read:
        try:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            continue  # Skip missing files (some implementations might skip base file)


# Example usage
if __name__ == "__main__":
    # Sample data
    # data = [{"id": i, "value": f"item_{i}"} for i in range(2500)]
    data = [
        {
            "depth": "data/depth/a.png",
            "mask": "data/mask/a.png",
            "image": "data/image/a.png",
            "prompt": "aaaaaaa",
        },
        {
            "depth": "data/depth/a.png",
            "mask": "data/mask/a.png",
            "image": "data/image/a.png",
            "prompt": "bbbbbbb",
        },
    ]
    # Initialize with 1000 lines/file
    path = "data/test.jsonl"

    # Write data (creates "test.jsonl" and "test.jsonl.001", "test.jsonl.002")
    write(data, path, max_lines_per_file=None)

    print(path)

    read_data = read(path)
    for line in read_data:
        print(line)
