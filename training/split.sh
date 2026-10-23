#!/usr/bin/env bash
#
# split.sh - Split a file into chunks of X lines each.
#
# Usage: ./split.sh <input_file> [lines_per_file]
#   input_file       Path to the file to split.
#   lines_per_file   Number of lines per output file (default: 20).
#
# Output files are named 1.json, 2.json, 3.json, ... and written next to
# the input file unless an absolute/relative output dir is provided via
# the SPLIT_OUT_DIR environment variable.
#
# Example:
#   ./split.sh raw/data.jsonl 20
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_file> [lines_per_file]" >&2
    exit 1
fi

INPUT="$1"
LINES_PER_FILE="${2:-20}"
OUT_DIR="${SPLIT_OUT_DIR:-$(dirname "$INPUT")}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: file not found: $INPUT" >&2
    exit 1
fi

if [[ ! -d "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
fi

mkdir -p "$OUT_DIR"

awk -v chunk="$LINES_PER_FILE" -v outdir="$OUT_DIR" '
{
    n = int((NR - 1) / chunk) + 1
    file = outdir "/" n ".json"
    print > file
}
' "$INPUT"

echo "Done. Split $(wc -l < "$INPUT") lines into files of $LINES_PER_FILE lines in $OUT_DIR"