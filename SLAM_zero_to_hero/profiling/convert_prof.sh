#!/bin/bash
# Convert .prof binary to JSON for Python analysis
# Usage: ./convert_prof.sh input.prof [output.json]
set -e
INPUT="$1"
if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input.prof> [output.json]"
  exit 1
fi
OUTPUT="${2:-${INPUT%.prof}.json}"
easy_profiler_converter "$INPUT" "$OUTPUT"
echo "Converted: $INPUT -> $OUTPUT"
