#!/usr/bin/env bash
# DEPRECATED -- kept so existing notes/muscle memory keep working.
#
# This used to prompt for a OneDrive direct-download URL, because the HKU-MARS
# team rotated those links. FAST-LIVO2's README now points at the Global-LVBA
# repository, which hosts the dataset on Google Drive with stable file IDs, so
# no manual URL is needed any more.
#
# Use ../download_fast_livo2.py instead. It knows every sequence's file ID and
# size, resumes interrupted downloads, checks the ROS bag magic bytes, and warns
# when a sequence needs a different calibration block than FAST-LIVO2 ships.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HERE}/../download_fast_livo2.py"

echo "This script is deprecated; forwarding to download_fast_livo2.py" >&2
echo >&2

if [ ! -f "${PY}" ]; then
    echo "Could not find ${PY}" >&2
    exit 1
fi

# Old usage was: ./download_fast_livo2_dataset.sh [SequenceName]
exec python3 "${PY}" "$@"
