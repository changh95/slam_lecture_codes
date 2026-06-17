#!/usr/bin/env bash
# Helper to fetch a FAST-LIVO2 reference bag.
# Pass a sequence name (e.g. Retail_Street) and the script will prompt for
# the current OneDrive direct-download URL. The HKU-MARS team rotates these
# links, so the canonical place to find them is the FAST-LIVO2 GitHub README.

set -euo pipefail

DEFAULT_DST="${HOME}/data/fast_livo2"
SEQ="${1:-Retail_Street}"

mkdir -p "${DEFAULT_DST}"

OUT="${DEFAULT_DST}/${SEQ}.bag"
if [ -f "${OUT}" ]; then
    echo "Already present: ${OUT}"
    exit 0
fi

cat <<EOF
About to fetch FAST-LIVO2 bag: ${SEQ}
Destination: ${OUT}

The HKU-MARS team distributes these via OneDrive (links rotate).
Open the FAST-LIVO2 README and copy the current direct-download URL for
'${SEQ}.bag', then paste it here:
   https://github.com/hku-mars/FAST-LIVO2#3-dataset
EOF

read -r -p "URL: " URL
if [ -z "${URL}" ]; then
    echo "No URL provided; aborting." >&2
    exit 1
fi

# OneDrive 'embed' URLs need ?download=1; if missing, append it.
case "${URL}" in
    *"download=1"*) ;;
    *) URL="${URL}&download=1" ;;
esac

echo "Downloading ${URL} -> ${OUT}"
wget --progress=bar:force -O "${OUT}.partial" "${URL}"
mv "${OUT}.partial" "${OUT}"
echo "OK: ${OUT}"
ls -lh "${OUT}"
