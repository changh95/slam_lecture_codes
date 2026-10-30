#!/usr/bin/env bash
# frontend_rerun in a browser: the image's default command.
#
# Starts a Rerun web viewer in the container (`rerun --serve-web`), prints its URL, runs
# frontend_rerun against it, and keeps the viewer up until Ctrl+C. The viewer's server buffers
# the recording, so a browser tab opened at any time, even after the run, gets all of it. Run the
# container with --network=host, so that a browser on the host reaches the two ports.
#
#   run_frontend_rerun.sh [frontend_rerun arguments]
#
# GRPC_PORT (default 9877, where frontend_rerun sends its data) and WEB_PORT (default 9090, the
# page the browser opens) pick other ports.
set -eu

GRPC_PORT=${GRPC_PORT:-9877}
WEB_PORT=${WEB_PORT:-9090}
FRONTEND_RERUN=$(dirname "$(readlink -f "$0")")/build/frontend_rerun

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

for port in "$GRPC_PORT" "$WEB_PORT"; do
    if listening "$port"; then
        echo "Port $port is already in use (another rerun viewer?)." \
             "Pick others with -e GRPC_PORT=... -e WEB_PORT=..." >&2
        exit 1
    fi
done

# The server keeps up to 25% of the RAM for late viewers; all of MH_01_easy is about 460 MB
log=$(mktemp)
rerun --serve-web --port "$GRPC_PORT" --web-viewer-port "$WEB_PORT" --server-memory-limit 25% \
    > "$log" 2>&1 &
server=$!
run=
trap 'kill $server $run 2> /dev/null || true; rm -f "$log"' EXIT
trap 'exit 0' INT TERM

for _ in $(seq 100); do
    listening "$GRPC_PORT" && listening "$WEB_PORT" && break
    sleep 0.1
done
if ! listening "$GRPC_PORT" || ! listening "$WEB_PORT"; then
    echo "The rerun web viewer did not start:" >&2
    cat "$log" >&2
    exit 1
fi

url="http://localhost:$WEB_PORT/?url=rerun+http://localhost:$GRPC_PORT/proxy"
echo
echo "  Open in a browser:  $url"
echo

# In the background, so that Ctrl+C (trapped above) stops it at once
RERUN_URL="rerun+http://127.0.0.1:$GRPC_PORT/proxy" "$FRONTEND_RERUN" "$@" &
run=$!
wait "$run"
run=

echo
echo "  The viewer stays up at  $url"
echo "  Press Ctrl+C to stop it."
wait "$server"
