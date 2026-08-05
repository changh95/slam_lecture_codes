#!/bin/bash
# setup_fastdds.sh — One-time kernel tuning for Fast DDS large-message transport.
# Run once (requires sudo). Settings persist across reboots via sysctl.d.
set -e

SYSCTL_CONF="/etc/sysctl.d/10-fastdds.conf"

echo "### Fast DDS kernel buffer tuning ###"
echo ""
echo "Current limits:"
sysctl net.core.rmem_max net.core.rmem_default \
       net.core.wmem_max net.core.wmem_default
echo ""

# ── Write persistent sysctl config ──
echo "[1/2] Writing $SYSCTL_CONF ..."
cat <<'EOF' | sudo tee "$SYSCTL_CONF" > /dev/null
# Fast DDS / ROS 2 — enlarged UDP socket buffers for pointcloud & image topics.
# max = ceiling any socket can request;  default = what sockets get if they don't ask.
# 2× Livox LiDAR + Oak-D Pro ≈ 80+ MB/s aggregate; need room for burst queueing.
net.core.rmem_max=67108864
net.core.rmem_default=8388608
net.core.wmem_max=67108864
net.core.wmem_default=4194304
EOF

# ── Apply immediately ──
echo "[2/2] Applying ..."
sudo sysctl --system > /dev/null 2>&1

echo ""
echo "New limits:"
sysctl net.core.rmem_max net.core.rmem_default \
       net.core.wmem_max net.core.wmem_default
echo ""
echo "### Done. These settings persist across reboots. ###"
