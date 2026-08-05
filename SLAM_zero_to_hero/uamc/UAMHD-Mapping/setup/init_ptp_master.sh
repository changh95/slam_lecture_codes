#!/bin/bash

# --- Configuration ---
IFACE="eno1"                  # NIC connected to both Livox Avia and MID360
PTPD_CONF="/etc/ptpd-master.conf"
PTPD_SERVICE="/etc/systemd/system/ptpd-master.service"

# --- Preflight ---
if ! command -v ptpd &>/dev/null; then
    echo "ptpd not found, installing..."
    sudo apt-get update && sudo apt-get install -y ptpd
fi

if ! ip link show "$IFACE" &>/dev/null; then
    echo "Error: Interface '$IFACE' not found. Available interfaces:"
    ip -o link show | awk -F': ' '{print "  " $2}'
    exit 1
fi

# Stop ptp4l if it was previously installed (doesn't work on this NIC)
if systemctl is-active --quiet ptp4l-master.service 2>/dev/null; then
    echo "Stopping old ptp4l-master service..."
    sudo systemctl disable --now ptp4l-master.service
fi

echo "### Setting up PTP Master on ${IFACE} for Livox Avia + MID360 ###"

# --- 1. Write ptpd master config ---
# ptpd uses gettimeofday() for TX timing instead of kernel SO_TIMESTAMPING,
# so it works on NICs where ptp4l fails with "timed out while polling for
# tx timestamp" (e.g. Intel I219 / eno1).
echo -e "\n[1] Writing ptpd master configuration..."
printf '%s\n' \
    '[ptpengine]'                        \
    "interface = $IFACE"                 \
    'preset = masteronly'                \
    'ip_mode = multicast'               \
    | sudo tee "$PTPD_CONF" > /dev/null
echo "  Config written to $PTPD_CONF"

# --- 2. Install systemd service ---
echo -e "\n[2] Installing ptpd service..."
printf '%s\n' \
    '[Unit]'                                               \
    'Description=PTPd Master (Livox LiDAR Time Sync)'     \
    'After=network-online.target'                          \
    'Wants=network-online.target'                          \
    ''                                                     \
    '[Service]'                                            \
    'Type=forking'                                         \
    "ExecStart=/usr/sbin/ptpd -c $PTPD_CONF"             \
    'Restart=always'                                       \
    ''                                                     \
    '[Install]'                                            \
    'WantedBy=multi-user.target'                           \
    | sudo tee "$PTPD_SERVICE" > /dev/null
echo "  Service installed at $PTPD_SERVICE"

# --- 3. Enable and start ---
echo -e "\n[3] Enabling and starting ptpd..."
sudo systemctl daemon-reload
sudo systemctl enable ptpd-master.service
sudo systemctl restart ptpd-master.service

echo -e "\n[4] Status:"
sudo systemctl status ptpd-master.service --no-pager

echo -e "\n### Done. Both LiDARs will auto-detect the PTP master on $IFACE and lock to it. ###"
echo "  Monitor:  sudo journalctl -fu ptpd-master"
echo "  Log:      sudo tail -f /var/log/ptpd-master.log"
echo "  Stop:     sudo systemctl stop ptpd-master"