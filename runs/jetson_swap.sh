#!/bin/bash
set -e

# =============================================================================
# Add swap space on SD card or eMMC for Jetson Orin
# =============================================================================
# The Jetson Orin has only 7.4GB shared RAM. Adding swap helps prevent OOM kills
# during training. Run this ONCE before training.
#
# Usage:
#   sudo bash runs/jetson_swap.sh           # 4GB swap on eMMC (default)
#   sudo bash runs/jetson_swap.sh /mnt/sd   # 4GB swap on mounted SD card
#   sudo bash runs/jetson_swap.sh /mnt/sd 8 # 8GB swap on SD card
# =============================================================================

SWAP_DIR="${1:-/var}"          # where to put the swapfile
SWAP_GB="${2:-4}"              # swap size in GB
SWAP_FILE="$SWAP_DIR/nanochat_swap"

echo "=== Current swap ==="
swapon --show
free -h
echo ""

if [ -f "$SWAP_FILE" ]; then
    echo "Swap file already exists at $SWAP_FILE"
    echo "To remove: sudo swapoff $SWAP_FILE && sudo rm $SWAP_FILE"
    exit 0
fi

echo "Creating ${SWAP_GB}GB swap at $SWAP_FILE ..."
dd if=/dev/zero of="$SWAP_FILE" bs=1G count="$SWAP_GB" status=progress
chmod 600 "$SWAP_FILE"
mkswap "$SWAP_FILE"
swapon "$SWAP_FILE"

echo ""
echo "=== Updated swap ==="
swapon --show
free -h

echo ""
echo "Swap added successfully!"
echo "To make persistent across reboots, add to /etc/fstab:"
echo "  $SWAP_FILE none swap sw 0 0"
echo ""
echo "To remove later:"
echo "  sudo swapoff $SWAP_FILE && sudo rm $SWAP_FILE"
