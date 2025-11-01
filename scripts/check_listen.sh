#!/bin/bash
# Check and fix PostgreSQL listen_addresses
# Run this ON LOGOS as: bash check_listen.sh

echo "=== Checking PostgreSQL Configuration ==="
echo ""

echo "1. Current listen_addresses setting:"
sudo grep -n "listen_addresses" /etc/postgresql/15/main/postgresql.conf | grep -v "^#"
echo ""

echo "2. Current listening ports:"
sudo ss -tlnp | grep 5432
echo ""

echo "=== Analysis ==="
if sudo ss -tlnp | grep -q "0.0.0.0:5432"; then
    echo "✅ PostgreSQL IS listening on all interfaces (0.0.0.0:5432)"
elif sudo ss -tlnp | grep -q "127.0.0.1:5432"; then
    echo "❌ PostgreSQL is ONLY listening on localhost (127.0.0.1:5432)"
    echo ""
    echo "FIX REQUIRED:"
    echo "1. Edit: sudo nano /etc/postgresql/15/main/postgresql.conf"
    echo "2. Find line with: #listen_addresses = 'localhost'"
    echo "3. Change to: listen_addresses = '*'"
    echo "4. Make sure there's NO # at the start!"
    echo "5. Save and restart: sudo systemctl restart postgresql"
else
    echo "⚠️  PostgreSQL may not be running"
fi
echo ""

echo "=== Quick Fix (if needed) ==="
echo "Run this to automatically fix listen_addresses:"
echo "  sudo sed -i \"s/#listen_addresses = 'localhost'/listen_addresses = '*'/g\" /etc/postgresql/15/main/postgresql.conf"
echo "  sudo systemctl restart postgresql"
echo "  sudo ss -tlnp | grep 5432"
