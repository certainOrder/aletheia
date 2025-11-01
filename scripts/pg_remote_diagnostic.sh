#!/bin/bash
# PostgreSQL Remote Connection Diagnostic & Fix Script
# Run this on logos (10.0.0.1) as chapinad

echo "=== PostgreSQL Remote Connection Diagnostics ==="
echo ""

echo "1. Check if PostgreSQL is listening on network interfaces:"
sudo ss -tlnp | grep 5432
echo ""

echo "2. Check postgresql.conf listen_addresses setting:"
sudo grep "^listen_addresses" /etc/postgresql/*/main/postgresql.conf || \
  echo "  (Setting is commented out - using default 'localhost' only)"
echo ""

echo "3. Check pg_hba.conf for remote connection rules:"
sudo grep -v "^#" /etc/postgresql/*/main/pg_hba.conf | grep -v "^$"
echo ""

echo "=== Required Fixes ==="
echo ""
echo "Fix 1: Edit postgresql.conf to listen on all interfaces"
echo "  sudo nano /etc/postgresql/15/main/postgresql.conf"
echo "  Find: #listen_addresses = 'localhost'"
echo "  Change to: listen_addresses = '*'"
echo ""

echo "Fix 2: Add remote connection rule to pg_hba.conf"
echo "  sudo nano /etc/postgresql/15/main/pg_hba.conf"
echo "  Add this line at the end:"
echo "  host    hearthminds     chapinad        10.0.0.0/24             md5"
echo ""

echo "Fix 3: Restart PostgreSQL"
echo "  sudo systemctl restart postgresql"
echo ""

echo "Fix 4: Verify it's listening on 0.0.0.0:5432"
echo "  sudo ss -tlnp | grep 5432"
echo ""
