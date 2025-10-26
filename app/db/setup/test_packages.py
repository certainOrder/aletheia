#!/usr/bin/env python3
"""
Test PostgreSQL package management functions.
"""
import sys
import logging
from app.db.setup.database import DatabaseSetup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_package_mgmt(host: str):
    setup = DatabaseSetup(host, nuclear=True)
    
    print("\nTest 1: Stop PostgreSQL")
    success = setup.stop_postgres()
    print(f"Stop PostgreSQL: {'Success' if success else 'Failed'}")
    
    print("\nTest 2: Remove PostgreSQL")
    success = setup.remove_postgres()
    print(f"Remove PostgreSQL: {'Success' if success else 'Failed'}")
    
    print("\nTest 3: Purge Data")
    success = setup.purge_data()
    print(f"Purge Data: {'Success' if success else 'Failed'}")
    
    print("\nTest 4: Install Packages")
    success = setup.install_packages()
    print(f"Install Packages: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_packages.py <host>")
        sys.exit(1)
        
    test_package_mgmt(sys.argv[1])