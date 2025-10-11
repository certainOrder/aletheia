"""
Command line interface for database setup.
"""
import argparse
import logging
import sys
from app.db.setup import DatabaseSetup

def main():
    parser = argparse.ArgumentParser(description='PostgreSQL Database Setup')
    parser.add_argument('host', help='Target host for database installation')
    parser.add_argument('--nuclear-reset', action='store_true',
                      help='Remove existing PostgreSQL installation before setup')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run setup
    setup = DatabaseSetup(args.host, nuclear=args.nuclear_reset)
    success = setup.setup()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()