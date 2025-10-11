#!/usr/bin/env python3
"""
Nuclear option test - complete uninstall/reinstall cycle
"""
import logging
import sys
from app.db.setup.database import DatabaseSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def nuclear_test(host: str):
    logger.info("Starting nuclear reset test...")
    
    # Initialize with nuclear option
    setup = DatabaseSetup(host, nuclear=True)
    
    # Run the full setup
    logger.info("Running full setup with nuclear option...")
    success = setup.setup()
    
    logger.info("=== Test Results ===")
    for i, result in enumerate(setup.results):
        logger.info(f"Step {i}: {'Success' if result else 'Failed'}")
    
    logger.info(f"Overall success: {success}")
    return success

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./nuclear_test.py <host>")
        sys.exit(1)
        
    success = nuclear_test(sys.argv[1])
    sys.exit(0 if success else 1)