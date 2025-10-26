#!/usr/bin/env python3
"""
Quick test of RemoteExecutor functionality.
"""
import sys
from app.db.utils.remote import RemoteExecutor

def test_remote(host: str):
    executor = RemoteExecutor(host)
    
    # Test 1: Simple command
    print("\nTest 1: Simple echo command")
    result = executor.run_command("echo 'Hello from remote'")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")
    
    # Test 2: Command with sudo
    print("\nTest 2: System service status (with sudo)")
    result = executor.run_command("systemctl status postgresql", use_sudo=True)
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_remote.py <host>")
        sys.exit(1)
        
    test_remote(sys.argv[1])