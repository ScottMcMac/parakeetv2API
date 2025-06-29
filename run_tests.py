#!/usr/bin/env python
"""Test runner script for parakeetv2API."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the test suite."""
    print("🧪 Running parakeetv2API Test Suite")
    print("=" * 50)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    
    # Test commands to run
    test_commands = [
        # Unit tests
        ["pytest", "tests/unit/", "-v", "--tb=short", "-m", "not slow"],
        
        # Integration tests  
        ["pytest", "tests/integration/", "-v", "--tb=short"],
        
        # E2E tests (excluding slow tests by default)
        ["pytest", "tests/e2e/", "-v", "--tb=short", "-m", "not slow and not e2e"],
    ]
    
    # Optional: Run with coverage
    if "--coverage" in sys.argv:
        test_commands = [
            ["pytest", "tests/", "-v", "--tb=short", "--cov=src", "--cov-report=html", "--cov-report=term", "-m", "not slow"]
        ]
    
    # Run slow tests if requested
    if "--slow" in sys.argv:
        test_commands.append(
            ["pytest", "tests/", "-v", "--tb=short", "-m", "slow"]
        )
    
    # Run E2E tests if requested
    if "--e2e" in sys.argv:
        test_commands.append(
            ["pytest", "tests/e2e/", "-v", "--tb=short", "-m", "e2e"]
        )
    
    total_failures = 0
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n📋 Running test batch {i}/{len(test_commands)}: {' '.join(cmd[2:])}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, cwd=project_root, check=False)
            if result.returncode != 0:
                total_failures += 1
                print(f"❌ Test batch {i} failed with exit code {result.returncode}")
            else:
                print(f"✅ Test batch {i} passed")
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running test batch {i}: {e}")
            total_failures += 1
    
    print("\n" + "=" * 50)
    if total_failures == 0:
        print("🎉 All test batches passed!")
        return 0
    else:
        print(f"💥 {total_failures}/{len(test_commands)} test batches failed")
        return 1


def print_help():
    """Print help information."""
    print("""
Usage: python run_tests.py [options]

Options:
    --coverage    Run tests with coverage reporting
    --slow        Include slow running tests
    --e2e         Include end-to-end tests
    --help        Show this help message

Examples:
    python run_tests.py                    # Run basic test suite
    python run_tests.py --coverage         # Run with coverage
    python run_tests.py --slow --e2e       # Run all tests including slow ones
    
Individual test categories:
    pytest tests/unit/                     # Unit tests only
    pytest tests/integration/              # Integration tests only
    pytest tests/e2e/                      # E2E tests only
    pytest tests/ -m "not slow"            # All tests except slow ones
    pytest tests/ -m "slow"                # Only slow tests
    pytest tests/ -m "e2e"                 # Only E2E tests
""")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)
    
    exit_code = run_tests()
    sys.exit(exit_code)