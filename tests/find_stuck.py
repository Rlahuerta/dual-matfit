"""
Script to identify stuck/hanging tests by running them individually with timeouts.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_with_timeout(test_path, timeout=30):
    """Run a single test file with timeout and return results."""
    try:
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-n", "4", "-v", "--tb=short"]
        
        print(f"\n{'='*60}")
        print(f"Testing: {test_path.name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Timeout: {timeout}s")
        print('='*60)
        
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            timeout=timeout, 
            capture_output=True, 
            text=True,
            cwd=Path.cwd()
        )
        
        elapsed = time.time() - start_time
        
        status = "PASSED" if result.returncode == 0 else "FAILED"
        print(f"Status: {status} (elapsed: {elapsed:.2f}s)")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-500:])  # Last 500 chars
            
        if result.stderr:
            print("STDERR:")
            print(result.stderr[-500:])  # Last 500 chars
            
        return {
            'file': test_path.name,
            'status': status,
            'elapsed': elapsed,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'hung': False
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {test_path.name} hung after {timeout}s")
        return {
            'file': test_path.name,
            'status': 'TIMEOUT',
            'elapsed': timeout,
            'returncode': None,
            'stdout': '',
            'stderr': '',
            'hung': True
        }
    except Exception as e:
        print(f"❌ ERROR: {test_path.name} - {str(e)}")
        return {
            'file': test_path.name,
            'status': 'ERROR',
            'elapsed': 0,
            'returncode': None,
            'stdout': '',
            'stderr': str(e),
            'hung': False
        }

def main():
    """Main function to test all test files."""
    test_dir = Path(__file__).parent

    timeout = 50 * 60 # seconds
    
    if not test_dir.exists():
        print(f"❌ Test directory {test_dir} not found!")
        return
    
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print(f"❌ No test files found in {test_dir}")
        return
    
    print(f"Found {len(test_files)} test files")
    
    results = []
    hung_tests = []
    failed_tests = []
    passed_tests = []
    
    for test_file in sorted(test_files):
        result = run_test_with_timeout(test_file, timeout=timeout)
        results.append(result)
        
        if result['hung']:
            hung_tests.append(result)
        elif result['status'] == 'FAILED' or result['status'] == 'ERROR':
            failed_tests.append(result)
        elif result['status'] == 'PASSED':
            passed_tests.append(result)
    
    # Summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print('='*80)
    
    print(f"\n✅ PASSED TESTS ({len(passed_tests)}):")
    for test in passed_tests:
        print(f"  - {test['file']} ({test['elapsed']:.2f}s)")
    
    print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
    for test in failed_tests:
        print(f"  - {test['file']} (exit code: {test['returncode']})")
    
    print(f"\n🔄 HUNG/TIMEOUT TESTS ({len(hung_tests)}):")
    for test in hung_tests:
        print(f"  - {test['file']} (timeout after {timeout}s)")
    
    if hung_tests:
        print(f"\n🚨 STUCK TESTS IDENTIFIED:")
        print("The following test files are hanging and need investigation:")
        for test in hung_tests:
            print(f"  - tests/{test['file']}")
        
        print(f"\nTo debug individual stuck tests, try:")
        for test in hung_tests:
            print(f"  timeout 5s python -m pytest tests/{test['file']}::TestClassName::test_method_name -v -s")
    
    if not hung_tests:
        print("\n✅ No hanging tests detected!")
    
    return hung_tests


if __name__ == "__main__":
    hung_tests = main()
    
    # Exit with error code if hung tests found
    if hung_tests:
        sys.exit(1)
    else:
        sys.exit(0)
