import time
import sys
import os
import bisect
from collections import defaultdict

# Optional psutil import for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def sieve_of_eratosthenes(limit):
    """Generate all primes up to limit using optimized sieve"""
    is_prime = bytearray([True] * (limit + 1))
    is_prime[0] = is_prime[1] = False
    
    sqrt_limit = int(limit**0.5) + 1
    for i in range(2, sqrt_limit):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]

def omega_optimized(n):
    """Fast omega (distinct prime divisors) using optimized trial division"""
    if n <= 1:
        return 0
    
    count = 0
    
    # Handle factor 2 efficiently with bit operations
    if n & 1 == 0:
        count += 1
        while n & 1 == 0:
            n >>= 1
    
    # Check odd divisors only
    f = 3
    while f * f <= n:
        if n % f == 0:
            count += 1
            while n % f == 0:
                n //= f
        f += 2
    
    # If n is still > 1, it's a prime factor
    if n > 1:
        count += 1
        
    return count

def clear_console():
    """Clear console if possible"""
    try:
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/macOS
            os.system('clear')
    except:
        pass

def progress_bar_advanced(current, total, bar_width=40, prefix="Progress"):
    """Advanced progress bar with percentage"""
    percent = 100.0 * current / total
    filled_length = int(bar_width * current // total)
    bar = '[' + '=' * filled_length + ' ' * (bar_width - filled_length) + ']'
    
    return f"{prefix} {percent:.2f}% {bar} {current:,}/{total:,}"

def get_memory_usage_mb():
    """Get current memory usage in MB (requires psutil)"""
    if not HAS_PSUTIL:
        return 0
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def test_conjecture_optimized(max_n=200_000_000, progress_step=None):
    """Optimized test of the conjecture using JS-style improvements"""
    
    if progress_step is None:
        progress_step = max_n // 100  # Update every 1%
    
    print(f"=== PROFESSIONAL COMPUTATIONAL VERIFICATION ===")
    print(f"Target: {max_n:,} (testing {max_n//2:,} even cases)")
    
    # Memory before
    mem_before = get_memory_usage_mb()
    
    # Generate primes with timing
    print(f"Generating primes up to {max_n:,}...")
    t0 = time.perf_counter()
    primes = sieve_of_eratosthenes(max_n)
    t1 = time.perf_counter()
    print(f"Primes generated: {len(primes):,} in {t1-t0:.2f}s")
    
    # Initialize counters
    tested = 0
    verified = 0
    s_prime = 0  # Cases where ω(s) = 1
    s_semi = 0   # Cases where ω(s) = 2
    failures = []
    
    # Start verification timing
    tv0 = time.perf_counter()
    total_cases = max_n // 2 - 2  # Even numbers from 6 to max_n
    
    print(f"\nStarting verification...")
    
    for N in range(6, max_n + 1, 2):
        tested += 1
        found_decomposition = False
        
        # Binary search optimization: find first prime > N/2
        min_prime = N // 2 + 1
        start_idx = bisect.bisect_left(primes, min_prime)
        
        # Check primes starting from the binary search position
        for i in range(start_idx, len(primes)):
            p = primes[i]
            if p >= N:
                break
                
            s = N - p
            w = omega_optimized(s)
            
            if w <= 2:
                verified += 1
                if w == 1:
                    s_prime += 1
                else:  # w == 2 or w == 0
                    s_semi += 1
                found_decomposition = True
                break
        
        if not found_decomposition:
            failures.append(N)
            print(f"\n❌ FAILURE: N = {N}")
        
        # Progress reporting with advanced bar
        if tested % progress_step == 0:
            elapsed = time.perf_counter() - tv0
            rate = tested / elapsed if elapsed > 0 else 0
            
            # Clear console for cleaner output
            clear_console()
            
            progress_msg = progress_bar_advanced(tested, total_cases)
            print(progress_msg)
            print(f"Rate: {rate:.0f} cases/s | Failures: {len(failures)}")
            
            if len(failures) > 0:
                print(f"Recent failures: {failures[-5:]}")
    
    # Final timing
    tv1 = time.perf_counter()
    verification_time = tv1 - tv0
    
    # Memory after
    mem_after = get_memory_usage_mb()
    
    # Final progress bar
    clear_console()
    print(progress_bar_advanced(total_cases, total_cases))
    
    print(f"\n=== RESULTS ===")
    print(f"Range tested: 6 to {max_n:,} (even only)")
    print(f"Total cases:    {tested:,}")
    print(f"Verified:       {verified:,}")
    print(f"Failures:       {len(failures)}")
    print(f"Success rate:   {100.0 * verified / tested:.6f}%")
    print(f"Verification time: {verification_time:.2f}s")
    print(f"Overall rate:      {tested / verification_time:,.0f} cases/s")
    
    if HAS_PSUTIL:
        print(f"Memory used:       {mem_after - mem_before:.2f} MB")
    else:
        print(f"Memory monitoring: unavailable (install psutil for memory stats)")
    
    if len(failures) == 0:
        print(f"\n🏆 All cases verified successfully!")
        print(f"  ω(s)=1: {s_prime:,} ({100.0 * s_prime / verified:.2f}%)")
        print(f"  ω(s)=2: {s_semi:,} ({100.0 * s_semi / verified:.2f}%)")
    else:
        print(f"\n❌ {len(failures)} counterexamples!")
        print(f"First few: {failures[:10]}")
    
    return failures

def test_small_examples_optimized():
    """Test small examples with optimized omega function"""
    print("Testing small examples with optimized omega:")
    
    test_cases = [6, 8, 10, 12, 14, 16, 18, 20]
    
    for N in test_cases:
        print(f"\nN = {N}:")
        found = False
        
        for p in range(N//2 + 1, N):
            if is_prime_simple(p):
                s = N - p
                omega_s = omega_optimized(s)
                if omega_s <= 2:
                    print(f"  {N} = {p} + {s}, ω({s}) = {omega_s}")
                    found = True
                    break
        
        if not found:
            print(f"  No decomposition found!")

def is_prime_simple(n):
    """Simple primality test for small numbers"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    # Test small examples first
    test_small_examples_optimized()
    
    print("\n" + "="*60)
    
    # Choose your test size:
    
    # Quick test (1 million)
    # test_conjecture_optimized(max_n=1_000_000, progress_step=10_000)
    
    # Medium test (10 million) 
    # test_conjecture_optimized(max_n=10_000_000, progress_step=100_000)
    
    # Large test (100 million)
    # test_conjecture_optimized(max_n=100_000_000, progress_step=1_000_000)
    
    # Full JS-equivalent test (200 million)
    test_conjecture_optimized(max_n=200_000_000, progress_step=2_000_000)