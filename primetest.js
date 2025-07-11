// === PROFESSIONAL COMPUTATIONAL VERIFICATION ===
// Target: 100 million even cases

const LIMIT = 200_000_000;          // Highest even N to test
const BAR_WIDTH = 40;                // Width of the ASCII progress bar
const PROGRESS_STEP = LIMIT / 100;   // Update bar every 0.01% (~10k cases)

// High-resolution timer
typeof process !== 'undefined' && process.hrtime && typeof require !== 'undefined';
const hrtime = () => {
  if (typeof process !== 'undefined' && process.hrtime) {
    const [s, ns] = process.hrtime();
    return s + ns / 1e9;
  } else {
    return performance.now() / 1000;
  }
};

// Binary search helper for lower bound
function lowerBound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Generate all primes up to LIMIT via sieve of Eratosthenes
const generatePrimes = (max) => {
  const isPrime = new Uint8Array(max + 1);
  isPrime.fill(1);
  isPrime[0] = isPrime[1] = 0;
  const root = Math.sqrt(max) | 0;
  for (let i = 2; i <= root; i++) {
    if (isPrime[i]) {
      for (let j = i * i; j <= max; j += i) isPrime[j] = 0;
    }
  }
  const primes = [];
  for (let i = 2; i <= max; i++) if (isPrime[i]) primes.push(i);
  return primes;
};

// Fast omega (distinct prime divisors) using trial division
const omega = (n) => {
  let cnt = 0;
  if ((n & 1) === 0) {
    cnt++;
    do { n >>= 1; } while ((n & 1) === 0);
  }
  for (let f = 3; f * f <= n; f += 2) {
    if (n % f === 0) {
      cnt++;
      do { n /= f; } while (n % f === 0);
    }
  }
  if (n > 1) cnt++;
  return cnt;
};

(async () => {
  console.log(`Starting verification up to N = ${LIMIT.toLocaleString()}`);
  const memBefore = typeof process !== 'undefined'
    ? process.memoryUsage().heapUsed
    : performance.memory?.usedJSHeapSize || 0;
  const t0 = hrtime();
  const primes = generatePrimes(LIMIT);
  const t1 = hrtime();
  console.log(`Primes generated: ${primes.length.toLocaleString()} in ${(t1 - t0).toFixed(2)}s`);

  let tested = 0, verified = 0, sPrime = 0, sSemi = 0;
  const failures = [];
  const tv0 = hrtime();

  for (let N = 6; N <= LIMIT; N += 2) {
    tested++;
    let ok = false;

    const startIdx = lowerBound(primes, N >>> 1);
    for (let i = startIdx; i < primes.length; i++) {
      const p = primes[i];
      if (p >= N) break;
      const w = omega(N - p);
      if (w <= 2) {
        verified++;
        w === 1 ? sPrime++ : sSemi++;
        ok = true;
        break;
      }
    }
    if (!ok) failures.push(N);

    if (tested % PROGRESS_STEP === 0) {
      const pct = tested / (LIMIT/2) * 100;
      const filled = Math.floor((pct/100) * BAR_WIDTH);
      const bar = '[' + '='.repeat(filled) + ' '.repeat(BAR_WIDTH - filled) + ']';
      const elapsed = hrtime() - tv0;
      // Clear console for environments that support it
      if (typeof console.clear === 'function') console.clear();
      console.log(`Progress ${pct.toFixed(2)}% ${bar}` +
                  ` ${tested.toLocaleString()}/${(LIMIT/2).toLocaleString()} evens` +
                  ` | ${(tested/elapsed).toFixed(0)} cases/s`);
    }
  }

  const tv1 = hrtime();
  console.log("\n=== RESULTS ===");
  console.log(`Range tested: 6 to ${LIMIT.toLocaleString()} (even only)`);
  console.log(`Total cases:    ${tested.toLocaleString()}`);
  console.log(`Verified:       ${verified.toLocaleString()}`);
  console.log(`Failures:       ${failures.length}`);
  console.log(`Success rate:   ${(100 * verified / tested).toFixed(6)}%`);
  console.log(`Verification time: ${(tv1 - tv0).toFixed(2)}s`);
  console.log(`Overall rate:      ${(tested / (tv1 - tv0)).toLocaleString()} cases/s`);
  console.log(`Memory used:       ${(((typeof process !== 'undefined'
    ? process.memoryUsage().heapUsed : performance.memory?.usedJSHeapSize || 0) - memBefore)
    /1024/1024).toFixed(2)} MB`);

  if (failures.length === 0) {
    console.log("\n🏆 All cases verified successfully.");
    console.log(`  ω(s)=1: ${sPrime.toLocaleString()} (${(100 * sPrime / verified).toFixed(2)}%)`);
    console.log(`  ω(s)=2: ${sSemi.toLocaleString()} (${(100 * sSemi / verified).toFixed(2)}%)`);
  } else {
    console.error(`\n❌ ${failures.length} counterexamples! First few: ${failures.slice(0, 10).join(', ')}`);
  }
})();
