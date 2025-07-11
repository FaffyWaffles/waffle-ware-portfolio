#!/usr/bin/env python3
"""
2p-Sandbox   —   empirical test of the “shift-by-2q” heuristic
For each odd prime p in [X, X+DELTA) and each odd prime q ≤ QMAX with q ≤ p/2,
check whether at least one of p − 2q or p + 2q is prime.

Python 3.8+   •   deterministic for every 64-bit input
"""

import math
from typing import List, Dict, Tuple

# ─────────────────────────────────────────── parameters ────
X      = 10**11         # start of p-range
DELTA  = 2_000_000      # width of p-range
QMAX   = 1_000_000      # upper limit for small primes q
REPORT = 100_000        # progress report interval (pairs)

# ───────────────────────── deterministic Miller–Rabin bases (≤2⁶⁴)
MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for 64-bit integers."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n & 1 == 0:
        return False

    # write n−1 = d·2ʳ
    d, r = n - 1, 0
    while d & 1 == 0:
        d >>= 1
        r += 1

    for a in MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ───────────────────── basic sieves ──────────────────────────────
def sieve_small(limit: int) -> List[int]:
    """Return list of all primes ≤ limit (simple bit-sieve)."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p*p : limit+1 : p] = b"\x00" * (((limit - p*p) // p) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]

def sieve_segment(start: int, length: int, base_primes: List[int]) -> List[int]:
    """Segmented sieve on [start, start+length)."""
    sieve = bytearray(b"\x01") * length
    end = start + length
    for p in base_primes:
        if p * p > end:
            break
        first = max(p * p, ((start + p - 1) // p) * p)
        sieve[first - start : : p] = b"\x00" * (((end - first - 1) // p) + 1)
    return [start + i for i, is_p in enumerate(sieve) if is_p and (start + i) >= 3]

# ───────────────────── main experiment ───────────────────────────
def run_sandbox() -> None:
    print(f"2p-Sandbox | X={X:,}  Δ={DELTA:,}  q≤{QMAX:,}")

    # 1) small primes (both q-list and base for segmented sieve)
    small_primes = sieve_small(QMAX)
    print(f"• small primes q  : {len(small_primes):,}")

    # 2) primes p in [X, X+Δ)
    base_for_seg = sieve_small(int((X + DELTA)**0.5) + 1)
    primes_p = sieve_segment(X, DELTA, base_for_seg)
    print(f"• large primes p  : {len(primes_p):,}")

    # quick estimate of total pairs
    total_pairs_est = sum(min(len(small_primes), p // 2 // 2) for p in primes_p)
    print(f"≈ will test ≤{total_pairs_est:,} (p,q) pairs\n")

    # statistics
    tot, succ = 0, 0
    max_gap, worst_info = 0, None
    gap_hist: Dict[int, int] = {}
    residue: Dict[int, Tuple[int, int]] = {}   # mod 30 ➜ (succ, tot)

    next_report = REPORT
    pairs_done = 0

    for p in primes_p:
        halfp = p // 2
        consec, gap_start = 0, None

        for q in small_primes:
            if q > halfp:
                break

            pairs_done += 1
            tot += 1

            if pairs_done >= next_report:
                rate = succ / tot if tot else 0
                print(f"{pairs_done:,} pairs | succ {succ:,} ({rate:.4%})")
                next_report += REPORT

            pm = p - 2 * q
            pp = p + 2 * q
            if pm < 3:           # negative / 1
                continue

            ok = (pm & 1 and is_prime(pm)) or (pp & 1 and is_prime(pp))
            r = q % 30
            s, t = residue.get(r, (0, 0))
            residue[r] = (s + ok, t + 1)

            if ok:
                succ += 1
                if consec:
                    gap_hist[consec] = gap_hist.get(consec, 0) + 1
                    if consec > max_gap:
                        max_gap = consec
                        worst_info = (p, gap_start, q, consec)
                consec, gap_start = 0, None
            else:
                if consec == 0:
                    gap_start = q
                consec += 1

        # tail streak for this p
        if consec:
            gap_hist[consec] = gap_hist.get(consec, 0) + 1
            if consec > max_gap:
                max_gap = consec
                worst_info = (p, gap_start, small_primes[-1], consec)

    # heuristic success probability at median p
    med_p = primes_p[len(primes_p)//2] if primes_p else X
    heur = 1 - (1 - 1 / math.log(med_p))**2

    # ─────────── results
    print("\n=====   RESULTS   =====")
    print(f"tests       : {tot:,}")
    print(f"successes   : {succ:,}  ({succ / tot:.6%})")
    print(f"heuristic   : {heur:.6f}")
    print(f"max gap     : {max_gap}")
    if worst_info:
        p_w, q_a, q_b, l = worst_info
        print(f"worst gap @ p={p_w:,}, q∈[{q_a:,},{q_b:,}] (len={l})")

    print("\nfirst 10 gap counts:")
    for g in sorted(gap_hist)[:10]:
        print(f" gap {g:2d} : {gap_hist[g]:,}")

    print("\nresidue-class success rates (mod 30):")
    for r in range(30):
        if r in residue:
            s, t = residue[r]
            print(f"  {r:2d}: {s:>7,}/{t:<7,} = {s/t:.4f}")

if __name__ == "__main__":
    run_sandbox()
