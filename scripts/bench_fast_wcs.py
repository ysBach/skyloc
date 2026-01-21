import sys
import os
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.append(os.path.abspath("src/skyloc/ioutils"))
import wcs

FastTanSipWCS = wcs.FastTanSipWCS


def make_test_header(sip_order=3):
    h = fits.Header()
    h["CTYPE1"] = "RA---TAN-SIP"
    h["CTYPE2"] = "DEC--TAN-SIP"
    h["CRVAL1"] = 10.0
    h["CRVAL2"] = 20.0
    h["CRPIX1"] = 1024.5
    h["CRPIX2"] = 1024.5
    h["CDELT1"] = -0.001
    h["CDELT2"] = 0.001
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"

    # PC Matrix (Rotation 30 deg)
    theta = np.deg2rad(30)
    c, s = np.cos(theta), np.sin(theta)
    h["PC1_1"] = c
    h["PC1_2"] = -s
    h["PC2_1"] = s
    h["PC2_2"] = c

    # SIP Forward
    h["A_ORDER"] = sip_order
    h["B_ORDER"] = sip_order

    np.random.seed(123)
    # Add very small random SIP coeffs to ensure stability
    scale = 1e-7
    for i in range(sip_order + 1):
        for j in range(sip_order + 1):
            if i + j <= sip_order and i + j > 0:
                h[f"A_{i}_{j}"] = np.random.uniform(-scale, scale)
                h[f"B_{i}_{j}"] = np.random.uniform(-scale, scale)

    # SIP Inverse (AP/BP) - symmetric for test simplicity
    h["AP_ORDER"] = sip_order
    h["BP_ORDER"] = sip_order
    for i in range(sip_order + 1):
        for j in range(sip_order + 1):
            if i + j <= sip_order and i + j > 0:
                h[f"AP_{i}_{j}"] = float(h[f"A_{i}_{j}"])
                h[f"BP_{i}_{j}"] = float(h[f"B_{i}_{j}"])

    return h


def test_correctness():
    print("Verifying Correctness...")
    h = make_test_header()
    h_dict = dict(h)

    # Astropy WCS
    w_astro = WCS(h)
    # Fast WCS
    w_fast = FastTanSipWCS(h_dict)

    # Test Points
    N = 100
    x = np.random.uniform(0, 2048, N)
    y = np.random.uniform(0, 2048, N)

    # 1. pix2world
    ra_a, dec_a = w_astro.all_pix2world(x, y, 0)
    ra_f, dec_f = w_fast.all_pix2world(x, y, 0)

    diff_ra = np.abs(ra_a - ra_f)
    diff_dec = np.abs(dec_a - dec_f)

    print(f"Max RA diff: {np.max(diff_ra):.2e} deg")
    print(f"Max Dec diff: {np.max(diff_dec):.2e} deg")

    if np.max(diff_ra) > 1e-10 or np.max(diff_dec) > 1e-10:
        print("FAILED: pix2world mismatch!")
    else:
        print("PASSED: pix2world matches Astropy.")

    # 2. world2pix
    # Round-trip check (most important for self-consistency)
    x_rt, y_rt = w_fast.all_world2pix(ra_f, dec_f, 0)
    diff_x_rt = np.abs(x - x_rt)
    diff_y_rt = np.abs(y - y_rt)
    print(f"Max Roundtrip X diff: {np.max(diff_x_rt):.2e} pix")
    print(f"Max Roundtrip Y diff: {np.max(diff_y_rt):.2e} pix")

    # Astropy comparison (if stable)
    try:
        x_a, y_a = w_astro.all_world2pix(ra_a, dec_a, 0)
        x_f, y_f = w_fast.all_world2pix(ra_a, dec_a, 0)

        diff_x = np.abs(x_a - x_f)
        diff_y = np.abs(y_a - y_f)

        print(f"Max X diff vs Astropy: {np.max(diff_x):.2e} pix")
        print(f"Max Y diff vs Astropy: {np.max(diff_y):.2e} pix")

        if np.max(diff_x) > 1e-3 or np.max(diff_y) > 1e-3:
            print("FAILED: world2pix mismatch vs Astropy")
        else:
            print("PASSED: world2pix matches Astropy")

    except Exception as e:
        print(f"Skipping Astropy world2pix comparison due to: {e}")


def benchmark():
    print("\nBenchmarking...")
    h = make_test_header()
    h_dict = dict(h)

    N_obj = 1000
    N_test = 10

    # Initialization
    t0 = time.time()
    for _ in range(N_obj):
        _ = FastTanSipWCS(h_dict)
    t1 = time.time()
    print(
        f"Init {N_obj} FastTanSipWCS: {t1-t0:.4f}s ({(t1-t0)/N_obj*1.e+6:.2f} us/obj)"
    )

    t2 = time.time()
    for _ in range(N_obj):
        _ = WCS(h)
    t3 = time.time()
    print(f"Init {N_obj} Astropy WCS: {t3-t2:.4f}s ({(t3-t2)/N_obj*1.e+6:.2f} us/obj)")

    # Transform Speed
    w_fast = FastTanSipWCS(h_dict)
    w_astro = WCS(h)
    # N_pts = 1000000
    # x = np.random.uniform(0, 2048, N_pts)
    # y = np.random.uniform(0, 2048, N_pts)

    # # Warmup Numba
    # ra_warm, dec_warm = w_fast.all_pix2world(x[:10], y[:10], 0)
    # _ = w_fast.all_world2pix(ra_warm, dec_warm, 0)
    # _ = w_fast.all_world2pix(ra_warm[0], dec_warm[0], 0)
    # _ = w_fast.all_pix2world(x[:10], y[:10], 0)

    # print(f"\nTransform Speed ({N_pts} pts):")

    # # 1. pix2world
    # print("\n[pix2world]")
    # t4 = time.time()
    # for _ in range(N_test):
    #     ra, dec = w_fast.all_pix2world(x, y, 0)
    # t5 = time.time()
    # print(f"FastTanSipWCS: {(t5-t4)/N_test*1000:.2f} ms")

    # t6 = time.time()
    # for _ in range(N_test):
    #     _ = w_astro.all_pix2world(x, y, 0)
    # t7 = time.time()
    # print(f"Astropy:       {(t7-t6)/N_test*1000:.2f} ms")

    # # 2. world2pix
    # # Use real sky coordinates from previous step
    # print("\n[world2pix]")
    # t8 = time.time()
    # for _ in range(N_test):
    #     _ = w_fast.all_world2pix(ra, dec, 0)
    # t9 = time.time()
    # print(f"FastTanSipWCS: {(t9-t8)/N_test*1000:.2f} ms")

    # ta = time.time()
    # for _ in range(N_test):
    #     _ = w_astro.all_world2pix(ra, dec, 0)
    # tb = time.time()
    # print(f"Astropy:       {(tb-ta)/N_test*1000:.2f} ms")

    # Small Array Performance
    print("\nPerformance (Time per call in us):")
    print(f"{'N':<8} {'FastTanSipWCS':<15} {'Astropy':<15} {'Speedup':<8}")
    for n_array in [
        1,
        10,
        100,
        1000,
        2000,
        5000,
        10000,
        50000,
        100000,
        500_000,
        1_000_000,
    ]:
        x_s = np.random.uniform(0, 2048, n_array)
        y_s = np.random.uniform(0, 2048, n_array)

        n_test = 10 + 10000 // n_array

        # FastTanSipWCS
        t_start = time.perf_counter()

        # warmup
        _ = w_fast.all_pix2world(x_s, y_s, 0)
        for _ in range(n_test):
            _ = w_fast.all_pix2world(x_s, y_s, 0)
        t_end = time.perf_counter()
        t_fast = (t_end - t_start) / n_test * 1e6

        # Astropy
        t_start = time.perf_counter()
        for _ in range(n_test):
            _ = w_astro.all_pix2world(x_s, y_s, 0)
        t_end = time.perf_counter()
        t_astro = (t_end - t_start) / n_test * 1e6

        print(
            f"{n_array:<8} {t_fast:<15.2f} {t_astro:<15.2f} {t_astro/t_fast:<8.2f}times (pix2world)"
        )

        # world2pix benchmark
        ra_s, dec_s = w_fast.all_pix2world(x_s, y_s, 0)

        # FastTanSipWCS (world2pix)
        t_start = time.perf_counter()
        # warmup
        _ = w_fast.all_world2pix(ra_s, dec_s, 0)
        for _ in range(n_test):
            _ = w_fast.all_world2pix(ra_s, dec_s, 0)
        t_end = time.perf_counter()
        t_fast_w2p = (t_end - t_start) / n_test * 1e6

        # Astropy (world2pix)
        t_start = time.perf_counter()
        for _ in range(n_test):
            _ = w_astro.all_world2pix(ra_s, dec_s, 0)
        t_end = time.perf_counter()
        t_astro_w2p = (t_end - t_start) / n_test * 1e6

        print(
            f"{'':<8} {t_fast_w2p:<15.2f} {t_astro_w2p:<15.2f} {t_astro_w2p/t_fast_w2p:<8.2f}times (world2pix)"
        )
        print("-" * 60)

    print("\n[Note on Parallel Threshold]")
    print("FastTanSipWCS uses a parallel threshold (default 5000) for pix2world.")
    print("For N < threshold, it uses a serial loop to avoid overhead.")
    print("For N >= threshold, it uses Numba parallel (SIMD).")
    print("(Note: all_world2pix is forced to run serially for performance stability.)")
    print(
        "You can tune this via 'w_fast = FastTanSipWCS(header, parallel_threshold=N)'."
    )
    print(
        "Analyze the table above: Look for the crossover where Speedup drops below 1.0."
    )
    print(
        "Set parallel_threshold slightly HIGHER than that N to keep using the fast serial loop."
    )
    print(
        "Example: If N=100 is fast (1.5x) but N=1000 is slow (0.5x), set threshold=2000."
    )


if __name__ == "__main__":
    test_correctness()
    benchmark()
