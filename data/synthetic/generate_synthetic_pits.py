"""
Volumetric Mining Auditor (VMA) - Synthetic Pit Benchmark Generator
Implements benchmark test cases E1, E2, E3, and E4 with exact analytical solutions
and verifies that Piecewise 2D Composite Simpson's 1/3 quadrature achieves < 0.010% relative error.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration import compute_volume_simpson_2d


def generate_pit_e1(
    R: float = 100.0,
    H: float = 25.0,
    N: int = 201,
) -> Dict[str, Any]:
    """
    Case E1: Circular Paraboloid Pit
      Equation: delta_z(x, y) = max(0, H * (1 - (x^2 + y^2) / R^2))
      Domain: [-R, R] x [-R, R]
      Analytical Volume: (1/2) * pi * R^2 * H
    """
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    r_sq = X**2 + Y**2
    delta_z = np.maximum(0.0, H * (1.0 - r_sq / (R**2)))
    analytical_volume = 0.5 * math.pi * (R**2) * H

    return {
        "case_id": "E1",
        "name": "Circular Paraboloid",
        "delta_z": delta_z,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "analytical_volume": analytical_volume,
        "params": {"R": R, "H": H, "N": N},
    }


def generate_pit_e2(
    a: float = 150.0,
    b: float = 80.0,
    H: float = 30.0,
    N: int = 201,
) -> Dict[str, Any]:
    """
    Case E2: Elliptical Paraboloid Pit
      Equation: delta_z(x, y) = max(0, H * (1 - (x/a)^2 - (y/b)^2))
      Domain: [-a, a] x [-b, b]
      Analytical Volume: (1/2) * pi * a * b * H
    """
    x = np.linspace(-a, a, N)
    y = np.linspace(-b, b, N)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    term = 1.0 - (X / a) ** 2 - (Y / b) ** 2
    delta_z = np.maximum(0.0, H * term)
    analytical_volume = 0.5 * math.pi * a * b * H

    return {
        "case_id": "E2",
        "name": "Elliptical Paraboloid",
        "delta_z": delta_z,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "analytical_volume": analytical_volume,
        "params": {"a": a, "b": b, "H": H, "N": N},
    }


def generate_pit_e3(
    R: float = 60.0,
    H: float = 120.0,
    N: int = 201,
) -> Dict[str, Any]:
    """
    Case E3: Deep Paraboloid Pit (High Aspect Ratio H/R = 2.0)
      Equation: delta_z(x, y) = max(0, H * (1 - (x^2 + y^2) / R^2))
      Domain: [-R, R] x [-R, R]
      Analytical Volume: (1/2) * pi * R^2 * H
    """
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    r_sq = X**2 + Y**2
    delta_z = np.maximum(0.0, H * (1.0 - r_sq / (R**2)))
    analytical_volume = 0.5 * math.pi * (R**2) * H

    return {
        "case_id": "E3",
        "name": "Deep Paraboloid",
        "delta_z": delta_z,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "analytical_volume": analytical_volume,
        "params": {"R": R, "H": H, "N": N},
    }


def generate_pit_e4(
    Lx: float = 100.0,
    Ly: float = 80.0,
    H: float = 25.0,
    c1: float = 0.15,
    c2: float = -0.10,
    N: int = 101,
) -> Dict[str, Any]:
    """
    Case E4: Asymmetric Polynomial Pit
      Equation: delta_z(x, y) = (1 - (x/Lx)^2) * (1 - (y/Ly)^2) * (H + c1*x + c2*y)
      Domain: [0, Lx] x [0, Ly]
      Analytical Volume:
        int_0^Lx int_0^Ly delta_z dx dy =
        (4/9)*H*Lx*Ly + (1/6)*c1*(Lx^2)*Ly + (1/6)*c2*Lx*(Ly^2)
    """
    x = np.linspace(0.0, Lx, N)
    y = np.linspace(0.0, Ly, N)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    poly_term = (1.0 - (X / Lx) ** 2) * (1.0 - (Y / Ly) ** 2)
    delta_z = np.maximum(0.0, poly_term * (H + c1 * X + c2 * Y))

    term_H = (4.0 / 9.0) * H * Lx * Ly
    term_c1 = (1.0 / 6.0) * c1 * (Lx**2) * Ly
    term_c2 = (1.0 / 6.0) * c2 * Lx * (Ly**2)
    analytical_volume = term_H + term_c1 + term_c2

    return {
        "case_id": "E4",
        "name": "Asymmetric Polynomial",
        "delta_z": delta_z,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "analytical_volume": analytical_volume,
        "params": {"Lx": Lx, "Ly": Ly, "H": H, "c1": c1, "c2": c2, "N": N},
    }


def run_all_benchmarks(verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Executes benchmark test cases E1, E2, E3, and E4, compares analytical ground truth
    vs VMA Simpson volume, and asserts relative error < 0.010%.
    """
    benchmarks = [
        generate_pit_e1(),
        generate_pit_e2(),
        generate_pit_e3(),
        generate_pit_e4(),
    ]

    results = {}
    threshold_pct = 0.010  # 0.010% maximum allowable relative error

    if verbose:
        print("\n" + "=" * 90)
        print(" VOLUMETRIC MINING AUDITOR (VMA) - ANALYTICAL QUADRATURE BENCHMARKS")
        print(" Tolerance Requirement: Relative Error < 0.010%")
        print("=" * 90)
        header = f"{'Case':<6} | {'Geometry Name':<24} | {'Analytical (m³)':<16} | {'VMA Simpson (m³)':<16} | {'Rel Error':<10} | {'Status'}"
        print(header)
        print("-" * 90)

    for b in benchmarks:
        cid = b["case_id"]
        name = b["name"]
        dz = b["delta_z"]
        dx = b["dx"]
        dy = b["dy"]
        v_analytical = b["analytical_volume"]

        v_simpson = compute_volume_simpson_2d(dz, dx, dy)
        rel_error_pct = abs(v_simpson - v_analytical) / v_analytical * 100.0
        passed = rel_error_pct < threshold_pct

        results[cid] = {
            "name": name,
            "analytical": v_analytical,
            "simpson": v_simpson,
            "rel_error_pct": rel_error_pct,
            "passed": passed,
        }

        if verbose:
            status_str = "PASS [✓]" if passed else "FAIL [✗]"
            row = (
                f"{cid:<6} | {name:<24} | {v_analytical:>16.3f} | {v_simpson:>16.3f} | "
                f"{rel_error_pct:>9.4f}% | {status_str}"
            )
            print(row)

        assert passed, (
            f"Benchmark {cid} ({name}) failed tolerance! "
            f"Rel Error: {rel_error_pct:.6f}% >= {threshold_pct}%"
        )

    if verbose:
        print("=" * 90)
        print(" ALL SYNTHETIC BENCHMARK TESTS SATISFIED (< 0.010% error threshold)")
        print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    run_all_benchmarks(verbose=True)
