#!/usr/bin/env python3
"""
Investment Platform — Repository Initialization & Health Check
All 11 repos on branch: claude/init-test-repos-oPogr
"""

import sys
import os
import importlib
import subprocess
import json
from pathlib import Path
from datetime import datetime

REPOS = {
    "Financial-Data":     {"layer": 0, "role": "Market data ingestion (yfinance fork)", "lang": "Python"},
    "open-bb":            {"layer": 0, "role": "Macro/alt data aggregation (OpenBB)", "lang": "Python/TS"},
    "QLIB":               {"layer": 1, "role": "Alpha factor research & backtesting", "lang": "Python"},
    "quant-trading":      {"layer": 1, "role": "Strategy library (TA + patterns)", "lang": "Python"},
    "ML-Macro-Market":    {"layer": 1, "role": "Cyclical/secular regime classifier", "lang": "Python"},
    "ai-hedgefund":       {"layer": 2, "role": "Multi-agent decision engine", "lang": "Python"},
    "Mav-Analysis":       {"layer": 2, "role": "MCP server for Claude tools", "lang": "Python"},
    "Air-LLM":            {"layer": 2, "role": "Lightweight LLM inference", "lang": "Python"},
    "AI-Newton":          {"layer": 2, "role": "Physics-inspired symbolic AI", "lang": "Rust/Python"},
    "hedgefund-tracker":  {"layer": 3, "role": "13F institutional tracker", "lang": "Python"},
    "Ruflo-agents":       {"layer": 3, "role": "Claude-flow orchestration engine", "lang": "TypeScript"},
}

BASE_DIR = Path("/home/user")
RESULTS = {}

def check_repo(name: str, info: dict) -> dict:
    repo_path = BASE_DIR / name
    result = {
        "name": name,
        "layer": info["layer"],
        "role": info["role"],
        "lang": info["lang"],
        "path_exists": repo_path.exists(),
        "branch": None,
        "file_count": 0,
        "status": "UNKNOWN",
        "errors": [],
    }

    if not repo_path.exists():
        result["status"] = "MISSING"
        result["errors"].append("Repository directory not found")
        return result

    # Check git branch
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_path, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["branch"] = branch
        if branch != "claude/init-test-repos-oPogr":
            result["errors"].append(f"Wrong branch: {branch}")
    except Exception as e:
        result["errors"].append(f"Git error: {e}")

    # Count Python/JS/TS/Rust files
    for ext in [".py", ".ts", ".js", ".rs"]:
        count = len(list(repo_path.rglob(f"*{ext}")))
        result["file_count"] += count

    # Language-specific checks
    if info["lang"].startswith("Python"):
        result.update(_check_python_repo(name, repo_path))
    elif info["lang"] == "TypeScript":
        result.update(_check_ts_repo(name, repo_path))
    elif info["lang"] == "Rust/Python":
        result.update(_check_rust_python_repo(name, repo_path))

    if not result["errors"]:
        result["status"] = "READY"
    elif result["status"] == "UNKNOWN":
        result["status"] = "PARTIAL"

    return result


def _check_python_repo(name: str, path: Path) -> dict:
    updates = {"import_check": False}
    checks = {
        "Financial-Data": ("yfinance", str(path)),
        "QLIB": ("qlib", str(path)),
        "ai-hedgefund": ("src.graph.state", str(path)),
        "Mav-Analysis": ("maverick_mcp", str(path)),
        "ML-Macro-Market": ("common.stockhistory", str(path)),
        "hedgefund-tracker": ("app", str(path)),
        "quant-trading": ("backtrader", None),
        "Air-LLM": ("air_llm", str(path)),
        "open-bb": (None, None),
    }
    mod, insert_path = checks.get(name, (None, None))
    if mod:
        try:
            if insert_path and insert_path not in sys.path:
                sys.path.insert(0, insert_path)
            importlib.import_module(mod)
            updates["import_check"] = True
        except ImportError as e:
            updates.setdefault("errors", []).append(f"Import: {e}")
    return updates


def _check_ts_repo(name: str, path: Path) -> dict:
    pkg = path / "package.json"
    if pkg.exists():
        with open(pkg) as f:
            data = json.load(f)
        return {
            "package_name": data.get("name"),
            "version": data.get("version"),
            "import_check": True,
        }
    return {"errors": ["package.json not found"]}


def _check_rust_python_repo(name: str, path: Path) -> dict:
    cargo = path / "Cargo.toml"
    return {
        "cargo_exists": cargo.exists(),
        "import_check": cargo.exists(),
    }


def run_all_checks():
    print("=" * 70)
    print("INVESTMENT PLATFORM — REPOSITORY HEALTH CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Branch: claude/init-test-repos-oPogr")
    print("=" * 70)

    layer_names = {
        0: "DATA INFRASTRUCTURE",
        1: "QUANTITATIVE ENGINE",
        2: "AI AGENT INTELLIGENCE",
        3: "PORTFOLIO MANAGEMENT",
    }

    results_by_layer = {}
    for name, info in REPOS.items():
        print(f"\nChecking {name}...", end=" ", flush=True)
        r = check_repo(name, info)
        RESULTS[name] = r
        layer = info["layer"]
        results_by_layer.setdefault(layer, []).append(r)
        status_icon = "✓" if r["status"] == "READY" else "⚠" if r["status"] == "PARTIAL" else "✗"
        print(f"{status_icon} {r['status']}")

    print("\n" + "=" * 70)
    print("SUMMARY BY LAYER")
    print("=" * 70)

    total_ready = 0
    for layer_num in sorted(results_by_layer.keys()):
        print(f"\nLAYER {layer_num} — {layer_names[layer_num]}")
        for r in results_by_layer[layer_num]:
            status = r["status"]
            icon = "✓" if status == "READY" else "⚠" if status == "PARTIAL" else "✗"
            branch_ok = "✓" if r.get("branch") == "claude/init-test-repos-oPogr" else "✗"
            print(f"  {icon} [{branch_ok} branch] {r['name']:<22} | {r['role']}")
            if r.get("errors"):
                for err in r["errors"][:2]:
                    print(f"      └─ {err}")
            if status == "READY":
                total_ready += 1

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_ready}/{len(REPOS)} repos READY")
    print("=" * 70)

    # Write JSON report
    report_path = BASE_DIR / "investment-platform" / "health_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "branch": "claude/init-test-repos-oPogr",
            "total": len(REPOS),
            "ready": total_ready,
            "results": RESULTS,
        }, f, indent=2)
    print(f"\nReport saved: {report_path}")
    return total_ready


if __name__ == "__main__":
    ready = run_all_checks()
    sys.exit(0 if ready == len(REPOS) else 1)
