"""
System Memory & Resource Monitor
==================================
Monitors:
  - Python process memory usage (RSS, VMS)
  - Context window token estimation (based on log file sizes)
  - Number of active agents
  - Log file sizes
  - Database size (SQLite/Redis)
  - API call counts and rate limits

Alerts when:
  - Process memory > 4GB (warning) or > 6GB (critical)
  - Log files > 1GB (rotate)
  - Rate limit approaching (>80% of limit)

Output: logs/system_health/YYYYMMDD.jsonl
"""

import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[2]
_HEALTH_DIR = _BASE_DIR / "logs" / "system_health"
_HEALTH_DIR.mkdir(parents=True, exist_ok=True)

# Alert thresholds
_MEM_WARNING_BYTES = 4 * 1024 ** 3   # 4 GB
_MEM_CRITICAL_BYTES = 6 * 1024 ** 3  # 6 GB
_LOG_ROTATE_BYTES = 1 * 1024 ** 3    # 1 GB
_RATE_LIMIT_WARNING_PCT = 80.0        # percent of limit used

# Approximate token-per-byte ratio for context window estimation
_BYTES_PER_TOKEN = 4  # ~4 bytes per token (UTF-8 English)

# Default API rate limits (requests per minute) — update per actual API plan
_DEFAULT_RATE_LIMITS: dict[str, int] = {
    "yfinance": 2000,       # unofficial; yfinance does not enforce
    "openai": 500,
    "anthropic": 500,
    "alpaca": 200,
    "polygon": 1000,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _bytes_to_mb(b: int) -> float:
    return round(b / (1024 ** 2), 2)


def _bytes_to_gb(b: int) -> float:
    return round(b / (1024 ** 3), 4)


# ---------------------------------------------------------------------------
# psutil with graceful fallback
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — memory stats will use /proc fallback")


def _get_process_memory() -> dict:
    """Return RSS and VMS for the current process in bytes."""
    if _PSUTIL_AVAILABLE:
        proc = _psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {"rss": mem.rss, "vms": mem.vms}

    # Fallback: read /proc/self/status on Linux
    rss = 0
    vms = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmSize:"):
                    vms = int(line.split()[1]) * 1024
    except Exception:
        pass
    return {"rss": rss, "vms": vms}


def _get_system_memory() -> dict:
    """Return system-wide memory stats."""
    if _PSUTIL_AVAILABLE:
        vm = _psutil.virtual_memory()
        return {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }
    # Fallback
    result = {"total": 0, "available": 0, "used": 0, "percent": 0.0}
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used = total - available
        result = {
            "total": total,
            "available": available,
            "used": used,
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        pass
    return result


def _get_cpu_percent() -> float:
    """Return CPU utilisation for the current process."""
    if _PSUTIL_AVAILABLE:
        try:
            proc = _psutil.Process(os.getpid())
            return proc.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    # /proc fallback returns 0 (requires two samples to be meaningful)
    return 0.0


# ---------------------------------------------------------------------------
# MemoryMonitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Monitors system resource usage and emits health snapshots."""

    def __init__(
        self,
        health_dir: str | Path | None = None,
        log_root: str | Path | None = None,
        rate_limits: dict[str, int] | None = None,
        api_call_counts: dict[str, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        health_dir : path
            Where to write JSONL health logs.
        log_root : path
            Root of the log directory to scan for sizes.
        rate_limits : dict[str, int]
            Map of service name → requests-per-minute limit.
        api_call_counts : dict[str, int]
            Current call counts this minute per service. Updated externally.
        """
        self.health_dir = Path(health_dir) if health_dir else _HEALTH_DIR
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self.log_root = Path(log_root) if log_root else (_BASE_DIR / "logs")
        self.rate_limits = rate_limits or _DEFAULT_RATE_LIMITS
        self.api_call_counts: dict[str, int] = api_call_counts or {}
        self._snapshot_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Capture a point-in-time resource snapshot.

        Returns
        -------
        dict
            Current resource usage including memory, CPU, log sizes,
            token estimation, and API call counts.
        """
        proc_mem = _get_process_memory()
        sys_mem = _get_system_memory()
        cpu = _get_cpu_percent()
        log_sizes = self._scan_log_sizes()
        token_est = self._estimate_context_tokens(log_sizes)
        api_stats = self._api_stats()
        db_sizes = self._db_sizes()

        snap = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "date": date.today().isoformat(),
            "process_memory": {
                "rss_bytes": proc_mem["rss"],
                "rss_mb": _bytes_to_mb(proc_mem["rss"]),
                "rss_gb": _bytes_to_gb(proc_mem["rss"]),
                "vms_bytes": proc_mem["vms"],
                "vms_mb": _bytes_to_mb(proc_mem["vms"]),
            },
            "system_memory": {
                "total_gb": _bytes_to_gb(sys_mem["total"]),
                "available_gb": _bytes_to_gb(sys_mem["available"]),
                "used_gb": _bytes_to_gb(sys_mem["used"]),
                "used_pct": sys_mem["percent"],
            },
            "cpu_pct": cpu,
            "log_sizes": log_sizes,
            "context_window_estimate": token_est,
            "api_stats": api_stats,
            "db_sizes": db_sizes,
            "psutil_available": _PSUTIL_AVAILABLE,
            "python_version": sys.version.split()[0],
        }

        self._log(snap)
        self._snapshot_count += 1
        return snap

    def check_health(self) -> dict:
        """Run a full health check and return warnings/errors.

        Returns
        -------
        dict
            Health status with 'status' (OK/WARNING/CRITICAL),
            list of 'warnings', and list of 'errors'.
        """
        snap = self.snapshot()
        warnings: list[str] = []
        errors: list[str] = []

        # Memory checks
        rss = snap["process_memory"]["rss_bytes"]
        if rss >= _MEM_CRITICAL_BYTES:
            errors.append(
                f"CRITICAL: Process RSS {_bytes_to_gb(rss):.2f} GB exceeds 6 GB limit"
            )
        elif rss >= _MEM_WARNING_BYTES:
            warnings.append(
                f"WARNING: Process RSS {_bytes_to_gb(rss):.2f} GB exceeds 4 GB threshold"
            )

        # Log file rotation checks
        for log_name, info in snap["log_sizes"].items():
            sz = info.get("bytes", 0)
            if sz >= _LOG_ROTATE_BYTES:
                warnings.append(
                    f"LOG ROTATE: {log_name} is {_bytes_to_mb(sz):.0f} MB — exceeds 1 GB"
                )

        # API rate limit checks
        for service, api_info in snap["api_stats"].items():
            used_pct = api_info.get("usage_pct", 0.0)
            if used_pct >= 100.0:
                errors.append(f"RATE LIMIT EXCEEDED: {service} at {used_pct:.1f}%")
            elif used_pct >= _RATE_LIMIT_WARNING_PCT:
                warnings.append(f"RATE LIMIT WARNING: {service} at {used_pct:.1f}%")

        # System memory
        sys_used_pct = snap["system_memory"]["used_pct"]
        if sys_used_pct >= 95.0:
            errors.append(f"SYSTEM MEMORY CRITICAL: {sys_used_pct:.1f}% used")
        elif sys_used_pct >= 85.0:
            warnings.append(f"SYSTEM MEMORY WARNING: {sys_used_pct:.1f}% used")

        status = "OK"
        if errors:
            status = "CRITICAL"
        elif warnings:
            status = "WARNING"

        return {
            "status": status,
            "warnings": warnings,
            "errors": errors,
            "snapshot": snap,
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    def get_summary(self) -> str:
        """Return a compact formatted string suitable for embedding in other reports."""
        snap = self.snapshot()
        pm = snap["process_memory"]
        sm = snap["system_memory"]
        health = self.check_health()

        lines = [
            "┌─ SYSTEM HEALTH ─────────────────────────────────────┐",
            f"│  Status       : {health['status']:<38}│",
            f"│  Process RSS  : {pm['rss_mb']:>8.1f} MB  ({pm['rss_gb']:.3f} GB)         │",
            f"│  Process VMS  : {pm['vms_mb']:>8.1f} MB                              │",
            f"│  System Mem   : {sm['used_pct']:>5.1f}% used  ({sm['available_gb']:.2f} GB free)       │",
            f"│  CPU          : {snap['cpu_pct']:>5.1f}%                                 │",
        ]

        ctx = snap.get("context_window_estimate", {})
        lines.append(
            f"│  Context Est. : ~{ctx.get('estimated_tokens', 0):,} tokens                    │"
        )

        if health["warnings"]:
            lines.append(f"│  Warnings     : {len(health['warnings'])} issue(s)                            │")
        if health["errors"]:
            lines.append(f"│  ERRORS       : {len(health['errors'])} critical issue(s)                    │")

        for msg in health["errors"][:3]:
            short = msg[:50]
            lines.append(f"│  !! {short:<50}│")
        for msg in health["warnings"][:3]:
            short = msg[:50]
            lines.append(f"│  ⚠  {short:<50}│")

        lines.append("└─────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_log_sizes(self) -> dict[str, dict]:
        """Walk log_root and compute per-subdirectory sizes."""
        sizes: dict[str, dict] = {}
        if not self.log_root.exists():
            return sizes
        try:
            for subdir in self.log_root.iterdir():
                if not subdir.is_dir():
                    continue
                total_bytes = 0
                file_count = 0
                for f in subdir.rglob("*"):
                    if f.is_file():
                        try:
                            total_bytes += f.stat().st_size
                            file_count += 1
                        except OSError:
                            pass
                sizes[subdir.name] = {
                    "bytes": total_bytes,
                    "mb": _bytes_to_mb(total_bytes),
                    "file_count": file_count,
                    "needs_rotation": total_bytes >= _LOG_ROTATE_BYTES,
                }
        except Exception as exc:
            logger.warning("_scan_log_sizes error: %s", exc)
        return sizes

    def _estimate_context_tokens(self, log_sizes: dict[str, dict]) -> dict:
        """Estimate context window token consumption from log file sizes."""
        total_bytes = sum(info.get("bytes", 0) for info in log_sizes.values())
        estimated_tokens = total_bytes // _BYTES_PER_TOKEN
        # Common LLM context limits
        limits = {
            "gpt-4-turbo": 128_000,
            "claude-3-opus": 200_000,
            "claude-3-5-sonnet": 200_000,
        }
        return {
            "total_log_bytes": total_bytes,
            "total_log_mb": _bytes_to_mb(total_bytes),
            "estimated_tokens": estimated_tokens,
            "context_usage_by_model": {
                model: round(estimated_tokens / limit * 100, 2)
                for model, limit in limits.items()
            },
        }

    def _api_stats(self) -> dict[str, dict]:
        """Compute API usage statistics from self.api_call_counts."""
        result = {}
        for service, limit in self.rate_limits.items():
            calls = self.api_call_counts.get(service, 0)
            usage_pct = (calls / limit * 100) if limit > 0 else 0.0
            result[service] = {
                "calls_this_window": calls,
                "limit": limit,
                "usage_pct": round(usage_pct, 2),
                "status": (
                    "CRITICAL" if usage_pct >= 100
                    else ("WARNING" if usage_pct >= _RATE_LIMIT_WARNING_PCT else "OK")
                ),
            }
        return result

    def _db_sizes(self) -> dict[str, dict]:
        """Detect SQLite database files in the project and return sizes."""
        sizes = {}
        try:
            for db_path in _BASE_DIR.rglob("*.db"):
                try:
                    sz = db_path.stat().st_size
                    sizes[str(db_path.relative_to(_BASE_DIR))] = {
                        "bytes": sz,
                        "mb": _bytes_to_mb(sz),
                    }
                except OSError:
                    pass
            for db_path in _BASE_DIR.rglob("*.sqlite"):
                try:
                    sz = db_path.stat().st_size
                    sizes[str(db_path.relative_to(_BASE_DIR))] = {
                        "bytes": sz,
                        "mb": _bytes_to_mb(sz),
                    }
                except OSError:
                    pass
        except Exception as exc:
            logger.debug("_db_sizes error: %s", exc)
        return sizes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _log(self, snap: dict) -> None:
        log_path = self.health_dir / f"{date.today().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(snap, default=str) + "\n")
        except Exception as exc:
            logger.warning("MemoryMonitor._log write error: %s", exc)

    def update_api_count(self, service: str, increment: int = 1) -> None:
        """Increment the API call counter for a given service.

        Call this from your API wrapper to track rate limit consumption.
        """
        self.api_call_counts[service] = self.api_call_counts.get(service, 0) + increment

    def reset_api_counts(self) -> None:
        """Reset all API counters (call at the start of each rate-limit window)."""
        self.api_call_counts = {}


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor = MemoryMonitor()
    monitor.update_api_count("yfinance", 45)
    monitor.update_api_count("openai", 410)
    health = monitor.check_health()
    print(json.dumps(health, indent=2, default=str))
    print()
    print(monitor.get_summary())
