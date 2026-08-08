#!/usr/bin/env python3
"""Reserve CUDA memory in a background process and release it on demand.

Examples:
    python3 gpu_memory_keeper.py
    python3 gpu_memory_keeper.py occupy --gpus 0,1
    python3 gpu_memory_keeper.py occupy --gpu 0 --memory 8G
    python3 gpu_memory_keeper.py status --gpu 0
    python3 gpu_memory_keeper.py release --gpu 0
    python3 gpu_memory_keeper.py list
    python3 gpu_memory_keeper.py release-all

The reservation protects memory from other CUDA processes, but it does not
make the GPU compute-exclusive. The process is intentionally kept separate
from the user's training/inference process so releasing the memory is safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PID_DIR = Path("/tmp")
DEFAULT_PID_GLOB = "gpu_memory_keeper_gpu*.json"
DEFAULT_PERCENT = 99.0
DEFAULT_KEEPER_PERCENT = 80.0
def systemd_unit_for(gpu: int) -> str:
    return f"gpu-memory-keeper@{gpu}.service"


def systemd_user_command(*arguments: str) -> list[str]:
    return ["systemctl", "--user", *arguments]


def systemd_user_check() -> tuple[bool, str | None]:
    """Check that a per-user systemd manager and user bus are reachable."""
    if shutil.which("systemctl") is None:
        return False, "systemctl 未找到 / systemctl was not found"
    try:
        result = subprocess.run(
            systemd_user_command("show-environment"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"无法连接 systemd 用户服务 / cannot reach systemd --user: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return False, f"systemd --user 不可用 / unavailable: {detail}"
    return True, None


def systemd_control(unit: str, *arguments: str) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            systemd_user_command(*arguments, unit),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return True, None


def systemd_guard_present(unit: str) -> bool:
    if shutil.which("systemctl") is None:
        return False
    for action in ("is-active", "is-enabled"):
        try:
            result = subprocess.run(
                systemd_user_command(action, unit),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode == 0:
            return True
    return False


def discover_systemd_guard_units() -> list[str]:
    if shutil.which("systemctl") is None:
        return []
    units: set[str] = set()
    commands = (
        systemd_user_command("list-units", "--all", "--plain", "--no-legend", "gpu-memory-keeper@*.service"),
        systemd_user_command("list-unit-files", "--plain", "--no-legend", "gpu-memory-keeper@*.service"),
    )
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in result.stdout.splitlines():
            unit = line.split(maxsplit=1)[0] if line.split() else ""
            if re.fullmatch(r"gpu-memory-keeper@\d+\.service", unit):
                units.add(unit)
    return sorted(units)


def systemd_unit_text(args: argparse.Namespace, gpu: int) -> str:
    """Build an independent service unit for one GPU."""
    unit = systemd_unit_for(gpu)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "occupy",
        "--gpu",
        str(gpu),
        "--foreground",
        "--systemd-unit",
        unit,
        *reservation_arguments(args),
    ]
    exec_start = shlex.join(command)
    return (
        "[Unit]\n"
        f"Description=GPU Memory Keeper for GPU {gpu}\n"
        "After=default.target\n\n"
        "StartLimitIntervalSec=0\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={exec_start}\n"
        "Restart=always\n"
        "RestartSec=3\n"
        "KillMode=control-group\n"
        "TimeoutStopSec=15\n"
        f"StandardOutput=append:/tmp/gpu_memory_keeper_gpu{gpu}.log\n"
        f"StandardError=append:/tmp/gpu_memory_keeper_gpu{gpu}.log\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def parse_size(value: str) -> int:
    """Parse a memory size such as 512M, 8G, or a byte count."""
    text = value.strip().upper()
    units = (("GIB", 1024**3), ("GB", 1024**3), ("G", 1024**3),
             ("MIB", 1024**2), ("MB", 1024**2), ("M", 1024**2),
             ("KIB", 1024), ("KB", 1024), ("K", 1024), ("B", 1))
    for suffix, multiplier in units:
        if text.endswith(suffix):
            number = text[:-len(suffix)].strip()
            break
    else:
        number, multiplier = text, 1

    try:
        amount = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid memory size: {value!r} (examples: 512M, 8G)"
        ) from exc
    if amount <= 0:
        raise argparse.ArgumentTypeError("memory size must be greater than zero")
    return int(amount * multiplier)


def parse_percent(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}") from exc
    if not 0 < percent < 100:
        raise argparse.ArgumentTypeError("percentage must be greater than 0 and less than 100")
    return percent


def parse_gpu_indexes(value: str) -> list[int]:
    indexes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        try:
            index = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid GPU list: {value!r} (example: 0,1,2)"
            ) from exc
        if index < 0:
            raise argparse.ArgumentTypeError("GPU indexes must not be negative")
        if index not in indexes:
            indexes.append(index)
    if not indexes:
        raise argparse.ArgumentTypeError("at least one GPU index is required")
    return indexes


def pid_file_for(gpu: int, pid_file: str | None) -> Path:
    if pid_file:
        return Path(pid_file).expanduser()
    return DEFAULT_PID_DIR / f"gpu_memory_keeper_gpu{gpu}.json"


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read PID file {path}: {exc}") from exc
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("pid"), int)
        or not isinstance(record.get("gpu"), int)
        or not isinstance(record.get("requested_bytes"), int)
    ):
        raise RuntimeError(f"invalid PID file: {path}")
    return record


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def discover_records() -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    for path in DEFAULT_PID_DIR.glob(DEFAULT_PID_GLOB):
        try:
            record = read_record(path)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if record is not None:
            records.append((path, record))
    records.sort(key=lambda item: (item[1].get("gpu", -1), item[1]["pid"]))
    return records, errors


def format_memory(byte_count: int) -> str:
    if byte_count >= 1024**3:
        return f"{byte_count / 1024**3:.2f} GiB"
    return f"{byte_count / 1024**2:.0f} MiB"


def calculate_initial_reservation(
    free_bytes: int,
    total_bytes: int,
    total_limit_percent: float,
    keeper_limit_percent: float,
) -> tuple[int, int]:
    safety_headroom = max(1, int(total_bytes * (100.0 - total_limit_percent) / 100.0))
    available_by_total = max(0, free_bytes - safety_headroom)
    keeper_limit_bytes = max(0, int(total_bytes * keeper_limit_percent / 100.0))
    return min(available_by_total, keeper_limit_bytes), safety_headroom


def calculate_monitor_plan(
    free_bytes: int,
    total_bytes: int,
    held_bytes: int,
    total_limit_percent: float,
    keeper_limit_percent: float,
    step_percent: float,
) -> tuple[int, int, int]:
    """Return (next_chunk, dynamic_target, safety_headroom) for monitor mode."""
    safety_headroom = max(1, int(total_bytes * (100.0 - total_limit_percent) / 100.0))
    available_by_total = max(0, free_bytes - safety_headroom)
    keeper_limit_bytes = max(0, int(total_bytes * keeper_limit_percent / 100.0))
    available_by_keeper = max(0, keeper_limit_bytes - held_bytes)
    available = min(available_by_total, available_by_keeper)
    step_bytes = max(1, int(total_bytes * step_percent / 100.0))
    return min(step_bytes, available), held_bytes + available, safety_headroom


def calculate_reservation_cap(
    free_bytes: int,
    total_bytes: int,
    held_bytes: int,
    total_limit_percent: float,
    keeper_limit_percent: float,
) -> int:
    """Return how much Keeper may hold after applying both live limits."""
    other_used_bytes = max(0, total_bytes - free_bytes - held_bytes)
    total_limit_bytes = int(total_bytes * total_limit_percent / 100.0)
    keeper_limit_bytes = int(total_bytes * keeper_limit_percent / 100.0)
    allowed_by_total = max(0, total_limit_bytes - other_used_bytes)
    return min(allowed_by_total, keeper_limit_bytes)


def apply_monitor_request(args: argparse.Namespace, record: dict[str, Any]) -> bool:
    request = record.pop("monitor_request", None)
    if not isinstance(request, dict):
        return False
    values = (
        request.get("total_limit_percent"),
        request.get("keeper_limit_percent"),
        request.get("step_percent"),
        request.get("interval"),
    )
    if not all(isinstance(value, (int, float)) and value > 0 for value in values):
        return False
    args.percent = float(values[0])
    args.keeper_percent = float(values[1])
    args.step_percent = float(values[2])
    args.interval = float(values[3])
    if request.get("enable_incremental", True):
        args.incremental = True
    return True


def query_gpu_metrics() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return [], "nvidia-smi was not found"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out"
    if result.returncode != 0:
        message = result.stderr.strip() or f"nvidia-smi exited with code {result.returncode}"
        return [], message

    metrics: list[dict[str, Any]] = []
    try:
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 5:
                raise ValueError(f"unexpected nvidia-smi row: {row!r}")
            index, name, used_mib, total_mib, utilization = row
            metrics.append(
                {
                    "gpu": int(index),
                    "name": name.strip(),
                    "used_mib": int(used_mib),
                    "total_mib": int(total_mib),
                    "utilization": int(utilization),
                }
            )
    except ValueError as exc:
        return [], f"cannot parse nvidia-smi output: {exc}"
    return metrics, None


def remove_own_pid_file(path: Path, pid: int) -> None:
    """Remove the file only when it still belongs to this process."""
    try:
        record = read_record(path)
        if record and record.get("pid") == pid:
            path.unlink(missing_ok=True)
    except (OSError, RuntimeError):
        # Cleanup must not hide the original allocation/termination result.
        pass


def write_pid_file(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def reserve_memory(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is required: python3 -m pip install torch", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is not available; cannot reserve GPU memory", file=sys.stderr)
        return 2
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        print(
            f"GPU index {args.gpu} is out of range (0-{torch.cuda.device_count() - 1})",
            file=sys.stderr,
        )
        return 2

    path = pid_file_for(args.gpu, args.pid_file)
    try:
        existing = read_record(path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if existing and process_exists(existing["pid"]):
        print(f"GPU {args.gpu} is already reserved by PID {existing['pid']}: {path}", file=sys.stderr)
        return 1
    if existing:
        path.unlink(missing_ok=True)

    torch.cuda.set_device(args.gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.gpu)
    initial_free_bytes = free_bytes
    incremental = args.incremental
    dynamic_monitor = incremental and args.memory is None
    initial_limit, total_headroom = calculate_initial_reservation(
        free_bytes,
        total_bytes,
        args.percent,
        args.keeper_percent,
    )
    requested: int | None
    if args.memory is None:
        requested = None if dynamic_monitor else initial_limit
    else:
        requested = args.memory
        maximum_allowed = min(free_bytes, initial_limit)
        if requested > maximum_allowed:
            print(
                f"requested {requested / 1024**2:.0f} MiB, but only "
                f"{maximum_allowed / 1024**2:.0f} MiB is available under the configured limits",
                file=sys.stderr,
            )
            return 1
    if requested is not None and requested <= 0 and not dynamic_monitor:
        print("GPU is already at the configured total or Keeper limit", file=sys.stderr)
        return 1

    stop = False
    monitor_requested = False

    def release_handler(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    def monitor_handler(_signum: int, _frame: Any) -> None:
        nonlocal monitor_requested
        monitor_requested = True

    # Install handlers before publishing the PID file, avoiding a signal race
    # between the control command and the keeper's wait loop.
    signal.signal(signal.SIGUSR1, release_handler)
    signal.signal(signal.SIGTERM, release_handler)
    signal.signal(signal.SIGINT, release_handler)
    signal.signal(signal.SIGUSR2, monitor_handler)

    if dynamic_monitor:
        first_chunk, target_bytes, safety_bytes = calculate_monitor_plan(
            free_bytes,
            total_bytes,
            0,
            args.percent,
            args.keeper_percent,
            args.step_percent,
        )
        step_bytes = max(1, int(total_bytes * args.step_percent / 100))
    else:
        assert requested is not None
        target_bytes = requested
        step_bytes = int(initial_free_bytes * args.step_percent / 100) if incremental else requested
        step_bytes = max(1, min(step_bytes, requested))
        safety_bytes = total_headroom
    tensors: list[tuple[Any, int]] = []
    held_bytes = 0

    def allocate_chunk(byte_count: int) -> None:
        tensor = torch.empty(
            (byte_count + 3) // 4,
            dtype=torch.float32,
            device=f"cuda:{args.gpu}",
        )
        torch.cuda.synchronize(args.gpu)
        tensors.append((tensor, byte_count))

    if not dynamic_monitor:
        first_chunk = min(step_bytes, requested)
    if first_chunk > 0:
        try:
            allocate_chunk(first_chunk)
            held_bytes = first_chunk
        except RuntimeError as exc:
            if not dynamic_monitor:
                print(f"CUDA allocation failed: {exc}", file=sys.stderr)
                return 1
            print(f"Incremental allocation paused: {exc}", file=sys.stderr, flush=True)

    record = {
        "pid": os.getpid(),
        "gpu": args.gpu,
        "requested_bytes": held_bytes,
        "target_bytes": target_bytes,
        "mode": "incremental" if incremental else "immediate",
        "dynamic_target": dynamic_monitor,
        "limit_percent": args.percent,
        "keeper_limit_percent": args.keeper_percent,
        "step_percent": args.step_percent,
        "interval": args.interval,
        "live_limit_updates": True,
        "supervisor_pid": args.supervisor_pid,
        "systemd_unit": getattr(args, "systemd_unit", None),
        "total_bytes_at_start": total_bytes,
        "pid_file": str(path),
    }

    def shrink_to(byte_limit: int) -> int:
        """Release whole allocations, then reclaim only the allowed remainder."""
        nonlocal held_bytes
        original_held = held_bytes
        while tensors and held_bytes > byte_limit:
            tensor, byte_count = tensors.pop()
            held_bytes -= byte_count
            del tensor
        torch.cuda.empty_cache()
        replacement = max(0, byte_limit - held_bytes)
        if replacement:
            try:
                allocate_chunk(replacement)
            except RuntimeError as exc:
                print(f"CUDA reallocation after shrink failed: {exc}", file=sys.stderr, flush=True)
            else:
                held_bytes += replacement
        return max(0, original_held - held_bytes)
    try:
        write_pid_file(path, record)
    except OSError as exc:
        tensors.clear()
        torch.cuda.empty_cache()
        print(f"cannot write PID file {path}: {exc}", file=sys.stderr)
        return 1

    if dynamic_monitor:
        print(
            f"Monitoring GPU {args.gpu}: reserved {held_bytes / 1024**2:.0f} MiB; "
            f"dynamic target {target_bytes / 1024**2:.0f} MiB, cap {args.percent:g}% of total "
            f"(PID {os.getpid()}).",
            flush=True,
        )
    else:
        assert requested is not None
        print(
            f"Reserved {held_bytes / 1024**2:.0f} / {requested / 1024**2:.0f} MiB "
            f"on GPU {args.gpu} (PID {os.getpid()}). "
            f"Run `python3 {Path(__file__).name} release --gpu {args.gpu}` to release.",
            flush=True,
        )
    next_growth = time.monotonic() + args.interval
    growth_error_reported = False
    try:
        while not stop:
            if monitor_requested:
                monitor_requested = False
                try:
                    monitor_record = read_record(path)
                except RuntimeError:
                    monitor_record = None
                request = monitor_record.get("monitor_request") if monitor_record else None
                enable_incremental = (
                    request.get("enable_incremental", True) if isinstance(request, dict) else True
                )
                if monitor_record and apply_monitor_request(args, monitor_record):
                    if enable_incremental:
                        dynamic_monitor = True
                        incremental = True
                        record["mode"] = "incremental"
                        record["dynamic_target"] = True
                    record["limit_percent"] = args.percent
                    record["keeper_limit_percent"] = args.keeper_percent
                    record["step_percent"] = args.step_percent
                    record["interval"] = args.interval
                    record.pop("monitor_request", None)
                    current_free, current_total = torch.cuda.mem_get_info(args.gpu)
                    reservation_cap = calculate_reservation_cap(
                        current_free,
                        current_total,
                        held_bytes,
                        args.percent,
                        args.keeper_percent,
                    )
                    released_bytes = shrink_to(reservation_cap) if held_bytes > reservation_cap else 0
                    if requested is not None and not dynamic_monitor:
                        requested = min(requested, reservation_cap)
                    target_bytes = min(target_bytes, reservation_cap)
                    record["requested_bytes"] = held_bytes
                    record["target_bytes"] = target_bytes
                    print(
                        f"GPU {args.gpu} live limits updated "
                        f"(total cap {args.percent:g}%, Keeper cap {args.keeper_percent:g}%, "
                        f"released {released_bytes / 1024**2:.0f} MiB)",
                        flush=True,
                    )
                    try:
                        write_pid_file(path, record)
                    except OSError as exc:
                        print(f"cannot update PID file {path}: {exc}", file=sys.stderr, flush=True)
                    next_growth = time.monotonic()
            if incremental and time.monotonic() >= next_growth:
                current_free, current_total = torch.cuda.mem_get_info(args.gpu)
                if dynamic_monitor:
                    chunk, new_target, safety_bytes = calculate_monitor_plan(
                        current_free,
                        current_total,
                        held_bytes,
                        args.percent,
                        args.keeper_percent,
                        args.step_percent,
                    )
                    target_bytes = new_target
                else:
                    assert requested is not None
                    safe_available = max(0, current_free - safety_bytes)
                    chunk = min(step_bytes, requested - held_bytes, safe_available)
                if chunk > 0:
                    try:
                        allocate_chunk(chunk)
                    except RuntimeError as exc:
                        if not growth_error_reported:
                            print(f"Incremental allocation paused: {exc}", file=sys.stderr, flush=True)
                            growth_error_reported = True
                    else:
                        held_bytes += chunk
                        record["requested_bytes"] = held_bytes
                        record["target_bytes"] = target_bytes
                        try:
                            write_pid_file(path, record)
                        except OSError as exc:
                            print(f"cannot update PID file {path}: {exc}", file=sys.stderr, flush=True)
                        print(
                            f"GPU {args.gpu} incremental reservation: "
                            f"{held_bytes / 1024**2:.0f} / {target_bytes / 1024**2:.0f} MiB",
                            flush=True,
                        )
                        growth_error_reported = False
                elif dynamic_monitor:
                    record["target_bytes"] = target_bytes
                    try:
                        write_pid_file(path, record)
                    except OSError as exc:
                        print(f"cannot update PID file {path}: {exc}", file=sys.stderr, flush=True)
                next_growth = time.monotonic() + args.interval
            time.sleep(0.2)
    finally:
        tensors.clear()
        torch.cuda.empty_cache()
        remove_own_pid_file(path, os.getpid())
        print(f"Released GPU {args.gpu} memory", flush=True)
    return 0


def reservation_arguments(args: argparse.Namespace) -> list[str]:
    arguments: list[str] = []
    if args.memory is not None:
        arguments.extend(("--memory", str(args.memory)))
    else:
        arguments.extend(("--percent", str(args.percent)))
    arguments.extend(("--keeper-percent", str(args.keeper_percent)))
    if args.incremental:
        arguments.extend(
            (
                "--incremental",
                "--step-percent",
                str(args.step_percent),
                "--interval",
                str(args.interval),
            )
        )
    if args.pid_file:
        arguments.extend(("--pid-file", args.pid_file))
    return arguments


def install_systemd_guard(args: argparse.Namespace, gpu_indexes: list[int]) -> int:
    """Install and start one systemd user service for each selected GPU."""
    available, error = systemd_user_check()
    if not available:
        print(error or "systemd --user is unavailable", file=sys.stderr)
        return 2

    unit_dir = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd" / "user"
    unit_paths = [unit_dir / systemd_unit_for(gpu) for gpu in gpu_indexes]
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
        for gpu, unit_path in zip(gpu_indexes, unit_paths):
            unit_path.write_text(systemd_unit_text(args, gpu), encoding="utf-8")
    except OSError as exc:
        print(f"cannot write systemd unit: {exc}", file=sys.stderr)
        return 1

    try:
        reload_result = subprocess.run(
            systemd_user_command("daemon-reload"),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"systemd daemon-reload failed: {exc}", file=sys.stderr)
        return 1
    if reload_result.returncode != 0:
        detail = reload_result.stderr.strip() or reload_result.stdout.strip()
        print(f"systemd daemon-reload failed: {detail}", file=sys.stderr)
        return 1

    started: list[str] = []
    for gpu in gpu_indexes:
        unit = systemd_unit_for(gpu)
        # Re-enable with the current reservation parameters when called again.
        ok, detail = systemd_control(unit, "enable", "--now")
        if not ok:
            for previous in started:
                systemd_control(previous, "disable", "--now")
            print(f"GPU {gpu} systemd guard failed: {detail}", file=sys.stderr)
            return 1
        started.append(unit)

    pending = set(gpu_indexes)
    deadline = time.monotonic() + 30
    while pending and time.monotonic() < deadline:
        ready: set[int] = set()
        for gpu in pending:
            try:
                record = read_record(pid_file_for(gpu, args.pid_file))
            except RuntimeError:
                record = None
            if (
                record
                and record.get("systemd_unit") == systemd_unit_for(gpu)
                and process_exists(record["pid"])
            ):
                ready.add(gpu)
        pending -= ready
        if pending:
            time.sleep(0.1)
    if pending:
        for unit in started:
            systemd_control(unit, "disable", "--now")
        indexes = ", ".join(map(str, sorted(pending)))
        print(
            f"GPU {indexes} systemd guard did not become ready; "
            "check /tmp/gpu_memory_keeper_gpu<index>.log",
            file=sys.stderr,
        )
        return 1

    print(
        "Systemd guard started for GPU(s): "
        + ", ".join(map(str, gpu_indexes))
        + " (independent per-GPU services)"
    )
    return 0


def update_systemd_guard_unit(args: argparse.Namespace, gpu: int) -> tuple[bool, str | None]:
    unit_dir = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd" / "user"
    unit_path = unit_dir / systemd_unit_for(gpu)
    try:
        unit_path.write_text(systemd_unit_text(args, gpu), encoding="utf-8")
        result = subprocess.run(
            systemd_user_command("daemon-reload"),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip() or "daemon-reload failed"
    return True, None


def supervise_memory(args: argparse.Namespace) -> int:
    stop = False
    monitor_requested = False
    child: subprocess.Popen[Any] | None = None
    state_path = pid_file_for(args.gpu, args.pid_file)

    def stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    def monitor_handler(_signum: int, _frame: Any) -> None:
        nonlocal monitor_requested
        monitor_requested = True

    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGUSR2, monitor_handler)

    try:
        while not stop:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "occupy",
                "--gpu",
                str(args.gpu),
                "--foreground",
                "--supervisor-pid",
                str(os.getpid()),
                *reservation_arguments(args),
            ]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True)
            while child.poll() is None and not stop:
                if monitor_requested:
                    monitor_requested = False
                    try:
                        record = read_record(state_path)
                    except RuntimeError:
                        record = None
                    if record and apply_monitor_request(args, record):
                        child.send_signal(signal.SIGUSR2)
                time.sleep(0.2)
            exit_code = child.wait()
            if stop:
                break
            print(
                f"GPU {args.gpu} worker PID {child.pid} exited with code {exit_code}; "
                f"restarting in {args.restart_delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            deadline = time.monotonic() + args.restart_delay
            while not stop and time.monotonic() < deadline:
                time.sleep(0.2)
    finally:
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        try:
            record = read_record(state_path)
            if record and record.get("supervisor_pid") == os.getpid():
                state_path.unlink(missing_ok=True)
        except (OSError, RuntimeError):
            pass
    return 0


def start_background(args: argparse.Namespace) -> int:
    gpu_indexes = args.gpus if args.gpus is not None else [args.gpu]
    if args.pid_file and len(gpu_indexes) > 1:
        print("--pid-file cannot be used with multiple GPUs", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print("PyTorch is required: python3 -m pip install torch", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA is not available; cannot reserve GPU memory", file=sys.stderr)
        return 2
    device_count = torch.cuda.device_count()
    invalid = [gpu for gpu in gpu_indexes if gpu < 0 or gpu >= device_count]
    if invalid:
        indexes = ", ".join(str(gpu) for gpu in invalid)
        print(
            f"GPU index(es) out of range: {indexes} (available: 0-{device_count - 1})",
            file=sys.stderr,
        )
        return 2

    for gpu in gpu_indexes:
        path = pid_file_for(gpu, args.pid_file)
        try:
            existing = read_record(path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if existing and process_exists(existing["pid"]):
            print(
                f"GPU {gpu} is already reserved by PID {existing['pid']}: {path}",
                file=sys.stderr,
            )
            return 1
        if existing:
            path.unlink(missing_ok=True)

    if getattr(args, "systemd_guard", False):
        return install_systemd_guard(args, gpu_indexes)

    workers: list[tuple[int, subprocess.Popen[Any], Path, Path]] = []
    for gpu in gpu_indexes:
        pid_path = pid_file_for(gpu, args.pid_file)
        log_path = DEFAULT_PID_DIR / f"gpu_memory_keeper_gpu{gpu}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "occupy",
            "--gpu",
            str(gpu),
            "--supervisor",
            "--restart-delay",
            str(args.restart_delay),
            *reservation_arguments(args),
        ]

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\nStarting GPU {gpu} reservation\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        workers.append((gpu, process, pid_path, log_path))

    pending = workers.copy()
    deadline = time.monotonic() + 30
    while pending and time.monotonic() < deadline:
        waiting: list[tuple[int, subprocess.Popen[Any], Path, Path]] = []
        for worker in pending:
            _gpu, process, pid_path, _log_path = worker
            if process.poll() is not None:
                continue
            try:
                record = read_record(pid_path)
            except RuntimeError:
                record = None
            if (
                not record
                or record.get("supervisor_pid") != process.pid
                or not process_exists(record["pid"])
            ):
                waiting.append(worker)
        pending = waiting
        if pending:
            time.sleep(0.1)

    failed = [worker for worker in workers if worker in pending or worker[1].poll() is not None]
    if failed:
        for _gpu, process, _pid_path, _log_path in workers:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _gpu, process, _pid_path, _log_path in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for gpu, _process, _pid_path, log_path in failed:
            print(f"GPU {gpu} reservation failed; see {log_path}", file=sys.stderr)
        return 1

    for gpu, process, pid_path, log_path in workers:
        record = read_record(pid_path)
        worker_pid = record["pid"] if record else "unknown"
        print(
            f"GPU {gpu} reservation started: supervisor PID {process.pid}, "
            f"worker PID {worker_pid}, log {log_path}"
        )
    return 0


def send_monitor_to_keeper(args: argparse.Namespace, enable_incremental: bool = True) -> int:
    path = pid_file_for(args.gpu, args.pid_file)
    try:
        record = read_record(path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not record or not process_exists(record["pid"]):
        print(
            f"GPU {args.gpu} must be occupied before incremental monitoring can start",
            file=sys.stderr,
        )
        return 1

    systemd_unit = record.get("systemd_unit")
    has_systemd_guard = isinstance(systemd_unit, str) and bool(systemd_unit)
    if not enable_incremental and not record.get("live_limit_updates") and not has_systemd_guard:
        print(
            "this running watchdog worker predates live limit updates; release and occupy it once",
            file=sys.stderr,
        )
        return 1

    request = {
        "total_limit_percent": args.percent,
        "keeper_limit_percent": args.keeper_percent,
        "step_percent": args.step_percent,
        "interval": args.interval,
        "enable_incremental": enable_incremental,
    }
    record["monitor_request"] = request
    try:
        write_pid_file(path, record)
    except OSError as exc:
        print(f"cannot update PID file {path}: {exc}", file=sys.stderr)
        return 1

    if has_systemd_guard:
        # Persist the new limits and the current mode across a later service restart.
        setattr(args, "memory", None)
        args.incremental = bool(record.get("dynamic_target")) or enable_incremental
        ok, detail = update_systemd_guard_unit(args, args.gpu)
        if not ok:
            record.pop("monitor_request", None)
            try:
                write_pid_file(path, record)
            except OSError:
                pass
            print(f"cannot update {systemd_unit}: {detail}", file=sys.stderr)
            return 1
        if not enable_incremental and not record.get("live_limit_updates"):
            ok, detail = systemd_control(systemd_unit, "restart")
            if not ok:
                print(f"cannot restart legacy worker {systemd_unit}: {detail}", file=sys.stderr)
                return 1
            deadline = time.monotonic() + args.wait_seconds
            while time.monotonic() < deadline:
                try:
                    updated = read_record(path)
                except RuntimeError:
                    updated = None
                if (
                    updated
                    and updated.get("live_limit_updates")
                    and process_exists(updated["pid"])
                ):
                    print(f"GPU {args.gpu} worker restarted with live limit support")
                    return 0
                time.sleep(0.05)
            print(
                f"timed out after {args.wait_seconds:g}s restarting GPU {args.gpu} worker",
                file=sys.stderr,
            )
            return 1
    supervisor_pid = record.get("supervisor_pid")
    supervisor_alive = isinstance(supervisor_pid, int) and process_exists(supervisor_pid)
    target_pid = supervisor_pid if supervisor_alive else record["pid"]
    try:
        os.kill(target_pid, signal.SIGUSR2)
    except OSError as exc:
        print(f"cannot signal PID {target_pid}: {exc}", file=sys.stderr)
        return 1

    action = "monitoring request" if enable_incremental else "live limit update"
    print(f"GPU {args.gpu} {action} sent to PID {target_pid}")
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        try:
            updated = read_record(path)
        except RuntimeError:
            updated = None
        if updated and "monitor_request" not in updated and process_exists(updated["pid"]):
            limits_applied = (
                updated.get("limit_percent") == args.percent
                and updated.get("keeper_limit_percent") == args.keeper_percent
            )
            mode_applied = bool(updated.get("dynamic_target")) if enable_incremental else True
            if limits_applied and mode_applied:
                if enable_incremental:
                    print(f"GPU {args.gpu} is now monitoring released memory")
                else:
                    print(f"GPU {args.gpu} live limits are active")
                return 0
        time.sleep(0.05)
    print(
        f"timed out after {args.wait_seconds:g}s waiting for GPU {args.gpu} to apply settings",
        file=sys.stderr,
    )
    return 1


def send_to_keeper(args: argparse.Namespace, signum: int) -> int:
    path = pid_file_for(args.gpu, args.pid_file)
    try:
        record = read_record(path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not record:
        unit = systemd_unit_for(args.gpu)
        if systemd_guard_present(unit):
            ok, detail = systemd_control(unit, "disable", "--now")
            if not ok:
                print(f"cannot stop systemd guard {unit}: {detail}", file=sys.stderr)
                return 1
            print(f"Systemd guard stopped and disabled: {unit}")
            print(f"GPU {args.gpu} memory is available")
            return 0
        print(f"GPU {args.gpu} is not reserved (no PID file: {path})")
        return 0
    worker_pid = record["pid"]
    systemd_unit = record.get("systemd_unit")
    if isinstance(systemd_unit, str) and systemd_unit:
        ok, detail = systemd_control(systemd_unit, "disable", "--now")
        if not ok:
            print(
                f"cannot stop systemd guard {systemd_unit}: {detail}",
                file=sys.stderr,
            )
            return 1
        print(f"Systemd guard stopped and disabled: {systemd_unit}")
        deadline = time.monotonic() + args.wait_seconds
        while time.monotonic() < deadline:
            if not process_exists(worker_pid):
                path.unlink(missing_ok=True)
                print(f"GPU {args.gpu} memory is available")
                return 0
            time.sleep(0.05)
        print(
            f"timed out after {args.wait_seconds:g}s waiting for PID {worker_pid} to exit",
            file=sys.stderr,
        )
        return 1

    supervisor_pid = record.get("supervisor_pid")
    supervisor_alive = isinstance(supervisor_pid, int) and process_exists(supervisor_pid)
    if not process_exists(worker_pid) and not supervisor_alive:
        path.unlink(missing_ok=True)
        print(f"Removed stale PID file: {path}")
        return 0
    target_pid = supervisor_pid if supervisor_alive else worker_pid
    target_signal = signal.SIGTERM if supervisor_alive else signum
    try:
        os.kill(target_pid, target_signal)
    except OSError as exc:
        print(f"cannot signal PID {target_pid}: {exc}", file=sys.stderr)
        return 1
    action = "Released" if signum == signal.SIGUSR1 else "Stopped"
    print(f"{action} request sent to PID {target_pid} (GPU {args.gpu})")
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if not process_exists(worker_pid) and not process_exists(target_pid):
            path.unlink(missing_ok=True)
            print(f"GPU {args.gpu} memory is available")
            return 0
        time.sleep(0.05)
    print(
        f"timed out after {args.wait_seconds:g}s waiting for PID {target_pid} to exit",
        file=sys.stderr,
    )
    return 1


def show_status(args: argparse.Namespace) -> int:
    metrics, metrics_error = query_gpu_metrics()
    if metrics_error:
        print(f"GPU metrics unavailable: {metrics_error}", file=sys.stderr)
    metric = next((item for item in metrics if item["gpu"] == args.gpu), None)
    if metric:
        memory_percent = metric["used_mib"] / metric["total_mib"] * 100
        print(f"GPU {args.gpu}: {metric['name']}")
        print(
            f"  VRAM: {metric['used_mib']} / {metric['total_mib']} MiB "
            f"({memory_percent:.1f}%)"
        )
        print(f"  GPU utilization: {metric['utilization']}%")
    elif not metrics_error:
        print(f"GPU index {args.gpu} was not found", file=sys.stderr)

    path = pid_file_for(args.gpu, args.pid_file)
    try:
        record = read_record(path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not record:
        print("  Keeper reservation: none")
        return 1 if metrics_error or metric is None else 0
    pid = record["pid"]
    if not process_exists(pid):
        print(f"  Keeper reservation: stale PID {pid} ({path})")
        return 1
    print(
        f"  Keeper reservation: {format_memory(record['requested_bytes'])}, "
        f"PID {pid}"
    )
    if record.get("systemd_unit"):
        print(f"  Systemd guard: {record['systemd_unit']}")
    return 1 if metrics_error or metric is None else 0


def list_all(json_output: bool = False) -> int:
    records, errors = discover_records()
    metrics, metrics_error = query_gpu_metrics()
    if metrics_error:
        errors.append(f"GPU metrics unavailable: {metrics_error}")

    records_by_gpu = {record["gpu"]: (path, record) for path, record in records}
    metrics_by_gpu = {metric["gpu"]: metric for metric in metrics}
    gpu_indexes = sorted(set(records_by_gpu) | set(metrics_by_gpu))
    gpu_data: list[dict[str, Any]] = []
    for gpu in gpu_indexes:
        metric = metrics_by_gpu.get(gpu)
        reservation_data = None
        reservation = records_by_gpu.get(gpu)
        if reservation:
            path, record = reservation
            reservation_data = {
                "active": process_exists(record["pid"]),
                "pid": record["pid"],
                "supervisor_pid": record.get("supervisor_pid"),
                "supervisor_active": (
                    isinstance(record.get("supervisor_pid"), int)
                    and process_exists(record["supervisor_pid"])
                ),
                "systemd_guard": bool(record.get("systemd_unit")),
                "systemd_unit": record.get("systemd_unit"),
                "requested_bytes": record["requested_bytes"],
                "target_bytes": record.get("target_bytes", record["requested_bytes"]),
                "mode": record.get("mode", "immediate"),
                "dynamic_target": bool(record.get("dynamic_target")),
                "limit_percent": record.get("limit_percent"),
                "keeper_limit_percent": record.get("keeper_limit_percent"),
                "state_file": str(path),
                "log_file": str(DEFAULT_PID_DIR / f"gpu_memory_keeper_gpu{gpu}.log"),
            }
        gpu_data.append(
            {
                "index": gpu,
                "name": metric["name"] if metric else None,
                "memory_used_mib": metric["used_mib"] if metric else None,
                "memory_total_mib": metric["total_mib"] if metric else None,
                "utilization_percent": metric["utilization"] if metric else None,
                "reservation": reservation_data,
            }
        )

    if json_output:
        print(json.dumps({"gpus": gpu_data, "errors": errors}))
        return 1 if errors else 0

    for error in errors:
        print(error, file=sys.stderr)
    if not gpu_indexes:
        print("No NVIDIA GPUs or GPU memory reservations found")
        return 1 if errors else 0

    print(
        f"{'GPU':>3}  {'NAME':<30}  {'VRAM USED/TOTAL':>20}  "
        f"{'MEM%':>6}  {'UTIL%':>6}  KEEPER RESERVATION"
    )
    active_bytes = 0
    active_count = 0
    for item in gpu_data:
        gpu = item["index"]
        if item["name"]:
            name = item["name"][:30]
            vram = f"{item['memory_used_mib']} / {item['memory_total_mib']} MiB"
            memory_percent = f"{item['memory_used_mib'] / item['memory_total_mib'] * 100:.1f}%"
            utilization = f"{item['utilization_percent']}%"
        else:
            name, vram, memory_percent, utilization = "unavailable", "N/A", "N/A", "N/A"

        reservation = item["reservation"]
        if reservation:
            requested = reservation["requested_bytes"]
            if reservation["active"]:
                active_count += 1
                active_bytes += requested
                guard = " [systemd]" if reservation.get("systemd_guard") else ""
                keeper = f"{format_memory(requested)}, PID {reservation['pid']}{guard}"
            else:
                keeper = f"stale PID {reservation['pid']} ({reservation['state_file']})"
        else:
            keeper = "-"
        print(
            f"{gpu:>3}  {name:<30}  {vram:>20}  "
            f"{memory_percent:>6}  {utilization:>6}  {keeper}"
        )
    print(f"Active: {active_count}, reserved: {format_memory(active_bytes)}")
    return 1 if errors else 0


def release_all(args: argparse.Namespace) -> int:
    records, errors = discover_records()
    for error in errors:
        print(error, file=sys.stderr)

    guard_units = set(discover_systemd_guard_units())
    stopped_guard_count = 0
    pending: list[tuple[Path, dict[str, Any], int]] = []
    for path, record in records:
        worker_pid = record["pid"]
        systemd_unit = record.get("systemd_unit")
        if isinstance(systemd_unit, str) and systemd_unit:
            guard_units.discard(systemd_unit)
            ok, detail = systemd_control(systemd_unit, "disable", "--now")
            if not ok:
                message = f"cannot stop systemd guard {systemd_unit}: {detail}"
                errors.append(message)
                print(message, file=sys.stderr)
                continue
            stopped_guard_count += 1
            pending.append((path, record, worker_pid))
            continue
        supervisor_pid = record.get("supervisor_pid")
        supervisor_alive = isinstance(supervisor_pid, int) and process_exists(supervisor_pid)
        if not process_exists(worker_pid) and not supervisor_alive:
            path.unlink(missing_ok=True)
            continue
        target_pid = supervisor_pid if supervisor_alive else worker_pid
        target_signal = signal.SIGTERM if supervisor_alive else signal.SIGUSR1
        try:
            os.kill(target_pid, target_signal)
        except OSError as exc:
            message = f"cannot signal PID {target_pid}: {exc}"
            errors.append(message)
            print(message, file=sys.stderr)
            continue
        pending.append((path, record, target_pid))

    for unit in sorted(guard_units):
        if not systemd_guard_present(unit):
            continue
        ok, detail = systemd_control(unit, "disable", "--now")
        if not ok:
            message = f"cannot stop systemd guard {unit}: {detail}"
            errors.append(message)
            print(message, file=sys.stderr)
            continue
        stopped_guard_count += 1

    if not pending:
        if stopped_guard_count:
            print("All reserved GPU memory is available")
        else:
            print("No active GPU memory reservations")
        return 1 if errors else 0

    print(f"Release request sent to {len(pending)} reservation(s)")
    deadline = time.monotonic() + args.wait_seconds
    while pending and time.monotonic() < deadline:
        still_running: list[tuple[Path, dict[str, Any], int]] = []
        for path, record, target_pid in pending:
            if process_exists(record["pid"]) or process_exists(target_pid):
                still_running.append((path, record, target_pid))
            else:
                path.unlink(missing_ok=True)
        pending = still_running
        if pending:
            time.sleep(0.05)

    if pending:
        pids = ", ".join(str(target_pid) for _, _record, target_pid in pending)
        print(
            f"timed out after {args.wait_seconds:g}s waiting for PID(s): {pids}",
            file=sys.stderr,
        )
        return 1
    print("All reserved GPU memory is available")
    return 1 if errors else 0


def interactive_menu() -> int:
    try:
        while True:
            print()
            list_all()
            metrics, metrics_error = query_gpu_metrics()
            if metrics_error:
                print(f"Cannot open manager: {metrics_error}", file=sys.stderr)
                return 1
            available = {metric["gpu"] for metric in metrics}
            if not available:
                print("No NVIDIA GPUs found", file=sys.stderr)
                return 1

            print()
            print(
                f"1) Occupy GPU(s) in background "
                f"(total cap: {DEFAULT_PERCENT:g}%, Keeper cap: {DEFAULT_KEEPER_PERCENT:g}%)"
            )
            print("2) Release selected GPU(s)")
            print("3) Release all GPU reservations")
            print("0) Exit")
            action = input("Select action: ").strip()
            if action in {"0", "q", "quit", "exit"}:
                return 0
            if action == "3":
                release_all(argparse.Namespace(wait_seconds=10.0))
                continue
            if action not in {"1", "2"}:
                print("Invalid action")
                continue

            raw_indexes = input("GPU indexes (comma-separated, e.g. 0,1): ").strip()
            try:
                gpu_indexes = parse_gpu_indexes(raw_indexes)
            except argparse.ArgumentTypeError as exc:
                print(str(exc), file=sys.stderr)
                continue
            invalid = [gpu for gpu in gpu_indexes if gpu not in available]
            if invalid:
                print(
                    f"GPU index(es) not available: {', '.join(map(str, invalid))}",
                    file=sys.stderr,
                )
                continue

            if action == "1":
                start_background(
                    argparse.Namespace(
                        gpu=0,
                        gpus=gpu_indexes,
                        pid_file=None,
                        memory=None,
                        percent=DEFAULT_PERCENT,
                        keeper_percent=DEFAULT_KEEPER_PERCENT,
                        incremental=False,
                        step_percent=10.0,
                        interval=5.0,
                        restart_delay=3.0,
                    )
                )
            else:
                for gpu in gpu_indexes:
                    send_to_keeper(
                        argparse.Namespace(gpu=gpu, pid_file=None, wait_seconds=10.0),
                        signal.SIGUSR1,
                    )
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
        subparser.add_argument("--pid-file", help="override the PID file path (excluded from global commands)")

    occupy = subparsers.add_parser("occupy", help="reserve memory in background")
    gpu_selection = occupy.add_mutually_exclusive_group()
    gpu_selection.add_argument("--gpu", type=int, default=0, help="single CUDA device index (default: 0)")
    gpu_selection.add_argument("--gpus", type=parse_gpu_indexes, help="comma-separated CUDA indexes, e.g. 0,1,2")
    occupy.add_argument("--pid-file", help="override the PID file path (single GPU only)")
    occupy.add_argument("--foreground", action="store_true", help="keep a single-GPU worker in the foreground")
    occupy.add_argument("--incremental", action="store_true", help="grow the reservation in configurable steps")
    occupy.add_argument(
        "--step-percent",
        type=parse_percent,
        default=10.0,
        help="monitor step as a percentage of total VRAM (default: 10)",
    )
    occupy.add_argument("--interval", type=float, default=5.0, help="seconds between increments (default: 5)")
    occupy.add_argument("--restart-delay", type=float, default=3.0, help="seconds before restarting a killed worker (default: 3)")
    occupy.add_argument(
        "--systemd-guard",
        action="store_true",
        help="run each selected GPU as a persistent systemd --user service",
    )
    occupy.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    occupy.add_argument("--supervisor-pid", type=int, help=argparse.SUPPRESS)
    occupy.add_argument("--systemd-unit", help=argparse.SUPPRESS)
    amount = occupy.add_mutually_exclusive_group()
    amount.add_argument("--memory", type=parse_size, help="exact amount to reserve, e.g. 8G or 512M")
    amount.add_argument(
        "--percent",
        type=parse_percent,
        default=DEFAULT_PERCENT,
        help="total GPU usage cap across all processes (default: 99)",
    )
    occupy.add_argument(
        "--keeper-percent",
        type=parse_percent,
        default=DEFAULT_KEEPER_PERCENT,
        help="maximum VRAM reserved by Keeper per GPU (default: 80)",
    )

    release = subparsers.add_parser("release", help="release memory and wait until it is available")
    add_common(release)
    release.add_argument("--wait-seconds", type=float, default=10.0, help="wait for memory to be released (default: 10)")
    stop = subparsers.add_parser("stop", help="stop the keeper process")
    add_common(stop)
    stop.add_argument("--wait-seconds", type=float, default=10.0, help="wait for the keeper to exit (default: 10)")
    monitor = subparsers.add_parser("monitor", help="switch an existing reservation to dynamic release monitoring")
    add_common(monitor)
    monitor.add_argument("--percent", type=parse_percent, default=DEFAULT_PERCENT, help="total GPU usage cap (default: 99)")
    monitor.add_argument("--keeper-percent", type=parse_percent, default=DEFAULT_KEEPER_PERCENT, help="Keeper VRAM cap per GPU (default: 80)")
    monitor.add_argument("--step-percent", type=parse_percent, default=10.0, help="monitor step as a percentage of total VRAM (default: 10)")
    monitor.add_argument("--interval", type=float, default=5.0, help="seconds between release checks (default: 5)")
    monitor.add_argument("--wait-seconds", type=float, default=10.0, help="wait for monitor mode to become active (default: 10)")
    configure = subparsers.add_parser(
        "configure",
        help="apply live limits without enabling incremental monitoring",
    )
    add_common(configure)
    configure.add_argument("--percent", type=parse_percent, default=DEFAULT_PERCENT, help="total GPU usage cap (default: 99)")
    configure.add_argument("--keeper-percent", type=parse_percent, default=DEFAULT_KEEPER_PERCENT, help="Keeper VRAM cap per GPU (default: 80)")
    configure.add_argument("--step-percent", type=parse_percent, default=10.0, help="monitor step as a percentage of total VRAM (default: 10)")
    configure.add_argument("--interval", type=float, default=5.0, help="seconds between release checks (default: 5)")
    configure.add_argument("--wait-seconds", type=float, default=10.0, help="wait for live settings to become active (default: 10)")
    status = subparsers.add_parser("status", help="show reservation status")
    add_common(status)
    subparsers.add_parser("menu", help="open the interactive GPU manager")
    list_parser = subparsers.add_parser("list", help="list reservations on all GPUs")
    list_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    release_all_parser = subparsers.add_parser("release-all", help="release reservations on all GPUs")
    release_all_parser.add_argument("--wait-seconds", type=float, default=10.0, help="wait for all memory to be released (default: 10)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command is None or args.command == "menu":
        return interactive_menu()
    if args.command == "occupy":
        if args.interval <= 0:
            print("--interval must be greater than zero", file=sys.stderr)
            return 2
        if args.restart_delay < 0:
            print("--restart-delay must not be negative", file=sys.stderr)
            return 2
        if args.systemd_guard and (args.foreground or args.supervisor):
            print("--systemd-guard cannot be combined with internal foreground modes", file=sys.stderr)
            return 2
        if args.supervisor:
            if args.gpus is not None:
                print("--supervisor cannot be used with --gpus", file=sys.stderr)
                return 2
            return supervise_memory(args)
        if args.foreground:
            if args.gpus is not None:
                print("--foreground cannot be used with --gpus", file=sys.stderr)
                return 2
            return reserve_memory(args)
        return start_background(args)
    if args.command == "release":
        if args.wait_seconds < 0:
            print("--wait-seconds must not be negative", file=sys.stderr)
            return 2
        return send_to_keeper(args, signal.SIGUSR1)
    if args.command == "stop":
        if args.wait_seconds < 0:
            print("--wait-seconds must not be negative", file=sys.stderr)
            return 2
        return send_to_keeper(args, signal.SIGTERM)
    if args.command == "monitor":
        if args.interval <= 0 or args.wait_seconds < 0:
            print("--interval must be greater than zero and --wait-seconds must not be negative", file=sys.stderr)
            return 2
        args.incremental = True
        args.memory = None
        return send_monitor_to_keeper(args)
    if args.command == "configure":
        if args.interval <= 0 or args.wait_seconds < 0:
            print("--interval must be greater than zero and --wait-seconds must not be negative", file=sys.stderr)
            return 2
        args.incremental = False
        args.memory = None
        return send_monitor_to_keeper(args, enable_incremental=False)
    if args.command == "list":
        return list_all(args.json)
    if args.command == "release-all":
        if args.wait_seconds < 0:
            print("--wait-seconds must not be negative", file=sys.stderr)
            return 2
        return release_all(args)
    return show_status(args)


if __name__ == "__main__":
    raise SystemExit(main())
