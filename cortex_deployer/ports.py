"""Pick a listen port. Windows Hyper-V often excludes mid-range ports."""

from __future__ import annotations

import errno
import socket

MIN_UNPRIVILEGED = 1024
_NEARBY = 32
_JUMPS = (18765, 17890, 19080, 23456, 28080, 34567, 45678)


def bind_unavailable(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    winerr = getattr(exc, "winerror", None)
    if winerr in {10013, 10048, 10049}:
        return True
    return exc.errno in {
        errno.EADDRINUSE,
        errno.EACCES,
        errno.EADDRNOTAVAIL,
        getattr(errno, "WSAEACCES", 10013),
        getattr(errno, "WSAEADDRINUSE", 10048),
    }


def candidate_ports(preferred: int) -> list[int]:
    """Unprivileged ports to try, then 0 so the OS assigns one."""
    seen: set[int] = set()
    out: list[int] = []

    def add(port: int) -> None:
        if port < MIN_UNPRIVILEGED or port > 65535 or port in seen:
            return
        seen.add(port)
        out.append(port)

    start = preferred if preferred >= MIN_UNPRIVILEGED else 18765
    add(start)
    for delta in range(1, _NEARBY + 1):
        add(start + delta)
    for jump in _JUMPS:
        for delta in range(0, 8):
            add(jump + delta)
    out.append(0)
    return out


def port_is_free(host: str, port: int) -> bool:
    if port < MIN_UNPRIVILEGED or port > 65535:
        return False
    listen = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((listen, port))
        return True
    except OSError as exc:
        if bind_unavailable(exc):
            return False
        raise


def pick_free_port(preferred: int = 0, host: str = "127.0.0.1") -> int:
    listen = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    start = int(preferred or 0)
    last: OSError | None = None
    for candidate in candidate_ports(start):
        if candidate == 0:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind((listen, 0))
                    return int(sock.getsockname()[1])
            except OSError as exc:
                last = exc
                continue
        try:
            if port_is_free(listen, candidate):
                return candidate
        except OSError as exc:
            last = exc
            continue
    raise OSError(f"no free listen port on {listen}: {last or 'exhausted'}")


def argv_port(argv: list[str] | tuple[str, ...]) -> int:
    args = list(argv)
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            try:
                return int(args[idx + 1])
            except ValueError:
                return 0
    return 0


def set_argv_port(argv: list[str] | tuple[str, ...], port: int) -> list[str]:
    out = list(argv)
    if "--port" in out:
        idx = out.index("--port")
        if idx + 1 < len(out):
            out[idx + 1] = str(int(port))
            return out
    out.extend(["--port", str(int(port))])
    return out


def bind_error_in_log(text: str) -> bool:
    low = (text or "").lower()
    return "couldn't bind" in low or "could not bind" in low or "address already in use" in low


def summarize_crash(text: str, *, limit: int = 8) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    tail = lines[-limit:] if lines else ["engine exited"]
    return "\n".join(tail)
