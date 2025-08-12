"""Kernel metrics via eBPF.

This module uses the :mod:`bcc` library to gather CPU, disk and network
statistics through eBPF probes.  If the kernel or runtime does not support
loading eBPF programs the functions fall back to reading data from
``/proc`` files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from bcc import BPF

    _BPF_AVAILABLE = True
except Exception:  # pragma: no cover - bcc may be missing
    BPF = None
    _BPF_AVAILABLE = False


@dataclass
class Metrics:
    cpu: Tuple[float, float, float]
    disk: Dict[str, int]
    net: Dict[str, int]


class KernelMetrics:
    """Collect metrics using eBPF with a /proc fallback."""

    def __init__(self) -> None:
        self.available = False
        if not _BPF_AVAILABLE:
            return
        program = """
        #include <uapi/linux/ptrace.h>
        #include <linux/blkdev.h>
        #include <linux/skbuff.h>

        BPF_HASH(cpu_switch, u32, u64);
        BPF_HASH(disk_bytes, u32, u64);
        BPF_HASH(net_bytes, u32, u64);

        int trace_cpu(struct pt_regs *ctx) {
            u32 cpu = bpf_get_smp_processor_id();
            u64 zero = 0, *val;
            val = cpu_switch.lookup_or_init(&cpu, &zero);
            (*val)++;
            return 0;
        }

        int trace_disk(struct pt_regs *ctx, struct request *req) {
            u32 cpu = bpf_get_smp_processor_id();
            u64 bytes = req->__data_len;
            u64 zero = 0, *val;
            val = disk_bytes.lookup_or_init(&cpu, &zero);
            (*val) += bytes;
            return 0;
        }

        int trace_net(struct pt_regs *ctx, struct sk_buff *skb) {
            u32 cpu = bpf_get_smp_processor_id();
            u64 len = skb->len;
            u64 zero = 0, *val;
            val = net_bytes.lookup_or_init(&cpu, &zero);
            (*val) += len;
            return 0;
        }
        """
        try:
            self.bpf = BPF(text=program)
            self.bpf.attach_kprobe(event="finish_task_switch", fn_name="trace_cpu")
            self.bpf.attach_kprobe(event="blk_account_io_start", fn_name="trace_disk")
            self.bpf.attach_kprobe(event="netif_receive_skb", fn_name="trace_net")
            self.available = True
        except Exception:
            # eBPF loading failed; keep available False to use /proc fallback
            self.available = False

    def cpu(self) -> Tuple[float, float, float]:
        """Return CPU metrics.

        When eBPF is active this returns a tuple where the first element is the
        number of context switches per CPU since startup.  Otherwise it falls
        back to :func:`os.getloadavg`.
        """

        if not self.available:
            return os.getloadavg()
        total = 0
        for val in self.bpf["cpu_switch"].values():
            total += val.value
        per_cpu = total / max(os.cpu_count() or 1, 1)
        return (float(per_cpu), 0.0, 0.0)

    def disk(self) -> Dict[str, int]:
        """Return disk throughput in bytes."""

        if not self.available:
            read = write = 0
            with open("/proc/diskstats") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 10:
                        read += int(parts[5]) * 512
                        write += int(parts[9]) * 512
            return {"read": read, "write": write}
        total = 0
        for val in self.bpf["disk_bytes"].values():
            total += val.value
        return {"bytes": int(total)}

    def net(self) -> Dict[str, int]:
        """Return network throughput in bytes."""

        if not self.available:
            rx = tx = 0
            with open("/proc/net/dev") as fh:
                for line in fh:
                    if ":" not in line:
                        continue
                    data = line.split(":", 1)[1].split()
                    if len(data) >= 9:
                        rx += int(data[0])
                        tx += int(data[8])
            return {"rx": rx, "tx": tx}
        total = 0
        for val in self.bpf["net_bytes"].values():
            total += val.value
        return {"bytes": int(total)}

    def collect(self) -> Metrics:
        """Convenience wrapper returning all metrics."""

        return Metrics(cpu=self.cpu(), disk=self.disk(), net=self.net())
