# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity: does a distributed tt-metal job end, end cleanly, and report it, after one failure?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. Every
test is one fault with one assertion, taken from the teardown plan of tt-metal #56902 (tt-blaze
tools/teardown_probe, lane L4): the rank program is ``sanity_probe.py`` next to this file. The faults are the
two kinds a real job meets: the code failing on one rank (an exception, a shape the descriptor cannot hold, a
descriptor that disagrees, a collective with a wrong or dead peer) and the user (a signal at tt-run).

Requirements, as in the plan: a failure never becomes a hang (R1), a hang ends by itself and leaves nothing
(R2), nobody kills processes by hand (R3), the exit code reports the failure (R4).

Modes. A row that passes today is a gate. A row that fails today is marked ``@known("P..: ...")`` with the
plan's error id and fix: it fails, so the job is red until the fix lands, and its failure message names the
error; when the fix lands the row passes and the marker is dropped. Each row has a deadline; a job that is still running at the
deadline is killed by its process group and counted as a hang.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_sanity -v   (from the repository root, with a
built tree and the python environment; the fabric CPU lane runs it as group ``teardown-sanity`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # sanity_common lives next to this file, whatever the import mode
from sanity_common import (
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    conflict_mapping,
    ended,
    fault_delivered,
    legacy_cmd,
    nonzero,
    rc_contains,
    run_case,
    solve_cmd,
    MGD_QUAD,
    PYTHON,
)


# ---------------------------------------------------------------------------------------------------------------
# rows that pass today (gates)
# ---------------------------------------------------------------------------------------------------------------
def test_clean():
    """Control: both ranks open the mock galaxy, one barrier, close, exit 0, nothing left."""
    r = run_case("clean", {}, DEADLINE_END)
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_dead_peer_barrier():
    """Default launcher (no ULFM): rank 1 dies before a barrier, so the whole job ends with a non-zero code and nothing
    is left. A hung (not dead) peer is the per-collective-timeout case instead (test_barrier_skipped_ends, P16)."""
    r = run_case("dead_peer_barrier", {"PROBE_COLL": "barrier_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_dead_peer_recv():
    """Default launcher: rank 1 dies before rank 0's recv, so the job ends non-zero instead of rank 0 waiting forever."""
    r = run_case("dead_peer_recv", {"PROBE_COLL": "recv_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_fabric_mismatch():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_prte_timeout():
    """A send nobody receives under PRRTE --timeout 60: the wall clock ends the job with 241 and nothing left.

    The timeout is longer than the ttnn import on a loaded runner, and the row checks that both ranks were past
    open when the clock fired, so it is the deadlock that was ended and not the start-up."""
    r = run_case(
        "send_no_recv+prte_timeout60", {"PROBE_COLL": "send_no_recv"}, DEADLINE_END + 60, cmd=legacy_cmd("--timeout 60")
    )
    ended(r)
    assert r.exit == 241, str(r)
    # Require positive evidence that the wall clock ended the intended deadlock, not a slow start-up: both ranks
    # must have reached the collective section (a ``coll_*`` marker), past open — not merely "not a startup marker",
    # which would also accept ``open_enter`` (entered open, never completed).
    for rank in (0, 1):
        assert r.reached.get(rank, "").startswith(
            "coll"
        ), f"rank {rank} had not reached the collective when the wall clock fired: {r}"
    clean(r)


def test_exit_nofinalize():
    """Rank 1 _exits with the mesh open (the crash path, no finalize): the job ends non-zero on the production line."""
    r = run_case(
        "exit_nofinalize_after_open", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_invalid_rank():
    """A receive from a rank that does not exist raises at once; the job goes on and ends."""
    r = run_case("recv_bad_rank", {"PROBE_COLL": "recv_bad_rank", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "RuntimeError")
    clean(r)


def test_use_before_init():
    """A collective before init_distributed_context raises 'not initialized' instead of crashing or hanging."""
    r = run_case("use_before_init", {"PROBE_SCENARIO": "use_before_init", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 1, "not initialized")
    assert r.exit == 0, str(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# rows that fail today. Each marker names the error of the plan and its fix; drop the marker with the fix.
# ---------------------------------------------------------------------------------------------------------------
def test_raise_after_open():
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


def test_bad_shape():
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


def test_bad_descriptor(tmp_path):
    r = run_case("conflicting_descriptor", {}, DEADLINE_HANG, cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)))
    ended(r)
    nonzero(r)
    clean(r)


def test_barrier_timeout():
    """P16, the per-collective timeout: rank 1 never joins the barrier, so rank 0 would wait forever. With
    TT_METAL_OPERATION_TIMEOUT_SECONDS set, the barrier gives up at the deadline and tears the whole job down (both
    ranks end, nothing left), rather than one rank raising while the peer keeps waiting. Ends well before the hang
    deadline, which proves the operation timeout fired, not the harness kill."""
    r = run_case(
        "barrier_skip", {"PROBE_COLL": "barrier_skip", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_send_timeout():
    """P16 on send: a send nobody receives gives up at TT_METAL_OPERATION_TIMEOUT_SECONDS and the job is torn down."""
    r = run_case(
        "send_no_recv", {"PROBE_COLL": "send_no_recv", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_sigkill_before_init():
    r = run_case("sigkill_before_init", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "before_init"}, DEADLINE_HANG)
    ended(r)
    clean(r)


def test_dead_peer_in_open():
    # A peer dies *inside* open_mesh_device (the control plane's distributed init). The kill is sent by the parent
    # harness, gated on the victim's ``open_enter`` marker, because open holds the GIL — a Python timer set in the
    # rank could not fire inside open. The job must end non-zero with nothing left.
    import signal

    r = run_case(
        "sigkill_in_open",
        {"PROBE_SCENARIO": "clean", "PROBE_VICTIM": "1"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGKILL, 0.2),
        signal_on_marker="open_enter",
    )
    ended(r)
    clean(r)
    nonzero(r)
    assert "FAULT NOT DELIVERED" not in r.signal_sent, r  # the victim really was killed inside open


def test_ctrl_c_in_deadlock():
    """SIGINT at tt-run while the ranks sit in the send/barrier deadlock: tt-run ends and nothing may be left."""
    r = run_case(
        "ctrl_c_in_deadlock",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "coll_ready"),  # wait until the ranks reach the collective, not a 45 s wall-clock guess
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


def test_ctrl_c_in_init():
    """SIGINT at tt-run while rank 0 waits in MPI_Init for a rank 1 that starts 75 s late: nothing may be left."""
    r = run_case(
        "ctrl_c_in_init",
        {"PROBE_START_DELAY_S": "75", "PROBE_START_DELAY_RANK": "1"},
        DEADLINE_END + 30,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "import_done"),  # rank 0 is up and about to block in init waiting for the late rank 1
        signal_after=5.0,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


def test_sigterm_at_ttrun():
    r = run_case(
        "sigterm_at_ttrun",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_HANG,
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


def test_solve_rank_killed():
    r = run_case("solve_rank_killed", {}, DEADLINE_HANG, cmd=solve_cmd(), phase1_kill=True)
    fault_delivered(r)  # the phase-1 (generate_rank_bindings) process really was found and killed
    ended(r)
    nonzero(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the rest of the plan's open errors that one host can show
# ---------------------------------------------------------------------------------------------------------------
def rank_saw(r, rank: int, marker: str) -> None:
    """The rank printed this marker at some point (not necessarily its last one)."""
    assert any(
        f"[probe r={rank} " in line and marker in line for line in r.log.splitlines()
    ), f"rank {rank} never printed {marker!r}; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"


def test_survivor_closes_ulfm():
    """Opt-in ULFM so the death is delivered as an error (the only mode in which the survivor runs on); rank 1 killed
    with the mesh open, rank 0 records the error and goes on to close: its close must complete."""
    r = run_case(
        "survivor_close_ulfm",
        {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open", "PROBE_ON_ERROR": "continue"},
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    rank_saw(r, 0, "close_done")
    ended(r)
    clean(r)


def test_abort_core_dump():
    """P8's shape: rank 1 aborts (SIGABRT) with the mesh open and the core limit raised, so the kernel writes a core
    that can take tens of seconds. The job must still end within 20 s with a non-zero code — it must not wait for the
    dying rank's core to finish. The fix is two-fold: the DistributedContext's own error path ends with ``_Exit``
    rather than ``std::abort`` (no core on that path), and the survivor's per-collective operation timeout tears the
    whole job down (``MPI_Abort`` → the launcher SIGKILLs the still-dumping rank), so teardown no longer waits on the
    core write however long it takes."""
    r = run_case(
        "abort_with_core",
        {
            "PROBE_SCENARIO": "abort",
            "PROBE_PHASE": "after_open",
            "PROBE_CORE_UNLIMITED": "1",
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8",
        },
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    assert r.wall_s < 20, f"P8: the aborting rank's core dump held the job for {r.wall_s:.0f} s\n{r.log[-2000:]}"
    clean(r)


def test_unsat_mgd():
    """A quad-galaxy descriptor asked of the dual-galaxy mock: phase 1 cannot place it and must end non-zero."""
    r = run_case("unsat_mgd", {}, DEADLINE_HANG, cmd=solve_cmd(mgd=MGD_QUAD))
    ended(r)
    nonzero(r)
    clean(r)


def test_late_rank():
    """P10: a peer that never joins the MPI wire-up hangs rank 0 inside MPI_Init, which the per-collective operation
    timeout cannot reach (MPI_Init is not a DistributedContext collective). Here rank 1 starts 20 s late. With
    TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS=5 rank 0's MPI_Init gives up at 5 s and exits, and the launcher ends the
    job — well before the 20 s late start, instead of rank 0 waiting the whole time."""
    r = run_case(
        "late_rank_init_timeout",
        {"PROBE_START_DELAY_S": "20", "PROBE_START_DELAY_RANK": "1", "TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS": "3"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    # Bounded: rank 0 gives up on MPI_Init at 3 s and exits; the rest is the launcher tearing the late rank down.
    # Well under the ~35 s it takes when rank 0 waits out the 20 s late start (the unbounded behaviour this fixes).
    assert (
        r.wall_s < 18
    ), f"rank 0 waited {r.wall_s:.0f} s for the late rank with the init timeout at 3 s\n{r.log[-2000:]}"
    clean(r)


def test_shape_mismatch():
    r = run_case("shape_mismatch", {"PROBE_MESH_SHAPE_VICTIM": "4x4"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_allgather_timeout():
    """The user's example on all_gather: rank 1 never joins, so the others would wait in distributed_context_allgather_int
    forever. With the operation timeout set, the call gives up at the deadline and aborts the job (every rank ends),
    not a local raise that leaves the peers waiting."""
    r = run_case(
        "allgather_missing_peer",
        {"PROBE_COLL": "allgather_skip", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_recv_timeout():
    """P16 on recv: rank 0 waits in recv for a peer that never sends; the per-call timeout cancels it at the deadline
    and tears the job down."""
    r = run_case(
        "recv_without_send", {"PROBE_COLL": "recv_no_send", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_slow_collective_ok():
    """The timeout must fire only on a hang, not on a legitimately slow collective. Rank 1 joins the barrier 4 s late
    with the operation timeout at 30 s: the barrier completes and the job exits 0. A per-call timeout that is too eager
    would tear this down, so this gate guards against that."""
    r = run_case(
        "slow_collective_ok",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "2", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "10"},
        DEADLINE_END,
    )
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_singleton_shm(tmp_path):
    """A tt-metal process nobody launched inits the distributed context as an MPI singleton. Open MPI would leave
    both a 16 MB shared-memory segment and its session directory (``/tmp/ompi.<host>.<uid>/...``) behind, because the
    launcher that reaps them never ran (#63870). The DistributedContext init asks for the ``self`` transport only (no
    shm segment) and steers the session directory into a private ``TMPDIR`` it removes at exit (nothing left). This is
    the less-invasive counterpart of tt-metal #63954, which fixes the same leak by not initialising MPI at all for a
    singleton; both are carried in this PR. Slow, because it imports ttnn; kept in Python like the rest of the suite.

    ``TMPDIR`` is pointed at a fresh directory so the session-dir check is scoped to this job (the shared
    ``/tmp/ompi.*`` tree can hold a concurrent job's subdirs)."""
    sess = tmp_path / "mpi"
    sess.mkdir()
    r = run_case(
        "singleton",
        {"TMPDIR": str(sess)},
        DEADLINE_END,
        cmd=[PYTHON, "-c", "import ttnn; ttnn.init_distributed_context()"],
    )
    ended(r)
    assert r.exit == 0, str(r)
    assert not r.shm_leaked, f"a tt-metal singleton left a shm segment {r.shm_leaked} (#63870)"
    session_dirs = sorted(p.name for p in sess.iterdir() if p.name.startswith(("tt_mpi_singleton", "ompi.")))
    assert not session_dirs, f"a tt-metal singleton left an Open MPI session directory {session_dirs} (#63870)"


# ---------------------------------------------------------------------------------------------------------------
# gaps the launcher covers today that the DistributedContext should own (plan P23-P25). Each row fails today and is a
# strict xfail naming the error it exposes, so the suite stays green and the row turns red (XPASS) the day the fix
# lands, which is the signal to drop the marker. The TODO at the line that should own the fix names the row.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="P23: a SIGINT to one rank only hangs its peers; fix: the Python abort hook treats KeyboardInterrupt as an error exit (abort 130)",
)
def test_sigint_one_rank():
    """SIGINT reaches rank 1 only (a user interrupts one pid, a debugger detaches): KeyboardInterrupt, the rank
    leaves with 130 and finalizes, its peer waits in the collective. The job must end non-zero with nothing left.
    Today the abort hook in distributed_nanobind.cpp excludes KeyboardInterrupt, so nothing ends the peer."""
    r = run_case("sigint_one_rank", {"PROBE_SCENARIO": "sigint", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P24: a rank's helper outlives it; fix: the rank's exit path ends its own process group (clean exit and abort alike)",
)
def test_helper_outlives_rank():
    """Rank 1 forks a helper that ignores SIGTERM, then finishes its work and exits cleanly. On an abort prterun kills
    the rank's whole process group, helper included; on a clean exit it only reaps the rank, so the helper lives on.
    Today only tt-run's session sweep ends it, on the launch host, after the launcher returned, and turns the clean
    job into exit 1; a direct mpirun or a remote host keeps it. The row asserts the helper is gone and the job is
    clean without tt-run having to sweep."""
    r = run_case("helper_outlives_rank", {"PROBE_SCENARIO": "fork_helper", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    clean(r)
    assert "process(es) of its job alive" not in r.log, (
        "the helper was ended by tt-run's sweep, not by the rank's own exit path\n" + r.log[-2000:]
    )
    assert r.exit == 0, str(r)


@pytest.mark.xfail(
    strict=True,
    reason="P25: a killed launcher leaves its session directory under /tmp; fix: tt-run sweeps this user's unowned launcher session dirs next to the shm sweep",
)
def test_launcher_killed_no_session_dir():
    """kill -9 of prterun while the ranks sit in a collective: tt-run ends the orphaned ranks (#65121), but prterun's
    exit path never ran, so its session directory under /tmp stays. Nothing may be left, directories included."""
    r = run_case(
        "launcher_killed",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
        kill_launcher=True,
    )
    fault_delivered(r)
    ended(r)
    nonzero(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the remaining gaps only the DistributedContext can close (plan P26-P30): one strict xfail each, a TODO at the owner
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="P26: a rank that exits non-zero without an exception (sys.exit(1)) hangs its peers, or the job ends 0 and the code is lost; fix: a SystemExit hook beside the excepthook aborts the job with that code",
)
def test_exit1_hangs_peer():
    """Rank 1 calls sys.exit(1) after open, the way click or a script's error return does. No exception reaches
    the hooks, the atexit finalize fences on the peers, and rank 0 waits in its collective forever."""
    r = run_case("exit1_after_open", {"PROBE_SCENARIO": "exit1", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P27: a rank that exits cleanly while its peers still work hangs them; fix: a bounded finalize (MPI_Ibarrier with the operation timeout) that aborts naming the early rank",
)
def test_early_exit0_hangs_peer():
    """Rank 1 returns early with exit 0 (a logic bug) while rank 0 enters the barrier. Nothing is an error, so
    nothing aborts: the finalize fence and the collective wait on each other. The job must still end."""
    r = run_case("exit0_after_open", {"PROBE_SCENARIO": "exit0", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P28: SIGTERM from the launcher runs no device teardown; fix: a SIGTERM handler that runs the MetalContext teardown before the rank dies (the simulator's path, generalised)",
)
def test_sigterm_runs_close():
    """SIGTERM reaches rank 1 with its mesh open (what the launcher sends every survivor when a job ends). The rank
    must close its device on the way out: on the mock that is the close_done marker; on hardware it is firmware
    back at base and the routers stopped (R5). Today the default disposition kills it where it stands."""
    r = run_case(
        "sigterm_with_mesh_open",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "8"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGTERM, 0.5),
        signal_on_marker="coll_ready",
    )
    fault_delivered(r)
    ended(r)
    nonzero(r)
    clean(r)
    rank_saw(r, 1, "close_done")


def test_bare_launch_dead_rank_leaves_no_shm():
    """P29 guard: tt-run --bare drops the TCP-only BTL line, so the shared-memory transport is back. A rank killed
    with the mesh open must still leave no sm_segment behind. If this row fails, the context should set the BTL
    default itself before MPI_Init instead of relying on tt-run's launch line."""
    cmd = legacy_cmd()
    cmd.insert(cmd.index(PYTHON), "--bare")
    r = run_case("bare_dead_rank", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open"}, DEADLINE_END, cmd=cmd)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P30: with no timeout set anywhere a hung collective never ends; policy: a default for TT_METAL_OPERATION_TIMEOUT_SECONDS, or tt-run setting it per rank",
)
def test_hang_without_any_timeout():
    """Rank 1 skips the barrier and nobody set a timeout: not the operation timeout, not PRRTE's --timeout. R2 says
    the job ends by itself; today both defaults are off, so it does not."""
    assert not os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS"), "this row needs the operation timeout unset"
    r = run_case("hang_no_timeout", {"PROBE_COLL": "barrier_skip"}, DEADLINE_HANG)
    ended(r)
    clean(r)
