"""The ISA simulator in Bend (bend/, opentpu/bend_sim.py; docs/bend.md): its laws, and its runs
against isasim's bit for bit (tests/bend_shadow.py's comparison: DRAM, TMEM, ACT RAM, registers,
instruction count, stalls, or the same error).

Without Bend 2 (~/.bend/bin/bend, or OTPU_BEND) the tests skip; with OTPU_BEND_BIN (a server
built elsewhere) only the checker's skip. pytest -p bend_shadow runs whole suites the same way."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import bend_shadow as BS
from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError

ROOT = Path(__file__).resolve().parent.parent
BEND = Path(os.environ.get("OTPU_BEND", str(Path.home() / ".bend" / "bin" / "bend")))
needs_bend = pytest.mark.skipif(not BEND.exists(), reason="no Bend 2 (OTPU_BEND)")
needs_server = pytest.mark.skipif(not BEND.exists() and not os.environ.get("OTPU_BEND_BIN"),
                                  reason="no Bend 2 and no OTPU_BEND_BIN")

# words that are not plain normals: zeros, denormals, infinities, NaNs, the edges of the range
SPECIAL = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x7F800000, 0xFF800000,
                    0x7FC00000, 0xFFC00001, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
                    0x00800000, 0x42F00000, 0xC2FE0000, 0x3F000000], np.uint32)


def check(path: str, how: str = "--check-only") -> str:
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    env["PATH"] = str(Path.home() / ".elan" / "bin") + os.pathsep + env.get("PATH", "")
    r = subprocess.run([str(BEND), str(ROOT / path), how], capture_output=True, text=True,
                       env=env, timeout=600)
    return r.stdout + r.stderr


def has_lean() -> bool:
    return (Path.home() / ".elan" / "bin" / "lean").exists() or shutil.which("lean") is not None


@pytest.fixture(scope="module", autouse=True)
def _servers():
    yield
    for m in BS._bend.values():
        m.close()
    BS._bend.clear()


def words(rng, n: int) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32).view(np.uint32)
    k = rng.random(n) < 0.04
    w[k] = SPECIAL[rng.integers(0, len(SPECIAL), k.sum())]
    return w


def both(m: Machine) -> str:
    """Run m on isasim and on the Bend machine from the same state: 'same', or 'error' when
    both stop on the same error (a mismatch raises)."""
    try:
        BS._shadow_run(m)
        return "same"
    except SimError:
        return "error"


# ------------------------------------------------------------------------------- the laws


@needs_bend
@pytest.mark.parametrize("path", ["bend/PROOF.bend", "bend/prog/PROOF.bend",
                                  "bend/cache/PROOF.bend"])
def test_laws(path):
    out = check(path)
    assert "ALL PROOFS CHECK" in out, out


@needs_bend
@pytest.mark.skipif(not has_lean(), reason="--verdict needs Lean 4 (elan)")
def test_laws_verdict():
    """The word laws again with BendTT's kernel, which has a proof in Lean (the program laws'
    sim.bend does not fit the kernel's memory)."""
    out = check("bend/PROOF.bend", "--verdict")
    assert "ALL PROOFS CHECK" in out, out


@needs_bend
@pytest.mark.parametrize("path", ["bend/fp32.bend", "bend/word.bend", "bend/sim.bend",
                                  "bend/machine.bend", "bend/cache/cache.bend"])
def test_pure(path):
    """Every def of the simulators proper is total and terminating (no @unsafe: only simpar.bend,
    the parallel MXU, and server.bend's IO use it; the cache's sweep, bend/cache/main.bend, only
    its file IO)."""
    out = check(path)
    assert "ALL PROOFS CHECK" in out, out


# ------------------------------------------------------------------------------- fp32


def fp32_reference(op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """opentpu/fp32.py's result words for bend/test/fpvec.bend's op codes."""
    from opentpu import fp32 as R
    fa, fb = a.view(np.float32), b.view(np.float32)
    w = lambda x: np.asarray(x, np.float32).view(np.uint32)        # noqa: E731
    if op in (0, 1, 2):
        return w((R.add, R.sub, R.mul)[op](fa, fb))
    if op == 13:
        return w(R.fmax(fa, fb))
    if op == 14:
        return w(R.fmin(fa, fb))
    if op in (5, 6, 7, 12):
        return w({5: R.exp2, 6: R.recip, 7: R.rsqrt, 12: R.log2}[op](fa))
    if op == 8:
        return w(R.i2f(a.view(np.int32)))
    if op == 9:
        return R.q8(fa).view(np.uint8).astype(np.uint32)
    if op == 10:
        return R.gt(fa, fb).astype(np.uint32)
    if op == 11:
        return w(R.fabs(fa))
    if op == 15:
        return np.array([I.f2i(int(x)) for x in a], np.uint32)
    if op == 17:
        return R.mx_offset(fa, fb).astype(np.uint32)
    return w(R.ftz(fa))


@needs_bend
def test_fp32_vectors(tmp_path):
    """fp32.bend against opentpu/fp32.py, op by op, on random words with the special ones mixed
    in (bend/test/fpvec.bend, built with bend -o)."""
    from opentpu import fp32 as R
    exe = ROOT / "build" / "fpvec"
    src = ROOT / "bend" / "test" / "fpvec.bend"
    deps = [src, ROOT / "bend" / "fp32.bend", ROOT / "bend" / "host.bend", ROOT / "bend" / "host.c"]
    if not exe.exists() or exe.stat().st_mtime < max(p.stat().st_mtime for p in deps):
        exe.parent.mkdir(exist_ok=True)
        subprocess.run([str(BEND), str(src), "-o", str(exe)], check=True, timeout=600,
                       env=dict(os.environ, BEND_NO_TELEMETRY="1"))
    rng = np.random.default_rng(7)
    n = 4096
    ops, refs, vec = [], [], []
    for op in (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        a, b = words(rng, n), words(rng, n)
        if op == 8:
            a = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
        if op == 17:                                   # 0 <= a <= amax, flushed
            a = R.fabs(a.view(np.float32)).view(np.uint32)
            b = R.fabs(b.view(np.float32)).view(np.uint32)
            a, b = np.minimum(a, b), np.maximum(a, b)
            a[(a & 0x7F800000) == 0x7F800000] = 0
            b[(b & 0x7F800000) == 0x7F800000] = 0x7F7FFFFF
        vec.append(np.stack([np.full(n, op, np.uint32), a, b, np.zeros(n, np.uint32)], 1))
        refs.append(fp32_reference(op, a, b))
        ops.append(op)
    v = np.concatenate(vec).reshape(-1)
    head = np.zeros(4, np.uint32)
    head[0] = v.size // 4
    np.concatenate([head, v]).astype(np.uint32).tofile(tmp_path / "vec.bin")
    subprocess.run([str(exe)], cwd=tmp_path, check=True, timeout=600)
    got = np.fromfile(tmp_path / "out.bin", np.uint32).reshape(len(ops), n)
    bad = {op: int((got[k] != refs[k]).sum()) for k, op in enumerate(ops)
           if not np.array_equal(got[k], refs[k])}
    assert not bad, f"mismatching vectors per op: {bad}"


# ------------------------------------------------------------------------------- VPU


@needs_server
def test_vop_vectors():
    """Every VOP function on random words with the special ones mixed in (the fp32 softfloat,
    the composites, the reductions), each operand mode."""
    cfg = Config(S=1, D=32, MCOLS=8, TMEM_WORDS=4096, DRAM_BYTES=1 << 12)
    rng = np.random.default_rng(1)
    rows, cols = 3, 37
    seen = set()
    for func in sorted(I.VFUNCS):
        for bmode in (I.B_FULL, I.B_ROW, I.B_COL, I.B_SCALAR):
            if func == I.V_ARGMAX:
                prog = [I.argmax(3000, 0, rows, cols, drs=2, ars=cols)]
            elif func == I.V_OUTER:
                prog = [I.outer(3000, 2000, 1000, 1500, rows, cols, drs=cols, brs=cols)]
            else:
                prog = [I.vop(func, 3000, 0, 1000, rows, cols, drs=cols, ars=cols, brs=cols,
                              bmode=bmode, imm=float(rng.standard_normal()))]
            m = Machine(cfg, [prog + [I.halt()]], [None])
            m.slices[0].tmem[:] = words(rng, cfg.TMEM_WORDS)
            seen.add(both(m))
    assert seen == {"same"}


# ------------------------------------------------------------------------------- STREAM


def stream_program(seed: int, cfg: Config) -> Machine:
    """A random STREAM (docs/stream.md), now and then a malformed one: bad ops, indices, rows
    or columns, words that are not float-safe, ranges past TMEM's or DRAM's end, rows that
    overlap one another in place."""
    rng = np.random.default_rng(seed)
    TW, DB, D = cfg.TMEM_WORDS, cfg.DRAM_BYTES, cfg.D
    rows = int(rng.integers(1, 7))
    cols = int(rng.choice([1, 2, 3, 17, 63, 64, 65, 100, 128, 130]))
    unary = (5, 6, 7, 8, 9)
    ops = []
    for _ in range(int(rng.integers(0, 6 if rng.random() < 0.9 else 16))):
        op = int(rng.integers(0, 11)) if rng.random() < 0.97 else int(rng.integers(11, 16))
        ops.append((op, int(rng.integers(0, 8)), int(rng.integers(0, 16)),
                    0 if op in unary and rng.random() < 0.7 else int(rng.integers(0, 16))))
    q_en = bool(rng.random() < 0.5)
    d = I.StreamDesc(
        rows=rows, cols=cols, a_en=bool(rng.random() < 0.6), a_op=int(rng.integers(0, 4)),
        a_idx=int(rng.integers(0, 4)), u_mode=int(rng.integers(0, 4)),
        g_src=int(rng.integers(0, 4)),
        g_idx=int(rng.integers(0, 4)) if rng.random() < 0.95 else int(rng.integers(0, 8)),
        b_src=int(rng.integers(0, 3)) if rng.random() < 0.97 else 3,
        b_idx=int(rng.integers(0, 4)) if rng.random() < 0.95 else int(rng.integers(0, 8)),
        d_reg=int(rng.integers(0, 8)), q_en=q_en, q_idx=int(rng.integers(0, 4)),
        q_self=bool(rng.random() < 0.3), out_p2=bool(rng.random() < 0.3),
        o_reg_en=(not q_en and rng.random() < 0.4) or rng.random() < 0.02,
        o_reg=int(rng.integers(0, 8)), rinit=int(rng.integers(0, 16)),
        rsave=int(rng.integers(0, 16)) if rng.random() < 0.5 else 0,
        srs=0 if rng.random() < 0.6 else int(rng.integers(1, 200)),
        drs=0 if rng.random() < 0.6 else int(rng.integers(1, 200)), ops=tuple(ops))
    if rng.random() < 0.1 and cols > 1:
        s_ = int(rng.integers(1, cols))
        d = replace(d, srs=s_, drs=s_)
    dw = d.words()
    if rng.random() < 0.03:
        dw[int(rng.integers(0, len(dw)))] ^= 1 << 30
    if rng.random() < 0.02:
        dw[0] = I._fsafe(cols << 12 if rng.random() < 0.5 else rows)
    zero = bool(rng.random() < 0.15)
    src_t, dst_t = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    nodst = bool(rng.random() < 0.15)

    def taddr(n):
        return int(rng.integers(0, max(1, TW - n))) if rng.random() < 0.98 else \
            int(rng.integers(TW - n // 2 - 1, TW + 4))

    def daddr(n):
        a = int(rng.integers(0, max(1, (DB - 4 * n) // D))) * D
        if rng.random() < 0.02:
            a = DB - 2 * n
        if rng.random() < 0.02:
            a += 4
        return a
    span_s = rows * (d.srs or cols) + cols
    span_d = rows * (d.drs or cols) + cols
    src = taddr(span_s) if src_t else daddr(rows * cols)
    r = rng.random()
    if d.srs and d.srs == d.drs and src_t and rng.random() < 0.8:
        dst_t, dst = True, src
    elif r < 0.3 and src_t == dst_t:
        dst = src
    elif r < 0.5 and src_t == dst_t:
        dst = max(0, src + int(rng.integers(-3, 4)) * (cols if src_t else D))
    else:
        dst = taddr(span_d) if dst_t else daddr(rows * cols)
    vec, xa = taddr(4 * cols), taddr(rows)
    ks = int(rng.integers(0, 4))
    ka, out = taddr(3 * ks + 1), taddr(rows)
    at = taddr(len(dw)) if rng.random() < 0.97 else TW - int(rng.integers(1, 6))
    ins = I.Instr(I.STREAM, flags=(I.F_SZERO if zero else 0) | (I.F_SRC_T if src_t else 0)
                  | (I.F_DST_T if dst_t else 0) | (I.F_NODST if nodst else 0),
                  w=[(at | ks << 16) & 0xFFFFFFFF, src & 0xFFFFFFFF, dst & 0xFFFFFFFF, vec, xa,
                     ka, out])
    m = Machine(cfg, [[ins, I.halt()]], [None] * cfg.S)
    s = m.slices[0]
    s.tmem[:] = words(rng, TW)
    s.dram.view(np.uint32)[:] = words(rng, DB // 4)
    n = min(len(dw), TW - at)
    if n > 0:
        s.tmem[at:at + n] = np.array(dw[:n], np.uint32)
    return m


@needs_server
def test_stream_fuzz():
    """Random STREAMs: the same end state, or the same error (the fuzzer that found isasim's
    unary-op crash: a unary op read its unused K operand)."""
    cfg = Config(S=1, D=64, TMEM_WORDS=4096, DRAM_BYTES=1 << 15, STREAM=True)
    n = int(os.environ.get("OTPU_BEND_FUZZ", "300"))
    out = [both(stream_program(seed, cfg)) for seed in range(n)]
    assert out.count("same") > n // 2, out


# ------------------------------------------------------------------------------- MXU


@needs_server
@pytest.mark.parametrize("m_rows,n,kb,wf", [(1, 300, 2, I.W8), (8, 64, 3, I.W8),
                                            (3, 1100, 1, I.W8), (5, 40, 4, I.W4I)])
def test_mxu_seq_par(m_rows, n, kb, wf):
    """The parallel MXU (bend/simpar.bend: rows and outputs split over the cores) and the
    sequential one (sim.bend's mm.seq) give isasim's MM bit for bit, accumulating too."""
    from opentpu.bend_sim import DRAM
    cfg = Config(S=1, D=128, MCOLS=8, TMEM_WORDS=1 << 16, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(m_rows * 1000 + n)
    K = 128 * kb
    rs = K if wf == I.W8 else K // 2
    prog = [I.mm(sa=0, ssa=0, out=0, n=n, kb=kb, rs=rs, ors=n, m=m_rows, ab=0, srs=0,
                 unit=True, acc=r > 0, wf=wf) for r in range(2)] + [I.halt()]
    dram = rng.integers(0, 256, cfg.DRAM_BYTES, dtype=np.uint8)
    mach = Machine(cfg, [prog], [dram.copy()])
    s = mach.slices[0]
    s.act[:] = rng.integers(-127, 128, s.act.shape).astype(s.act.dtype)
    act = s.act.copy()
    mach.run()
    ref = s.tmem[:m_rows * n].copy()
    b = BS._server(cfg)
    for seq in (True, False):
        b.seq_mxu = seq
        b.put(0, DRAM, 0, dram)
        b.set_act(0, act, s.ascale, s.adoff)
        b.load([prog])
        b.run()
        assert np.array_equal(b.tmem(0)[:m_rows * n], ref), "sequential" if seq else "parallel"
    b.seq_mxu = False


# ------------------------------------------------------------------------------- the expert cache


@needs_bend
def test_cache_sweep():
    """bend/cache/ against tools/offload/cachesim.py's simulate: the same misses of every token,
    each policy Bend has (static, lru, lru_layer, lfu, opt), on random skewed traces from one to
    four layers, with and without a profile; and no cache above its slots."""
    import importlib.util
    mods = {}
    for name in ("cachesim", "cache_bend"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools/offload" / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    cs, cb = mods["cachesim"], mods["cache_bend"]
    rng = np.random.default_rng(3)
    for trial in range(12):
        T, L, E = int(rng.integers(1, 60)), int(rng.integers(1, 5)), int(rng.integers(2, 12))
        k = int(rng.integers(1, E + 1))
        p = rng.dirichlet(np.ones(E) * 0.3, size=L)
        req = np.stack([np.stack([rng.choice(E, k, replace=False, p=p[j]) + j * E
                                  for j in range(L)]) for _ in range(T)])
        prof = cs.freq(req, E * L)
        jobs = [(C, pol, cs.top_set(prof, E * L if pol == "lru_layer" else C),
                 prof if trial % 3 or pol == "lru_layer" else None)
                for C in sorted({k, L * k, max(k, E * L // 3), E * L})
                for pol in ("static", "lru", "lru_layer", "lfu", "opt")]
        for (C, pol, warm, prior), (miss, peak) in zip(jobs, cb.sweep(req, jobs)):
            assert miss.tolist() == cs.simulate(req, C, pol, warm, prior)[0].tolist(), (trial, pol)
            assert pol == "static" or peak <= max(C, len(warm)), (trial, pol, peak)
