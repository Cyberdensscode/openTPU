# Laws about whole programs (docs/bend.md "Laws"): each runs on a slice of sim.bend, and the
# values are isasim's for the same program. PROOF.bend proves them by evaluation in the checker
# (`bend bend/prog/PROOF.bend --check-only`); the laws about words are ../LAWS.bend.
#
# Bend has no hex literals: the words are decimal, their hex or value in the comments.
import Base
import ../sim.bend as S

# A slice of the configuration isasim.Config(S=1, D=8, MCOLS=1, ACT_BLOCKS=1, TMEM_WORDS=64,
# DRAM_BYTES=256, IMEM_WORDS=64) running a program from pc 0, as the server loads one: the
# instructions' words at IMEM, the program's length at PLEN.
def cfg() -> S.Cf:
  S.Cf{1, 8, 1, 1, 64, 255, 64, 1, 0, 0, 0, 0, 2, 3}

def put(xs: List<U32>, c: Array<U32>, +i: U32) -> Array<U32>:
  match xs:
    case Nil{}:
      c
    case Con{w, t}:
      put(t, c[i] <- w, (i + 1 : U32))

def plen(c: Array<U32>, +n: U32) -> Array<U32>:
  c[21] <- n

def boot(xs: List<U32>, +n: U32) -> S.Sl:
  S.Sl{plen(put(xs, [0 : U32*8192n], 4096), n), [0 : U32*64n], [0 : U32*64n], [0 : U32*64n],
    [0 : U32*64n], [0 : U32*64n], cfg()}

# the program xs of n instructions, run until it stops (64 instructions at most)
def run(xs: List<U32>, +n: U32) -> S.Sl:
  S.go(~(s => i => S.mm.seq(s, i)), 64n, S.runs(boot(xs, n)))

# the word read (r's), then the words at ix
def get.go(ix: List<U32>, r: Array<U32> & U32) -> List<U32>:
  match ix:
    case Nil{}:
      (a, v) = r
      [v]
    case Con{+i, t}:
      (a, v) = r
      v <> get.go(t, Array.get(U32, a, i))

# the words at ix
def get(ix: List<U32>, a: Array<U32>) -> List<U32>:
  match ix:
    case Nil{}:
      []
    case Con{+i, t}:
      get.go(t, Array.get(U32, a, i))

# R1, R2, ICOUNT, STATE (RUN 0, HALT 1, ERROR 4), ERR
def regs(s: S.Sl) -> List<U32>:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      get([1, 2, 17, 18, 19], c)

# TMEM words
def tmem(s: S.Sl, ix: List<U32>) -> List<U32>:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      get(ix, t)

# DRAM words (four little-endian bytes each)
def dram(s: S.Sl, ix: List<U32>) -> List<U32>:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      get(ix, d)

# LI R1, 7; ADDI R2, R1, 5; HALT
law prog_scalar:
  {regs(run([1048578, 7, 0, 0, 0, 0, 0, 0, 2097411, 5, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0], 3)) == [7, 12, 3, 1, 0] : List<U32>}

# LI R1, 0; LOOP 1, 10 { ADDI R1, R1, 3 }; HALT: 13 instructions
law prog_loop:
  {regs(run([1048578, 0, 0, 0, 0, 0, 0, 0, 4, 1, 10, 0, 0, 0, 0, 0,
    1048835, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4)) == [30, 0, 13, 1, 0] : List<U32>}

# LI R0, 9; ADDI R1, R0, 1; HALT: R0 reads 0
law prog_r0:
  {regs(run([2, 9, 0, 0, 0, 0, 0, 0, 1048579, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0], 3)) == [1, 0, 3, 1, 0] : List<U32>}

# LI R1, 7 and no HALT: the slice stops with error 1 (pc past the end of the program)
law prog_end:
  {regs(run([1048578, 7, 0, 0, 0, 0, 0, 0], 1)) == [7, 0, 1, 4, 1] : List<U32>}

# VOP FILL T[0..4) = 1.5; VOP FILL T[4..8) = 2.25; VOP ADD T[8..12) = T[0..4) + T[4..8); HALT
law prog_vop:
  {tmem(run([48, 0, 0, 0, 262145, 4, 51183616, 1069547520, 48, 4, 0, 0, 262145, 4, 51183616,
    1074790400, 48, 8, 0, 4, 262145, 262148, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4),
    [7, 8, 11, 12]) == [1074790400, 1081081856, 1081081856, 0] : List<U32>}

# VOP FILL T[0..4) = -0.75; ST DRAM[16..32) = T[0..4); LD T[32..36) = DRAM[16..32); HALT
law prog_ldst:
  {tmem(run([48, 0, 0, 0, 262145, 4, 51183616, 3208642560, 17, 16, 0, 4, 0, 0, 0, 0,
    16, 16, 32, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4), [31, 32, 35, 36])
    == [0, 3208642560, 3208642560, 0] : List<U32>}

law prog_ldst_dram:
  {dram(run([48, 0, 0, 0, 262145, 4, 51183616, 3208642560, 17, 16, 0, 4, 0, 0, 0, 0,
    16, 16, 32, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4), [3, 4, 7, 8])
    == [0, 3208642560, 3208642560, 0] : List<U32>}
