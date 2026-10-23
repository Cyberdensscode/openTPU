// The Bend ISA simulator's server effects (bend/server.bend): the command pipe from the host
// (fd 3) and the reply pipe to it (fd 4), with the bulk data read straight into and written
// straight out of Array<U32> buffers (opentpu/bend_sim.py drives it).
//
// These use the runtime's internals (heap_alloc, term_blk, blk_loc: Bend 2.0.38's
// bend2/comp.ts output); bend/EFFECTS.md in the Bend guide says there is no ABI promise, so
// rebuild and recheck them with every Bend update. An Array<U32> is a BUF block of packed u32
// words: blk_loc gives its first u64 word in the heap.

#include <unistd.h>
#include <errno.h>

static u32 srv_h[16];        // the last command's header
static u32 srv_r[16];        // the reply header being built

static int srv_full(int fd, void* p, u64 n, int wr) {
  char* q = (char*)p;
  while (n > 0) {
    ssize_t k = wr ? write(fd, q, n > (1u << 30) ? (1u << 30) : n)
                   : read(fd, q, n > (1u << 30) ? (1u << 30) : n);
    if (k < 0 && errno == EINTR) {
      continue;
    }
    if (k <= 0) {
      return -1;
    }
    q += k;
    n -= (u64)k;
  }
  return 0;
}

// the next command's 16 header words; its word 0 (0, QUIT, at the pipe's end)
Term srv_hdr_run(Env e, Term* f, IoWork* w) {
  if (srv_full(3, srv_h, sizeof srv_h, 0) != 0) {
    memset(srv_h, 0, sizeof srv_h);
  }
  return (Term)srv_h[0];
}

Term srv_arg_run(Env e, Term* f, IoWork* w) {
  return (Term)srv_h[(u32)f[0] & 15];
}

// a fresh array of 2^depth words, zero filled
Term srv_mk_run(Env e, Term* f, IoWork* w) {
  u32 c = (u32)f[0];
  u64 loc = heap_alloc(e, c == 0 ? 0 : c - 1);
  memset((u32*)(e.mem + loc), 0, 4ull << c);
  return term_blk(false, c, loc);
}

// dir 0: n bytes from the command pipe into the array's bytes from off; dir 1: the array's
// bytes off .. off + n to the reply pipe. The array is handed back.
Term srv_io_run(Env e, Term* f, IoWork* w) {
  Term a = f[0];
  u32 dir = (u32)f[1];
  u64 off = (u32)f[2];
  u64 n = (u32)f[3];
  char* p = (char*)(e.mem + blk_loc(e.mem, a)) + off;
  if (srv_full(dir ? 4 : 3, p, n, dir) != 0) {
    _exit(3);
  }
  return a;
}

Term srv_put_run(Env e, Term* f, IoWork* w) {
  srv_r[(u32)f[0] & 15] = (u32)f[1];
  return term_pak(CID(Unit), 0);
}

// the reply header, then cleared for the next
Term srv_flush_run(Env e, Term* f, IoWork* w) {
  if (srv_full(4, srv_r, sizeof srv_r, 1) != 0) {
    _exit(3);
  }
  memset(srv_r, 0, sizeof srv_r);
  return term_pak(CID(Unit), 0);
}

static void __attribute__((constructor)) srv_use(void) {
  io_eff(CID(Srv.hdr), srv_hdr_run);
  io_eff(CID(Srv.arg), srv_arg_run);
  io_eff(CID(Srv.mk), srv_mk_run);
  io_eff(CID(Srv.io), srv_io_run);
  io_eff(CID(Srv.put), srv_put_run);
  io_eff(CID(Srv.flush), srv_flush_run);
}
