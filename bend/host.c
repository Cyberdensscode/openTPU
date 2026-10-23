// The host side of the Bend simulators (bend/host.bend): files and pipes in and out of
// Array<U32> buffers without a list in between.
//
// These use the runtime's internals (heap_alloc, term_blk, blk_loc: Bend 2.0.38's
// bend2/comp.ts output); bend/EFFECTS.md in the Bend guide says there is no ABI promise, so
// rebuild and recheck them with every Bend update.
//
// Host.load(path, depth): the file into a fresh Array<U32> of 2^depth words (a BUF block,
// zero filled past the file). Host.save(path, a, n): a's first n words to the file.

Term host_load_run(Env e, Term* f, IoWork* w) {
  u64 len;
  char* path = io_cstr(e, f[0], &len);
  u32 c = (u32)f[1];
  u64 loc = heap_alloc(e, c == 0 ? 0 : c - 1);
  u32* p = (u32*)(e.mem + loc);
  memset(p, 0, 4ull << c);
  FILE* h = path ? fopen(path, "rb") : NULL;
  if (h) {
    size_t got = fread(p, 1, 4ull << c, h);
    (void)got;
    fclose(h);
  }
  free(path);
  return term_blk(false, c, loc);
}

Term host_save_run(Env e, Term* f, IoWork* w) {
  u64 len;
  char* path = io_cstr(e, f[0], &len);
  Term a = f[1];
  u32 n = (u32)f[2];
  u32* p = (u32*)(e.mem + blk_loc(e.mem, a));
  FILE* h = path ? fopen(path, "wb") : NULL;
  if (h) {
    fwrite(p, 4, n, h);
    fclose(h);
  }
  free(path);
  return a;
}

static void __attribute__((constructor)) host_mem_use(void) {
  io_eff(CID(Host.load), host_load_run);
  io_eff(CID(Host.save), host_save_run);
}
