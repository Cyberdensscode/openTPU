# The host effects of the Bend simulators (host.c): word arrays from and to files and pipes.
import Base

# the file into a fresh array of 2^depth words, zero filled past its end
def Host.load(path: String, depth: U32) -> IO(Array<U32>):
  import "./host.c"

# the array's first n words to the file; the array is handed back
def Host.save(path: String, a: Array<U32>, n: U32) -> IO(Array<U32>):
  import "./host.c"
