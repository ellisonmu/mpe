# LPC-10 (FS-1015 / FED-STD-1015) reference codec

`nuke`/`unnuke` are the standard DoD 2400 bps LPC-10 encoder/decoder, built
from the public-domain reference source (originally released Oct 1993),
via the C port maintained at <https://github.com/jafingerhut/lpc10>
(BSD-3-Clause, see `LICENSE`). Used by `lpc10_wer.py` as an external-codec
reference point alongside this project's own MPE algorithms.

- 180 samples/frame, 54 bits/frame (byte-aligned to 7 bytes/frame on disk)
  at 8 kHz -> exactly 2400 bps.
- Input/output is **raw, headerless, little-endian 16-bit signed PCM at
  8 kHz mono** via stdin/stdout -- not WAV, and not through `sox`. A
  trailing partial frame (< 180 samples) is silently dropped by the
  encoder, so callers should pad input to a multiple of 180 samples first.

## Prebuilt binaries

`nuke.exe` / `unnuke.exe` are a Windows (x86_64, MinGW-w64/UCRT) build.
They depend only on standard Windows 10+ UCRT api-set DLLs -- nothing
extra to install.

They carry one portability patch beyond upstream `src/nuke.c` /
`src/unnuke.c`: `_setmode(..., _O_BINARY)` on stdin/stdout, guarded by
`#ifdef _WIN32`. Without it, Windows' default text-mode stdio corrupts the
binary audio stream (any 0x1A byte in the samples is read as EOF), so the
tools silently produce zero output. This isn't needed on Linux/macOS.

## Rebuilding (e.g. for the Jetson / any Linux box)

The upstream repo is tested on Ubuntu x86_64 and aarch64, so it should
build as-is on the Jetson:

```bash
cd src/lpc10
gcc -O -c *.c -I..
ar rcs ../liblpc10.a *.o
cd ..
gcc -O -c nuke.c unnuke.c bitio.c sampleio.c
gcc -o nuke   nuke.o   bitio.o sampleio.o -L. -llpc10 -lm
gcc -o unnuke unnuke.o bitio.o sampleio.o -L. -llpc10 -lm
```

(On Linux, stdio is binary by default, so the `_setmode`/`_WIN32` block in
`nuke.c`/`unnuke.c` is a no-op there -- no changes needed.)

On Windows with MinGW-w64 under a path containing a space (e.g.
`C:\Users\Your Name\...`), the linker step needs `-B <short-8.3-path-to-
mingw's-lib-dir>` to work around a known space-in-path bug when resolving
`default-manifest.o`; see git history / ask if you hit this again.
