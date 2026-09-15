# sstv-auto

`sstv-auto` decodes an SSTV image out of an audio file without being told
anything about it.

```bash
sstv-auto audio.wav
```

No mode, timing, frequency offset, sample rate, channel or image size has to be
supplied. The tool finds the transmission, works out which SSTV mode it is, and
writes the reconstructed image.

## What "automatic" means here

Two independent paths find the mode, and the tool prefers the authoritative one:

1. **VIS header.** When the header is intact, it *is* the mode. Its parity bit
   is checked, and a header that fails parity is treated as a hint rather than
   as truth.
2. **Line-sync period.** When the header is missing or destroyed, the tool
   measures the repeated line-sync pulse. Each mode has a distinctive line
   period, so a run of sync pulses identifies the mode on its own. This is the
   path that matters for CTF recordings with a stripped header.

Everything else is measured rather than assumed:

* **Frequency offset** comes from the sync pulses and from the VIS leader tone,
  so a mistuned receiver is corrected rather than silently mistuned.
* **Clock rate** comes from a least-squares fit of the sync chain, so a
  transmitter running a percent fast or slow is corrected. The VIS burst
  supplies an independent estimate of the same quantity.
* **Start position** is taken from the first sync pulse of the fitted chain, not
  from an assumed offset.

## How candidates are ranked

A wrong mode can still decode into *an* image, so the decoder does not trust a
single decode. For each hypothesis it:

1. extracts a low-resolution grid from the measured signal,
2. re-encodes that grid back into SSTV audio,
3. measures the re-rendered audio, and
4. compares it against the original recording at the same sample positions.

This works because the renderer and the extractor are inverses of one another by
construction, so a hypothesis that re-renders audio matching the recording tone
for tone is genuinely explained by that mode, and one that does not is rejected.
Weak matches are dropped rather than reported, and a candidate whose image is
flat, fully clipped or noise-like is rejected outright.

## Supported modes

| Family  | Modes                       | VIS codes           |
|---------|-----------------------------|---------------------|
| Martin  | M1, M2                      | `0x2C`, `0x28`      |
| Scottie | S1, S2, DX                  | `0x3C`, `0x38`, `0x4C` |
| Robot   | 24, 36, 72                  | `0x04`, `0x08`, `0x0C` |
| PD      | PD-120, PD-180, PD-240      | `0x5F`, `0x60`, `0x61` |

Mode geometry is data-driven: every timing constant lives in one table in
`src/modes.rs` and is consumed by the extractor, the renderer and the ranking, so
they cannot drift apart.

## Audio input

Decoding is pure Rust via [Symphonia](https://github.com/pdeljanov/Symphonia):
WAV/PCM, FLAC, MP3, AAC/MP4, Ogg/Vorbis, AIFF, CAF and MKV. **No external
programs are required** — the tool works on a machine that has only a Rust
toolchain installed.

Stereo and multichannel files are handled by scoring each channel on how much
SSTV-band energy it carries and selecting the best one, or by explicit
`--channel` selection.

## Install

```bash
cargo build --release
```

The binary lands at `target/release/sstv-auto` (`sstv-auto.exe` on Windows).

## Usage

```bash
# Fully automatic
sstv-auto audio.wav

# Choose the output directory
sstv-auto challenge.flac -o out

# Show per-stage diagnostics and the evidence behind the decision
sstv-auto audio.wav --verbose

# Pick a specific channel
sstv-auto stereo.wav --channel right

# Decode only from the VIS header, skipping blind recovery
sstv-auto audio.wav --no-blind

# Force a mode when the recording is too damaged to identify automatically
sstv-auto audio.wav --mode robot36

# Also write the images of rejected candidates, for inspection
sstv-auto audio.wav --keep-candidates
```

Output:

```
audio_sstv/
├── 001-robot36.png
└── report.json
```

`report.json` records the detected mode, how it was determined, confidence,
signal agreement, coverage, image start time, frequency offset, clock rate,
matched sync pulses, image dimensions and output path, plus any warnings.

## The `audio.wav` validation result

`audio.wav` (36.91 s, 44.1 kHz mono) is the acceptance fixture. Run against the
release binary:

```
$ ./target/release/sstv-auto audio.wav --verbose
audio: 36.91s, 44100 Hz, 1 channel(s) -> channel 0 (mono)
detect: 245 sync pulse(s), 1 VIS burst(s), 9 hypothesis/hypotheses, frequency offset -0.0 Hz
rank: 9 candidate(s)
  Robot 36   by vis               score 1.000 (agreement 0.818, timing 0.998, coverage 1.00)
  Robot 24   by sync-period       score 0.919 (agreement 0.854, timing 0.999, coverage 1.00)
  Robot 36   by sync-period       score 0.919 (agreement 0.854, timing 0.999, coverage 1.00)
  ...
Decoded 3 images:
  audio_sstv/001-robot36.png  Robot 36 (VIS 0x08)  confidence 100%
      detected by vis | 320x240 | image starts 0.916s | offset -0.0 Hz | clock nominal
```

The decoded image is 320x240 and reads **`flag{SSTV_and_R0b0t36}`**. Measured
properties of the output PNG: per-channel variance about 10,880, standard
deviation 104.4, 5,925 distinct colours — unambiguous real image content rather
than an empty or noise frame.

Removing the VIS header from the fixture and re-running still decodes the same
image through the sync-period path, which is the CTF scenario the blind recovery
exists for.

## Known limitations

* **The outermost few columns of every line, and the first row, carry a thin
  coloured fringe.** This comes from the raster backend's per-pixel FFT window
  overlapping the sync pulse and neighbouring channel at line boundaries. It is
  measurable: fed a *perfect* synthetic transmission of a flat grey image, the
  backend still produces about 1.3% out-of-range edge pixels, all within roughly
  six columns of a line edge or in row 0, while the interior decodes exactly.
  This crate's own grid extraction compensates for the effect, but the final
  image is produced by the backend at full resolution, so the fringe survives.
  It carries no image information.
* **The last several milliseconds of each line cannot be measured.** A
  short-time frequency window wide enough to resolve tones is wider than the
  tail of a line, so the final columns are filled from the last trustworthy
  measurement rather than from a contaminated one.
* **Robot 24 and Robot 36 are indistinguishable without a VIS header.** They
  share a 150 ms line period, 240 rows, the same pixel time and the same
  alternating chroma layout; on the wire they differ only in their VIS code. A
  recording made as either mode decodes to the same picture, and the report says
  so explicitly.
* **Recovery is information-limited.** An unknown or private mode, severe
  clipping, heavy time warping, missing image data, destructive filtering or a
  signal below the noise floor may be unrecoverable. `sstv-auto` targets
  aggressive automatic recovery of standard analog SSTV from CTF and radio
  captures. It does not claim to decode every possible audio file, and it will
  report "no SSTV transmission found" rather than inventing an image.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release
```

### Layout

| Module        | Responsibility                                          |
|---------------|---------------------------------------------------------|
| `audio`       | Container/codec decoding, channel selection, levels      |
| `dsp`         | FFT analysis, the `Track` and `Trajectory` models        |
| `modes`       | Data-driven mode geometry and the level/frequency map    |
| `vis`         | VIS detection, parity validation, synthesis              |
| `sync`        | Sync-pulse detection, period fitting, clock recovery     |
| `raster`      | Recovering a pixel grid from a measured signal           |
| `synth`       | Rendering a grid back to audio (the inverse of `raster`) |
| `autodetect`  | Mode inference and candidate ranking                     |
| `backend`     | Driving the full-resolution raster decoder               |
| `score`       | Image plausibility metrics                               |
| `pipeline`    | End-to-end orchestration                                 |

Unit tests live beside the code they cover; `tests/end_to_end.rs` runs the whole
pipeline over synthetic transmissions and over `audio.wav` when it is present.

### On the `audio.wav` fixture and CI

`audio.wav` is deliberately **not committed**: it is a 3 MB binary and it is a
challenge artefact rather than source. The end-to-end test skips it with a
printed notice when the file is absent, so CI stays green without it, and runs
it for real wherever the file exists. The assertions in that test are about what
was recovered — real dimensions, real pixel variance, a valid VIS detection —
and never about the file's size, checksum or expected pixels, so it remains a
black-box check rather than a fixture-specific one.

## License

MIT.
