//! `sstv-auto` — automatic SSTV detection and decoding.
//!
//! The crate is split into layers that can each be tested on their own:
//!
//! * [`audio`] — container/codec ingestion, channel selection, level safety.
//! * [`dsp`] — FFT analysis producing a compact time-frequency [`dsp::Track`]
//!   and the [`dsp::Trajectory`] model every later stage reads.
//! * [`modes`] — data-driven SSTV mode geometry and the level/frequency map.
//! * [`vis`] — VIS header detection and synthesis.
//! * [`sync`] — line-sync pulse detection, period fitting, clock recovery.
//! * [`raster`] — recovering a pixel grid from a measured signal.
//! * [`synth`] — rendering a grid back to audio (the inverse of [`raster`]).
//! * [`autodetect`] — mode/offset/clock inference and candidate ranking.
//! * [`backend`] — driving the full-resolution raster decoder.
//! * [`score`] — image plausibility metrics used to reject false positives.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod dsp;
pub mod modes;
