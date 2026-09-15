//! DSP primitives shared by detection, inference and reconstruction.
//!
//! Two design rules govern this module:
//!
//! 1. **No per-sample allocation on the hot path.** Frequency analysis runs
//!    over a small sliding window; the result is stored as *windowed
//!    medians*, so a whole-file pass allocates `O(frames)`, not
//!    `O(samples)`.
//! 2. **Frames stay short.** A STFT frame must be short enough that the test
//!    tone looks stationary *around each pixel centre* (pixel times run from
//!    138 us to 1081 us) yet long enough for usable frequency resolution.
//!    Those constraints pull in opposite directions, so the FFT is
//!    zero-padded to recover fine bin spacing without widening the window,
//!    and [`Track::hz_at_sample`] interpolates between neighbouring windowed
//!    medians.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// Hann analysis window length in seconds.
///
/// # Choosing this value
///
/// Two constraints pull in opposite directions:
///
/// * **Time.** The window has to be short enough to sit inside the shortest
///   sync pulse any supported mode transmits, which is Martin 1/2 at 4.862 ms.
///   A window longer than that can never resolve a Martin sync: the window
///   always spans image tones on either side, the dominant frequency is the
///   image tone, and the mode becomes undetectable without a VIS header.
/// * **Frequency.** The window has to be long enough that the zero-padded FFT
///   still resolves tones a few tens of hertz apart, since mode inference
///   compares measured pulse frequencies against nominal ones.
///
/// 4 ms satisfies both: it fits inside a Martin sync with margin, and it still
/// spans between four and twenty nine transmitted pixels, so a per-pixel level
/// estimate averages rather than chasing individual transitions. A longer
/// window was measured to lose Martin 1 entirely, which is why this is 4 ms
/// rather than the more comfortable 8 ms.
pub const FRAME_SECONDS: f64 = 0.004;

/// Zero-padding factor applied to the Hann window before the FFT.
///
/// Zero padding buys frequency resolution without widening the window in time,
/// which is exactly the trade this module needs: the window is kept short so
/// that brief sync pulses remain visible, and the padding recovers the fine
/// bin spacing that a short window would otherwise cost.
const ZERO_PAD: usize = 8;

/// Number of neighbouring frames folded into each reported median.
const MEDIAN_RADIUS: usize = 1;

/// A symmetric Hann window of `len` points.
#[must_use]
pub fn hann(len: usize) -> Vec<f32> {
    match len {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..len)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (len - 1) as f64;
                (0.5 - 0.5 * x.cos()) as f32
            })
            .collect(),
    }
}

/// Sliding frequency analysis over a signal.
pub struct Analyzer {
    rate: u32,
    window_len: usize,
    hop: usize,
    fft_len: usize,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    bin_hz: f64,
}

impl Analyzer {
    /// Create an analyzer for a working sample rate.
    #[must_use]
    pub fn new(rate: u32) -> Self {
        let window_len = ((FRAME_SECONDS * f64::from(rate)).round() as usize).max(32);
        let hop = (window_len / 2).max(1);
        let fft_len = (window_len * ZERO_PAD).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        Self {
            rate,
            window_len,
            hop,
            fft_len,
            window: hann(window_len),
            fft,
            bin_hz: f64::from(rate) / fft_len as f64,
        }
    }

    /// Hann window length in samples.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window_len
    }

    /// Analysis hop in samples.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// FFT length including zero padding.
    #[must_use]
    pub fn fft_len(&self) -> usize {
        self.fft_len
    }

    /// Frequency resolution of one FFT bin, Hz.
    #[must_use]
    pub fn bin_hz(&self) -> f64 {
        self.bin_hz
    }

    /// Rate this analyzer was built for.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Highest analysable frequency (Nyquist), Hz.
    #[must_use]
    pub fn nyquist(&self) -> f64 {
        f64::from(self.rate) / 2.0
    }

    /// Analyse a whole signal into a [`Track`].
    ///
    /// Memory grows with duration, not with sample count: roughly 32 bytes
    /// per analysis hop.
    #[must_use]
    pub fn track(&self, samples: &[f32]) -> Track {
        let frame_count = if samples.len() >= self.window_len {
            (samples.len() - self.window_len) / self.hop + 1
        } else {
            0
        };
        let mut peak_hz = Vec::with_capacity(frame_count);
        let mut peak_level = Vec::with_capacity(frame_count);
        let mut tone_likeness = Vec::with_capacity(frame_count);
        let mut frame_rms = Vec::with_capacity(frame_count);

        if frame_count == 0 {
            return Track {
                rate: self.rate,
                hop: self.hop,
                window_len: self.window_len,
                bin_hz: self.bin_hz,
                peak_hz,
                peak_level,
                tone_likeness,
                frame_rms,
            };
        }

        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft_len];
        let mut spectrum = vec![0.0_f64; self.fft_len / 2 + 1];
        // Bins below this index are ignored: a DC offset or mains hum would
        // otherwise dominate every quiet frame.
        let first_bin = ((60.0 / self.bin_hz).ceil() as usize).max(1);

        for frame in 0..frame_count {
            let start = frame * self.hop;
            let slice = &samples[start..start + self.window_len];

            for slot in scratch.iter_mut() {
                *slot = Complex32::new(0.0, 0.0);
            }
            for ((slot, sample), w) in scratch.iter_mut().zip(slice.iter()).zip(self.window.iter())
            {
                *slot = Complex32::new(sample * w, 0.0);
            }
            self.fft.process(&mut scratch);

            let mut total = 0.0_f64;
            let mut best = 0.0_f64;
            let mut best_bin = first_bin;
            for (bin, value) in spectrum.iter_mut().enumerate() {
                let energy = if bin == 0 {
                    0.0
                } else {
                    let c = scratch[bin];
                    f64::from(c.re) * f64::from(c.re) + f64::from(c.im) * f64::from(c.im)
                };
                *value = energy;
                if bin >= first_bin {
                    total += energy;
                    if energy > best {
                        best = energy;
                        best_bin = bin;
                    }
                }
            }

            // Parabolic interpolation around the peak recovers sub-bin
            // resolution; the ranking compares measured pulse frequencies
            // against nominal ones, so this matters.
            let refined = if best_bin > 0 && best_bin + 1 < spectrum.len() && best > 0.0 {
                let a = spectrum[best_bin - 1];
                let b = spectrum[best_bin];
                let c = spectrum[best_bin + 1];
                let denom = a - 2.0 * b + c;
                let delta = if denom.abs() > f64::EPSILON {
                    (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                };
                (best_bin as f64 + delta) * self.bin_hz
            } else {
                best_bin as f64 * self.bin_hz
            };

            // Tone likeness: a pure tone concentrates its windowed energy
            // into a narrow peak; broadband noise spreads it across the whole
            // band. The peak's *neighbourhood* is summed rather than a single
            // bin, because zero padding smears a Hann peak over roughly
            // `4 * ZERO_PAD` bins. Comparing one bin against the total would
            // report a pure tone as noise.
            let lobe = (2 * ZERO_PAD).max(1);
            let lo = best_bin.saturating_sub(lobe).max(first_bin);
            let hi = (best_bin + lobe + 1).min(spectrum.len());
            let peak_energy: f64 = spectrum[lo..hi].iter().sum();
            let purity = if total > 0.0 {
                peak_energy / total
            } else {
                0.0
            };

            peak_hz.push(refined);
            peak_level.push(best.sqrt());
            tone_likeness.push(purity.clamp(0.0, 1.0));
            frame_rms.push(rms(slice));
        }

        Track {
            rate: self.rate,
            hop: self.hop,
            window_len: self.window_len,
            bin_hz: self.bin_hz,
            peak_hz,
            peak_level,
            tone_likeness,
            frame_rms,
        }
    }
}

/// A compact time-frequency description of a signal: dominant frequency,
/// peak magnitude, tone likeness and amplitude per analysis frame.
#[derive(Debug, Clone)]
pub struct Track {
    rate: u32,
    hop: usize,
    window_len: usize,
    bin_hz: f64,
    peak_hz: Vec<f64>,
    peak_level: Vec<f64>,
    tone_likeness: Vec<f64>,
    frame_rms: Vec<f64>,
}

impl Track {
    /// Sample rate the track was measured at.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Hop between frames, in samples.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Analysis window length in samples.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window_len
    }

    /// Number of analysable frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.peak_hz.len()
    }

    /// Whether the track contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.peak_hz.is_empty()
    }

    /// Frequency resolution of the underlying FFT.
    #[must_use]
    pub fn bin_hz(&self) -> f64 {
        self.bin_hz
    }

    /// Median dominant frequency in Hz around `frame`.
    ///
    /// The median is deliberately preferred over the mean: a frame straddling
    /// a colour transition contains two tones, and their mean is a frequency
    /// the transmitter never sent.
    #[must_use]
    pub fn hz_at(&self, frame: usize) -> f64 {
        self.median_of(&self.peak_hz, frame)
    }

    /// Median dominant frequency at a fractional sample position.
    #[must_use]
    pub fn hz_at_sample(&self, sample: f64) -> f64 {
        self.interpolate(|t, f| t.median_of(&t.peak_hz, f), sample)
    }

    /// Median amplitude at a fractional sample position.
    #[must_use]
    pub fn level_at_sample(&self, sample: f64) -> f64 {
        self.interpolate(|t, f| t.median_of(&t.frame_rms, f), sample)
    }

    /// Median tone likeness at a fractional sample position.
    #[must_use]
    pub fn tone_likeness_at_sample(&self, sample: f64) -> f64 {
        self.interpolate(|t, f| t.median_of(&t.tone_likeness, f), sample)
    }

    /// Median peak magnitude at a fractional sample position.
    #[must_use]
    pub fn peak_level_at_sample(&self, sample: f64) -> f64 {
        self.interpolate(|t, f| t.median_of(&t.peak_level, f), sample)
    }

    fn median_of(&self, series: &[f64], frame: usize) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let frame = frame.min(series.len() - 1);
        let lo = frame.saturating_sub(MEDIAN_RADIUS);
        let hi = (frame + MEDIAN_RADIUS + 1).min(series.len());
        let window = &series[lo..hi];
        match window.len() {
            1 => window[0],
            2 => (window[0] + window[1]) * 0.5,
            _ => {
                // Small fixed-size sort: no allocation on this path.
                let mut a = window[0];
                let mut b = window[1];
                let mut c = window[2];
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                if b > c {
                    std::mem::swap(&mut b, &mut c);
                }
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                b
            }
        }
    }

    fn interpolate(&self, select: impl Fn(&Self, usize) -> f64, sample: f64) -> f64 {
        if self.peak_hz.is_empty() || !sample.is_finite() {
            return 0.0;
        }
        let position = (sample / self.hop as f64).max(0.0);
        let lower = position.floor();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let lower_index = lower as usize;
        if lower_index + 1 >= self.peak_hz.len() {
            return select(self, self.peak_hz.len() - 1);
        }
        let a = select(self, lower_index);
        let b = select(self, lower_index + 1);
        a + (b - a) * (position - lower)
    }
}

/// Root mean square of a slice.
#[must_use]
pub fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples
        .iter()
        .map(|x| {
            let x = f64::from(*x);
            x * x
        })
        .sum();
    (sum / samples.len() as f64).sqrt()
}

/// Peak absolute value of a slice.
#[must_use]
pub fn peak(samples: &[f32]) -> f64 {
    samples
        .iter()
        .copied()
        .map(|x| f64::from(x.abs()))
        .fold(0.0_f64, f64::max)
}

/// Median of a slice, sorting it in place. Returns 0 for an empty slice.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Mean of a slice. Returns 0 for an empty slice.
#[must_use]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Scale a signal so its peak sits at `target`, in place.
///
/// Returns the gain applied. A silent signal is left untouched rather than
/// amplified into numerical noise.
pub fn normalize_to_peak(samples: &mut [f32], target: f32) -> f32 {
    let current = peak(samples);
    if current <= 1.0e-9 {
        return 1.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let gain = (f64::from(target) / current) as f32;
    if (gain - 1.0).abs() > 1.0e-6 {
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
    gain
}

/// Remove the mean of a signal, in place.
pub fn remove_dc(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    #[allow(clippy::cast_possible_truncation)]
    let mean = (samples.iter().map(|x| f64::from(*x)).sum::<f64>() / samples.len() as f64) as f32;
    for sample in samples.iter_mut() {
        *sample -= mean;
    }
}

/// Linear-interpolated resampler.
///
/// Used to move a signal to the analysis rate. Linear interpolation is
/// adequate there because the analysis band is narrow and bounded well below
/// Nyquist; it is never used on a path that affects pixel fidelity.
#[must_use]
pub fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == 0 || output_rate == 0 {
        return Vec::new();
    }
    if input_rate == output_rate {
        return samples.to_vec();
    }
    let output_len = ((samples.len() as f64 * f64::from(output_rate) / f64::from(input_rate))
        .floor() as usize)
        .max(1);
    let mut out = Vec::with_capacity(output_len);
    let ratio = f64::from(input_rate) / f64::from(output_rate);
    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let left_f = pos.floor();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let left = left_f as usize;
        #[allow(clippy::cast_possible_truncation)]
        let frac = (pos - left_f) as f32;
        let a = samples.get(left).copied().unwrap_or(0.0);
        let b = samples.get(left + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

/// Grid step of the trajectory model, in seconds.
const MODEL_STEP_SECONDS: f64 = 0.001;

/// A measured signal reduced to a low-level model: dominant frequency over
/// time, plus the amplitude envelope and a tone-likeness figure.
///
/// This is the shared currency between VIS detection, mode inference, grid
/// extraction and candidate validation. Everything downstream reads the
/// signal exclusively through this view, so the layer that *decides* what the
/// recording contains and the layer that *checks* that decision can never
/// disagree about the evidence.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Sample positions the model is defined at, ascending.
    pub xs: Vec<f64>,
    /// Dominant frequency in Hz at each position.
    pub hz: Vec<f64>,
    /// Amplitude envelope at each position.
    pub level: Vec<f64>,
    /// Tone likeness at each position, `0.0..=1.0`.
    pub tone: Vec<f64>,
    rate: u32,
}

impl Trajectory {
    /// Build a trajectory from an analysis track.
    #[must_use]
    pub fn from_track(track: &Track, sample_count: usize) -> Self {
        let rate = track.rate();
        if track.is_empty() {
            return Self {
                xs: Vec::new(),
                hz: Vec::new(),
                level: Vec::new(),
                tone: Vec::new(),
                rate,
            };
        }
        let step = ((MODEL_STEP_SECONDS * f64::from(rate)).round() as usize).max(1);
        let steps = sample_count / step + 1;
        let mut xs = Vec::with_capacity(steps);
        let mut hz = Vec::with_capacity(steps);
        let mut level = Vec::with_capacity(steps);
        let mut tone = Vec::with_capacity(steps);
        for i in 0..steps {
            let x = (i * step) as f64;
            xs.push(x);
            hz.push(track.hz_at_sample(x));
            level.push(track.level_at_sample(x));
            tone.push(track.tone_likeness_at_sample(x));
        }
        Self {
            xs,
            hz,
            level,
            tone,
            rate,
        }
    }

    /// Build a trajectory from a known frequency model.
    ///
    /// Used by tests across the crate to exercise inference and
    /// reconstruction against an exactly known signal.
    #[must_use]
    pub fn from_model(xs: Vec<f64>, hz: Vec<f64>, rate: u32) -> Self {
        let len = xs.len().min(hz.len());
        Self {
            xs: xs[..len].to_vec(),
            hz: hz[..len].to_vec(),
            level: vec![1.0; len],
            tone: vec![1.0; len],
            rate,
        }
    }

    /// Build a trajectory from explicit series.
    #[must_use]
    pub fn from_series(
        xs: Vec<f64>,
        hz: Vec<f64>,
        level: Vec<f64>,
        tone: Vec<f64>,
        rate: u32,
    ) -> Self {
        let len = xs.len().min(hz.len()).min(level.len()).min(tone.len());
        Self {
            xs: xs[..len].to_vec(),
            hz: hz[..len].to_vec(),
            level: level[..len].to_vec(),
            tone: tone[..len].to_vec(),
            rate,
        }
    }

    /// Working sample rate of the trajectory.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Number of model points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Whether the model is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Last sample position covered by the model.
    #[must_use]
    pub fn last_sample(&self) -> f64 {
        self.xs.last().copied().unwrap_or(0.0)
    }

    /// Interpolated frequency at a sample position.
    #[must_use]
    pub fn hz_at(&self, sample: f64) -> f64 {
        self.sample_at(&self.hz, sample)
    }

    /// Interpolated amplitude at a sample position.
    #[must_use]
    pub fn level_at(&self, sample: f64) -> f64 {
        self.sample_at(&self.level, sample)
    }

    /// Interpolated tone likeness at a sample position.
    #[must_use]
    pub fn tone_at(&self, sample: f64) -> f64 {
        self.sample_at(&self.tone, sample)
    }

    fn sample_at(&self, series: &[f64], sample: f64) -> f64 {
        if series.is_empty() || !sample.is_finite() {
            return 0.0;
        }
        if self.xs.len() < 2 {
            return series[0];
        }
        let step = self.xs[1] - self.xs[0];
        if step <= 0.0 {
            return series[0];
        }
        let position = (sample - self.xs[0]) / step;
        if position <= 0.0 {
            return series[0];
        }
        let lower = position.floor();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let index = lower as usize;
        if index + 1 >= series.len() {
            return series[series.len() - 1];
        }
        let a = series[index];
        let b = series[index + 1];
        a + (b - a) * (position - lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: u32, seconds: f64) -> Vec<f32> {
        let n = (f64::from(rate) * seconds) as usize;
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    #[test]
    fn analyzer_recovers_a_pure_tone() {
        let rate = 22_050;
        let analyzer = Analyzer::new(rate);
        let signal = tone(1200.0, rate, 0.5);
        let track = analyzer.track(&signal);
        assert!(track.len() > 50);
        for frame in 10..track.len() - 10 {
            assert!(
                (track.hz_at(frame) - 1200.0).abs() < 10.0,
                "frame {frame} measured {}",
                track.hz_at(frame)
            );
        }
    }

    #[test]
    fn analyzer_resolves_frequencies_across_the_image_band() {
        let rate = 22_050;
        let analyzer = Analyzer::new(rate);
        let mut previous = f64::NEG_INFINITY;
        for hz in [1500.0, 1700.0, 1900.0, 2100.0, 2300.0] {
            let track = analyzer.track(&tone(hz, rate, 0.1));
            let measured = track.hz_at(track.len() / 2);
            assert!(
                (measured - hz).abs() < 10.0,
                "expected {hz}, measured {measured}"
            );
            assert!(measured > previous);
            previous = measured;
        }
    }

    #[test]
    fn zero_padding_beats_raw_bin_spacing() {
        // The whole point of zero padding: measured accuracy must be much
        // finer than one FFT bin.
        let rate = 44_100;
        let analyzer = Analyzer::new(rate);
        assert!(analyzer.bin_hz() < 40.0);
        let track = analyzer.track(&tone(1234.0, rate, 0.2));
        let measured = track.hz_at(track.len() / 2);
        assert!((measured - 1234.0).abs() < 10.0, "measured {measured}");
    }

    #[test]
    fn track_interpolates_between_frames() {
        let rate = 22_050;
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&tone(1200.0, rate, 0.2));
        let mid = track.hop() as f64 * 4.5;
        assert!((track.hz_at_sample(mid) - 1200.0).abs() < 10.0);
    }

    #[test]
    fn track_survives_out_of_range_queries() {
        let analyzer = Analyzer::new(22_050);
        let track = analyzer.track(&tone(1200.0, 22_050, 0.1));
        // Past the end clamps; nonsense input returns 0 rather than panicking.
        assert!(track.hz_at_sample(1.0e12) > 0.0);
        assert_eq!(track.hz_at_sample(f64::NAN), 0.0);
        assert!(track.hz_at_sample(-5.0) > 0.0);
    }

    #[test]
    fn tone_likeness_separates_tone_from_noise() {
        let rate = 22_050;
        let analyzer = Analyzer::new(rate);
        let tone_track = analyzer.track(&tone(1200.0, rate, 0.3));
        // Deterministic pseudo-noise, no RNG dependency.
        let noise: Vec<f32> = (0..rate / 5)
            .map(|i| {
                let x = (i as f64 * 12.9898).sin() * 43_758.545_3;
                (x - x.floor()) as f32 * 2.0 - 1.0
            })
            .collect();
        let noise_track = analyzer.track(&noise);
        let t = tone_track.tone_likeness_at_sample(10_000.0);
        let n = noise_track.tone_likeness_at_sample(10_000.0);
        assert!(t > n * 2.0, "tone {t} should dominate noise {n}");
        assert!(t > 0.1, "tone likeness of a pure tone was {t}");
    }

    #[test]
    fn hann_window_is_symmetric_and_bounded() {
        let w = hann(65);
        assert_eq!(w.len(), 65);
        for (a, b) in w.iter().zip(w.iter().rev()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(w.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(hann(0).is_empty());
        assert_eq!(hann(1), vec![1.0]);
    }

    #[test]
    fn median_and_mean_handle_degenerate_input() {
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(median(&mut [3.0]), 3.0);
        assert_eq!(median(&mut [3.0, 1.0]), 2.0);
        assert_eq!(median(&mut [5.0, 1.0, 3.0]), 3.0);
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_rescales_and_reports_gain() {
        let mut samples = vec![0.25_f32, -0.5, 0.125];
        let gain = normalize_to_peak(&mut samples, 1.0);
        assert!((gain - 2.0).abs() < 1e-6);
        assert!((peak(&samples) - 1.0).abs() < 1e-6);
        let mut silent = vec![0.0_f32; 16];
        assert_eq!(normalize_to_peak(&mut silent, 1.0), 1.0);
    }

    #[test]
    fn remove_dc_centres_a_signal() {
        let mut samples = vec![1.0_f32, 1.0, 1.0, 1.0];
        remove_dc(&mut samples);
        assert!(samples.iter().all(|x| x.abs() < 1e-6));
        let mut empty: Vec<f32> = Vec::new();
        remove_dc(&mut empty);
    }

    #[test]
    fn resample_preserves_a_tone() {
        let rate = 44_100;
        let signal = tone(1200.0, rate, 0.3);
        let moved = resample_linear(&signal, rate, 22_050);
        let analyzer = Analyzer::new(22_050);
        let track = analyzer.track(&moved);
        let mid = track.len() / 2;
        assert!((track.hz_at(mid) - 1200.0).abs() < 20.0);
    }

    #[test]
    fn analyzer_handles_signals_shorter_than_one_frame() {
        let analyzer = Analyzer::new(22_050);
        assert!(analyzer.track(&[]).is_empty());
        assert!(analyzer.track(&[0.0_f32; 8]).is_empty());
        let empty = analyzer.track(&[]);
        assert_eq!(empty.hz_at_sample(0.0), 0.0);
        assert!(empty.median_of(&[], 0) == 0.0);
    }

    #[test]
    fn analyzer_uses_a_bounded_frame() {
        let analyzer = Analyzer::new(16_000);
        assert!(analyzer.window_len() <= 256);
        assert_eq!(analyzer.hop(), analyzer.window_len() / 2);
        assert!(analyzer.fft_len() >= analyzer.window_len() * ZERO_PAD);
    }

    #[test]
    fn trajectory_interpolates_and_clamps() {
        let trajectory = Trajectory::from_model(
            vec![0.0, 10.0, 20.0, 30.0],
            vec![1000.0, 1100.0, 1200.0, 1300.0],
            22_050,
        );
        assert_eq!(trajectory.hz_at(0.0), 1000.0);
        assert!((trajectory.hz_at(5.0) - 1050.0).abs() < 1e-9);
        assert!((trajectory.hz_at(25.0) - 1250.0).abs() < 1e-9);
        // Clamped outside the modelled range.
        assert_eq!(trajectory.hz_at(-100.0), 1000.0);
        assert_eq!(trajectory.hz_at(1.0e9), 1300.0);
        assert_eq!(trajectory.hz_at(f64::NAN), 0.0);
    }

    #[test]
    fn trajectory_from_track_tracks_a_sweep() {
        let rate = 22_050;
        let n = rate as usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                // Phase is the integral of a linearly rising frequency, so
                // the tone stays continuous.
                (2.0 * PI * (1500.0 * t + 400.0 * t * t)).sin() as f32
            })
            .collect();
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        let quarter = trajectory.hz_at(0.25 * f64::from(rate));
        let three_quarter = trajectory.hz_at(0.75 * f64::from(rate));
        assert!(three_quarter > quarter + 200.0);
        assert!((quarter - 1700.0).abs() < 40.0, "quarter {quarter}");
        assert!(
            (three_quarter - 2100.0).abs() < 40.0,
            "three_quarter {three_quarter}"
        );
    }

    #[test]
    fn trajectory_handles_empty_and_single_point_models() {
        let empty = Trajectory::from_model(Vec::new(), Vec::new(), 22_050);
        assert!(empty.is_empty());
        assert_eq!(empty.hz_at(0.0), 0.0);
        assert_eq!(empty.last_sample(), 0.0);

        let single = Trajectory::from_model(vec![5.0], vec![1200.0], 22_050);
        assert_eq!(single.len(), 1);
        assert_eq!(single.hz_at(0.0), 1200.0);
        assert_eq!(single.hz_at(1.0e6), 1200.0);

        // A degenerate x-spacing must not divide by zero.
        let degenerate = Trajectory::from_model(vec![0.0, 0.0], vec![1.0, 2.0], 22_050);
        assert_eq!(degenerate.hz_at(0.0), 1.0);
    }

    #[test]
    fn trajectory_from_series_truncates_to_the_shortest_input() {
        let trajectory = Trajectory::from_series(
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0],
            22_050,
        );
        assert_eq!(trajectory.len(), 1);
    }
}
