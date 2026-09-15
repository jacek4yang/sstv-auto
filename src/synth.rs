//! Re-encoding a grid back into SSTV audio.
//!
//! [`render`] is the inverse of [`crate::raster::extract`]: it walks the same
//! channel layout at the same per-pixel timing, so a grid recovered from a
//! recording re-encodes into audio that maps back to the same grid.
//!
//! That inverse relationship is the basis of the whole blind path. Instead of
//! guessing a mode and trusting whatever image comes out, the decoder:
//!
//! 1. infers mode, frequency offset and clock from measurable signal
//!    properties,
//! 2. extracts a grid,
//! 3. re-encodes that grid with this module,
//! 4. measures the result and compares it against the *original* recording.
//!
//! A wrong hypothesis cannot score well, because steps 1 and 4 measure the
//! same physical quantity through the same code path. What survives is
//! genuinely explained by the inferred mode.

use std::f64::consts::TAU;

use crate::modes::{Family, Mode, PORCH_HZ, SYNC_HZ, level_to_hz};
use crate::raster::{Grid, Palette};

/// One contiguous tone run inside a radio line.
struct Segment {
    /// Where this segment starts, relative to the start of the radio line, in
    /// samples.
    start: f64,
    /// Palette component each pixel is taken from.
    component: usize,
    /// Grid row the pixels come from.
    row: u32,
    /// Samples occupied by one emitted pixel.
    pixel_samples: f64,
}

/// Render a grid into continuous-phase SSTV audio.
///
/// * `rate` is the output sample rate in Hz.
/// * `frequency_shift_hz` is added to every tone, modelling receiver
///   mistuning.
/// * `clock_rate` scales every duration; `None` means nominal speed, while a
///   recording playing 2% fast is reproduced with `Some(1.02)`.
///
/// The result contains image audio only: no VIS header, no leading silence.
///
/// # Fidelity
///
/// Pixel boundaries are computed in floating point and only rounded when a
/// tone is actually emitted. Rounding a *per-pixel* sample count first would
/// quantise the pixel clock, and the error would accumulate across the line:
/// at 44.1 kHz a Scottie DX pixel is 47.6 samples, so rounding to 48 drifts by
/// more than half a pixel every hundred columns and the render would no
/// longer line up with what [`crate::raster::extract`] measures.
#[must_use]
pub fn render(
    grid: &Grid,
    mode: &Mode,
    rate: u32,
    frequency_shift_hz: f64,
    clock_rate: Option<f64>,
) -> Vec<f32> {
    let clock = clock_rate.unwrap_or(1.0);
    let sr = f64::from(rate);
    let layout = crate::raster::channel_layout(mode);

    // Line boundaries are derived from the absolute nominal position of each
    // line rather than accumulated, so rounding cannot drift over a long
    // transmission.
    let line_seconds = mode.line_seconds * clock;
    let total = (f64::from(mode.radio_lines) * line_seconds * sr).round() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(total);
    let mut phase = 0.0_f64;

    // Each line is written against its absolute nominal boundaries. Pixel
    // boundaries inside a line are computed in floating point and rounded
    // only at the point a tone is emitted, so a pixel width that is not a
    // whole number of samples does not accumulate error across the line.
    for line in 0..mode.radio_lines {
        let line_start = out.len();
        let line_end = (f64::from(line + 1) * line_seconds * sr).round() as usize;
        let line_len = line_end.saturating_sub(line_start);
        if line_len == 0 {
            continue;
        }

        let mut segments: Vec<Segment> = Vec::with_capacity(layout.len() + 1);

        let sync_samples = mode.sync_seconds * sr * clock;
        if sync_samples >= 1.0 {
            segments.push(Segment {
                start: mode.sync_offset_seconds() * sr * clock,
                component: 0,
                row: 0,
                pixel_samples: -sync_samples, // negative marks a constant tone
            });
        }

        for (wire_index, (channel_start, double_pixel)) in layout.iter().enumerate() {
            let (Some(component), Some(row)) = (
                wire_component(mode, line, wire_index),
                wire_row(mode, line, wire_index),
            ) else {
                continue;
            };
            let mut pixel_samples = mode.pixel_seconds * sr * clock * f64::from(mode.stride);
            if *double_pixel {
                pixel_samples *= 2.0;
            }
            if pixel_samples < 1.0 {
                continue;
            }
            segments.push(Segment {
                start: channel_start * sr * clock,
                component,
                row,
                pixel_samples,
            });
        }

        segments.sort_by(|a, b| a.start.total_cmp(&b.start));

        for segment in &segments {
            let written = out.len() - line_start;
            if segment.start as usize >= line_len {
                break;
            }
            let gap = segment.start - written as f64;
            if gap >= 1.0 {
                emit_tone(
                    &mut out,
                    &mut phase,
                    rate,
                    PORCH_HZ + frequency_shift_hz,
                    gap.round() as usize,
                );
            }

            if segment.pixel_samples < 0.0 {
                // A constant tone: a sync pulse.
                emit_tone(
                    &mut out,
                    &mut phase,
                    rate,
                    SYNC_HZ + frequency_shift_hz,
                    (-segment.pixel_samples).round() as usize,
                );
                continue;
            }

            // One tone per pixel. The boundary of pixel `x` is measured from
            // the channel's own start, so error stays bounded by half a
            // sample no matter how many pixels precede it.
            //
            // Robot 36/24 finish the chroma channel exactly on the line
            // boundary with no slack at all, and floating point can put the
            // final boundary a few ulps past the line length. Dropping that
            // last pixel would leave a gap that the next line's sync pulse
            // fills, which decodes as a hard colour break. The final pixel is
            // therefore clamped to the line instead of discarded.
            let channel_start_index = out.len() - line_start;
            let mut previous = 0usize;
            for x in 0..grid.width {
                let boundary = segment.pixel_samples * (f64::from(x) + 1.0);
                let is_last = x + 1 == grid.width;
                let projected = channel_start_index as f64 + boundary;
                if projected > line_len as f64 && !is_last {
                    break;
                }
                let level = grid
                    .pixel(x, segment.row)
                    .map_or(128.0, |pixel| f64::from(pixel[segment.component]));
                let target = (boundary.round() as usize).min(line_len - channel_start_index);
                let samples = target.saturating_sub(previous);
                if samples > 0 {
                    emit_tone(
                        &mut out,
                        &mut phase,
                        rate,
                        level_to_hz(level) + frequency_shift_hz,
                        samples,
                    );
                    previous = target;
                }
                if is_last {
                    break;
                }
            }
            if out.len() - line_start >= line_len {
                break;
            }
        }

        // Pad whatever is left of the line. The final channel rarely ends
        // exactly on a line boundary.
        let produced = out.len() - line_start;
        if produced < line_len {
            emit_tone(
                &mut out,
                &mut phase,
                rate,
                PORCH_HZ + frequency_shift_hz,
                line_len - produced,
            );
        }
    }

    out
}

/// Which grid row a wire channel of one radio line belongs to.
///
/// This mirrors [`crate::raster::extract`], which writes PD's two luma
/// channels into consecutive rows of the same radio line. If the two ever
/// disagree, a grid recovered from a recording will not re-render into
/// comparable audio.
///
/// Returns `None` for a channel the layout does not define.
#[must_use]
pub fn wire_row(mode: &Mode, line: u32, wire_index: usize) -> Option<u32> {
    match mode.family {
        // Y(row0), Cr, Cb, Y(row1).
        Family::Pd => match wire_index {
            0..=2 => Some(line * 2),
            3 => Some(line * 2 + 1),
            _ => None,
        },
        _ => Some(line),
    }
}

/// Which stored palette component a wire channel carries.
///
/// Wire order is family specific, and Robot 36/24 additionally rotate the
/// chroma channel with line parity:
///
/// | family           | wire order        | stored as                    |
/// |------------------|-------------------|------------------------------|
/// | PD / Robot 72    | `Y, Cb, Cr`       | `[y, cb, cr]`                |
/// | Robot 36/24      | `Y, Cr` (even) `Y, Cb` (odd) | `[y, cb, cr]`     |
/// | Scottie / Martin | `G, B, R`         | `[r, g, b]`                  |
///
/// Returns `None` for a wire/palette combination that has no meaning, so a
/// mismatch is reported rather than silently encoded wrong.
#[must_use]
pub fn wire_component(mode: &Mode, line: u32, wire_index: usize) -> Option<usize> {
    match (mode.family, mode.family.palette()) {
        // PD wire order is `Y(row0), Cr, Cb, Y(row1)` — note Cr comes *before*
        // Cb here, the opposite of the stored `[y, cb, cr]` order, so the two
        // chroma indices are crossed.
        (Family::Pd, Palette::YCbCr) => match wire_index {
            0 => Some(0), // Y(row0)
            1 => Some(2), // Cr
            2 => Some(1), // Cb
            3 => Some(0), // Y(row1)
            _ => None,
        },
        // Robot 72 wire order is `Y, U, V`, i.e. Y then Cb then Cr.
        (Family::RobotSequential, Palette::YCbCr) => match wire_index {
            0 => Some(0), // Y
            1 => Some(1), // Cb (U)
            2 => Some(2), // Cr (V)
            _ => None,
        },
        (Family::RobotAlternating, Palette::YCbCr) => match wire_index {
            0 => Some(0), // Y
            // Even radio lines transmit Cr, odd lines transmit Cb. The
            // extractor reconstructs the missing one from the neighbour.
            1 if line % 2 == 0 => Some(2),
            1 => Some(1),
            _ => None,
        },
        (Family::Sequential, Palette::Rgb) => match wire_index {
            0 => Some(1), // G
            1 => Some(2), // B
            2 => Some(0), // R
            _ => None,
        },
        _ => None,
    }
}

/// Emit `count` samples of a continuous-phase tone.
fn emit_tone(out: &mut Vec<f32>, phase: &mut f64, rate: u32, hz: f64, count: usize) {
    if count == 0 {
        return;
    }
    let increment = TAU * hz / f64::from(rate);
    for _ in 0..count {
        out.push((phase.sin() * 0.70) as f32);
        *phase = (*phase + increment).rem_euclid(TAU);
    }
}

/// Render image audio preceded by the VIS burst for `mode`, ready to hand to
/// the raster backend.
#[must_use]
pub fn render_with_vis(
    grid: &Grid,
    mode: &Mode,
    rate: u32,
    frequency_shift_hz: f64,
    clock_rate: Option<f64>,
) -> Vec<f32> {
    let mut out = crate::vis::synthesize(mode.vis_code, rate, frequency_shift_hz);
    out.extend(render(grid, mode, rate, frequency_shift_hz, clock_rate));
    out
}

/// Width of one block in [`test_grid`], in pixels. Wide enough that the
/// analysis window never spans two blocks at once.
pub const BLOCK_X: u32 = 64;

/// Height of one block in [`test_grid`], in rows.
pub const BLOCK_Y: u32 = 16;

/// A deterministic test pattern used by round-trip tests across the crate.
///
/// # Why the pattern is smooth
///
/// The analysis window is about 28 pixels wide at typical pixel rates. A
/// per-pixel tone therefore cannot be recovered exactly when neighbouring
/// pixels differ sharply: the window contains several tones and the dominant
/// one wins. That is a property of short-time frequency analysis, not a
/// defect, so the test pattern uses **blocky** content whose features are much
/// wider than the analysis window. Sharp per-pixel edges are deliberately not
/// asserted on, because the metric could not honestly support them.
#[must_use]
pub fn test_grid(mode: &Mode) -> Grid {
    let mut levels = Vec::new();
    for y in 0..mode.grid_h {
        for x in 0..mode.grid_w {
            let block_x = x / BLOCK_X;
            let block_y = y / BLOCK_Y;
            // Levels chosen to exercise the full range without saturation:
            // luma sweeps with x, chroma with y.
            let luma = (30 + (block_x * 60) % 200) as u8;
            let chroma = (40 + (block_y * 55) % 180) as u8;
            match mode.family.palette() {
                // [r, g, b]
                Palette::Rgb => levels.extend_from_slice(&[luma, chroma, 190]),
                // [y, cb, cr]
                Palette::YCbCr => levels.extend_from_slice(&[luma, 128, chroma.max(30)]),
            }
        }
    }
    Grid {
        width: mode.grid_w,
        height: mode.grid_h,
        palette: mode.family.palette(),
        levels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Analyzer, Trajectory, median};
    use crate::modes::{describe, hz_to_level};
    use slowrx::SstvMode;

    /// Measure the level a steady tone of `level` decodes back to, through
    /// the same analysis path the decoder uses.
    fn measured_level(level: u8, rate: u32) -> f64 {
        let count = (0.20 * f64::from(rate)) as usize;
        let hz = level_to_hz(f64::from(level));
        let signal: Vec<f32> = (0..count)
            .map(|i| (TAU * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect();
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let mut probes = [
            track.hz_at_sample(count as f64 * 0.3),
            track.hz_at_sample(count as f64 * 0.5),
            track.hz_at_sample(count as f64 * 0.7),
        ];
        hz_to_level(median(&mut probes))
    }

    #[test]
    fn the_analysis_path_is_level_linear() {
        // Guard for the whole architecture: if the measurement path were not
        // level linear, comparing a re-rendered signal against the original
        // would be meaningless.
        let rate = 22_050;
        let mut previous = f64::NEG_INFINITY;
        let mut pairs = Vec::new();
        for level in [0u8, 32, 64, 96, 128, 160, 192, 224, 255] {
            let measured = measured_level(level, rate);
            assert!(
                measured > previous,
                "level {level} measured {measured}, not above {previous}"
            );
            previous = measured;
            pairs.push((f64::from(level), measured));
        }
        assert!((pairs[0].1).abs() < 20.0, "level 0 measured {}", pairs[0].1);
        let last = pairs[pairs.len() - 1];
        assert!(
            (last.1 - 255.0).abs() < 20.0,
            "level 255 measured {}",
            last.1
        );

        let n = pairs.len() as f64;
        let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let num: f64 = pairs.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
        let den: f64 = pairs.iter().map(|p| (p.0 - mean_x) * (p.0 - mean_x)).sum();
        let slope = num / den;
        assert!(
            (0.9..=1.1).contains(&slope),
            "level response slope {slope} is not close to 1 across the range"
        );
    }

    #[test]
    fn render_produces_the_expected_duration() {
        let rate = 22_050;
        for mode in crate::modes::all() {
            let grid = test_grid(&mode);
            let signal = render(&grid, &mode, rate, 0.0, None);
            let seconds = signal.len() as f64 / f64::from(rate);
            assert!(
                (seconds - mode.image_seconds()).abs() < 0.02,
                "{} rendered {seconds}s, expected {}s",
                mode.name,
                mode.image_seconds()
            );
        }
    }

    #[test]
    fn render_stays_inside_the_sstv_band() {
        let rate = 22_050;
        let analyzer = Analyzer::new(rate);
        for mode in crate::modes::all() {
            let grid = test_grid(&mode);
            let signal = render(&grid, &mode, rate, 0.0, None);
            let track = analyzer.track(&signal);
            for frame in 0..track.len() {
                let hz = track.hz_at(frame);
                assert!(
                    (900.0..=2500.0).contains(&hz),
                    "{} frame {frame} at {hz} Hz is outside the SSTV band",
                    mode.name
                );
            }
        }
    }

    #[test]
    fn render_applies_a_frequency_offset() {
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        let analyzer = Analyzer::new(rate);
        for shift in [-100.0, 0.0, 100.0] {
            let signal = render(&grid, &mode, rate, shift, None);
            let track = analyzer.track(&signal);
            let found =
                (0..track.len()).any(|frame| (track.hz_at(frame) - (1200.0 + shift)).abs() < 25.0);
            assert!(found, "no sync pulse at offset {shift}");
        }
    }

    #[test]
    fn render_with_vis_is_detectable_by_our_own_detector() {
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        let signal = render_with_vis(&grid, &mode, rate, 0.0, None);
        let expected = crate::modes::VIS_TOTAL_SECONDS + mode.image_seconds();
        let actual = signal.len() as f64 / f64::from(rate);
        assert!((actual - expected).abs() < 0.05, "rendered {actual}s");

        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        let hit = crate::vis::find(&trajectory)
            .into_iter()
            .find(|h| h.code == mode.vis_code);
        assert!(hit.is_some(), "rendered VIS prefix was not detectable");
    }

    #[test]
    fn clock_scaling_changes_duration_proportionally() {
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        let nominal = render(&grid, &mode, rate, 0.0, None).len();
        let fast = render(&grid, &mode, rate, 0.0, Some(1.05)).len();
        let ratio = fast as f64 / nominal as f64;
        assert!((ratio - 1.05).abs() < 0.01, "ratio {ratio}");
    }

    #[test]
    fn line_boundaries_do_not_drift_over_a_long_mode() {
        // Regression guard: accumulating rounded line lengths drifts by
        // hundreds of samples across a 250 line mode, which would misalign
        // every late line during round-trip comparison.
        let rate = 44_100;
        for mode in crate::modes::all() {
            let grid = test_grid(&mode);
            let signal = render(&grid, &mode, rate, 0.0, None);
            let expected = (mode.image_seconds() * f64::from(rate)).round() as usize;
            let drift = signal.len().abs_diff(expected);
            assert!(
                drift <= 2,
                "{} drifted {drift} samples over {} radio lines",
                mode.name,
                mode.radio_lines
            );
        }
    }

    #[test]
    fn silent_grid_renders_a_valid_signal() {
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = Grid {
            width: mode.grid_w,
            height: mode.grid_h,
            palette: mode.family.palette(),
            levels: vec![0; mode.grid_pixels() * 3],
        };
        let signal = render(&grid, &mode, rate, 0.0, None);
        assert!(!signal.is_empty());
        assert!(signal.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn render_tolerates_a_degenerate_grid() {
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = Grid {
            width: 0,
            height: 0,
            palette: Palette::YCbCr,
            levels: Vec::new(),
        };
        let signal = render(&grid, &mode, rate, 0.0, None);
        assert!(signal.iter().all(|s| s.is_finite()));
        assert!(!signal.is_empty(), "line structure must still be emitted");
    }

    #[test]
    fn wire_component_matches_each_family() {
        // PD wire order is Y(row0), Cr, Cb, Y(row1), so the chroma indices
        // are crossed relative to the stored [y, cb, cr].
        let pd = describe(SstvMode::Pd120).expect("pd120");
        assert_eq!(wire_component(&pd, 0, 0), Some(0), "Y(row0)");
        assert_eq!(wire_component(&pd, 0, 1), Some(2), "Cr");
        assert_eq!(wire_component(&pd, 0, 2), Some(1), "Cb");
        assert_eq!(wire_component(&pd, 0, 3), Some(0), "Y(row1)");
        assert_eq!(wire_component(&pd, 0, 4), None);

        let s1 = describe(SstvMode::Scottie1).expect("scottie1");
        // Wire order G, B, R into a stored [r, g, b].
        assert_eq!(wire_component(&s1, 0, 0), Some(1));
        assert_eq!(wire_component(&s1, 0, 1), Some(2));
        assert_eq!(wire_component(&s1, 0, 2), Some(0));

        // Robot 36/24 rotate the chroma channel with line parity.
        let r36 = describe(SstvMode::Robot36).expect("robot36");
        assert_eq!(wire_component(&r36, 0, 1), Some(2), "even line sends Cr");
        assert_eq!(wire_component(&r36, 1, 1), Some(1), "odd line sends Cb");
        assert_eq!(wire_component(&r36, 2, 1), Some(2));
    }
}
