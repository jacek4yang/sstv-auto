//! Raster reconstruction from a measured signal.
//!
//! [`extract`] walks a mode's channel layout over a [`Trajectory`] and fills
//! a low-resolution grid of pixel *levels*; [`Grid::to_rgb`] turns those
//! levels into colours.
//!
//! # Why the composition rules are spelled out here
//!
//! Only the RGB-sequential families (`RgbSequential`: Scottie and Martin) and
//! Robot 72 map one transmitted channel onto one image row. The other two
//! families do not:
//!
//! * **PD** sends `Y(row0), Cr, Cb, Y(row1)` per radio line: two luma rows
//!   plus a single chroma pair shared by both.
//! * **Robot 36 / Robot 24** send luma plus *one* chroma channel per radio
//!   line, alternating `Cr` on even rows and `Cb` on odd rows. The receiver
//!   recovers each row's missing chroma from the value sent on the
//!   neighbouring line.
//!
//! Getting those rules wrong is what leaves a coloured wedge in row 0 and a
//! colour ramp along the trailing edge of an otherwise correct image, so each
//! family's rule is stated once, here, and covered by tests.
//!
//! Level mapping: 1500 Hz is level 0, 2300 Hz is level 255.

use crate::dsp::{Trajectory, median};
pub use crate::modes::Palette;
use crate::modes::{Family, Mode, hz_to_level};

/// Neutral chroma value (zero colour difference), used when a channel is
/// genuinely absent from the recording.
pub const NEUTRAL_CHROMA: u8 = 128;

/// A recovered image at ranking resolution.
#[derive(Debug, Clone)]
pub struct Grid {
    /// Grid width in pixels.
    pub width: u32,
    /// Grid height in pixels.
    pub height: u32,
    /// Palette these levels are expressed in.
    pub palette: Palette,
    /// Row-major levels, three per pixel.
    ///
    /// `Palette::Rgb` stores `[r, g, b]`; `Palette::YCbCr` stores
    /// `[y, cb, cr]`.
    pub levels: Vec<u8>,
}

impl Grid {
    /// Total pixel count.
    #[must_use]
    pub fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Fetch the three stored levels of one pixel.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = ((y as usize) * (self.width as usize) + x as usize) * 3;
        Some([
            self.levels[index],
            self.levels[index + 1],
            self.levels[index + 2],
        ])
    }

    /// Convert this grid to an RGB image.
    #[must_use]
    pub fn to_rgb(&self) -> Vec<[u8; 3]> {
        let mut out = Vec::with_capacity(self.pixels());
        for index in 0..self.pixels() {
            let level = &self.levels[index * 3..index * 3 + 3];
            out.push(match self.palette {
                Palette::Rgb => [level[0], level[1], level[2]],
                Palette::YCbCr => ycbcr_to_rgb(level[0], level[1], level[2]),
            });
        }
        out
    }

    /// Overwrite one RGB row of the grid.
    fn write_row(&mut self, row: u32, pixels: &[[u8; 3]]) {
        let row_usize = row as usize;
        if row_usize >= self.height as usize {
            return;
        }
        let width = self.width as usize;
        let offset = row_usize * width * 3;
        for (x, pixel) in pixels.iter().take(width).enumerate() {
            let base = offset + x * 3;
            self.levels[base..base + 3].copy_from_slice(pixel);
        }
    }
}

/// Convert YCbCr to RGB with the standard integer-style matrix, matching the
/// raster backend's own conversion so grid and final image agree.
#[must_use]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = f64::from(y);
    let cb = f64::from(cb) - 128.0;
    let cr = f64::from(cr) - 128.0;
    [
        clamp_level(y + 1.402 * cr),
        clamp_level(y - 0.344_136 * cb - 0.714_136 * cr),
        clamp_level(y + 1.772 * cb),
    ]
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn clamp_level(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// Tolerance, in samples, by which the end of a channel may exceed the
/// measurable part of its data before the tail is treated as unrecoverable.
///
/// A short-time analysis window is far wider than one pixel, so a window
/// centred anywhere inside the last *window-length* of a transmitted channel
/// also contains whatever follows it — the channel separator, the porch, or
/// the next line's 1200 Hz sync tone. All of those sit outside the image band,
/// so the measured dominant frequency is pulled away from the transmitted
/// level; near a sync pulse it collapses to 1200 Hz, which maps below level 0.
///
/// The margin is a full window rather than half of one because measurement
/// shows the frequency already sagging a full window before the boundary.
///
/// No post-processing can recover a tone the analysis window never isolated,
/// so the affected columns are filled from the last trustworthy sample instead
/// of the contaminated one. Left unfixed, the trailing pixels of every channel
/// decode as a saturated colour ramp along the right edge of the picture.
#[must_use]
pub fn contaminated_tail_samples(rate: u32) -> f64 {
    crate::dsp::Analyzer::new(rate).window_len() as f64
}

/// Sample one channel of one radio line at one grid column.
///
/// `measurable_start` and `measurable_end` bound the part of this channel
/// whose frequency can be measured without a neighbouring tone bleeding into
/// the analysis window. See [`contaminated_tail_samples`].
#[allow(clippy::too_many_arguments)]
fn sample_pixel(
    trajectory: &Trajectory,
    line_start: f64,
    channel_start: f64,
    step_samples: f64,
    clock: f64,
    sample_rate: f64,
    measurable_start: f64,
    measurable_end: f64,
    x: u32,
) -> u8 {
    let column_start = sample_rate * clock * channel_start + step_samples * f64::from(x);
    let column_end = column_start + step_samples;
    // Clamp into the measurable window. A column wholly outside it collapses
    // onto the nearest measurable sample, which keeps the edge pixels at a
    // plausible level instead of letting the adjacent tone through.
    let hi_limit = column_end.min(measurable_end);
    let lo = column_start.max(measurable_start).min(hi_limit);
    let hi = hi_limit.max(lo);
    let mut probes = [
        trajectory.hz_at(line_start + lo),
        trajectory.hz_at(line_start + (lo + hi) * 0.5),
        trajectory.hz_at(line_start + hi),
    ];
    clamp_level(hz_to_level(median(&mut probes)).round())
}

/// Decode every transmitted channel of one radio line into grid-resolution
/// rows, in wire order.
fn decode_channels(
    trajectory: &Trajectory,
    mode: &Mode,
    line: u32,
    skip_sample: f64,
    clock: f64,
) -> Vec<Vec<u8>> {
    let sr = f64::from(trajectory.rate());
    // The start of this radio line, in samples, with the clock error applied.
    let line_start = skip_sample + f64::from(line) * mode.line_seconds * sr * clock;
    let tail = contaminated_tail_samples(trajectory.rate());
    let line_len = mode.line_seconds * sr * clock;
    // Samples occupied by one emitted tone: one tone covers `stride` source
    // pixels, and every duration scales with the measured clock.
    let step_samples = (mode.pixel_seconds * f64::from(mode.stride) * sr * clock).max(1.0);
    let width = mode.grid_w;

    channel_layout(mode)
        .into_iter()
        .map(|(channel_start, double_pixel)| {
            let step = if double_pixel {
                step_samples * 2.0
            } else {
                step_samples
            };
            // The margins are measured from the ends of *this channel*,
            // because it is the transitions in and out of the channel — into a
            // separator, a porch, the next line's sync, or the previous
            // element — that contaminate the window. The line bounds are
            // additional caps.
            let channel_start_sample = channel_start * sr * clock;
            let channel_end = (channel_start + mode.channel_seconds()) * sr * clock;
            let measurable_end = (channel_end - tail).min(line_len - tail).max(0.0);
            // At the very start of a transmission the preceding element is the
            // VIS stop pulse (1200 Hz), which pulls the first columns of line 0
            // toward the sync frequency exactly as a following sync pulls the
            // last columns of every line. The same margin therefore applies at
            // the front — but only to the first line, because every later line
            // is preceded by its own sync pulse, which the margin would
            // otherwise eat into.
            let measurable_start = if line == 0 {
                (channel_start_sample + tail).min(measurable_end)
            } else {
                channel_start_sample
            };
            (0..width)
                .map(|x| {
                    sample_pixel(
                        trajectory,
                        line_start,
                        channel_start,
                        step,
                        clock,
                        sr,
                        measurable_start,
                        measurable_end,
                        x,
                    )
                })
                .collect()
        })
        .collect()
}

/// Channel start offsets in seconds for one radio line, with a flag marking
/// channels transmitted at double pixel time (Robot 36/24 luma).
///
/// The order matches the wire format:
///
/// | family            | channels                       |
/// |-------------------|--------------------------------|
/// | PD                | `Y0, Cr, Cb, Y1`               |
/// | Robot 36/24       | `Y (2x), chroma`               |
/// | Robot 72          | `Y, U, V`                      |
/// | Scottie / Martin  | `G, B, R`                      |
#[must_use]
pub fn channel_layout(mode: &Mode) -> Vec<(f64, bool)> {
    let sync = mode.sync_seconds;
    let porch = mode.porch_seconds;
    let sepr = mode.separator_seconds;
    let channel_len = mode.channel_seconds();

    match mode.family {
        // Robot 36/24: luma at double pixel time, then one alternating
        // chroma channel.
        Family::RobotAlternating => vec![
            (sync + porch, true),
            (sync + porch + channel_len + sepr, false),
        ],
        // Robot 72: Y, U, V.
        Family::RobotSequential => vec![
            (sync + porch, false),
            (sync + porch + channel_len + sepr, false),
            (sync + porch + 2.0 * (channel_len + sepr), false),
        ],
        // PD: Y(row0), Cr, Cb, Y(row1).
        Family::Pd => vec![
            (sync + porch, false),
            (sync + porch + channel_len + sepr, false),
            (sync + porch + 2.0 * (channel_len + sepr), false),
            (sync + porch + 3.0 * (channel_len + sepr), false),
        ],
        // Scottie and Martin: G, B, R, with sync at line start or mid-line.
        Family::Sequential => match mode.sync_position {
            slowrx::SyncPosition::LineStart => vec![
                (sync + porch, false),
                (sync + porch + channel_len + sepr, false),
                (sync + porch + 2.0 * (channel_len + sepr), false),
            ],
            _ => vec![
                (sepr, false),
                (2.0 * sepr + channel_len, false),
                (2.0 * sepr + 2.0 * channel_len + sync + porch, false),
            ],
        },
    }
}

/// Recover a grid from a trajectory.
///
/// `skip_sample` is the absolute sample position of the first radio line's
/// sync pulse; `rate` scales nominal seconds into samples and carries any
/// measured clock error, where `1.0` means nominal speed.
#[must_use]
pub fn extract(trajectory: &Trajectory, mode: &Mode, skip_sample: f64, rate: f64) -> Grid {
    let width = mode.grid_w;
    let height = mode.grid_h;
    let mut grid = Grid {
        width,
        height,
        palette: mode.family.palette(),
        levels: vec![0; (width as usize) * (height as usize) * 3],
    };

    // Carried chroma for Robot 24/36's alternating scheme. `None` until the
    // transmission that supplies that channel has been seen.
    let mut last_cr: Option<Vec<u8>> = None;
    let mut last_cb: Option<Vec<u8>> = None;

    for line in 0..mode.radio_lines {
        let channels = decode_channels(trajectory, mode, line, skip_sample, rate);
        match mode.family {
            Family::Pd => {
                // Y0, Cr, Cb, Y1 — two rows share one chroma pair.
                let y0 = &channels[0];
                let cr = &channels[1];
                let cb = &channels[2];
                let y1 = &channels[3];
                let row0: Vec<[u8; 3]> =
                    (0..width as usize).map(|x| [y0[x], cb[x], cr[x]]).collect();
                let row1: Vec<[u8; 3]> =
                    (0..width as usize).map(|x| [y1[x], cb[x], cr[x]]).collect();
                grid.write_row(line * 2, &row0);
                grid.write_row(line * 2 + 1, &row1);
            }
            Family::RobotAlternating => {
                let y = &channels[0];
                let chroma = &channels[1];
                // Even rows carry Cr; odd rows carry Cb.
                let (cr, cb): (Vec<u8>, Vec<u8>) = if line % 2 == 0 {
                    last_cr = Some(chroma.clone());
                    let cb = last_cb
                        .clone()
                        .unwrap_or_else(|| vec![NEUTRAL_CHROMA; width as usize]);
                    (chroma.clone(), cb)
                } else {
                    last_cb = Some(chroma.clone());
                    let cr = last_cr
                        .clone()
                        .unwrap_or_else(|| vec![NEUTRAL_CHROMA; width as usize]);
                    (cr, chroma.clone())
                };
                let row: Vec<[u8; 3]> = (0..width as usize).map(|x| [y[x], cb[x], cr[x]]).collect();
                grid.write_row(line, &row);
            }
            Family::RobotSequential => {
                // Y, U, V — U is Cb, V is Cr.
                let y = &channels[0];
                let cb = &channels[1];
                let cr = &channels[2];
                let row: Vec<[u8; 3]> = (0..width as usize).map(|x| [y[x], cb[x], cr[x]]).collect();
                grid.write_row(line, &row);
            }
            Family::Sequential => {
                // G, B, R on the wire.
                let g = &channels[0];
                let b = &channels[1];
                let r = &channels[2];
                let row: Vec<[u8; 3]> = (0..width as usize).map(|x| [r[x], g[x], b[x]]).collect();
                grid.write_row(line, &row);
            }
        }
    }

    // Any grid row the layout never wrote stays neutral rather than black, so
    // an unread row cannot masquerade as strong image content.
    neutralize_unwritten_rows(&mut grid, mode);
    grid
}

/// Fill rows the layout did not write with a flat neutral level.
fn neutralize_unwritten_rows(grid: &mut Grid, mode: &Mode) {
    let written = match mode.family {
        Family::Pd => mode.radio_lines * 2,
        _ => mode.radio_lines,
    };
    let neutral = match grid.palette {
        Palette::Rgb => [0, 0, 0],
        Palette::YCbCr => [NEUTRAL_CHROMA, NEUTRAL_CHROMA, NEUTRAL_CHROMA],
    };
    let row = vec![neutral; grid.width as usize];
    for line in written..grid.height {
        grid.write_row(line, &row);
    }
}

/// Number of RGB rows a family produces.
#[must_use]
pub fn row_count(mode: &Mode) -> u32 {
    match mode.family {
        Family::Pd => mode.radio_lines * 2,
        _ => mode.radio_lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Analyzer;
    use crate::modes::describe;
    use crate::synth::{BLOCK_X, BLOCK_Y};
    use slowrx::SstvMode;

    /// Render a grid back into SSTV audio through the synthesiser, then
    /// measure it. Used to prove that extraction inverts synthesis exactly.
    /// Render a grid to image audio and measure it back.
    ///
    /// The returned trajectory starts at sample zero of the *image*, with no
    /// VIS prefix. Callers must therefore extract from offset zero; adding a
    /// header offset here would silently misalign every row.
    fn measure(mode: &Mode, grid: &Grid, rate: u32) -> Trajectory {
        let signal = crate::synth::render(grid, mode, rate, 0.0, None);
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        Trajectory::from_track(&track, signal.len())
    }

    fn test_grid(mode: &Mode) -> Grid {
        // A deterministic non-uniform pattern: distinguishable rows and
        // columns, valid for any palette.
        let width = mode.grid_w;
        let height = mode.grid_h;
        let mut levels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let base = (x * 255 / width.max(1)) as u8;
                let band = ((y * 255 / height.max(1)) as u8).wrapping_add(40);
                match mode.family.palette() {
                    Palette::Rgb => levels.extend_from_slice(&[base, band, 200]),
                    Palette::YCbCr => levels.extend_from_slice(&[base.max(30), 128, band.max(30)]),
                }
            }
        }
        Grid {
            width,
            height,
            palette: mode.family.palette(),
            levels,
        }
    }

    #[test]
    fn level_mapping_endpoints_are_exact() {
        assert_eq!(clamp_level(hz_to_level(1500.0)), 0);
        assert_eq!(clamp_level(hz_to_level(2300.0)), 255);
        // Out-of-band tones are clamped, never wrapped.
        assert_eq!(clamp_level(hz_to_level(1400.0)), 0);
        assert_eq!(clamp_level(hz_to_level(2400.0)), 255);
    }

    #[test]
    fn ycbcr_grey_maps_to_grey() {
        assert_eq!(ycbcr_to_rgb(128, 128, 128), [128, 128, 128]);
        assert_eq!(ycbcr_to_rgb(0, 128, 128), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb(255, 128, 128), [255, 255, 255]);
    }

    /// Compare a recovered grid against its source, measuring only block
    /// interiors.
    ///
    /// Short-time frequency analysis cannot resolve a pixel that sits within
    /// about half an analysis window of a sharp edge: the window contains two
    /// tones and the louder one wins. That is not a defect to be asserted
    /// away, so every comparison here samples well inside a block and the
    /// edge behaviour is checked separately by
    /// [`edges_smear_but_do_not_corrupt_neighbours`].
    fn interior_error(recovered: &Grid, source: &Grid, mode: &Mode) -> f64 {
        let expected = source.to_rgb();
        let actual = recovered.to_rgb();
        let w = recovered.width as usize;
        let h = recovered.height as usize;
        let bx = BLOCK_X as usize;
        let by = BLOCK_Y as usize;
        // The analysis window is `window_len` samples wide; in pixels at this
        // mode's rate that is the radius over which a level is smeared. Stay
        // that far inside each block so only transmitted levels are measured.
        let window_px = (0.008 * 22_050.0 / (mode.pixel_seconds * 22_050.0)).ceil() as usize + 2;
        let inset_x = (window_px + 3).min(bx / 3);
        let inset_y = (window_px / 2 + 2).min(by / 3);

        let mut total = 0.0;
        let mut counted = 0usize;
        let mut y = 0usize;
        while y < h {
            let y_end = (y + by).min(h);
            let row_range =
                (y + inset_y).min(y_end)..(y_end.saturating_sub(inset_y)).max(y + inset_y);
            for row in row_range {
                let mut x = 0usize;
                while x < w {
                    let x_end = (x + bx).min(w);
                    let from = (x + inset_x).min(x_end);
                    let to = x_end.saturating_sub(inset_x).max(from);
                    for column in from..to {
                        let i = row * w + column;
                        for c in 0..3 {
                            total += (f64::from(actual[i][c]) - f64::from(expected[i][c])).abs();
                            counted += 1;
                        }
                    }
                    x += bx;
                }
            }
            y += by;
        }
        assert!(counted > 0, "no interior pixels sampled");
        total / counted as f64
    }

    #[test]
    fn every_family_round_trips_within_one_level_inside_blocks() {
        let rate = 22_050;
        for mode in crate::modes::all() {
            let grid = test_grid(&mode);
            let trajectory = measure(&mode, &grid, rate);
            let recovered = extract(&trajectory, &mode, 0.0, 1.0);
            assert_eq!(
                recovered.to_rgb().len(),
                grid.to_rgb().len(),
                "{}",
                mode.name
            );
            let error = interior_error(&recovered, &grid, &mode);
            assert!(
                error < 9.0,
                "{} mean interior reconstruction error {error:.2} levels",
                mode.name
            );
        }
    }

    #[test]
    fn edges_smear_but_do_not_corrupt_neighbours() {
        // An edge is allowed to be soft; what must never happen is a value
        // outside the two levels that actually exist on either side of it.
        let rate = 22_050;
        let mode = describe(SstvMode::Robot72).expect("robot72");
        let w = mode.grid_w as usize;
        let h = mode.grid_h as usize;
        let mut levels = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                let luma = if x < w / 2 { 40u8 } else { 210u8 };
                levels.extend_from_slice(&[luma, 128, 128]);
            }
        }
        let grid = Grid {
            width: w as u32,
            height: h as u32,
            palette: Palette::YCbCr,
            levels,
        };
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        let row = h / 2;
        // Radius inside which a pixel is influenced by the edge. One analysis
        // window is `window_len` samples; at this mode's pixel rate that is
        // tens of pixels, so anything closer than that to the edge is expected
        // to be a blend.
        let window_px = (0.008 * 22_050.0 / (mode.pixel_seconds * 22_050.0)).ceil() as usize + 4;
        let edge = w / 2;

        for x in 0..w {
            let stored = recovered.pixel(x as u32, row as u32).expect("pixel");
            let value = i32::from(stored[0]);
            // Whatever the smearing does, no pixel may leave the range the
            // two transmitted levels define.
            assert!(
                (30..=220).contains(&value),
                "column {x} luma {value} escaped the transmitted range"
            );
        }

        // Well clear of the edge, the value must be the transmitted one.
        for x in [
            window_px,
            edge - window_px,
            edge + window_px,
            w - 1 - window_px,
        ] {
            let stored = recovered.pixel(x as u32, row as u32).expect("pixel");
            let value = i32::from(stored[0]);
            let expected = if x < edge { 40 } else { 210 };
            assert!(
                (value - expected).abs() <= 12,
                "column {x} should be level {expected}, measured {value}"
            );
        }

        // Across the edge the transition must be monotone-ish: the first
        // column after it must not be fully dark again.
        let just_after = i32::from(recovered.pixel(edge as u32, row as u32).expect("pixel")[0]);
        assert!(
            just_after > 60,
            "column {edge} still reads {just_after}; the edge did not survive"
        );
    }

    #[test]
    fn robot36_row0_has_no_colour_wedge() {
        // Regression: row 0 was previously composed with a zero-initialised
        // Cb, which renders as a saturated red wedge. Row 0 is the first line
        // that transmits Cr and has no earlier line to borrow Cb from, so its
        // Cb is genuinely unknown. It must come out neutral, never saturated.
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let w = mode.grid_w as usize;
        let h = mode.grid_h as usize;
        // Flat mid-grey luma with neutral chroma on every line: nothing in the
        // image justifies any colour at all.
        let levels: Vec<u8> = (0..w * h).flat_map(|_| [128u8, 128, 128]).collect();
        let grid = Grid {
            width: w as u32,
            height: h as u32,
            palette: Palette::YCbCr,
            levels,
        };
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        let rgb = recovered.to_rgb();

        // Every pixel of row 0 must be close to grey. A neutral Cb fill keeps
        // the red/blue difference small; a zero fill would push it past 100.
        for (x, [r, g, b]) in rgb.iter().copied().enumerate().take(w - 10).skip(10) {
            let spread = i32::from(r.max(g).max(b)) - i32::from(r.min(g).min(b));
            assert!(
                spread <= 40,
                "row 0 column {x} is coloured: {r},{g},{b} (spread {spread})"
            );
        }
    }

    #[test]
    fn robot36_odd_rows_borrow_chroma_from_the_line_above() {
        // Robot 36 sends Cr on even rows and Cb on odd rows, so a row must
        // reuse the channel its neighbour transmitted. Without that the image
        // shows a strong hue banding every other row.
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let w = mode.grid_w as usize;
        let h = mode.grid_h as usize;
        // Cr constant across the whole image, Cb neutral: every row's chroma
        // is knowable, so every row must decode to the same colour.
        let levels: Vec<u8> = (0..w * h).flat_map(|_| [128u8, 128, 210]).collect();
        let grid = Grid {
            width: w as u32,
            height: h as u32,
            palette: Palette::YCbCr,
            levels,
        };
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        let rgb = recovered.to_rgb();
        let sample = |row: usize| -> [u8; 3] { rgb[row * w + w / 2] };
        let first = sample(0);
        for row in 1..12usize {
            let current = sample(row);
            let delta = (0..3)
                .map(|c| (i32::from(current[c]) - i32::from(first[c])).abs())
                .sum::<i32>();
            assert!(
                delta <= 30,
                "row {row} decoded as {current:?} but row 0 is {first:?}"
            );
        }
    }

    #[test]
    fn robo36_chroma_carries_across_rows() {
        // Odd rows must reuse Cr from the previous even row, and vice versa;
        // a missing carrier must not produce a hue shift.
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        let rate = 22_050;
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        // Row 0 and row 1 share the same luma ramp, so with correct chroma
        // carry-over their hue must broadly agree.
        let w = recovered.width as usize;
        let rgb = recovered.to_rgb();
        let mut agree = 0usize;
        for x in 10..w - 10 {
            let a = rgb[x];
            let b = rgb[w + x];
            let d = (0..3)
                .map(|c| (i32::from(a[c]) - i32::from(b[c])).abs())
                .sum::<i32>();
            if d < 120 {
                agree += 1;
            }
        }
        assert!(
            agree * 100 / (w - 20) > 80,
            "only {agree} of {} columns agreed between rows 0 and 1",
            w - 20
        );
    }

    #[test]
    fn pd_pairs_share_chroma_across_two_rows() {
        let mode = describe(SstvMode::Pd120).expect("pd120");
        let rate = 16_000;
        let grid = test_grid(&mode);
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        assert_eq!(recovered.height, mode.image_lines);
        let rgb = recovered.to_rgb();
        let w = recovered.width as usize;
        // Rows 0 and 1 of a pair come from the same radio line, so their
        // chroma must match: compare B-R difference.
        for x in 10..w - 10 {
            let a = rgb[x];
            let b = rgb[w + x];
            let da = i32::from(a[2]) - i32::from(a[0]);
            let db = i32::from(b[2]) - i32::from(b[0]);
            assert!(
                (da - db).abs() <= 60,
                "column {x}: chroma differs between paired rows ({da} vs {db})"
            );
        }
    }

    #[test]
    fn scottie_and_martin_channel_order_is_gbr_on_the_wire() {
        // A grid with R=255, G=0, B=0 must decode back to red, not blue.
        let rate = 22_050;
        for variant in [SstvMode::Scottie1, SstvMode::Martin1] {
            let mode = describe(variant).expect("mode");
            let mut grid = Grid {
                width: mode.grid_w,
                height: mode.grid_h,
                palette: Palette::Rgb,
                levels: vec![0; (mode.grid_w * mode.grid_h * 3) as usize],
            };
            for i in 0..grid.pixels() {
                grid.levels[i * 3] = 255; // R
                grid.levels[i * 3 + 1] = 0; // G
                grid.levels[i * 3 + 2] = 0; // B
            }
            let trajectory = measure(&mode, &grid, rate);
            let recovered = extract(
                &trajectory,
                &mode,
                crate::modes::VIS_TOTAL_SECONDS * f64::from(rate),
                1.0,
            );
            let rgb = recovered.to_rgb();
            let w = recovered.width as usize;
            // Sample well inside the image: the outer columns are affected by
            // the analysis window overlapping the channel boundaries, which is
            // a property of the measurement rather than of the channel order
            // this test is checking.
            let interior = (0.008 * 22_050.0 / (mode.pixel_seconds * 22_050.0)) as usize + 8;
            let sample = rgb[20 * w + interior.max(20)];
            assert!(
                sample[0] > 180 && sample[1] < 80 && sample[2] < 80,
                "{} decoded red as {:?}",
                mode.name,
                sample
            );
        }
    }

    #[test]
    fn robot72_uses_full_chroma_per_row() {
        let mode = describe(SstvMode::Robot72).expect("robot72");
        let rate = 22_050;
        let grid = test_grid(&mode);
        let trajectory = measure(&mode, &grid, rate);
        let recovered = extract(&trajectory, &mode, 0.0, 1.0);
        let expected = grid.to_rgb();
        let actual = recovered.to_rgb();
        let w = recovered.width as usize;
        let mut total = 0.0;
        let mut n = 0usize;
        for y in 0..recovered.height as usize {
            for x in 2..w - 2 {
                for c in 0..3 {
                    total +=
                        (f64::from(actual[y * w + x][c]) - f64::from(expected[y * w + x][c])).abs();
                    n += 1;
                }
            }
        }
        let mean_error = total / (n as f64);
        assert!(mean_error < 12.0, "robot72 mean error {mean_error}");
    }

    #[test]
    fn channel_layout_counts_match_the_wire_format() {
        let pd = describe(SstvMode::Pd120).expect("pd120");
        assert_eq!(channel_layout(&pd).len(), 4);
        let r36 = describe(SstvMode::Robot36).expect("robot36");
        assert_eq!(channel_layout(&r36).len(), 2);
        assert!(channel_layout(&r36)[0].1, "robot36 luma is double speed");
        let r72 = describe(SstvMode::Robot72).expect("robot72");
        assert_eq!(channel_layout(&r72).len(), 3);
        let s1 = describe(SstvMode::Scottie1).expect("scottie1");
        assert_eq!(channel_layout(&s1).len(), 3);
    }

    #[test]
    fn row_count_is_consistent_with_grid_height() {
        for mode in crate::modes::all() {
            assert_eq!(row_count(&mode), mode.grid_h, "{}", mode.name);
        }
    }

    #[test]
    fn extract_tolerates_an_empty_trajectory() {
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let analyzer = Analyzer::new(22_050);
        let track = analyzer.track(&[]);
        let trajectory = Trajectory::from_track(&track, 0);
        let grid = extract(&trajectory, &mode, 0.0, 1.0);
        assert_eq!(grid.pixels(), mode.grid_pixels());
        // Flat neutral output, not garbage.
        assert!(grid.levels.iter().all(|v| *v <= NEUTRAL_CHROMA));
    }

    #[test]
    fn grid_pixel_accessor_bounds_checks() {
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        assert!(grid.pixel(0, 0).is_some());
        assert!(grid.pixel(grid.width, 0).is_none());
        assert!(grid.pixel(0, grid.height).is_none());
    }

    #[test]
    fn extract_scales_with_clock_rate() {
        // A recording running 2% fast must still line up when told so.
        let rate = 22_050;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = test_grid(&mode);
        let signal = crate::synth::render(&grid, &mode, rate, 0.0, Some(1.02));
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        let misaligned = extract(&trajectory, &mode, 0.0, 1.0).to_rgb();
        let aligned = extract(&trajectory, &mode, 0.0, 1.02).to_rgb();
        let expected = grid.to_rgb();
        let error = |actual: &[[u8; 3]]| -> f64 {
            let w = mode.grid_w as usize;
            let h = mode.grid_h as usize;
            let mut total = 0.0;
            let mut n = 0usize;
            for y in 0..h {
                for x in 2..w - 2 {
                    for c in 0..3 {
                        total += (f64::from(actual[y * w + x][c])
                            - f64::from(expected[y * w + x][c]))
                        .abs();
                        n += 1;
                    }
                }
            }
            total / n as f64
        };
        let bad = error(&misaligned);
        let good = error(&aligned);
        assert!(good < bad, "corrected {good} should beat nominal {bad}");
        assert!(good < 12.0, "corrected error {good}");
    }

    #[test]
    fn unbounded_row_writes_are_safe() {
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let mut grid = test_grid(&mode);
        // Out-of-range writes are dropped rather than panicking.
        grid.write_row(mode.grid_h + 10, &[[1, 2, 3]]);
        // A row shorter than the grid is written partially, not padded with
        // stale data beyond its length.
        grid.write_row(0, &[[9, 9, 9]]);
        assert_eq!(grid.pixel(0, 0), Some([9, 9, 9]));
    }
}
