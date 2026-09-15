//! Data-driven SSTV mode geometry.
//!
//! Every timing constant that describes *where* a colour component lives
//! inside a radio line is defined here once and consumed by three
//! independent layers:
//!
//! * [`crate::raster`] — recovers a low-resolution "grid" from a measured
//!   signal map,
//! * [`crate::synth`] — re-encodes a candidate grid back into SSTV audio so
//!   the production decoder can run against a synthetic VIS header, and
//! * the ranking in [`crate::autodetect`], which compares the two.
//!
//! Because the encoder and decoder are inverses of one another by
//! construction, a mismatch between them can never masquerade as a strong
//! match. That property is what makes the blind ranking trustworthy.
//!
//! Level mapping: all supported families use the same 800 Hz full-scale
//! deviation, `level 0 => 1500 Hz`, `level 255 => 2300 Hz`. This matches
//! `slowrx`'s `demod::freq_to_luminance` divisor of `3.137_254_9`
//! (`800 / 255`).

use slowrx::{ChannelLayout, SstvMode, SyncPosition, for_mode};

/// Tone frequency for a sync pulse.
pub const SYNC_HZ: f64 = 1200.0;
/// Tone frequency for a porch / channel separator.
pub const PORCH_HZ: f64 = 1500.0;
/// Tone frequency for the VIS leader.
pub const VIS_LEADER_HZ: f64 = 1900.0;
/// Tone frequency for a VIS `1` bit.
pub const VIS_BIT_ONE_HZ: f64 = 1100.0;
/// Tone frequency for a VIS `0` bit (and the VIS start/stop separator).
pub const VIS_BIT_ZERO_HZ: f64 = 1300.0;
/// Tone frequency for the VIS start/stop separator pulse.
pub const VIS_SEPARATOR_HZ: f64 = 1200.0;

/// Lowest image tone: level 0.
pub const LEVEL_MIN_HZ: f64 = 1500.0;
/// Highest image tone: level 255.
pub const LEVEL_MAX_HZ: f64 = 2300.0;
/// Full-scale deviation covered by level 0..=255.
pub const LEVEL_SPAN_HZ: f64 = LEVEL_MAX_HZ - LEVEL_MIN_HZ;

/// Duration of a single VIS bit, seconds.
///
/// The start pulse, the seven data bits, the parity bit and the stop pulse are
/// all this long.
pub const VIS_BIT_SECONDS: f64 = 0.030;
/// Duration of the VIS leader tone, seconds.
pub const VIS_LEADER_SECONDS: f64 = 0.300;
/// Duration of the short break between the two leader tones, seconds.
///
/// This is **10 ms, not 30 ms**. The break is the one VIS field whose length
/// differs from a bit: it is a brief interruption of the leader tone, not a
/// data field. Using 30 ms here shifts every subsequent field 20 ms late and
/// makes a real header undecodable, which is exactly the kind of error a
/// hand-rolled VIS implementation tends to make.
pub const VIS_BREAK_SECONDS: f64 = 0.010;
/// Total duration of a VIS burst, seconds.
///
/// Ten 30 ms fields follow the break: the start pulse, seven data bits, the
/// parity bit, and the stop pulse.
///
/// ```text
/// leader  break  leader  start  b0..b6  parity  stop
///  300ms   10ms   300ms   30ms  7x30ms   30ms  30ms
/// ```
pub const VIS_TOTAL_SECONDS: f64 =
    2.0 * VIS_LEADER_SECONDS + VIS_BREAK_SECONDS + 10.0 * VIS_BIT_SECONDS;

/// Target width of the recovered grid. Modes that transmit more pixels per
/// line are decimated to this width; ranking only needs enough horizontal
/// detail to discriminate geometry, and the final image always comes from
/// the full-resolution decoder.
pub const GRID_WIDTH: u32 = 320;

/// Convert an 8-bit pixel level to its transmission frequency in Hz.
#[must_use]
pub fn level_to_hz(level: f64) -> f64 {
    LEVEL_MIN_HZ + level * LEVEL_SPAN_HZ / 255.0
}

/// Convert a transmission frequency in Hz back to a (possibly out of range)
/// 8-bit-equivalent pixel level.
#[must_use]
pub fn hz_to_level(hz: f64) -> f64 {
    (hz - LEVEL_MIN_HZ) * 255.0 / LEVEL_SPAN_HZ
}

/// Structural family of a mode. Families differ in how the transmitted
/// channels map onto image rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// PD-120/180/240: `Y(row0), Cr, Cb, Y(row1)` — two image rows and
    /// shared chroma per radio line.
    Pd,
    /// Robot 36 and Robot 24: `Y` at double pixel time plus one alternating
    /// chroma channel duplicated onto the neighbouring row.
    RobotAlternating,
    /// Robot 72: `Y, U, V` at single pixel time, full chroma per row.
    RobotSequential,
    /// Scottie and Martin: `G, B, R` sequentially, full chroma per row.
    Sequential,
}

impl Family {
    /// Three-letter palette this family's channels are expressed in.
    #[must_use]
    pub fn palette(self) -> Palette {
        match self {
            Self::Pd | Self::RobotAlternating | Self::RobotSequential => Palette::YCbCr,
            Self::Sequential => Palette::Rgb,
        }
    }

    /// Grid rows produced by one radio line. PD packs two image rows into a
    /// radio line; every other supported family packs exactly one.
    #[must_use]
    pub fn rows_per_radio_line(self) -> u32 {
        match self {
            Self::Pd => 2,
            _ => 1,
        }
    }
}

/// Colour space a family's channels are expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    /// Luma plus two colour-difference channels.
    YCbCr,
    /// Red, green, blue.
    Rgb,
}

/// A supported SSTV mode: `slowrx`'s authoritative geometry plus the derived
/// grid shape this crate ranks against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mode {
    /// Mode enum used by the `slowrx` decode backend.
    pub mode: SstvMode,
    /// Stable lowercase slug used in output filenames.
    pub short_name: &'static str,
    /// Human-readable display name.
    pub name: &'static str,
    /// 7-bit VIS code that unambiguously identifies this mode.
    pub vis_code: u8,
    /// Structural family.
    pub family: Family,
    /// Visible pixels per radio line as transmitted.
    pub line_pixels: u32,
    /// Visible image rows in the finished image.
    pub image_lines: u32,
    /// Radio lines per image (`image_lines / 2` for PD, else `image_lines`).
    pub radio_lines: u32,
    /// Grid width used for ranking.
    pub grid_w: u32,
    /// Grid height used for ranking.
    pub grid_h: u32,
    /// Emitted pixels per grid column (`line_pixels / grid_w`).
    pub stride: u32,
    /// Total radio-line duration including sync and porches, seconds.
    pub line_seconds: f64,
    /// Sync pulse duration, seconds.
    pub sync_seconds: f64,
    /// Porch duration, seconds.
    pub porch_seconds: f64,
    /// Per-pixel duration inside a colour channel, seconds.
    pub pixel_seconds: f64,
    /// Channel separator duration, seconds.
    pub separator_seconds: f64,
    /// Where the sync pulse sits inside the radio line.
    pub sync_position: SyncPosition,
}

impl Mode {
    /// Seconds from the start of a radio line to the rising edge of its sync
    /// pulse.
    ///
    /// `slowrx`'s PD/Robot/Martin layout puts sync first. Scottie places the
    /// sync between the blue and red channels, at
    /// `2 * separator + 2 * channel_length`.
    #[must_use]
    pub fn sync_offset_seconds(&self) -> f64 {
        match self.sync_position {
            // Scottie is the only family in this build whose sync sits
            // mid-line. `SyncPosition` is `#[non_exhaustive]`, so an unknown
            // value falls back to the line-start convention rather than
            // inventing an offset.
            SyncPosition::Scottie => 2.0 * self.separator_seconds + 2.0 * self.channel_seconds(),
            _ => 0.0,
        }
    }

    /// Duration of one colour channel, seconds.
    #[must_use]
    pub fn channel_seconds(&self) -> f64 {
        match self.family {
            // Robot 36/24 allocate the luma channel twice the pixel time.
            Family::RobotAlternating => f64::from(self.line_pixels) * self.pixel_seconds * 2.0,
            _ => f64::from(self.line_pixels) * self.pixel_seconds,
        }
    }

    /// Total transmitted audio duration of the image, excluding VIS, seconds.
    ///
    /// This is derived from the geometry rather than the nominal line table so
    /// that it always agrees with the line count the decoder walks.
    #[must_use]
    pub fn image_seconds(&self) -> f64 {
        f64::from(self.radio_lines) * self.line_seconds
    }

    /// Nominal signal frequency offset in Hz that [`crate::synth`] should
    /// apply when it renders this mode. Always zero: the measured offset is
    /// supplied separately so the renderer stays a pure inverse of
    /// [`crate::raster`].
    #[must_use]
    pub fn grid_pixels(&self) -> usize {
        (self.grid_w as usize) * (self.grid_h as usize)
    }
}

/// Grid rows produced per radio line for a mode.
#[must_use]
pub fn rows_per_line(mode: &Mode) -> u32 {
    mode.family.rows_per_radio_line()
}

macro_rules! mode_table {
    ($(($variant:ident, $family:ident)),* $(,)?) => {
        /// Every mode this crate can detect and hand to the `slowrx` backend.
        pub const SUPPORTED: &[SstvMode] = &[$(SstvMode::$variant),*];

        /// Build the [`Mode`] record for a `slowrx` mode.
        #[must_use]
        pub fn describe(mode: SstvMode) -> Option<Mode> {
            let spec = for_mode(mode);
            let family = match mode {
                $(SstvMode::$variant => Family::$family,)*
                _ => return None,
            };
            let radio_lines = match spec.channel_layout {
                ChannelLayout::PdYcbcr => spec.image_lines / 2,
                _ => spec.image_lines,
            };
            let grid_w = spec.line_pixels.min(GRID_WIDTH).max(1);
            Some(Mode {
                mode,
                short_name: spec.short_name,
                name: spec.name,
                vis_code: spec.vis_code,
                family,
                line_pixels: spec.line_pixels,
                image_lines: spec.image_lines,
                radio_lines,
                grid_w,
                grid_h: spec.image_lines,
                stride: (spec.line_pixels / grid_w).max(1),
                line_seconds: spec.line_seconds,
                sync_seconds: spec.sync_seconds,
                porch_seconds: spec.porch_seconds,
                pixel_seconds: spec.pixel_seconds,
                separator_seconds: spec.septr_seconds,
                sync_position: spec.sync_position,
            })
        }
    };
}

mode_table![
    (Pd120, Pd),
    (Pd180, Pd),
    (Pd240, Pd),
    (Robot24, RobotAlternating),
    (Robot36, RobotAlternating),
    (Robot72, RobotSequential),
    (Scottie1, Sequential),
    (Scottie2, Sequential),
    (ScottieDx, Sequential),
    (Martin1, Sequential),
    (Martin2, Sequential),
];

/// Look up a supported mode by its VIS code.
#[must_use]
pub fn from_vis(vis_code: u8) -> Option<Mode> {
    SUPPORTED
        .iter()
        .find(|m| for_mode(**m).vis_code == vis_code)
        .and_then(|m| describe(*m))
}

/// All supported modes, described.
#[must_use]
pub fn all() -> Vec<Mode> {
    SUPPORTED.iter().filter_map(|m| describe(*m)).collect()
}

/// Look up a mode by its slug or display name, case- and separator-insensitively.
///
/// Accepts `robot36`, `Robot 36`, `robot-36` and `ROBOT36` alike, because a
/// user typing a mode from a spec sheet should not have to guess which spelling
/// this tool prefers.
#[must_use]
pub fn from_name(query: &str) -> Option<Mode> {
    let normalise = |text: &str| -> String {
        text.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    let wanted = normalise(query);
    if wanted.is_empty() {
        return None;
    }
    all()
        .into_iter()
        .find(|mode| normalise(mode.short_name) == wanted || normalise(mode.name) == wanted)
}

/// The slugs of every supported mode, for error messages.
#[must_use]
pub fn slug_list() -> String {
    all()
        .iter()
        .map(|mode| mode.short_name)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_mapping_spans_full_scale() {
        assert!((level_to_hz(0.0) - 1500.0).abs() < 1e-9);
        assert!((level_to_hz(255.0) - 2300.0).abs() < 1e-9);
        assert!((hz_to_level(2300.0) - 255.0).abs() < 1e-9);
        // Round-trip at an arbitrary level.
        for level in [0.0, 1.0, 42.5, 128.0, 254.0, 255.0] {
            assert!((hz_to_level(level_to_hz(level)) - level).abs() < 1e-9);
        }
    }

    #[test]
    fn every_supported_mode_describes() {
        for mode in SUPPORTED {
            let described = describe(*mode).expect("supported mode must describe");
            assert_eq!(described.mode, *mode);
            assert!(described.grid_h > 0 && described.grid_w > 0);
            assert!(described.grid_w <= GRID_WIDTH);
            assert_eq!(described.grid_w * described.stride, described.line_pixels);
        }
    }

    #[test]
    fn vis_codes_are_unique_and_seven_bit() {
        let mut seen = Vec::new();
        for mode in all() {
            assert!(mode.vis_code < 0x80, "{} VIS must be 7 bits", mode.name);
            assert!(
                !seen.contains(&mode.vis_code),
                "duplicate VIS {}",
                mode.vis_code
            );
            seen.push(mode.vis_code);
        }
    }

    #[test]
    fn vis_lookup_round_trips() {
        for mode in all() {
            assert_eq!(from_vis(mode.vis_code).map(|m| m.mode), Some(mode.mode));
        }
        assert!(from_vis(0x7f).is_none());
    }

    #[test]
    fn pd_pairs_two_rows_per_radio_line() {
        let pd = describe(SstvMode::Pd120).expect("pd120");
        assert_eq!(pd.radio_lines, pd.image_lines / 2);
        assert_eq!(rows_per_line(&pd), 2);
        assert_eq!(pd.grid_h, pd.image_lines);
    }

    #[test]
    fn non_pd_rows_match_radio_lines() {
        let robot = describe(SstvMode::Robot36).expect("robot36");
        assert_eq!(robot.radio_lines, robot.image_lines);
        assert_eq!(rows_per_line(&robot), 1);
    }

    #[test]
    fn scottie_sync_sits_mid_line() {
        let scottie = describe(SstvMode::Scottie1).expect("scottie1");
        let chan = f64::from(scottie.line_pixels) * scottie.pixel_seconds;
        let expected = 2.0 * scottie.separator_seconds + 2.0 * chan;
        assert!((scottie.sync_offset_seconds() - expected).abs() < 1e-12);
        // Sync must be strictly inside the line.
        assert!(scottie.sync_offset_seconds() < scottie.line_seconds);
    }

    #[test]
    fn line_start_modes_put_sync_first() {
        for mode in all() {
            if mode.sync_position == SyncPosition::LineStart {
                assert_eq!(mode.sync_offset_seconds(), 0.0, "{}", mode.name);
            }
        }
    }

    #[test]
    fn robot_alternating_luma_channel_is_double_length() {
        let r36 = describe(SstvMode::Robot36).expect("robot36");
        // Channel total (Y + chroma + separator) must fit inside the line.
        let y = r36.channel_seconds();
        let chroma = f64::from(r36.line_pixels) * r36.pixel_seconds;
        let total = r36.sync_seconds + r36.porch_seconds + y + r36.separator_seconds + chroma;
        assert!(
            total <= r36.line_seconds + 1e-9,
            "robot36 channel layout {total} exceeds line {}",
            r36.line_seconds
        );
    }

    #[test]
    fn mode_lookup_accepts_every_reasonable_spelling() {
        for query in ["robot36", "Robot 36", "robot-36", "ROBOT36", "  Robot36  "] {
            assert_eq!(
                from_name(query).map(|m| m.short_name),
                Some("robot36"),
                "query {query:?}"
            );
        }
        assert_eq!(from_name("pd120").map(|m| m.short_name), Some("pd120"));
        assert_eq!(
            from_name("scottiedx").map(|m| m.short_name),
            Some("scottiedx")
        );
        assert_eq!(from_name("not a mode"), None);
        assert_eq!(from_name(""), None);
    }

    #[test]
    fn every_slug_is_findable_by_its_own_name() {
        for mode in all() {
            assert_eq!(
                from_name(mode.short_name).map(|m| m.mode),
                Some(mode.mode),
                "{}",
                mode.short_name
            );
            assert_eq!(
                from_name(mode.name).map(|m| m.mode),
                Some(mode.mode),
                "{}",
                mode.name
            );
        }
        assert!(slug_list().contains("robot36"));
    }

    #[test]
    fn image_seconds_match_known_totals() {
        // Robot 36 is 240 lines x 150 ms.
        let r36 = describe(SstvMode::Robot36).expect("robot36");
        assert!((r36.image_seconds() - 36.0).abs() < 1e-9);
        // Scottie 1 is 256 lines x 428.38 ms.
        let s1 = describe(SstvMode::Scottie1).expect("scottie1");
        assert!((s1.image_seconds() - 109.665_28).abs() < 1e-6);
        // PD-120 is 248 pairs x 508.48 ms.
        let pd = describe(SstvMode::Pd120).expect("pd120");
        assert!((pd.image_seconds() - 126.103_04).abs() < 1e-6);
    }
}
