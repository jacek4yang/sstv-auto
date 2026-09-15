//! Machine-readable and human-readable reporting.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// Detection summary for one decoded image.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    /// Human-readable mode name.
    pub mode: String,
    /// Stable lowercase mode slug.
    pub mode_slug: String,
    /// 7-bit VIS code for the mode.
    pub vis_code: u8,
    /// How the mode was determined (`vis`, `vis-parity-failed`, `sync-period`).
    pub detected_by: String,
    /// 0.0..=1.0 confidence in the detection.
    pub confidence: f64,
    /// How well a re-render of the recovered image matched the recording.
    pub signal_agreement: f64,
    /// Fraction of the expected transmission present in the recording.
    pub coverage: f64,
    /// Whether the whole transmission was present.
    pub complete: bool,
    /// Start of the image body, in seconds from the beginning of the file.
    pub image_start_seconds: f64,
    /// Measured receiver frequency offset, Hz.
    pub frequency_offset_hz: f64,
    /// Measured transmitter clock scale, where 1.0 is nominal.
    pub clock_rate: f64,
    /// Clock error as a percentage, for readability.
    pub clock_error_percent: f64,
    /// Number of sync pulses consistent with this hypothesis.
    pub matched_syncs: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Plausibility score of the final image, 0.0..=1.0.
    pub image_quality: f64,
    /// Evidence notes gathered while evaluating this hypothesis.
    pub evidence: Vec<String>,
    /// Other modes that would decode identically without a VIS header.
    pub ambiguous_with: Vec<String>,
    /// Path of the written PNG.
    pub output_png: PathBuf,
}

/// Facts about the input audio.
#[derive(Debug, Clone, Serialize)]
pub struct AudioReport {
    /// Input path as given on the command line.
    pub input: PathBuf,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Sample rate used for analysis, Hz.
    pub analysis_rate_hz: u32,
    /// Channel count of the source file.
    pub source_channels: usize,
    /// Which channel was used.
    pub selected_channel: String,
    /// Peak level after normalisation.
    pub normalized_peak: f32,
}

/// The complete run report, written as `report.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Version of the producing tool.
    pub tool_version: String,
    /// Input facts.
    pub audio: AudioReport,
    /// Number of hypotheses evaluated.
    pub hypotheses_evaluated: usize,
    /// Detections, best first.
    pub detections: Vec<Detection>,
    /// Human-readable warnings and notes.
    pub warnings: Vec<String>,
}

impl Report {
    /// Serialise to pretty JSON.
    ///
    /// # Errors
    ///
    /// Fails if the report cannot be serialised, which would indicate a bug in
    /// a `Serialize` implementation rather than bad input.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).context("serialize report to JSON")
    }

    /// Write the report to `path`.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be written.
    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_json()?).with_context(|| format!("write {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Report {
        Report {
            tool_version: "0.2.0".to_owned(),
            audio: AudioReport {
                input: PathBuf::from("in.wav"),
                duration_seconds: 36.9,
                analysis_rate_hz: 44_100,
                source_channels: 1,
                selected_channel: "channel 0 (mono)".to_owned(),
                normalized_peak: 0.95,
            },
            hypotheses_evaluated: 3,
            detections: vec![Detection {
                mode: "Robot 36".to_owned(),
                mode_slug: "robot36".to_owned(),
                vis_code: 0x08,
                detected_by: "vis".to_owned(),
                confidence: 0.95,
                signal_agreement: 0.91,
                coverage: 1.0,
                complete: true,
                image_start_seconds: 0.93,
                frequency_offset_hz: 4.2,
                clock_rate: 1.001,
                clock_error_percent: 0.1,
                matched_syncs: 240,
                width: 320,
                height: 240,
                image_quality: 0.8,
                evidence: vec!["VIS parity ok".to_owned()],
                ambiguous_with: vec!["Robot 24".to_owned()],
                output_png: PathBuf::from("out/001-robot36.png"),
            }],
            warnings: vec!["example warning".to_owned()],
        }
    }

    #[test]
    fn report_serialises_with_every_documented_field() {
        let json = sample().to_json().expect("serialise");
        let text = String::from_utf8(json).expect("utf-8");
        for key in [
            "tool_version",
            "audio",
            "hypotheses_evaluated",
            "detections",
            "warnings",
            "mode",
            "mode_slug",
            "vis_code",
            "detected_by",
            "confidence",
            "signal_agreement",
            "coverage",
            "complete",
            "image_start_seconds",
            "frequency_offset_hz",
            "clock_rate",
            "clock_error_percent",
            "matched_syncs",
            "width",
            "height",
            "image_quality",
            "evidence",
            "ambiguous_with",
            "output_png",
            "duration_seconds",
            "analysis_rate_hz",
            "source_channels",
            "selected_channel",
            "normalized_peak",
        ] {
            assert!(text.contains(key), "report is missing `{key}`");
        }
    }

    #[test]
    fn report_round_trips_through_json() {
        let original = sample();
        let json = original.to_json().expect("serialise");
        let value: serde_json::Value = serde_json::from_slice(&json).expect("parse");
        assert_eq!(value["detections"][0]["vis_code"], 8);
        assert_eq!(value["audio"]["source_channels"], 1);
    }

    #[test]
    fn report_can_be_written_to_disk() {
        let dir = std::env::temp_dir().join("sstv_auto_report_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("report.json");
        sample().write(&path).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("Robot 36"));
        let _ = std::fs::remove_file(&path);
    }
}
