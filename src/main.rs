//! `sstv-auto` — automatically find and decode an SSTV image in an audio file.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use sstv_auto::audio::ChannelChoice;
use sstv_auto::pipeline::{self, DEFAULT_MAX_CANDIDATES, Options};

#[derive(Debug, Parser)]
#[command(
    name = "sstv-auto",
    version,
    about = "Find and decode an SSTV image in an audio file",
    long_about = "Decode SSTV without being told the mode, the timing, the frequency offset, \
                  the sample rate or the image size.\n\n\
                  The VIS header is used when it is present and intact. When it is missing or \
                  damaged, the line-sync period is measured and used to infer the mode instead, \
                  and the frequency offset and clock error are taken from the signal itself.\n\n\
                  Audio is decoded internally (WAV, FLAC, MP3, AAC/MP4, Ogg/Vorbis, AIFF, CAF, \
                  MKV); no external programs are required.",
    after_help = "EXAMPLES:\n  \
                  sstv-auto audio.wav\n  \
                  sstv-auto challenge.mp3 -o out\n  \
                  sstv-auto audio.wav --verbose\n  \
                  sstv-auto audio.wav --channel right\n  \
                  sstv-auto audio.wav --mode robot36"
)]
struct Cli {
    /// Audio file to decode.
    #[arg(value_name = "AUDIO")]
    input: PathBuf,

    /// Directory for the decoded PNGs and report.json.
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Which audio channel to analyse: `auto`, `mix`, `left`, `right`, or a
    /// zero-based index.
    #[arg(long, default_value = "auto", value_name = "WHICH")]
    channel: String,

    /// Restrict decoding to one mode, e.g. `robot36` or `pd120`, instead of
    /// detecting it. Useful when a recording is too damaged for automatic
    /// detection but the mode is known.
    #[arg(long, value_name = "MODE")]
    mode: Option<String>,

    /// Skip the no-VIS recovery pass, decoding only from a VIS header.
    #[arg(long)]
    no_blind: bool,

    /// Maximum number of ranked candidates to decode at full resolution.
    #[arg(long, default_value_t = DEFAULT_MAX_CANDIDATES, value_name = "N")]
    max_candidates: usize,

    /// Also write the images of candidates that were rejected.
    #[arg(long)]
    keep_candidates: bool,

    /// Print per-stage diagnostics to stderr.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let channel = ChannelChoice::parse(&cli.channel).with_context(|| {
        format!(
            "invalid --channel value `{}`; expected auto, mix, left, right, or an index",
            cli.channel
        )
    })?;

    let output_dir = cli.output.clone().unwrap_or_else(|| {
        let stem = cli
            .input
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("sstv");
        cli.input.with_file_name(format!("{stem}_sstv"))
    });

    let options = Options {
        channel,
        blind: !cli.no_blind,
        max_candidates: cli.max_candidates.max(1),
        keep_candidates: cli.keep_candidates,
        forced_mode: cli.mode.clone(),
        verbose: cli.verbose,
    };

    let outcome = pipeline::run(&cli.input, &output_dir, &options)?;

    for warning in &outcome.warnings {
        eprintln!("warning: {warning}");
    }

    if outcome.detections.is_empty() {
        if outcome.plausible_sstv {
            println!("SSTV was detected but could not be decoded into a usable image.");
        } else {
            println!("No SSTV transmission found.");
        }
        println!("Report: {}", outcome.report_path.display());
        std::process::exit(2);
    }

    println!(
        "Decoded {} image{}:",
        outcome.detections.len(),
        if outcome.detections.len() == 1 {
            ""
        } else {
            "s"
        }
    );
    for detection in &outcome.detections {
        println!(
            "  {}  {} (VIS 0x{:02x})  confidence {:.0}%",
            detection.output_png.display(),
            detection.mode,
            detection.vis_code,
            detection.confidence * 100.0
        );
        println!(
            "      detected by {} | {}x{} | image starts {:.3}s | offset {:+.1} Hz | clock {}",
            detection.detected_by,
            detection.width,
            detection.height,
            detection.image_start_seconds,
            detection.frequency_offset_hz,
            if detection.clock_error_percent.abs() < 0.05 {
                "nominal".to_owned()
            } else {
                format!("{:+.2}%", detection.clock_error_percent)
            }
        );
        if !detection.complete {
            println!(
                "      note: only {:.0}% of the transmission is present",
                detection.coverage * 100.0
            );
        }
        if !detection.ambiguous_with.is_empty() {
            println!(
                "      note: without a VIS header this is indistinguishable from {}",
                detection.ambiguous_with.join(", ")
            );
        }
        if cli.verbose {
            for note in &detection.evidence {
                println!("      evidence: {note}");
            }
        }
    }
    println!("Report: {}", outcome.report_path.display());
    Ok(())
}
