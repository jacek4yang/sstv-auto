//! Image plausibility metrics.
//!
//! These exist to answer one question: *could a human read this image?* They
//! are used to reject false positives from the blind path, where a wrong
//! hypothesis can otherwise still produce a PNG full of plausible-looking
//! noise.
//!
//! Every metric here is deliberately cheap and reasoning-free. None of them
//! look for "an SSTV picture"; they measure properties any real photograph or
//! rendered flag graphic has and pure noise does not:
//!
//! * a wide but bounded range of levels,
//! * strong correlation between neighbouring pixels, because a transmitted
//!   image is smooth at pixel scale while misalignment destroys that
//!   correlation,
//! * a small fraction of clipped pixels.

/// A set of independent measurements of an image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quality {
    /// Mean absolute difference between horizontally adjacent pixels,
    /// normalised. Lower is smoother.
    pub horizontal_smoothness: f64,
    /// Mean absolute difference between vertically adjacent pixels,
    /// normalised.
    pub vertical_smoothness: f64,
    /// Fraction of pixels that are fully clipped (0 or 255 in every channel).
    pub clipped_ratio: f64,
    /// Standard deviation of luma, normalised.
    pub contrast: f64,
    /// Fraction of luma variance that is *not* explained by white noise.
    pub structure: f64,
    /// Aggregate score in `0.0..=1.0`.
    pub score: f64,
}

/// Luma of an RGB triple.
#[must_use]
pub fn luma(pixel: [u8; 3]) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

/// Measure an image given as row-major RGB pixels.
///
/// Returns a zeroed [`Quality`] for an image too small to measure, which
/// callers treat as "unproven" rather than "good".
#[must_use]
pub fn measure(pixels: &[[u8; 3]], width: u32, height: u32) -> Quality {
    let empty = Quality {
        horizontal_smoothness: 0.0,
        vertical_smoothness: 0.0,
        clipped_ratio: 1.0,
        contrast: 0.0,
        structure: 0.0,
        score: 0.0,
    };
    let w = width as usize;
    let h = height as usize;
    if w < 4 || h < 4 || pixels.len() < w * h {
        return empty;
    }

    let mut sum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    let mut clipped = 0usize;
    let mut horizontal = 0.0_f64;
    let mut horizontal_count = 0usize;
    let mut vertical = 0.0_f64;
    let mut vertical_count = 0usize;
    // Sum of squared *differences* between neighbours, used to separate
    // spatial structure from white noise.
    let mut neighbour_squares = 0.0_f64;
    let mut neighbour_count = 0usize;

    for y in 0..h {
        for x in 0..w {
            let index = y * w + x;
            let value = luma(pixels[index]);
            sum += value;
            sum_squares += value * value;
            if pixels[index].iter().all(|c| *c <= 1 || *c >= 254) {
                clipped += 1;
            }
            if x + 1 < w {
                let d = value - luma(pixels[index + 1]);
                horizontal += d.abs();
                neighbour_squares += d * d;
                neighbour_count += 1;
                horizontal_count += 1;
            }
            if y + 1 < h {
                let d = value - luma(pixels[index + w]);
                vertical += d.abs();
                neighbour_squares += d * d;
                neighbour_count += 1;
                vertical_count += 1;
            }
        }
    }

    let n = (w * h) as f64;
    let mean = sum / n;
    let variance = (sum_squares / n - mean * mean).max(0.0);
    let stddev = variance.sqrt();

    let mean_horizontal = horizontal / horizontal_count.max(1) as f64;
    let mean_vertical = vertical / vertical_count.max(1) as f64;

    // For white noise, the expected squared difference between independent
    // samples is twice the variance. A structured image has far less.
    let expected_noise_difference = 2.0 * variance;
    let observed_difference = neighbour_squares / neighbour_count.max(1) as f64;
    let structure = if expected_noise_difference > f64::EPSILON {
        (1.0 - observed_difference / expected_noise_difference).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // A transmitted image has neighbouring-pixel differences far below the
    // difference between black and white. Sweeping sliders live around 8-60.
    //
    // Smoothness alone cannot be trusted, though: a completely flat image has
    // zero neighbour difference and would score perfectly. It is therefore
    // gated on there being real variation, so flatness only counts as
    // evidence once the picture has something in it.
    let smoothness = 1.0 - ((mean_horizontal + mean_vertical) * 0.5 / 128.0).clamp(0.0, 1.0);

    let contrast = (stddev / 64.0).clamp(0.0, 1.0);
    let clipped_ratio = clipped as f64 / n;

    // Weights: structural coherence carries the most weight because it is the
    // one metric white noise cannot fake. Smoothness and contrast follow, but
    // both are gated on contrast so a featureless frame collapses to zero
    // instead of scoring on its silence.
    let score = (0.45 * structure
        + 0.25 * smoothness * contrast
        + 0.18 * contrast
        + 0.12 * (1.0 - clipped_ratio) * contrast)
        .clamp(0.0, 1.0);

    Quality {
        horizontal_smoothness: mean_horizontal,
        vertical_smoothness: mean_vertical,
        clipped_ratio,
        contrast,
        structure,
        score,
    }
}

/// Whether an image is flat enough that it cannot be real content.
#[must_use]
pub fn is_degenerate(quality: &Quality) -> bool {
    // Almost every pixel clipped, or no variation at all: a flat fill, a
    // single sync bar, or an empty buffer.
    quality.contrast < 0.03 || quality.clipped_ratio > 0.8
}

/// Whether an image looks like noise rather than a transmitted picture.
///
/// `structure` is the discriminating metric: for white noise the squared
/// difference between neighbouring samples approaches twice the variance, so
/// the measured structure collapses to zero. A real picture, however noisy,
/// retains correlation between adjacent pixels. Smoothness is required as
/// well so that a genuinely flat field is not mislabelled as noise.
#[must_use]
pub fn looks_like_noise(quality: &Quality) -> bool {
    quality.structure < 0.25
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: u32, height: u32, value: u8) -> (Vec<[u8; 3]>, u32, u32) {
        (
            vec![[value, value, value]; (width * height) as usize],
            width,
            height,
        )
    }

    fn gradient(width: u32, height: u32) -> (Vec<[u8; 3]>, u32, u32) {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 255 / width.max(1)) as u8).wrapping_add((y * 7) as u8);
                pixels.push([v, v, v]);
            }
        }
        (pixels, width, height)
    }

    fn noise(width: u32, height: u32) -> (Vec<[u8; 3]>, u32, u32) {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        let mut state = 0x1234_5678_u32;
        for _ in 0..width * height {
            // xorshift keeps this deterministic without a dependency.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = (state >> 24) as u8;
            pixels.push([v, v, v]);
        }
        (pixels, width, height)
    }

    #[test]
    fn flat_image_scores_zero_and_is_degenerate() {
        let (pixels, w, h) = flat(64, 64, 128);
        let quality = measure(&pixels, w, h);
        assert!(is_degenerate(&quality));
        assert_eq!(quality.score, 0.0);
    }

    #[test]
    fn gradient_scores_far_above_noise() {
        let (gradient_pixels, w, h) = gradient(128, 128);
        let (noise_pixels, _, _) = noise(128, 128);
        let gradient = measure(&gradient_pixels, w, h);
        let noise = measure(&noise_pixels, w, h);
        assert!(
            gradient.score > noise.score + 0.2,
            "gradient {} vs noise {}",
            gradient.score,
            noise.score
        );
        assert!(
            noise.score < 0.4,
            "noise must not score as plausible content: {}",
            noise.score
        );
        assert!(!is_degenerate(&gradient));
        assert!(looks_like_noise(&noise), "noise scored {noise:?}");
        assert!(!looks_like_noise(&gradient));
    }

    #[test]
    fn noise_is_detected_as_noise() {
        let (pixels, w, h) = noise(128, 128);
        let quality = measure(&pixels, w, h);
        assert!(
            quality.structure < 0.4,
            "white noise showed structure {}",
            quality.structure
        );
        assert!(looks_like_noise(&quality));
    }

    #[test]
    fn a_realistic_image_has_high_structure() {
        // A smooth two-tone picture with text-like blocks.
        let (mut pixels, w, h) = gradient(160, 120);
        for y in 40..80 {
            for x in 20..140 {
                pixels[(y * w + x) as usize] = [250, 250, 250];
            }
        }
        let quality = measure(&pixels, w, h);
        assert!(
            quality.structure > 0.5,
            "structure was {}",
            quality.structure
        );
        assert!(quality.score > 0.6, "score was {}", quality.score);
        assert!(!is_degenerate(&quality));
        assert!(!looks_like_noise(&quality));
    }

    #[test]
    fn tiny_images_are_reported_as_unproven() {
        for (w, h) in [(0, 0), (1, 1), (3, 3), (2, 100)] {
            let (pixels, _, _) = flat(w.max(1), h.max(1), 100);
            let quality = measure(&pixels, w, h);
            assert_eq!(quality.score, 0.0, "{w}x{h} should be unproven");
        }
    }

    #[test]
    fn short_pixel_buffers_do_not_panic() {
        // Fewer pixels than width*height must be treated as unmeasurable.
        let quality = measure(&[[0, 0, 0]; 4], 64, 64);
        assert_eq!(quality.score, 0.0);
    }

    #[test]
    fn clipping_is_penalised() {
        let (white, w, h) = flat(64, 64, 255);
        let mut pixels = white;
        // Half the image is real gradient, half is blown out.
        for (index, pixel) in pixels.iter_mut().enumerate() {
            if index % 2 == 0 {
                *pixel = [120, 130, 110];
            }
        }
        let clipped = measure(&pixels, w, h);
        let (gradient_pixels, _, _) = gradient(64, 64);
        let clean = measure(&gradient_pixels, w, h);
        assert!(clipped.clipped_ratio > 0.4);
        assert!(clipped.score < clean.score);
    }

    #[test]
    fn luma_matches_rec709_weights() {
        assert!((luma([255, 255, 255]) - 255.0).abs() < 1e-9);
        assert!((luma([0, 0, 0])).abs() < 1e-9);
        // Green must dominate.
        assert!(luma([0, 255, 0]) > luma([255, 0, 0]));
        assert!(luma([255, 0, 0]) > luma([0, 0, 255]));
    }
}
