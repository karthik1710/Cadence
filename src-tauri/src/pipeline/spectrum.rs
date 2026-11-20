//! The small amount of frequency-domain machinery the audio stages need.
//!
//! Three callers share it: the noise classifier (what *kind* of noise is in
//! the room), the audio profile (band levels, hum, spectral tilt) and the
//! model-free noise reduction in [`crate::pipeline::enhance`], which needs a
//! real analysis/synthesis STFT rather than just magnitudes.
//!
//! It is hand-written for the same reason `preprocess::Biquad` is: one
//! radix-2 FFT is eighty lines and no supply chain, whereas a DSP crate is a
//! dependency with its own API-stability risk in a project whose whole point
//! is that it ships no runtime it did not choose deliberately.

/// In-place iterative radix-2 Cooley-Tukey FFT.
///
/// `re`/`im` must be the same length and a power of two. `inverse` runs the
/// conjugate transform *and* the 1/N scaling, so a forward transform followed
/// by an inverse one returns the input.
pub fn fft(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    if n < 2 || !n.is_power_of_two() {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0f32 } else { -1.0f32 };
    let mut len = 2usize;
    while len <= n {
        let angle = sign * 2.0 * std::f32::consts::PI / len as f32;
        let (wl_im, wl_re) = angle.sin_cos();
        let mut start = 0usize;
        while start < n {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..len / 2 {
                let a = start + k;
                let b = a + len / 2;
                let t_re = re[b] * w_re - im[b] * w_im;
                let t_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = w_re * wl_re - w_im * wl_im;
                w_im = w_re * wl_im + w_im * wl_re;
                w_re = next_re;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for i in 0..n {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

/// Periodic Hann window. The periodic form (divisor `n`, not `n - 1`) is the
/// one that satisfies constant-overlap-add, which [`stft_process`] relies on.
pub fn hann(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// One frame as magnitudes, `n/2 + 1` bins from DC to Nyquist. The frame is
/// windowed and zero-padded to the next power of two, so callers can hand it
/// any frame length.
pub fn magnitudes(frame: &[f32], window: &[f32]) -> Vec<f32> {
    if frame.is_empty() {
        return Vec::new();
    }
    let n = frame.len().next_power_of_two();
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    for (i, &s) in frame.iter().enumerate() {
        re[i] = s * window.get(i).copied().unwrap_or(1.0);
    }
    fft(&mut re, &mut im, false);
    (0..=n / 2)
        .map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt())
        .collect()
}

/// A magnitude spectrum averaged over frames, plus the frequency each bin
/// stands for.
#[derive(Debug, Clone, Default)]
pub struct AverageSpectrum {
    /// Mean magnitude per bin, DC to Nyquist.
    pub bins: Vec<f32>,
    pub bin_hz: f32,
    /// How many frames went into the mean. Zero means nothing was analysed,
    /// and every measure below then reports a neutral value rather than a
    /// number derived from an empty spectrum.
    pub frames: usize,
}

impl AverageSpectrum {
    pub fn nyquist_hz(&self) -> f32 {
        self.bin_hz * (self.bins.len().max(1) - 1) as f32
    }

    /// Summed magnitude from `lo_hz` up to `hi_hz`.
    pub fn band_energy(&self, lo_hz: f32, hi_hz: f32) -> f32 {
        if self.bins.is_empty() || self.bin_hz <= 0.0 {
            return 0.0;
        }
        let first = (lo_hz / self.bin_hz).ceil().max(0.0) as usize;
        let last = ((hi_hz / self.bin_hz).floor() as usize).min(self.bins.len() - 1);
        if first > last {
            return 0.0;
        }
        self.bins[first..=last].iter().sum()
    }

    /// That band as a share of the whole spectrum, which is the form every
    /// threshold in the noise classifier is written against.
    pub fn band_fraction(&self, lo_hz: f32, hi_hz: f32) -> f32 {
        let total: f32 = self.bins.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        self.band_energy(lo_hz, hi_hz) / total
    }

    /// How far the loudest bin near `hz` stands above the background either
    /// side of it. A tonal component (mains hum, a fan whine) gives a large
    /// ratio; broadband noise gives roughly 1.
    pub fn tonal_ratio(&self, hz: f32, tolerance_hz: f32) -> f32 {
        if self.bins.is_empty() || self.bin_hz <= 0.0 {
            return 0.0;
        }
        let bin = |f: f32| -> usize {
            ((f / self.bin_hz).round().max(0.0) as usize).min(self.bins.len() - 1)
        };
        let centre = bin(hz);
        let width = ((tolerance_hz / self.bin_hz).ceil() as usize).max(1);
        let lo = centre.saturating_sub(width);
        let hi = (centre + width).min(self.bins.len() - 1);
        let peak = self.bins[lo..=hi].iter().fold(0f32, |a, &b| a.max(b));

        // The background is sampled a few bins out on each side, so a peak is
        // never compared against its own skirt.
        let skirt = width * 4;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for i in lo.saturating_sub(skirt)..lo {
            sum += self.bins[i];
            count += 1;
        }
        for i in (hi + 1)..=(hi + skirt).min(self.bins.len() - 1) {
            sum += self.bins[i];
            count += 1;
        }
        if count == 0 || sum <= 0.0 {
            return 0.0;
        }
        peak / (sum / count as f32)
    }

    /// Geometric mean over arithmetic mean: 1.0 is white noise, near 0 is a
    /// pure tone. The standard "is this noise or is this a signal" measure.
    pub fn flatness(&self) -> f32 {
        // DC is skipped: a residual offset would otherwise dominate the
        // geometric mean and make every spectrum look tonal.
        let bins = if self.bins.len() > 1 {
            &self.bins[1..]
        } else {
            &self.bins[..]
        };
        if bins.is_empty() {
            return 0.0;
        }
        let mut log_sum = 0.0f64;
        let mut sum = 0.0f64;
        for &b in bins {
            log_sum += ((b as f64) + 1e-12).ln();
            sum += b as f64;
        }
        let geometric = (log_sum / bins.len() as f64).exp();
        let arithmetic = sum / bins.len() as f64;
        if arithmetic <= 0.0 {
            return 0.0;
        }
        (geometric / arithmetic).clamp(0.0, 1.0) as f32
    }

    /// Energy above 4 kHz over energy below it, in dB. Positive is a hissy,
    /// top-heavy spectrum; strongly negative is a rumbly one.
    pub fn tilt_db(&self) -> f32 {
        let low = self.band_energy(100.0, 4_000.0);
        let high = self.band_energy(4_000.0, self.nyquist_hz().max(4_001.0));
        if low <= 0.0 || high <= 0.0 {
            return 0.0;
        }
        20.0 * (high / low).log10()
    }
}

/// Averages the magnitude spectrum over the frames `keep` accepts.
///
/// `keep` receives a frame index and that frame's RMS; returning false skips
/// it. That is how the noise classifier looks at *only* the quiet frames — the
/// room between words — without needing a VAD model to be installed.
///
/// At most `max_frames` frames are analysed, spread evenly across the whole
/// recording. A 60-minute meeting has hundreds of thousands of frames and
/// every caller here wants a spectral average, not all of them.
pub fn average_spectrum<F>(
    samples: &[f32],
    sample_rate: u32,
    frame_len: usize,
    max_frames: usize,
    mut keep: F,
) -> AverageSpectrum
where
    F: FnMut(usize, f32) -> bool,
{
    let frame_len = frame_len.next_power_of_two().max(2);
    if samples.len() < frame_len || sample_rate == 0 || max_frames == 0 {
        return AverageSpectrum::default();
    }
    let window = hann(frame_len);
    let total_frames = samples.len() / frame_len;
    // Stride rather than truncate: analysing the first N frames of a meeting
    // would describe its opening minute, not the meeting.
    let stride = (total_frames / max_frames).max(1);

    let mut accumulator = vec![0.0f32; frame_len / 2 + 1];
    let mut frames = 0usize;
    for index in (0..total_frames).step_by(stride) {
        let frame = &samples[index * frame_len..(index + 1) * frame_len];
        let energy = crate::pipeline::audio_profile::rms(frame);
        if !keep(index, energy) {
            continue;
        }
        for (slot, value) in accumulator.iter_mut().zip(magnitudes(frame, &window)) {
            *slot += value;
        }
        frames += 1;
    }

    let bin_hz = sample_rate as f32 / frame_len as f32;
    if frames == 0 {
        return AverageSpectrum {
            bin_hz,
            ..AverageSpectrum::default()
        };
    }
    for slot in accumulator.iter_mut() {
        *slot /= frames as f32;
    }
    AverageSpectrum {
        bins: accumulator,
        bin_hz,
        frames,
    }
}

/// Analysis/synthesis STFT with a square-root Hann window at 50% overlap.
///
/// `modify` is called once per frame with that frame's magnitudes and phases
/// (`n/2 + 1` bins each) and may rewrite the magnitudes in place; the frames
/// are then overlap-added back into a signal the same length as the input.
///
/// sqrt-Hann on both the analysis and the synthesis side squares to a plain
/// Hann, which sums to exactly 1.0 at 50% overlap — so a `modify` that changes
/// nothing reconstructs the input rather than a rippled copy of it. That
/// property is what makes the noise reduction below safe to leave enabled:
/// with the threshold never crossed it is a no-op on the samples, not a
/// slightly-degraded pass.
pub fn stft_process<F>(samples: &[f32], frame_len: usize, mut modify: F) -> Vec<f32>
where
    F: FnMut(&mut [f32], &[f32]),
{
    let n = frame_len.next_power_of_two().max(4);
    let hop = n / 2;
    if samples.len() < n {
        return samples.to_vec();
    }
    let window: Vec<f32> = hann(n).into_iter().map(|w| w.sqrt()).collect();

    let mut output = vec![0.0f32; samples.len()];
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    let mut magnitude = vec![0.0f32; n / 2 + 1];
    let mut phase = vec![0.0f32; n / 2 + 1];

    let mut start = 0usize;
    while start + n <= samples.len() {
        for i in 0..n {
            re[i] = samples[start + i] * window[i];
            im[i] = 0.0;
        }
        fft(&mut re, &mut im, false);

        for bin in 0..=n / 2 {
            magnitude[bin] = (re[bin] * re[bin] + im[bin] * im[bin]).sqrt();
            phase[bin] = im[bin].atan2(re[bin]);
        }
        modify(&mut magnitude, &phase);

        // Rebuild the full spectrum from the (possibly rewritten) magnitudes
        // and the untouched phases, mirroring the upper half so the inverse
        // transform comes back real.
        for bin in 0..=n / 2 {
            let (sin, cos) = phase[bin].sin_cos();
            re[bin] = magnitude[bin] * cos;
            im[bin] = magnitude[bin] * sin;
            if bin > 0 && bin < n / 2 {
                re[n - bin] = re[bin];
                im[n - bin] = -im[bin];
            }
        }
        im[0] = 0.0;
        im[n / 2] = 0.0;
        fft(&mut re, &mut im, true);

        for i in 0..n {
            output[start + i] += re[i] * window[i];
        }
        start += hop;
    }

    // The leading and trailing half-frames never got an overlap partner, so
    // they keep the original samples rather than fading out.
    for i in 0..hop.min(output.len()) {
        output[i] = samples[i];
    }
    for i in (start + hop).min(samples.len())..samples.len() {
        output[i] = samples[i];
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sample_rate: u32, seconds: f32, freq: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Deterministic pseudo-noise, so no spectral test can ever flake.
    fn noise(n: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 12_345u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((state >> 16) as f32 / 32_768.0) - 1.0) * amplitude
            })
            .collect()
    }

    #[test]
    fn a_forward_then_inverse_transform_returns_the_input() {
        let mut re: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let original = re.clone();
        let mut im = vec![0.0f32; 64];
        fft(&mut re, &mut im, false);
        fft(&mut re, &mut im, true);
        for (a, b) in original.iter().zip(re.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn a_non_power_of_two_is_left_alone_rather_than_corrupted() {
        let mut re = vec![1.0f32, 2.0, 3.0];
        let mut im = vec![0.0f32; 3];
        fft(&mut re, &mut im, false);
        assert_eq!(re, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_pure_tone_lands_in_its_own_bin() {
        let sr = 16_000;
        let spectrum = average_spectrum(&tone(sr, 1.0, 1_000.0, 0.5), sr, 512, 100, |_, _| true);
        let peak = spectrum
            .bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // 16 kHz over 512 is 31.25 Hz per bin, so 1 kHz is bin 32.
        assert_eq!(peak, 32);
        assert!((spectrum.bin_hz - 31.25).abs() < 0.01);
    }

    #[test]
    fn flatness_separates_a_tone_from_noise() {
        let sr = 16_000;
        let tonal = average_spectrum(&tone(sr, 1.0, 440.0, 0.5), sr, 512, 100, |_, _| true);
        let flat = average_spectrum(&noise(sr as usize, 1.0), sr, 512, 100, |_, _| true);
        assert!(tonal.flatness() < 0.1, "tone flatness {}", tonal.flatness());
        assert!(flat.flatness() > 0.3, "noise flatness {}", flat.flatness());
    }

    #[test]
    fn a_mains_tone_shows_a_tonal_ratio_and_a_neighbour_does_not() {
        let sr = 16_000;
        let spectrum = average_spectrum(&tone(sr, 1.0, 50.0, 0.2), sr, 2048, 100, |_, _| true);
        assert!(spectrum.tonal_ratio(50.0, 8.0) > 10.0);
        assert!(spectrum.tonal_ratio(3_000.0, 50.0) < 5.0);
    }

    #[test]
    fn tilt_is_positive_for_hiss_and_negative_for_rumble() {
        let sr = 16_000;
        let hiss = average_spectrum(&tone(sr, 1.0, 6_000.0, 0.3), sr, 512, 100, |_, _| true);
        let rumble = average_spectrum(&tone(sr, 1.0, 120.0, 0.3), sr, 512, 100, |_, _| true);
        assert!(hiss.tilt_db() > 0.0);
        assert!(rumble.tilt_db() < 0.0);
    }

    #[test]
    fn band_fraction_is_a_share_of_the_whole() {
        let sr = 16_000;
        let spectrum = average_spectrum(&tone(sr, 1.0, 200.0, 0.4), sr, 512, 100, |_, _| true);
        assert!(spectrum.band_fraction(0.0, 400.0) > 0.5);
        assert!(spectrum.band_fraction(4_000.0, 8_000.0) < 0.1);
    }

    #[test]
    fn keeping_no_frames_yields_an_empty_but_usable_spectrum() {
        let sr = 16_000;
        let spectrum = average_spectrum(&tone(sr, 1.0, 200.0, 0.4), sr, 512, 100, |_, _| false);
        assert_eq!(spectrum.frames, 0);
        assert_eq!(spectrum.band_energy(0.0, 8_000.0), 0.0);
        assert_eq!(spectrum.flatness(), 0.0);
        assert_eq!(spectrum.tonal_ratio(50.0, 5.0), 0.0);
        assert_eq!(spectrum.tilt_db(), 0.0);
    }

    #[test]
    fn frames_are_sampled_across_the_whole_recording_not_just_the_start() {
        let sr = 16_000;
        let mut samples = tone(sr, 1.0, 200.0, 0.4);
        samples.extend(tone(sr, 1.0, 6_000.0, 0.4));
        // 20 frames out of ~62: if it truncated instead of striding, only the
        // 200 Hz opening would be in the average.
        let spectrum = average_spectrum(&samples, sr, 512, 20, |_, _| true);
        assert!(spectrum.band_fraction(5_000.0, 7_000.0) > 0.2);
    }

    #[test]
    fn an_untouched_stft_reconstructs_the_signal() {
        let sr = 16_000;
        let samples = tone(sr, 0.5, 300.0, 0.4);
        let out = stft_process(&samples, 512, |_, _| {});
        assert_eq!(out.len(), samples.len());
        // Away from the un-overlapped edges, reconstruction is exact to
        // floating-point noise.
        for i in 512..samples.len() - 512 {
            assert!(
                (out[i] - samples[i]).abs() < 1e-3,
                "sample {i}: {} vs {}",
                out[i],
                samples[i]
            );
        }
    }

    #[test]
    fn an_stft_that_zeroes_every_bin_silences_the_interior() {
        let sr = 16_000;
        let samples = tone(sr, 0.5, 300.0, 0.4);
        let out = stft_process(&samples, 512, |mag, _| mag.fill(0.0));
        for i in 1_000..samples.len() - 1_000 {
            assert!(out[i].abs() < 1e-3, "sample {i} was {}", out[i]);
        }
    }

    #[test]
    fn audio_shorter_than_one_frame_passes_straight_through() {
        let samples = vec![0.1f32; 100];
        assert_eq!(stft_process(&samples, 512, |mag, _| mag.fill(0.0)), samples);
    }
}
