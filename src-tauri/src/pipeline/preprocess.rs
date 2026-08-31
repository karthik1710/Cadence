use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Downmixes to mono and resamples to 16kHz via linear interpolation.
///
/// Audio preprocessing for ASR input doesn't need broadcast-grade sinc
/// resampling — whisper.cpp is robust to this — and a hand-written linear
/// resampler avoids pulling in a heavier DSP dependency with its own
/// API-stability risk.
pub fn resample_to_16k_mono(input_path: &Path, output_path: &Path) -> Result<(), String> {
    let mut reader = WavReader::open(input_path).map_err(|e| e.to_string())?;
    let spec = reader.spec();

    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "unsupported WAV format for preprocessing: {:?}/{}-bit (expected 16-bit PCM, which is all audio.rs ever writes)",
            spec.sample_format, spec.bits_per_sample
        ));
    }

    let channels = spec.channels as usize;
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let mono: Vec<f32> = samples
        .chunks(channels.max(1))
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum as f32) / (frame.len().max(1) as f32)
        })
        .collect();

    let resampled = if spec.sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        linear_resample(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
    };

    let out_spec = WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output_path, out_spec).map_err(|e| e.to_string())?;
    for sample in resampled {
        let clamped = sample.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(clamped).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(())
}

fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 {
        return Vec::new();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsamples_to_expected_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = linear_resample(&input, 8000, 16000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn identity_when_same_rate() {
        let input = vec![0.0, 1.0, 2.0];
        let out = linear_resample(&input, 16000, 16000);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn downsamples_to_expected_length() {
        let input: Vec<f32> = (0..16000).map(|i| i as f32).collect();
        let out = linear_resample(&input, 16000, 8000);
        assert_eq!(out.len(), 8000);
    }
}
