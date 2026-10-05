// Copyright 2026 Jordan Schneider
//
// This file is part of softcast-rs.
//
// softcast-rs is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// softcast-rs is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// softcast-rs. If not, see <https://www.gnu.org/licenses/>.

#![cfg(target_vendor = "apple")]

use crate::decoder::*;
use crate::encoder::*;
use crate::noise::*;
use crate::sync::*;
use crate::utils::dump_file::*;
use num_complex::Complex32;

pub fn run_simulation(
    mut encoder: FileReaderEncoder,
    mut decoder: FileWriterDecoder,
    attenuation: f32,
    clamp: bool,
    noise_power: f32,
    dump: bool,
    decode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut mpsc_writer, mpsc_reader) = MPSCWriter::new_channel(0x400); // 8MiB

    let abort_token = AbortToken::new();
    let abort_token_clone = abort_token.clone();

    let decoder_result = std::thread::spawn(move || {
        if decode {
            let mut transformer =
                SignalTransformer::new(mpsc_reader, attenuation, clamp, noise_power, dump);
            let dump_join = transformer.dump_join.take();
            let result = decoder
                .run(transformer, abort_token_clone)
                .map_err(|e| e.to_string());
            eprintln!("decoder result: {:?}", result);
            if let Some(dump_join) = dump_join {
                let _ = dump_join.join(); // ignore err
            }
            result
        } else {
            if !dump {
                return Err(
                    "Specify --dump when --disable--ofdm. No decode is expected in this case."
                        .into(),
                );
            }
            let mut dumper =
                SignalTransformer::new(mpsc_reader, attenuation, clamp, noise_power, true);
            let dumper_join = dumper.dump_join.take().unwrap();
            let _ = dumper.into_iter().count(); // drain the iterator to dump
            let _ = dumper_join.join();
            Ok(())
        }
    });
    encoder.run(&mut mpsc_writer, abort_token)?;
    drop(mpsc_writer); // finishes the decoder thread
    let _ = decoder_result.join().map_err(|_| "thread panic'd")?; // TODO: preserve inner error

    Ok(())
}

struct SignalTransformer {
    iter: Box<dyn Iterator<Item = Box<[Complex32]>>>,
    pub dump_join: Option<std::thread::JoinHandle<()>>,
}
impl SignalTransformer {
    fn new(
        mpsc_reader: MPSCReader,
        attenuation: f32,
        clamp: bool,
        noise_power: f32,
        dump: bool,
    ) -> Self {
        let transformer = mpsc_reader.into_iter().map(move |mut c32_slice| {
            c32_slice.as_mut().attenuate(attenuation);
            if clamp {
                c32_slice.as_mut().clamp_magnitude(1.0);
            }
            c32_slice
        });

        let transformer: Box<dyn Iterator<Item = Box<[Complex32]>>> = if noise_power > 0.0 {
            let noise_iter = AdditiveWhiteGaussianNoise::new(transformer, noise_power, 0);
            Box::new(noise_iter)
        } else {
            Box::new(transformer)
        };
        let mut dump_join = None;
        let transformer = if dump {
            let (sender, receiver) = std::sync::mpsc::channel::<Box<[Complex32]>>();
            dump_join = Some(std::thread::spawn(move || {
                let mut dump_file = create_dump_file(false);

                while let Ok(complex32_symbols) = receiver.recv() {
                    let _ = write_complex32_symbols(&mut dump_file, &complex32_symbols);
                }
            }));
            let dump_iter = transformer.inspect(move |buf| {
                let buf_copy = buf.clone();
                let _ = sender.send(buf_copy); // ingore err
            });
            Box::new(dump_iter)
        } else {
            transformer
        };

        Self {
            iter: transformer,
            dump_join,
        }
    }
}

trait Attenuate {
    fn attenuate(&mut self, by: f32);
}
impl Attenuate for &mut [Complex32] {
    fn attenuate(&mut self, multiplicand: f32) {
        self.iter_mut().for_each(|value| *value *= multiplicand);
    }
}
trait Clamp {
    fn clamp_magnitude(&mut self, max_magnitude: f32);
}
impl Clamp for &mut [Complex32] {
    fn clamp_magnitude(&mut self, max_magnitude: f32) {
        self.iter_mut().for_each(|value| {
            if value.norm() > max_magnitude {
                *value /= value.norm();
            }
        });
    }
}

impl Complex32Reader for SignalTransformer {
    fn into_iter(self) -> impl Iterator<Item = Box<[Complex32]>> {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Decoder thread does not exit, so this loops forever."]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_simulate() {
        use super::*;

        let infile = "sample-media/bipbop-1920x1080-5s.mp4";
        let outfile = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = std::fs::remove_file(outfile);
        let gop_len = 2;
        let compression_ratio = 0.01;
        let noise_power = 0.0;
        let y_chunk_dimensions = (48, 30, 1);
        let c_chunk_dimensions = (40, 30, 1);
        let encoder = FileReaderEncoder::with_file(
            infile.into(),
            gop_len,
            PerPixelConfiguration {
                compression_ratio: compression_ratio,
                chunk_dimensions: y_chunk_dimensions,
            },
            PerPixelConfiguration {
                compression_ratio: compression_ratio,
                chunk_dimensions: c_chunk_dimensions,
            },
            PerPixelConfiguration {
                compression_ratio: compression_ratio,
                chunk_dimensions: c_chunk_dimensions,
            },
            true,
            true,
            None,
        )
        .expect("Failed to create encoder.");

        let asset_resolution = encoder.asset_resolution();
        let frame_rate = encoder.frame_rate();
        let decoder = FileWriterDecoder::try_new(
            outfile.into(),
            asset_resolution,
            frame_rate,
            gop_len,
            encoder.y_chunk_dimensions(),
            encoder.cb_chunk_dimensions(),
            encoder.cr_chunk_dimensions(),
            true,
            true,
            None,
        )
        .expect("Failed to create decoder.");
        run_simulation(encoder, decoder, 1.0, false, noise_power, false, true)
            .expect("run_simulation failed.");
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_encode_decode() {
        use super::*;
        let infile = "sample-media/bipbop-768x432-5s.mp4";
        let outfile = "/tmp/bipbop-768x432-5s.mp4";
        let _ = std::fs::remove_file(outfile);
        let gop_len = 2;
        let per_pixel_config = PerPixelConfiguration {
            compression_ratio: 0.006,
            chunk_dimensions: (40, 30, 1),
        };
        let mut tap = MacroBlockTap::default();
        let tap_receiver = tap.take_receiver();
        let mut encoder = FileReaderEncoder::with_file(
            infile.into(),
            gop_len,
            per_pixel_config.clone(),
            per_pixel_config.clone(),
            per_pixel_config.clone(),
            true,
            true,
            Some(tap),
        )
        .expect("Failed to create encoder.");

        let mut decoder = FileWriterDecoder::try_new(
            outfile.into(),
            encoder.asset_resolution(),
            encoder.frame_rate(),
            gop_len,
            encoder.y_chunk_dimensions(),
            encoder.cb_chunk_dimensions(),
            encoder.cr_chunk_dimensions(),
            true,
            true,
            Some(tap_receiver),
        )
        .expect("Failed to create decoder.");

        let (mut mpsc_writer, mpsc_reader) = MPSCWriter::new_channel(0x400); // 8MiB
        let (stats_sender, stats_receiver) = std::sync::mpsc::sync_channel(0);
        let abort_token_e = AbortToken::new();
        let abort_token_d = abort_token_e.clone();
        let _join_handle = std::thread::spawn(move || {
            decoder
                .run(mpsc_reader, abort_token_d)
                .expect_err("decoder.run() failed.");
            let stats = decoder.final_stats().expect("Failed to grab final stats.");
            stats_sender.send(stats).expect("Failed to send stats.");
        });
        encoder
            .run(&mut mpsc_writer, abort_token_e)
            .expect("encoder.run() failed.");
        drop(mpsc_writer); // finishes the decoder thread

        let Statistics {
            y_psnr,
            cb_psnr,
            cr_psnr,
            weighted_total_psnr,
        } = stats_receiver.recv().expect("Failed to receive stats.");
        assert!(y_psnr.is_normal());
        assert!(cb_psnr.is_normal());
        assert!(cr_psnr.is_normal());
        assert!(weighted_total_psnr.is_normal());
        assert!(y_psnr > 20.0, "y_psnr too low: {y_psnr}");
        assert!(cb_psnr > 20.0, "cb_psnr too low: {cb_psnr}");
        assert!(cr_psnr > 20.0, "cr_psnr too low: {cr_psnr}");
        assert!(
            weighted_total_psnr > 20.0,
            "weighted_total_psnr too low: {weighted_total_psnr}"
        );
    }
}
