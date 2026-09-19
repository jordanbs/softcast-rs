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
    noise_power: f32,
    dump: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut mpsc_writer, mpsc_reader) = MPSCWriter::new_channel(0x400); // 8MiB

    let abort_token = AbortToken::new();
    let abort_token_clone = abort_token.clone();

    let decoder_result = std::thread::spawn(move || {
        let mut transformer = SignalTransformer::new(mpsc_reader, noise_power, dump);
        let dump_join = transformer.dump_join.take();
        let result = decoder
            .run(transformer, abort_token_clone)
            .map_err(|e| e.to_string());
        eprintln!("decoder result: {:?}", result);
        if let Some(dump_join) = dump_join {
            let _ = dump_join.join(); // ignore err
        }
        result
    });
    encoder.run(&mut mpsc_writer, abort_token)?;
    drop(mpsc_writer); // finishes the decocder thread
    let _ = decoder_result.join().map_err(|_| "thread panic'd")?; // TODO: preserve inner error

    Ok(())
}

struct SignalTransformer {
    iter: Box<dyn Iterator<Item = Box<[Complex32]>>>,
    pub dump_join: Option<std::thread::JoinHandle<()>>,
}
impl SignalTransformer {
    fn new(mpsc_reader: MPSCReader, noise_power: f32, dump: bool) -> Self {
        let mut transformer: Box<dyn Iterator<Item = Box<[Complex32]>>> =
            Box::new(mpsc_reader.into_iter());
        if 0.0 < noise_power {
            let noise_iter = AdditiveWhiteGaussianNoise::new(transformer, noise_power, 0);
            transformer = Box::new(noise_iter);
        }
        let mut dump_join = None;
        if dump {
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
            transformer = Box::new(dump_iter);
        }

        Self {
            iter: transformer,
            dump_join,
        }
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
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )
        .expect("Failed to create encoder.");
        let asset_resolution = encoder.asset_resolution();
        let frame_rate = encoder.frame_rate();
        let decoder = FileWriterDecoder::try_new(
            outfile.into(),
            asset_resolution,
            frame_rate,
            gop_len,
            encoder.y_chunk_dimensions,
            encoder.cb_chunk_dimensions,
            encoder.cr_chunk_dimensions,
        )
        .expect("Failed to create decoder.");
        run_simulation(encoder, decoder).expect("run_simulation failed.");
    }
}
