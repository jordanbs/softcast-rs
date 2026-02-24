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

use crate::decoder::*;
use crate::encoder::*;
use crate::sync::*;

pub fn run_simulation(
    mut encoder: FileReaderEncoder,
    mut decoder: FileWriterDecoder,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mpsc_writer, mpsc_reader) = MPSCWriter::new_channel(0x400); // 8MiB
    let decoder_result = std::thread::spawn(move || {
        let result = decoder.run(mpsc_reader).map_err(|e| e.to_string());
        eprintln!("decoder result: {:?}", result);
        result
    });
    encoder.run(mpsc_writer)?;

    let _ = decoder_result.join().map_err(|_| "thread panic'd")?; // TODO: preserve inner error

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_simulate() {
        let infile = "sample-media/bipbop-1920x1080-5s.mp4";
        let outfile = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = std::fs::remove_file(outfile);
        let gop_len = 2;
        let compression_ratio = 0.01;
        let noise_power = 0.0;
        let y_chunk_dimensions = (48, 30, 1);
        let c_chunk_dimensions = (40, 30, 1);
        let encoder = FileReaderEncoder::try_new(
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
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )
        .expect("Failed to create decoder.");
        run_simulation(encoder, decoder).expect("run_simulation failed.");
    }
}
