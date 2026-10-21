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

use clap::Parser;
use half::f16;
use softcast_rs::pixel_buffer::PixelComponentType;
use softcast_rs::utils::parse_dimensions_2d;

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_parser = parse_dimensions_2d)]
    resolution: (usize, usize),

    gop_len: usize,

    y_compression: f64,
    cbcr_compression: f64,

    #[arg(long, value_parser = parse_dimensions_2d, default_value = "24x24")]
    min_chunk_size: (usize, usize),
    #[arg(long, value_parser = parse_dimensions_2d, default_value = "64x64")]
    max_chunk_size: (usize, usize),
}

struct ChunkDimInfo {
    dim: (usize, usize),
    num_chunks: usize,
    num_slices: usize,
    overhead_pixels: usize,
}

fn minimium_overhead(
    frame_dim: (usize, usize),
    gop_len: usize,
    compression_ratio: f64,
    min_chunk_dim: (usize, usize),
    max_chunk_dim: (usize, usize),
) -> Result<ChunkDimInfo, String> {
    let w_candidates = frame_dim
        .0
        .factors_in_range(min_chunk_dim.0..max_chunk_dim.0);
    let h_candidates = frame_dim
        .1
        .factors_in_range(min_chunk_dim.1..max_chunk_dim.1);
    if w_candidates.is_empty() {
        return Err(format!(
            "No factors of {} in ({}..{})",
            frame_dim.0, min_chunk_dim.0, max_chunk_dim.0,
        ));
    }
    if h_candidates.is_empty() {
        return Err(format!(
            "No factors of {} in ({}..{})",
            frame_dim.1, min_chunk_dim.1, max_chunk_dim.1,
        ));
    }
    let num_pixels = frame_dim.0 * frame_dim.1;
    let mut winner = None;
    let mut winner_num_chunks = None;
    let mut winner_num_slices = None;
    let mut min_overhead = std::usize::MAX;
    for w in &w_candidates {
        for h in &h_candidates {
            assert_eq!(num_pixels % (w * h), 0);
            let num_chunks =
                (compression_ratio * (gop_len * (num_pixels / (w * h))) as f64).ceil() as usize;
            let num_slices = num_chunks.next_power_of_two();
            let overhead = w * h * (num_slices - num_chunks);
            if overhead < min_overhead {
                min_overhead = overhead;
                winner = Some((*w, *h));
                winner_num_chunks = Some(num_chunks);
                winner_num_slices = Some(num_slices);
            }
        }
    }

    Ok(ChunkDimInfo {
        dim: winner.expect("Winner not initialized."),
        num_chunks: winner_num_chunks.unwrap(),
        num_slices: winner_num_slices.unwrap(),
        overhead_pixels: min_overhead,
    })
}

trait FactorsInRange
where
    Self: Sized,
{
    fn factors_in_range(self, range: std::ops::Range<Self>) -> Vec<Self>;
}
impl FactorsInRange for usize {
    fn factors_in_range(self, range: std::ops::Range<Self>) -> Vec<Self> {
        range.filter(|n| self % n == 0).collect()
    }
}

struct PacketLenInfo {
    metadata_len: usize,
    analog_len: usize,
}
fn packet_len(
    chunk_dim: (usize, usize),
    frame_dim: (usize, usize),
    gop_len: usize,
    num_chunks: usize, // number of included chunks
    num_slices: usize,
) -> PacketLenInfo {
    assert_eq!((frame_dim.0 * frame_dim.1) % (chunk_dim.0 * chunk_dim.1), 0);
    // number of all chunks, including those not included
    let chunks_per_gop = gop_len * (frame_dim.0 * frame_dim.1) / (chunk_dim.0 * chunk_dim.1);

    const ENTROPY_COMPRESSION_RATIO: f64 = 0.04; // arbitrary... based on observation.
    let metadata_bitmap_len = ((chunks_per_gop / 8) as f64).ceil() as usize;
    let metadata_means_len = chunks_per_gop * size_of::<u8>();
    let medatata_energies_len = num_chunks * size_of::<f16>(); // energy only included for included chunks
    const SYMBOLS_PER_BYTE: usize = 34; // including FEC and CRC
    let metadata_len = SYMBOLS_PER_BYTE
        * (ENTROPY_COMPRESSION_RATIO
            * (metadata_bitmap_len + metadata_means_len + medatata_energies_len) as f64)
            .ceil() as usize;

    let analog_len = num_slices * chunk_dim.0 * chunk_dim.1 / 2;

    PacketLenInfo {
        metadata_len,
        analog_len,
    }
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let y_frame_dim = args.resolution;
    let c_frame_dim = (
        args.resolution.0 / PixelComponentType::Cb.interleave_step(),
        args.resolution.1 / PixelComponentType::Cb.vertical_subsampling(),
    );

    let ChunkDimInfo {
        dim: y_chunk_dim,
        num_chunks: y_num_chunks,
        num_slices: y_num_slices,
        overhead_pixels: y_overhead,
    } = minimium_overhead(
        y_frame_dim,
        args.gop_len,
        args.y_compression,
        args.min_chunk_size,
        args.max_chunk_size,
    )?;

    let ChunkDimInfo {
        dim: c_chunk_dim,
        num_chunks: c_num_chunks,
        num_slices: c_num_slices,
        overhead_pixels: c_overhead,
    } = minimium_overhead(
        c_frame_dim,
        args.gop_len,
        args.cbcr_compression,
        args.min_chunk_size,
        args.max_chunk_size,
    )?;

    let PacketLenInfo {
        metadata_len: y_metadata_len,
        analog_len: y_analog_len,
    } = packet_len(
        y_chunk_dim,
        y_frame_dim,
        args.gop_len,
        y_num_chunks,
        y_num_slices,
    );
    let PacketLenInfo {
        metadata_len: c_metadata_len,
        analog_len: c_analog_len,
    } = packet_len(
        c_chunk_dim,
        c_frame_dim,
        args.gop_len,
        c_num_chunks,
        c_num_slices,
    );
    let metadata_len = y_metadata_len + 2 * c_metadata_len + 4 /* length header */;
    let analog_len = y_analog_len + 2 * c_analog_len;
    let packet_len = metadata_len + analog_len;

    println!(
        "packet_len:0x{:x}\tmetadata_len:0x{:x}\tanalog_len:0x{:x}",
        packet_len, metadata_len, analog_len
    );
    println!(
        "y   :{}x{}\toverhead:{y_overhead}\tnum_chunks:{y_num_chunks}\tnum_slices:{y_num_slices}",
        y_chunk_dim.0, y_chunk_dim.1,
    );
    println!(
        "cbcr:{}x{}\toverhead:{c_overhead}\tnum_chunks:{c_num_chunks}\tnum_slices:{c_num_slices}",
        c_chunk_dim.0, c_chunk_dim.1,
    );

    Ok(())
}
