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

use clap::{Parser, Subcommand};
use num_complex::Complex32;
use softcast_rs::asset_reader_writer::asset_reader::*;
use softcast_rs::asset_reader_writer::pixel_buffer::*;
use softcast_rs::asset_reader_writer::transform_block_3d::*;
use softcast_rs::asset_reader_writer::*;
use softcast_rs::channel_coding::slice::*;
use softcast_rs::compressor::*;
use softcast_rs::framing::*;
use softcast_rs::metadata_coding::packetizer::*;
use softcast_rs::metadata_coding::*;
use softcast_rs::modulation::metadata::*;
use softcast_rs::modulation::slices::*;
use softcast_rs::source_coding::chunk::*;
use softcast_rs::source_coding::power_scaling::*;
use softcast_rs::source_coding::transform_block_3d_dct::*;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Encode {
        #[arg(value_hint = clap::ValueHint::FilePath)]
        #[arg(value_parser = validate_file_exists)]
        infile: std::path::PathBuf,

        #[arg(short)]
        compression_ratio: f64,
    },
    Decode {
        #[arg(value_hint = clap::ValueHint::FilePath)]
        outfile: std::path::PathBuf,
    },
}

fn validate_file_exists(path: &str) -> Result<std::path::PathBuf, String> {
    let path: std::path::PathBuf = path.into();
    if !path.try_exists().map_err(|e| e.to_string())? {
        return Err(format!("File does not exist: {}", path.display()));
    }
    if !path.is_file() {
        return Err(format!("Not a regular file: {}", path.display()));
    }
    Ok(path)
}

fn encode(infile: std::path::PathBuf, compression_ratio: f64) -> Result<(), String> {
    let mut reader = AssetReader::new(infile.to_str().ok_or("Invalid path.")?);
    const GOP_LENGTH: usize = 90; // TODO: Remove this generic.

    fn frame_encoder<PixelType: HasPixelComponentType>(
        dct_components: &mut TransformBlock3DDCT<GOP_LENGTH, PixelType>,
        compression_ratio: f64,
    ) -> impl Iterator<Item = OFDMSymbol> {
        let chunks: Box<_> = dct_components.chunks_iter().collect();

        // metadata
        let metadata_bitmap = MetadataBitmap::new(&chunks, compression_ratio);
        let chunk_metadata_iter = chunks.iter().map(|chunk| &chunk.metadata);
        let compressed_metadata = CompressedMetadata::new(&metadata_bitmap, chunk_metadata_iter);
        let packetizer: Packetizer = compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        // slices
        let num_included_chunks = metadata_bitmap.values.count_ones();
        let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);
        let slice_modulator: SliceModulator<'_, _, _, _> = PowerScaler::new(compressor)
            .into_slice_iter(num_included_chunks)
            .map(|slice_and_chunk_metadata| slice_and_chunk_metadata.slice)
            .into();

        // ofdm
        let ofdm_framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into(); // TODO: interleave
        ofdm_framer
    }

    let macro_block_3d_iter: MacroBlock3DIterator<GOP_LENGTH, _> =
        reader.pixel_buffer_iter().into();
    for macro_block in macro_block_3d_iter {
        let MacroBlock3D {
            y_components,
            cb_components,
            cr_components,
        } = macro_block;

        let mut y_dct: TransformBlock3DDCT<_, _> = y_components.into();
        let y_framer = frame_encoder(&mut y_dct, compression_ratio);

        let mut cb_dct: TransformBlock3DDCT<_, _> = cb_components.into();
        let cb_framer = frame_encoder(&mut cb_dct, compression_ratio);

        let mut cr_dct: TransformBlock3DDCT<_, _> = cr_components.into();
        let cr_framer = frame_encoder(&mut cr_dct, compression_ratio);

        let encoder = y_framer.chain(cb_framer).chain(cr_framer);
    }

    todo!()
}

fn decode(outfile: std::path::PathBuf) -> Result<(), String> {
    todo!()
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    match args.command {
        Commands::Encode {
            infile,
            compression_ratio,
        } => {
            let _ = encode(infile, compression_ratio);
        }
        Commands::Decode { outfile } => {
            let _ = decode(outfile);
        }
    }
    todo!()
}
