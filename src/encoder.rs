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

use crate::asset_reader_writer::asset_reader::*;
use crate::asset_reader_writer::pixel_buffer::*;
use crate::asset_reader_writer::transform_block_3d::*;
use crate::asset_reader_writer::*;
use crate::channel_coding::slice::*;
use crate::compressor::*;
use crate::framing::*;
use crate::metadata_coding::packetizer::*;
use crate::metadata_coding::*;
use crate::modulation::metadata::*;
use crate::modulation::slices::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use num_complex::Complex32;
use rand::Rng;

pub trait Complex32Consumer {
    // consumes buf, so it can be sent without copies
    fn consume(&mut self, buf: Box<[Complex32]>) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct FileReaderEncoder {
    macro_block_3d_iter: MacroBlock3DIterator<IntoPixelBufferIterator>,
    compression_ratio: f64,
    noise_power: f32,
    y_chunk_dimensions: (usize, usize, usize),
    cb_chunk_dimensions: (usize, usize, usize),
    cr_chunk_dimensions: (usize, usize, usize),
    asset_resolution: (usize, usize),
    frame_rate: f64,
}

impl FileReaderEncoder {
    pub fn try_new(
        in_path: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        noise_power: f32,
        y_chunk_dimensions: (usize, usize, usize),
        cb_chunk_dimensions: (usize, usize, usize),
        cr_chunk_dimensions: (usize, usize, usize),
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = AssetReader::new(in_path);
        let frame_rate = reader.frame_rate()?;
        let asset_resolution = reader.resolution()?;

        println!(
            "Asset resolution: {}x{}",
            asset_resolution.0, asset_resolution.1
        );
        println!("Asset framerate: {}", frame_rate);

        let pb_iter: IntoPixelBufferIterator = reader.into();

        let asset_resolution = (asset_resolution.0 as usize, asset_resolution.1 as usize);

        let y_chunk_dimensions =
            chunk_dimensions_sizer(y_chunk_dimensions, asset_resolution, PixelComponentType::Y);
        let cb_chunk_dimensions = chunk_dimensions_sizer(
            cb_chunk_dimensions,
            asset_resolution,
            PixelComponentType::Cb,
        );
        let cr_chunk_dimensions = chunk_dimensions_sizer(
            cr_chunk_dimensions,
            asset_resolution,
            PixelComponentType::Cr,
        );

        Ok(Self {
            macro_block_3d_iter: pb_iter.into_macro_block_3d_iter(gop_len),
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            cb_chunk_dimensions,
            cr_chunk_dimensions,
            asset_resolution,
            frame_rate,
        })
    }

    pub fn asset_resolution(&self) -> (usize, usize) {
        self.asset_resolution
    }
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }
}

fn ofdm_framer<PixelType: HasPixelComponentType>(
    dct_components: &mut TransformBlock3DDCT<PixelType>,
    compression_ratio: f64,
    chunk_dimensions: (usize, usize, usize),
) -> impl Iterator<Item = OFDMSymbol> {
    let chunks: Box<_> = dct_components.chunks_iter(chunk_dimensions).collect();

    // metadata
    let metadata_bitmap = MetadataBitmap::new(&chunks, compression_ratio);
    let chunk_metadata_iter = chunks.iter().map(|chunk| &chunk.metadata);
    let compressed_metadata = CompressedMetadata::new(&metadata_bitmap, chunk_metadata_iter);
    let packetizer: Packetizer = compressed_metadata.into();
    let metadata_modulator: MetadataModulator<_> = packetizer.into();

    // slices
    let num_included_chunks = metadata_bitmap.values.count_ones();
    let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);
    let slice_modulator: SliceModulator<'_, _, _> = PowerScaler::new(compressor)
        .into_slice_iter(num_included_chunks)
        .map(|slice_and_chunk_metadata| slice_and_chunk_metadata.slice)
        .into();

    // ofdm
    let ofdm_framer: OFDMFrameGenerator<_> =
        metadata_modulator.flatten().chain(slice_modulator).into(); // TODO: interleave
    ofdm_framer
}

impl FileReaderEncoder {
    pub fn run<W: Complex32Consumer>(
        &mut self,
        mut ofdm_symbol_writer: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for macro_block in self.macro_block_3d_iter.by_ref() {
            // encoder
            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
                ..
            } = macro_block;

            let mut y_dct = y_components.into();
            let y_framer = ofdm_framer(&mut y_dct, self.compression_ratio, self.y_chunk_dimensions);

            let mut cb_dct = cb_components.into();
            let cb_framer = ofdm_framer(
                &mut cb_dct,
                self.compression_ratio,
                self.cb_chunk_dimensions,
            );

            let mut cr_dct = cr_components.into();
            let cr_framer = ofdm_framer(
                &mut cr_dct,
                self.compression_ratio,
                self.cr_chunk_dimensions,
            );

            let encoder = y_framer.chain(cb_framer).chain(cr_framer);

            let noise_power = self.noise_power; // should sqrt?
            let encoder_plus_noise = encoder.map(|mut ofdmsymbol| {
                if 0.0 >= noise_power {
                    return ofdmsymbol;
                }
                let mut rng = rand::rng();
                for iq in ofdmsymbol.time_domain_symbols.iter_mut() {
                    let i_distortion = rng.random_range(-noise_power..noise_power);
                    let q_distortion = rng.random_range(-noise_power..noise_power);
                    iq.re += i_distortion;
                    iq.im += q_distortion;
                }
                ofdmsymbol
            });

            for symbol in encoder_plus_noise {
                ofdm_symbol_writer.consume(symbol.time_domain_symbols)?;
            }
        }
        Ok(())
    }
}

fn max_factor_at_or_below(limit: usize, value: usize) -> usize {
    assert!(limit > 0);
    (1..=limit)
        .rev()
        .find(|i| value.is_multiple_of(*i))
        .unwrap()
}

fn chunk_dimensions_sizer(
    proposed_chunk_dimensions: (usize, usize, usize), // (width, height, len)
    asset_resolution: (usize, usize),
    pixel_type: PixelComponentType,
) -> (usize, usize, usize) {
    let (asset_width, asset_height) = asset_resolution;
    let chunk_width = max_factor_at_or_below(proposed_chunk_dimensions.0, asset_width);
    let chunk_height = max_factor_at_or_below(proposed_chunk_dimensions.1, asset_height);
    let chunk_len = 1; // only supports 1

    println!(
        "Chunk dimensions for {:<2}: {}x{}x{}",
        pixel_type.to_string(),
        chunk_width,
        chunk_height,
        chunk_len
    );

    // rval is (len, height, width) in conformance with ndarray
    (chunk_len, chunk_height, chunk_width)
}
