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
use crate::asset_reader_writer::asset_writer::*;
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
use crate::modulation::*;
use crate::source_coding::chunk::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use rand::Rng;

pub struct EncoderDecoderSimulator {
    macro_block_3d_iter: MacroBlock3DIterator<IntoPixelBufferIterator>,
    gop_len: usize,
    compression_ratio: f64,
    noise_power: f32,
    asset_resolution: (usize, usize),
    asset_writer: AssetWriter,
}

impl EncoderDecoderSimulator {
    pub fn try_new(
        in_path: std::path::PathBuf,
        out_path: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        noise_power: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = AssetReader::new(in_path);
        let frame_rate = reader.frame_rate()?;
        let asset_resolution = reader.resolution()?;
        let pb_iter: IntoPixelBufferIterator = reader.into();

        let writer_settings = AssetWritterSettings {
            path: std::path::PathBuf::from(out_path),
            codec: Codec::H264,
            resolution: asset_resolution,
            frame_rate: frame_rate,
        };
        let asset_resolution = (asset_resolution.0 as usize, asset_resolution.1 as usize);
        let writer = AssetWriter::load_new(writer_settings)?;
        Ok(Self {
            macro_block_3d_iter: pb_iter.into_macro_block_3d_iter(gop_len),
            gop_len,
            compression_ratio,
            noise_power,
            asset_resolution,
            asset_writer: writer,
        })
    }
}

fn ofdm_framer<PixelType: HasPixelComponentType>(
    dct_components: &mut TransformBlock3DDCT<PixelType>,
    compression_ratio: f64,
) -> (impl Iterator<Item = OFDMSymbol>, (usize, usize, usize)) {
    let chunks: Box<_> = dct_components.chunks_iter().collect();
    let first_chunk = chunks.first().expect("No chunks.");
    let chunk_dim = first_chunk.values.dim();

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
    (ofdm_framer, chunk_dim)
}

impl EncoderDecoderSimulator {
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.asset_writer.start_writing()?;

        while let Some(macro_block) = self.macro_block_3d_iter.next() {
            // encoder
            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
                ..
            } = macro_block;

            let mut y_dct = y_components.into();
            let (y_framer, y_chunk_dim) = ofdm_framer(&mut y_dct, self.compression_ratio);

            let mut cb_dct = cb_components.into();
            let (cb_framer, cb_chunk_dim) = ofdm_framer(&mut cb_dct, self.compression_ratio);

            let mut cr_dct = cr_components.into();
            let (cr_framer, cr_chunk_dim) = ofdm_framer(&mut cr_dct, self.compression_ratio);

            let encoder = y_framer.chain(cb_framer).chain(cr_framer);

            let noise_power = self.noise_power; // should sqrt?
            let mut encoder_plus_noise = encoder.map(|mut ofdmsymbol| {
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

            //decoder
            let y_dct_out = into_transform_block_3d_dct(
                &mut encoder_plus_noise,
                self.gop_len,
                self.asset_resolution,
                y_chunk_dim,
            );
            let cb_dct_out = into_transform_block_3d_dct(
                &mut encoder_plus_noise,
                self.gop_len,
                self.asset_resolution,
                cb_chunk_dim,
            );
            let cr_dct_out = into_transform_block_3d_dct(
                &mut encoder_plus_noise,
                self.gop_len,
                self.asset_resolution,
                cr_chunk_dim,
            );

            let new_macro_block_3d = MacroBlock3D {
                y_components: y_dct_out.into(),
                cb_components: cb_dct_out.into(),
                cr_components: cr_dct_out.into(),
                gop_len: self.gop_len,
            };
            let pixel_buffer_iter: transform_block_3d::PixelBufferIterator<_> =
                new_macro_block_3d.into();

            for pixel_buffer in pixel_buffer_iter {
                self.asset_writer.append_pixel_buffer(pixel_buffer)?;
                self.asset_writer.wait_for_writer_to_be_ready()?;
            }
        }
        self.asset_writer.finish_writing()?;
        Ok(())
    }
}

fn slices_allocation<PixelType: HasPixelComponentType>(
    gop_len: usize,
    asset_resolution: (usize, usize),
    chunk_dim: (usize, usize, usize),
    num_padding_slices: usize,
) -> ndarray::Array3<f32> {
    let (frame_width, frame_height) = (
        asset_resolution.0 / PixelType::TYPE.interleave_step(),
        asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
    );
    let chunks_per_gop =
        (gop_len * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

    let allocation_gop_length_with_padding =
        (((chunks_per_gop + num_padding_slices) * chunk_dim.0 * chunk_dim.1 * chunk_dim.2) as f64
            / (frame_width * frame_height) as f64)
            .ceil() as usize;

    ndarray::Array3::zeros((
        allocation_gop_length_with_padding,
        frame_height,
        frame_width,
    ))
}

fn into_transform_block_3d_dct<PixelType: HasPixelComponentType, O: Iterator<Item = OFDMSymbol>>(
    ofdm_symbol_iter: &mut O,
    gop_len: usize,
    asset_resolution: (usize, usize),
    chunk_dim: (usize, usize, usize),
) -> TransformBlock3DDCT<PixelType> {
    let (frame_width, frame_height) = (
        asset_resolution.0 / PixelType::TYPE.interleave_step(),
        asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
    );
    let chunks_per_gop =
        (gop_len * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

    let synchonizer: OFDMFrameSynchronizer<_> = ofdm_symbol_iter.into();
    let metadata_demodulator: MetadataDemodulator<_> = synchonizer.into();
    let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

    let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
    let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
        .by_ref()
        .take(chunks_per_gop)
        .map(|r| r.unwrap())
        .collect();
    assert!(!chunk_metadatas.is_empty());

    let metadata_bitmap = metadata_decompressor
        .take_metadata_bitmap()
        .expect("Failed to decode metadata_bitmap");

    let included_chunk_metadatas: Box<_> = metadata_bitmap
        .values
        .iter_ones()
        .map(|idx| chunk_metadatas[idx])
        .collect();

    let num_included_chunks = metadata_bitmap.values.count_ones();
    let num_included_slices = num_included_chunks.next_power_of_two();

    let synchronizer: OFDMFrameSynchronizer<_> =
        metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing

    let mut dct_allocation = slices_allocation::<PixelType>(
        gop_len,
        asset_resolution,
        chunk_dim,
        num_included_slices - num_included_chunks,
    );
    let slice_demodulator: SliceDemodulator<'_, PixelType, _> = SliceDemodulator::new(
        chunk_dim,
        metadata_bitmap,
        synchronizer,
        &mut dct_allocation,
    );

    let mut slice_and_metadatas = vec![];
    let mut included_chunk_metadatas_iter = included_chunk_metadatas.into_iter();
    for slice in slice_demodulator.take(num_included_slices) {
        // there will be more slices than chunk_metadatas
        let chunk_metadata = included_chunk_metadatas_iter.next().unwrap_or_default();
        let slice_and_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
        slice_and_metadatas.push(slice_and_metadata);
    }
    let slice_and_chunk_metadata_iter = slice_and_metadatas.into_iter();

    let chunks_iter = slice_and_chunk_metadata_iter
        .into_chunks_iter(num_included_chunks)
        .take(num_included_chunks);
    let power_descaler = PowerScaler::inverse(chunks_iter);
    let _chunks: Box<_> = power_descaler.collect(); // discard.. runs fwht

    TransformBlock3DDCT::from_chunks_owned(
        dct_allocation,
        &chunk_metadatas,
        gop_len,
        asset_resolution,
        chunk_dim,
    )
}
