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

use crate::asset_reader_writer::asset_writer::*;
use crate::asset_reader_writer::transform_block_3d::*;
use crate::asset_reader_writer::*;
use crate::channel_coding::slice::*;
use crate::framing::*;
use crate::metadata_coding::packetizer::*;
use crate::metadata_coding::*;
use crate::modulation::metadata::*;
use crate::modulation::slices::*;
use crate::modulation::*;
use crate::source_coding::chunk::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use num_complex::Complex32;

pub trait Complex32Reader {
    fn into_iter(self) -> impl Iterator<Item = Box<[Complex32]>>;
}

pub struct FileWriterDecoder {
    asset_writer: AssetWriter,
    asset_resolution: (usize, usize),
    gop_len: usize,
    y_chunk_dim: (usize, usize, usize),
    cb_chunk_dim: (usize, usize, usize),
    cr_chunk_dim: (usize, usize, usize),
    started_writing: bool,
}
impl FileWriterDecoder {
    pub fn try_new(
        out_path: std::path::PathBuf,
        asset_resolution: (usize, usize),
        frame_rate: f64,
        gop_len: usize,
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let writer_settings = AssetWritterSettings {
            path: out_path,
            codec: Codec::H264,
            resolution: (asset_resolution.0 as i32, asset_resolution.1 as i32),
            frame_rate,
        };

        let y_chunk_dim = chunk_dimensions_inverter(y_chunk_dim);
        let cb_chunk_dim = chunk_dimensions_inverter(cb_chunk_dim);
        let cr_chunk_dim = chunk_dimensions_inverter(cr_chunk_dim);

        let writer = AssetWriter::load_new(writer_settings)?;
        Ok(Self {
            asset_resolution,
            gop_len,
            y_chunk_dim,
            cb_chunk_dim,
            cr_chunk_dim,
            asset_writer: writer,
            started_writing: false,
        })
    }

    pub fn run<R: Complex32Reader>(
        &mut self,
        complex32_reader: R,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.asset_writer.start_writing()?;
        self.started_writing = true;

        let mut frame_synchronizer: OFDMFrameSynchronizer<_> = complex32_reader.into_iter().into();

        let mut gops_received = 0;
        eprintln!("GOPS Received: {}", gops_received);

        loop {
            let y_dct_out = into_transform_block_3d_dct(
                &mut frame_synchronizer,
                self.gop_len,
                self.asset_resolution,
                self.y_chunk_dim,
            )?;

            gops_received += 1;
            eprintln!("GOPS Received: {}", gops_received);

            let cb_dct_out = into_transform_block_3d_dct(
                &mut frame_synchronizer,
                self.gop_len,
                self.asset_resolution,
                self.cb_chunk_dim,
            )?;
            frame_synchronizer.reset();

            let cr_dct_out = into_transform_block_3d_dct(
                &mut frame_synchronizer,
                self.gop_len,
                self.asset_resolution,
                self.cr_chunk_dim,
            )?;
            frame_synchronizer.reset();

            let new_macro_block_3d = MacroBlock3D {
                y_components: y_dct_out.into(),
                cb_components: cb_dct_out.into(),
                cr_components: cr_dct_out.into(),
                gop_len: self.gop_len,
            };
            frame_synchronizer.reset();

            let pixel_buffer_iter: transform_block_3d::PixelBufferIterator<_> =
                new_macro_block_3d.into();

            for pixel_buffer in pixel_buffer_iter {
                self.asset_writer.append_pixel_buffer(pixel_buffer)?;
                self.asset_writer.wait_for_writer_to_be_ready()?;
            }
        }
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

fn into_transform_block_3d_dct<
    PixelType: HasPixelComponentType,
    O: Iterator<Item = QuadratureSymbol>,
>(
    synchronizer: &mut O,
    gop_len: usize,
    asset_resolution: (usize, usize),
    chunk_dim: (usize, usize, usize),
) -> Result<TransformBlock3DDCT<PixelType>, Box<dyn std::error::Error>> {
    let (frame_width, frame_height) = (
        asset_resolution.0 / PixelType::TYPE.interleave_step(),
        asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
    );
    let chunks_per_gop =
        (gop_len * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

    let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
    let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

    let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
    let mut chunk_metadatas: Vec<ChunkMetadata> = Vec::with_capacity(chunks_per_gop);
    for metadata_result in metadata_decompressor.by_ref().take(chunks_per_gop) {
        chunk_metadatas.push(metadata_result.map_err(|e| e.to_string())?);
    }
    if chunks_per_gop != chunk_metadatas.len() {
        // EOF
        return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
    }

    let metadata_bitmap = metadata_decompressor
        .take_metadata_bitmap()
        .map_err(|_| "Failed to decode metadata_bitmap")?; // TODO: don't discard error

    let included_chunk_metadatas: Box<_> = metadata_bitmap
        .values
        .iter_ones()
        .map(|idx| chunk_metadatas[idx])
        .collect();

    let num_included_chunks = metadata_bitmap.values.count_ones();
    let num_included_slices = num_included_chunks.next_power_of_two();

    let synchronizer = metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing

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

    let dct = TransformBlock3DDCT::from_chunks_owned(
        dct_allocation,
        &chunk_metadatas,
        gop_len,
        asset_resolution,
        chunk_dim,
    );
    Ok(dct)
}

impl Drop for FileWriterDecoder {
    fn drop(&mut self) {
        if self.started_writing {
            self.asset_writer
                .finish_writing()
                .expect("Failed to finish writing.");
        }
    }
}

fn chunk_dimensions_inverter(chunk_dimensions: (usize, usize, usize)) -> (usize, usize, usize) {
    (chunk_dimensions.2, chunk_dimensions.1, chunk_dimensions.0)
}
