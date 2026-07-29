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

use crate::asset_reader_writer::asset_writer::*;
use crate::asset_reader_writer::*;
use crate::channel_coding::slice::*;
use crate::config::*;
use crate::framing::*;
use crate::metadata_coding::packetizer::*;
use crate::metadata_coding::*;
use crate::modulation::metadata::*;
use crate::modulation::slices::*;
use crate::modulation::*;
use crate::pixel_buffer::transform_block_3d::*;
use crate::pixel_buffer::*;
use crate::source_coding::chunk::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use crate::sync::*;
use crate::utils::*;
use ndarray_stats::DeviationExt;

pub struct FileWriterDecoder {
    asset_writer: AssetWriter,
    asset_resolution: (usize, usize),
    gop_len: usize,
    y_chunk_dim: (usize, usize, usize),
    cb_chunk_dim: (usize, usize, usize),
    cr_chunk_dim: (usize, usize, usize),
    started_writing: bool,
    original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>, // to compute PSNR
}
impl FileWriterDecoder {
    pub fn try_new(
        out_path: std::path::PathBuf,
        asset_resolution: (usize, usize),
        frame_rate: f64,
        gop_len: usize,
        y_chunk_dim: (usize, usize, usize), // length, height, width
        cb_chunk_dim: (usize, usize, usize), // length, height, width
        cr_chunk_dim: (usize, usize, usize), // length, height, width
        original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let writer_settings = AssetWritterSettings {
            path: out_path,
            codec: Codec::H264,
            resolution: (asset_resolution.0 as i32, asset_resolution.1 as i32),
            frame_rate,
        };

        let writer = AssetWriter::load_new(writer_settings)?;
        Ok(Self {
            asset_resolution,
            gop_len,
            y_chunk_dim,
            cb_chunk_dim,
            cr_chunk_dim,
            asset_writer: writer,
            started_writing: false,
            original_macro_block_3ds,
        })
    }

    pub fn run<R: Complex32Reader>(
        &mut self,
        complex32_reader: R,
        abort_token: AbortToken,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.asset_writer.start_writing()?;
        self.started_writing = true;

        let mut frame_synchronizer: OFDMFrameSynchronizer<_> = complex32_reader.into_iter().into();
        frame_synchronizer.abort_token = Some(abort_token);

        let mut decoder = Decoder::new(
            frame_synchronizer,
            self.asset_resolution,
            self.gop_len,
            self.y_chunk_dim,
            self.cb_chunk_dim,
            self.cr_chunk_dim,
            self.original_macro_block_3ds.take(),
        );

        loop {
            if let Err(err) = self.run_loop_inner(&mut decoder) {
                // compute and print stats
                let Statistics {
                    y_psnr,
                    cb_psnr,
                    cr_psnr,
                    weighted_total_psnr,
                } = decoder.stats.finalize();
                let cumulative_snr = decoder.signal_stats.signal_to_noise_db();
                println!("Cumulative SNR: {cumulative_snr:.2}");
                println!(
                    "PSNR: {weighted_total_psnr:.2} dB\t{y_psnr:.2} Y dB\t{cb_psnr:.2} Cb dB\t{cr_psnr:.2} Cr dB"
                );
                return Err(err);
            }
        }
    }

    fn run_loop_inner<O: OFDMFrameSynchronizerTrait>(
        &mut self,
        decoder: &mut Decoder<O>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixel_buffer_iter = decoder.next_pb_iter()?;
        for pixel_buffer in pixel_buffer_iter {
            self.asset_writer.append_pixel_buffer(pixel_buffer)?;
            self.asset_writer.wait_for_writer_to_be_ready()?;
        }
        Ok(())
    }
}
struct Decoder<O: OFDMFrameSynchronizerTrait> {
    frame_synchronizer: O,
    asset_resolution: (usize, usize),
    gop_len: usize,
    y_chunk_dim: (usize, usize, usize),
    cb_chunk_dim: (usize, usize, usize),
    cr_chunk_dim: (usize, usize, usize),
    snr: f64,
    gops_received: usize,
    original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
    stats: PartialStatistics,
    signal_stats: SignalStats,
}
impl<O: OFDMFrameSynchronizerTrait> Decoder<O> {
    fn next_pb_iter(
        &mut self,
    ) -> Result<impl Iterator<Item = CVPixelBufferWrapper>, Box<dyn std::error::Error>> {
        let gop = self.next_gop()?;
        let iter = gop.into_iter();
        Ok(iter)
    }

    pub fn new(
        frame_synchronizer: O,
        asset_resolution: (usize, usize),
        gop_len: usize,
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
        original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
    ) -> Self {
        Self {
            frame_synchronizer,
            asset_resolution,
            gop_len,
            y_chunk_dim,
            cb_chunk_dim,
            cr_chunk_dim,
            snr: 0.0,
            gops_received: 0,
            original_macro_block_3ds,
            stats: PartialStatistics::default(),
            signal_stats: SignalStats::default(),
        }
    }

    pub fn next_gop(&mut self) -> Result<Box<[CVPixelBufferWrapper]>, Box<dyn std::error::Error>> {
        let y_dct_out = into_transform_block_3d_dct(
            &mut self.frame_synchronizer,
            self.gop_len,
            self.asset_resolution,
            self.y_chunk_dim,
            self.snr, // a bit stale
        )
        .inspect_err(|_err| {
            println!(
                "Fatal SNR: {:.2}",
                self.frame_synchronizer.signal_to_noise_db()
            )
        })?;

        self.snr = self.frame_synchronizer.signal_to_noise_ratio();
        self.signal_stats += self.frame_synchronizer.current_signal_stats();
        self.frame_synchronizer.reset();
        self.frame_synchronizer.reset_seeking_frame_index();

        self.gops_received += 1;
        eprintln!("Y GOPS Received: {}", self.gops_received);

        let cb_dct_out = into_transform_block_3d_dct(
            &mut self.frame_synchronizer,
            self.gop_len,
            self.asset_resolution,
            self.cb_chunk_dim,
            self.snr,
        )
        .inspect_err(|_err| {
            println!(
                "Fatal SNR: {:.2}",
                self.frame_synchronizer.signal_to_noise_db()
            )
        })?;
        self.snr = self.frame_synchronizer.signal_to_noise_ratio();
        self.signal_stats += self.frame_synchronizer.current_signal_stats();
        self.frame_synchronizer.reset();
        self.frame_synchronizer.reset_seeking_frame_index();
        eprintln!("Cb GOPS Received: {}", self.gops_received);

        let cr_dct_out = into_transform_block_3d_dct(
            &mut self.frame_synchronizer,
            self.gop_len,
            self.asset_resolution,
            self.cr_chunk_dim,
            self.snr,
        )
        .inspect_err(|_err| {
            println!(
                "Fatal SNR: {:.2}",
                self.frame_synchronizer.signal_to_noise_db()
            )
        })?;
        self.snr = self.frame_synchronizer.signal_to_noise_ratio();
        self.signal_stats += self.frame_synchronizer.current_signal_stats();
        self.frame_synchronizer.reset();
        self.frame_synchronizer.reset_seeking_frame_index();
        eprintln!("Cr GOPS Received: {}", self.gops_received);

        let new_macro_block_3d = MacroBlock3D {
            y_components: y_dct_out.into(),
            cb_components: cb_dct_out.into(),
            cr_components: cr_dct_out.into(),
            gop_len: self.gop_len,
        };

        if let Some(mb_receiver) = &mut self.original_macro_block_3ds {
            let original_mb = mb_receiver.recv()?;
            let y_psnr = original_mb
                .y_components
                .values()
                .mean_sq_err(new_macro_block_3d.y_components.values())?
                .psnr(u8::MAX as f64);
            let cb_psnr = original_mb
                .cb_components
                .values()
                .mean_sq_err(new_macro_block_3d.cb_components.values())?
                .psnr(u8::MAX as f64);
            let cr_psnr = original_mb
                .cr_components
                .values()
                .mean_sq_err(new_macro_block_3d.cr_components.values())?
                .psnr(u8::MAX as f64);

            self.stats.y_psnr_partial_sum += y_psnr;
            self.stats.cb_psnr_partial_sum += cb_psnr;
            self.stats.cr_psnr_partial_sum += cr_psnr;
            self.stats.sample_count += 1;
            println!("PSNR: {y_psnr:.2} Y dB\t{cb_psnr:.2} Cb dB\t{cr_psnr:.2} Cr dB");
        }

        let pixel_buffer_iter: transform_block_3d::PixelBufferIterator<_, _> =
            new_macro_block_3d.into();
        let gop = pixel_buffer_iter.collect();
        Ok(gop)
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
    snr: f64,
) -> Result<TransformBlock3DDCT<PixelType>, Box<dyn std::error::Error>> {
    let (frame_width, frame_height) = (
        asset_resolution.0 / PixelType::TYPE.interleave_step(),
        asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
    );
    let chunks_per_gop =
        (gop_len * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

    let de_whitener = Whitener::new(
        synchronizer,
        Config::get().per_pixel_type::<PixelType>().whiten_length,
        data_symbols_per_ofdm_symbol(),
        true,
    );

    let metadata_demodulator: MetadataDemodulator<_> = de_whitener.into();
    let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

    let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
    let mut chunk_metadatas: Vec<ChunkMetadata> = Vec::with_capacity(chunks_per_gop);
    for metadata_result in metadata_decompressor.by_ref().take(chunks_per_gop) {
        chunk_metadatas.push(metadata_result.map_err(|e| e.to_string())?);
    }
    if chunks_per_gop != chunk_metadatas.len() {
        // EOF
        let count_chunk_metadatas = chunk_metadatas.len();
        let pixel_type = PixelType::TYPE;
        eprintln!(
            "Number of chunk metadatas for {pixel_type} {count_chunk_metadatas} does not match chunks per GOP {chunks_per_gop}.",
        );
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
    println!("{num_included_chunks} chunks | {num_included_slices} slices");

    let de_whitener = metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing

    let mut dct_allocation = slices_allocation::<PixelType>(
        gop_len,
        asset_resolution,
        chunk_dim,
        num_included_slices - num_included_chunks,
    );
    let slice_demodulator: SliceDemodulator<'_, PixelType, _> =
        SliceDemodulator::new(chunk_dim, metadata_bitmap, de_whitener, &mut dct_allocation);

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
    let power_descaler = PowerScaler::inverse(chunks_iter, snr);
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

#[derive(Default)]
struct PartialStatistics {
    y_psnr_partial_sum: f64,
    cb_psnr_partial_sum: f64,
    cr_psnr_partial_sum: f64,
    sample_count: usize,
}
impl PartialStatistics {
    fn finalize(&self) -> Statistics {
        let sample_count = self.sample_count as f64;
        Statistics {
            y_psnr: self.y_psnr_partial_sum / sample_count,
            cb_psnr: self.cb_psnr_partial_sum / sample_count,
            cr_psnr: self.cr_psnr_partial_sum / sample_count,
            weighted_total_psnr: (6.0 * self.y_psnr_partial_sum
                + self.cb_psnr_partial_sum
                + self.cr_psnr_partial_sum)
                / (8.0 * sample_count),
        }
    }
}
struct Statistics {
    y_psnr: f64,
    cb_psnr: f64,
    cr_psnr: f64,
    weighted_total_psnr: f64,
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
