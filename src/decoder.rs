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
use crate::compressor::*;
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
use std::{cell::Cell, rc::Rc};

pub struct FileWriterDecoder {
    asset_writer: AssetWriter,
    asset_resolution: (usize, usize),
    gop_len: usize,
    y_chunk_dim: (usize, usize, usize),
    cb_chunk_dim: (usize, usize, usize),
    cr_chunk_dim: (usize, usize, usize),
    hadamard: bool,
    started_writing: bool,
    original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>, // to compute PSNR
    final_stats: std::cell::OnceCell<Statistics>,
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
        hadamard: bool,
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
            hadamard,
            asset_writer: writer,
            started_writing: false,
            original_macro_block_3ds,
            final_stats: std::cell::OnceCell::new(),
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
            self.hadamard,
            self.original_macro_block_3ds.take(),
        );

        loop {
            if let Err(err) = self.run_loop_inner(&mut decoder) {
                // compute and print stats
                self.final_stats
                    .set(decoder.stats.finalize())
                    .expect("Already initialized.");
                let Statistics {
                    y_psnr,
                    cb_psnr,
                    cr_psnr,
                    weighted_total_psnr,
                } = self.final_stats.get().cloned().unwrap();
                let cumulative_snr = decoder.signal_to_noise_db();
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

    pub fn final_stats(&self) -> Result<Statistics, Box<dyn std::error::Error>> {
        self.final_stats
            .get()
            .cloned()
            .ok_or("Stats not yet finalized".into())
    }
}
struct Decoder<O: OFDMFrameSynchronizerTrait> {
    frame_synchronizer: O,
    asset_resolution: (usize, usize),
    gop_len: usize,
    y_chunk_dim: (usize, usize, usize),
    cb_chunk_dim: (usize, usize, usize),
    cr_chunk_dim: (usize, usize, usize),
    hadamard: bool,
    gops_received: usize,
    original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
    stats: PartialStatistics,
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
        hadamard: bool,
        original_macro_block_3ds: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
    ) -> Self {
        Self {
            frame_synchronizer,
            asset_resolution,
            gop_len,
            y_chunk_dim,
            cb_chunk_dim,
            cr_chunk_dim,
            hadamard,
            gops_received: 0,
            original_macro_block_3ds,
            stats: PartialStatistics::default(),
        }
    }

    fn next_gop(&mut self) -> Result<Box<[CVPixelBufferWrapper]>, Box<dyn std::error::Error>> {
        let (y_dct, cb_dct, cr_dct) = self.frame_synchronizer.next_gop(
            self.gop_len,
            self.asset_resolution,
            self.y_chunk_dim,
            self.cb_chunk_dim,
            self.cr_chunk_dim,
            self.hadamard,
            self.frame_synchronizer.signal_stats(),
        )?;
        self.frame_synchronizer.reset();
        self.frame_synchronizer.reset_seeking_frame_index();

        self.gops_received += 1;
        eprintln!("GOPS Received: {}", self.gops_received);

        let new_macro_block_3d = MacroBlock3D {
            y_components: y_dct.into(),
            cb_components: cb_dct.into(),
            cr_components: cr_dct.into(),
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

    fn signal_to_noise_db(&self) -> f64 {
        self.frame_synchronizer
            .signal_stats()
            .get()
            .signal_to_noise_db()
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

trait SignalDecoder {
    fn next_gop(
        &mut self,
        gop_len: usize,
        asset_resolution: (usize, usize),
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
        hadamard: bool,
        signal_stats: Rc<Cell<SignalStats>>,
    ) -> Result<
        (
            TransformBlock3DDCT<YPixelComponentType>,
            TransformBlock3DDCT<CbPixelComponentType>,
            TransformBlock3DDCT<CrPixelComponentType>,
        ),
        Box<dyn std::error::Error>,
    >;
    fn de_whiten<'a>(&'a mut self) -> impl Iterator<Item = QuadratureSymbol> + 'a;
    fn next_metadatas(
        &mut self,
        gop_len: usize,
        asset_resolution: (usize, usize),
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
    ) -> Result<
        (
            MetadataInfo<YPixelComponentType>,
            MetadataInfo<CbPixelComponentType>,
            MetadataInfo<CrPixelComponentType>,
        ),
        Box<dyn std::error::Error>,
    >;
    fn next_dct<PixelType: HasPixelComponentType>(
        &mut self,
        metadata: MetadataInfo<PixelType>,
        gop_len: usize,
        asset_resolution: (usize, usize),
        chunk_dim: (usize, usize, usize),
        hadamard: bool,
        signal_stats: Rc<Cell<SignalStats>>,
    ) -> Result<TransformBlock3DDCT<PixelType>, Box<dyn std::error::Error>>;
}
impl<O: Iterator<Item = QuadratureSymbol>> SignalDecoder for O {
    fn next_gop(
        &mut self,
        gop_len: usize,
        asset_resolution: (usize, usize),
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
        hadamard: bool,
        signal_stats: Rc<Cell<SignalStats>>,
    ) -> Result<
        (
            TransformBlock3DDCT<YPixelComponentType>,
            TransformBlock3DDCT<CbPixelComponentType>,
            TransformBlock3DDCT<CrPixelComponentType>,
        ),
        Box<dyn std::error::Error>,
    > {
        let mut clear_signal = self.de_whiten();
        let (y_metadata, cb_metadata, cr_metadata) = clear_signal.next_metadatas(
            gop_len,
            asset_resolution,
            y_chunk_dim,
            cb_chunk_dim,
            cr_chunk_dim,
        )?;
        let y_dct = clear_signal.next_dct(
            y_metadata,
            gop_len,
            asset_resolution,
            y_chunk_dim,
            hadamard,
            signal_stats.clone(),
        )?;
        let cb_dct = clear_signal.next_dct(
            cb_metadata,
            gop_len,
            asset_resolution,
            cb_chunk_dim,
            hadamard,
            signal_stats.clone(),
        )?;
        let cr_dct = clear_signal.next_dct(
            cr_metadata,
            gop_len,
            asset_resolution,
            cr_chunk_dim,
            hadamard,
            signal_stats.clone(),
        )?;

        Ok((y_dct, cb_dct, cr_dct))
    }

    fn de_whiten<'a>(&'a mut self) -> impl Iterator<Item = QuadratureSymbol> + 'a {
        // If whiten_len == 0, skip whitening.
        let Config {
            frame_length: _,
            whiten_length,
            whiten_rounds,
        } = Config::get();
        let coerced: Box<dyn Iterator<Item = QuadratureSymbol>> = if 0 != whiten_length {
            let de_whitener = Whitener::new(
                self,
                NUM_SUBCARRIERS,
                (1 + whiten_length) / NUM_SUBCARRIERS,
                whiten_rounds,
                true,
            );
            Box::new(de_whitener)
        } else {
            Box::new(self)
        };
        coerced
    }

    fn next_metadatas(
        &mut self,
        gop_len: usize,
        asset_resolution: (usize, usize),
        y_chunk_dim: (usize, usize, usize),
        cb_chunk_dim: (usize, usize, usize),
        cr_chunk_dim: (usize, usize, usize),
    ) -> Result<
        (
            MetadataInfo<YPixelComponentType>,
            MetadataInfo<CbPixelComponentType>,
            MetadataInfo<CrPixelComponentType>,
        ),
        Box<dyn std::error::Error>,
    > {
        let demodulator: MetadataDemodulator<_> = self.by_ref().into();
        let depacketizer: Depacketizer<_> = demodulator.into();

        fn chunks_per_gop(
            gop_len: usize,
            asset_resolution: (usize, usize),
            chunk_dim: (usize, usize, usize),
            pixel_type: PixelComponentType,
        ) -> usize {
            let (frame_width, frame_height) = (
                asset_resolution.0 / pixel_type.interleave_step(),
                asset_resolution.1 / pixel_type.vertical_subsampling(),
            );
            (gop_len * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2)
        }

        let y_chunks_per_gop = chunks_per_gop(
            gop_len,
            asset_resolution,
            y_chunk_dim,
            PixelComponentType::Y,
        );
        let cb_chunks_per_gop = chunks_per_gop(
            gop_len,
            asset_resolution,
            cr_chunk_dim,
            PixelComponentType::Cb,
        );
        let cr_chunks_per_gop = chunks_per_gop(
            gop_len,
            asset_resolution,
            cb_chunk_dim,
            PixelComponentType::Cr,
        );

        let mut y_decompressor = MetadataDecompressor::new(depacketizer, y_chunks_per_gop);

        fn next_metadata<PixelType: HasPixelComponentType, R: std::io::Read>(
            chunks_per_gop: usize,
            decompressor: &mut MetadataDecompressor<R>,
        ) -> Result<MetadataInfo<PixelType>, Box<dyn std::error::Error>> {
            let parse_result: Result<Box<[ChunkMetadata]>, _> =
                decompressor.take(chunks_per_gop).collect(); // using take for a size hint
            let chunk_metadatas = parse_result.map_err(|e| e.to_string())?;

            let metadata_bitmap = decompressor
                .take_metadata_bitmap()
                .map_err(|e| e.to_string())?;

            if chunks_per_gop != chunk_metadatas.len() {
                // EOF
                let count_chunk_metadatas = chunk_metadatas.len();
                let pixel_type = PixelType::TYPE;
                eprintln!(
                    "Number of chunk metadatas for {pixel_type} {count_chunk_metadatas} does not match chunks per GOP {chunks_per_gop}.",
                );
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }

            Ok(MetadataInfo::new(metadata_bitmap, chunk_metadatas.into()))
        }

        let y = next_metadata(y_chunks_per_gop, &mut y_decompressor)?;

        let mut cb_decompressor = y_decompressor.into_next(cb_chunks_per_gop);
        let cb = next_metadata(cb_chunks_per_gop, &mut cb_decompressor)?;

        let mut cr_decompressor = cb_decompressor.into_next(cr_chunks_per_gop);
        let cr = next_metadata(cr_chunks_per_gop, &mut cr_decompressor)?;

        Ok((y, cb, cr))
    }

    fn next_dct<PixelType: HasPixelComponentType>(
        &mut self,
        metadata: MetadataInfo<PixelType>,
        gop_len: usize,
        asset_resolution: (usize, usize),
        chunk_dim: (usize, usize, usize),
        hadamard: bool,
        signal_stats: Rc<Cell<SignalStats>>,
    ) -> Result<TransformBlock3DDCT<PixelType>, Box<dyn std::error::Error>> {
        let included_chunk_metadatas: Box<_> = metadata
            .bitmap
            .values
            .iter_ones()
            .map(|idx| metadata.values[idx])
            .collect();

        let num_included_chunks = metadata.bitmap.values.count_ones();
        let num_included_slices = if hadamard {
            num_included_chunks.next_power_of_two()
        } else {
            num_included_chunks
        };
        println!("{num_included_chunks} chunks | {num_included_slices} slices");

        let mut dct_allocation = slices_allocation::<PixelType>(
            gop_len,
            asset_resolution,
            chunk_dim,
            num_included_slices - num_included_chunks,
        );
        let slice_demodulator: SliceDemodulator<'_, PixelType, _> =
            SliceDemodulator::new(chunk_dim, metadata.bitmap, self, &mut dct_allocation);

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
            .into_chunks_iter(num_included_chunks, hadamard)
            .take(num_included_chunks);
        let power_descaler = PowerScaler::inverse(chunks_iter, signal_stats);
        let _chunks: Box<_> = power_descaler.collect(); // discard.. runs fwht

        let dct = TransformBlock3DDCT::from_chunks_owned(
            dct_allocation,
            &metadata.values,
            gop_len,
            asset_resolution,
            chunk_dim,
        );
        Ok(dct)
    }
}

struct MetadataInfo<PixelType: HasPixelComponentType> {
    bitmap: MetadataBitmap,
    values: Box<[ChunkMetadata]>,
    _marker: std::marker::PhantomData<PixelType>,
}
impl<PixelType: HasPixelComponentType> MetadataInfo<PixelType> {
    pub fn new(bitmap: MetadataBitmap, values: Box<[ChunkMetadata]>) -> Self {
        Self {
            bitmap,
            values,
            _marker: std::marker::PhantomData,
        }
    }
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
#[derive(Debug, Copy, Clone)]
pub struct Statistics {
    pub y_psnr: f64,
    pub cb_psnr: f64,
    pub cr_psnr: f64,
    pub weighted_total_psnr: f64,
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
