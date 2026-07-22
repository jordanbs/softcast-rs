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

#[cfg(target_vendor = "apple")]
use crate::asset_reader_writer::asset_reader::*;
use crate::channel_coding::slice::*;
use crate::compressor::*;
use crate::config::*;
use crate::framing::*;
use crate::metadata_coding::packetizer::*;
use crate::metadata_coding::*;
use crate::modulation::metadata::*;
use crate::modulation::slices::*;
use crate::noise::*;
use crate::pixel_buffer::transform_block_3d::*;
use crate::pixel_buffer::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use crate::sync::*;
use num_complex::Complex32;

#[cfg(target_vendor = "apple")]
pub type FileReaderEncoder = Encoder<IntoPixelBufferIterator, CVPixelBufferWrapper>;

pub struct Encoder<I: Iterator<Item = PB>, PB: PixelBuffer> {
    macro_block_3d_iter: MacroBlock3DIterator<I, PB>,
    compression_ratio: f64,
    noise_power: f32,
    pub y_chunk_dimensions: (usize, usize, usize),
    pub cb_chunk_dimensions: (usize, usize, usize),
    pub cr_chunk_dimensions: (usize, usize, usize),
    asset_resolution: (usize, usize),
    frame_rate: f64,
    macro_block_tap: Option<MacroBlockTap>,
}

impl<I: Iterator<Item = PB>, PB: PixelBuffer> Encoder<I, PB> {
    #[cfg(target_vendor = "apple")]
    pub fn with_file(
        in_path: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        noise_power: f32,
        y_chunk_dimensions: (usize, usize, usize),
        cb_chunk_dimensions: (usize, usize, usize),
        cr_chunk_dimensions: (usize, usize, usize),
        macro_block_tap: Option<MacroBlockTap>,
    ) -> Result<Encoder<IntoPixelBufferIterator, CVPixelBufferWrapper>, Box<dyn std::error::Error>>
    {
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

        Ok(Encoder::new(
            pb_iter,
            gop_len,
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            cb_chunk_dimensions,
            cr_chunk_dimensions,
            asset_resolution,
            frame_rate,
            macro_block_tap,
        ))
    }

    pub fn new(
        pb_iter: I,
        gop_len: usize,
        compression_ratio: f64,
        noise_power: f32,
        y_chunk_dimensions: (usize, usize, usize),
        cb_chunk_dimensions: (usize, usize, usize),
        cr_chunk_dimensions: (usize, usize, usize),
        asset_resolution: (usize, usize),
        frame_rate: f64,
        macro_block_tap: Option<MacroBlockTap>,
    ) -> Self {
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

        Self {
            macro_block_3d_iter: MacroBlock3DIterator::new(pb_iter, gop_len),
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            cb_chunk_dimensions,
            cr_chunk_dimensions,
            asset_resolution,
            frame_rate,
            macro_block_tap,
        }
    }

    pub fn asset_resolution(&self) -> (usize, usize) {
        self.asset_resolution
    }
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }
    pub fn run(
        &mut self,
        ofdm_symbol_writer: &mut dyn Complex32Consumer,
        abort_token: AbortToken,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let count_symbols_arc = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
        for macro_block in self.macro_block_3d_iter.by_ref() {
            if let Some(tap) = &mut self.macro_block_tap {
                let clone = macro_block.clone();
                tap.writer.send(clone)?;
            }

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

            let count_symbols_arc_clone = count_symbols_arc.clone();
            let encoder = y_framer
                .chain(cb_framer)
                .chain(cr_framer)
                .map(|ofdmframe| ofdmframe.into_box_complex32_slice())
                .inspect(|iqs| {
                    count_symbols_arc_clone
                        .fetch_add(iqs.len() as i64, std::sync::atomic::Ordering::Relaxed);
                });

            let mut dyn_encoder: Box<dyn Iterator<Item = Box<[Complex32]>>> =
                if self.noise_power == 0.0 {
                    Box::new(encoder)
                } else {
                    let noise_encoder = AdditiveWhiteGaussianNoise::new(encoder, self.noise_power);
                    Box::new(noise_encoder)
                };

            while let Some(frame) = dyn_encoder.next() {
                ofdm_symbol_writer.consume(frame, true)?;
                if abort_token.is_aborted() {
                    return Err("Encoder aborted.".into());
                }
            }
            eprintln!(
                "Cumulative Symbols Transmitted: {}",
                count_symbols_arc.load(std::sync::atomic::Ordering::Relaxed)
            );
        }
        Ok(())
    }
}

fn ofdm_framer<PixelType: HasPixelComponentType>(
    dct_components: &mut TransformBlock3DDCT<PixelType>,
    compression_ratio: f64,
    chunk_dimensions: (usize, usize, usize),
) -> impl Iterator<Item = OFDMFrame> {
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

    let frequency_domain_signal = metadata_modulator.flatten().chain(slice_modulator);
    let whitener = Whitener::new(
        frequency_domain_signal,
        Config::get().per_pixel_type::<PixelType>().whiten_length,
        data_symbols_per_ofdm_symbol(),
        false,
    );

    // ofdm
    let ofdm_framer: OFDMFrameGenerator<_> = whitener.into();
    ofdm_framer
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

pub struct MacroBlockTap {
    writer: std::sync::mpsc::SyncSender<MacroBlock3D>,
    reader: Option<std::sync::mpsc::Receiver<MacroBlock3D>>,
}
impl Default for MacroBlockTap {
    fn default() -> Self {
        let (writer, reader) = std::sync::mpsc::sync_channel(1); // limit to 1 macro block at a time
        Self {
            writer,
            reader: Some(reader),
        }
    }
}
impl MacroBlockTap {
    pub fn take_receiver(&mut self) -> std::sync::mpsc::Receiver<MacroBlock3D> {
        self.reader.take().expect("reader already taken")
    }
}
