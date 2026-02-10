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

pub trait OFDMSymbolWriter {
    fn write(&mut self, symbol: OFDMSymbol) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait OFDMSymbolReader {
    fn into_iter(self) -> impl Iterator<Item = OFDMSymbol>;
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
    pub fn run<W: OFDMSymbolWriter>(
        &mut self,
        mut ofdm_symbol_writer: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        while let Some(macro_block) = self.macro_block_3d_iter.next() {
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
                ofdm_symbol_writer.write(symbol)?;
            }
        }
        Ok(())
    }
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
            path: std::path::PathBuf::from(out_path),
            codec: Codec::H264,
            resolution: (asset_resolution.0 as i32, asset_resolution.1 as i32),
            frame_rate: frame_rate,
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

    pub fn run<R: OFDMSymbolReader>(
        &mut self,
        ofdm_symbol_reader: R,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.asset_writer.start_writing()?;
        self.started_writing = true;

        let mut ofdm_symbol_iter = ofdm_symbol_reader.into_iter();

        loop {
            let y_dct_out = into_transform_block_3d_dct(
                &mut ofdm_symbol_iter,
                self.gop_len,
                self.asset_resolution,
                self.y_chunk_dim,
            )?;
            let cb_dct_out = into_transform_block_3d_dct(
                &mut ofdm_symbol_iter,
                self.gop_len,
                self.asset_resolution,
                self.cb_chunk_dim,
            )?;
            let cr_dct_out = into_transform_block_3d_dct(
                &mut ofdm_symbol_iter,
                self.gop_len,
                self.asset_resolution,
                self.cr_chunk_dim,
            )?;

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
    }
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

pub fn run_simulation(
    mut encoder: FileReaderEncoder,
    mut decoder: FileWriterDecoder,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mpsc_writer, mpsc_reader) = MPSCWriter::new_channel();
    let decoder_result = std::thread::spawn(move || {
        let result = decoder.run(mpsc_reader).map_err(|e| e.to_string());
        eprintln!("decoder result: {:?}", result);
        result
    });
    encoder.run(mpsc_writer)?;

    let _ = decoder_result.join().map_err(|_| "thread panic'd")?; // TODO: preserve inner error

    Ok(())
}

struct MPSCWriter {
    sender: std::sync::mpsc::SyncSender<OFDMSymbol>,
}
impl OFDMSymbolWriter for MPSCWriter {
    fn write(&mut self, symbol: OFDMSymbol) -> Result<(), Box<dyn std::error::Error>> {
        self.sender.send(symbol).map_err(|e| e.into())
    }
}
impl MPSCWriter {
    pub fn new_channel() -> (Self, MPSCReader) {
        let (sender, receiver) = std::sync::mpsc::sync_channel(0x80); // 64 KiB
        let writer = Self { sender };
        let reader = MPSCReader { receiver };
        (writer, reader)
    }
}

struct MPSCReader {
    receiver: std::sync::mpsc::Receiver<OFDMSymbol>,
}

impl OFDMSymbolReader for MPSCReader {
    fn into_iter(self) -> impl Iterator<Item = OFDMSymbol> {
        self.receiver.into_iter()
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
) -> Result<TransformBlock3DDCT<PixelType>, Box<dyn std::error::Error>> {
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

    let dct = TransformBlock3DDCT::from_chunks_owned(
        dct_allocation,
        &chunk_metadatas,
        gop_len,
        asset_resolution,
        chunk_dim,
    );
    Ok(dct)
}

fn max_factor_at_or_below(limit: usize, value: usize) -> usize {
    assert!(limit > 0);
    (1..=limit).rev().find(|i| value % i == 0).unwrap()
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

fn chunk_dimensions_inverter(chunk_dimensions: (usize, usize, usize)) -> (usize, usize, usize) {
    (chunk_dimensions.2, chunk_dimensions.1, chunk_dimensions.0)
}
