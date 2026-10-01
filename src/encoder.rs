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
use crate::modulation::QuadratureSymbol;
use crate::modulation::metadata::*;
use crate::modulation::slices::*;
use crate::pixel_buffer::transform_block_3d::*;
use crate::pixel_buffer::*;
use crate::source_coding::chunk::*;
use crate::source_coding::power_scaling::*;
use crate::source_coding::transform_block_3d_dct::*;
use crate::sync::*;

#[cfg(target_vendor = "apple")]
pub type FileReaderEncoder = Encoder<IntoPixelBufferIterator, CVPixelBufferWrapper>;

#[derive(Copy, Clone)]
pub struct PerPixelConfiguration {
    pub compression_ratio: f64,
    pub chunk_dimensions: (usize, usize, usize),
}

pub struct Encoder<I: Iterator<Item = PB>, PB: PixelBuffer> {
    macro_block_3d_iter: MacroBlock3DIterator<I, PB>,
    y_config: PerPixelConfiguration,
    cb_config: PerPixelConfiguration,
    cr_config: PerPixelConfiguration,
    asset_resolution: (usize, usize),
    frame_rate: f64,
    macro_block_tap: Option<MacroBlockTap>,
    hadamard: bool,
    ofdm: bool,
}

impl<I: Iterator<Item = PB>, PB: PixelBuffer> Encoder<I, PB> {
    #[cfg(target_vendor = "apple")]
    pub fn with_file(
        in_path: std::path::PathBuf,
        gop_len: usize,
        y_config: PerPixelConfiguration,
        cb_config: PerPixelConfiguration,
        cr_config: PerPixelConfiguration,
        hadamard: bool,
        ofdm: bool,
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
            y_config,
            cb_config,
            cr_config,
            asset_resolution,
            frame_rate,
            hadamard,
            ofdm,
            macro_block_tap,
        ))
    }

    pub fn new(
        pb_iter: I,
        gop_len: usize,
        mut y_config: PerPixelConfiguration,
        mut cb_config: PerPixelConfiguration,
        mut cr_config: PerPixelConfiguration,
        asset_resolution: (usize, usize),
        frame_rate: f64,
        hadamard: bool,
        ofdm: bool,
        macro_block_tap: Option<MacroBlockTap>,
    ) -> Self {
        y_config.chunk_dimensions = chunk_dimensions_sizer(
            y_config.chunk_dimensions,
            asset_resolution,
            PixelComponentType::Y,
        );
        cb_config.chunk_dimensions = chunk_dimensions_sizer(
            cb_config.chunk_dimensions,
            asset_resolution,
            PixelComponentType::Cb,
        );
        cr_config.chunk_dimensions = chunk_dimensions_sizer(
            cr_config.chunk_dimensions,
            asset_resolution,
            PixelComponentType::Cr,
        );

        Self {
            macro_block_3d_iter: MacroBlock3DIterator::new(pb_iter, gop_len),
            y_config,
            cb_config,
            cr_config,
            asset_resolution,
            frame_rate,
            macro_block_tap,
            hadamard,
            ofdm,
        }
    }

    pub fn asset_resolution(&self) -> (usize, usize) {
        self.asset_resolution
    }
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }
    pub fn y_chunk_dimensions(&self) -> (usize, usize, usize) {
        self.y_config.chunk_dimensions
    }
    pub fn cb_chunk_dimensions(&self) -> (usize, usize, usize) {
        self.cb_config.chunk_dimensions
    }
    pub fn cr_chunk_dimensions(&self) -> (usize, usize, usize) {
        self.cr_config.chunk_dimensions
    }
    pub fn run(
        &mut self,
        ofdm_symbol_writer: &mut dyn Complex32Consumer,
        abort_token: AbortToken,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut count_symbols = 0;
        for macro_block in self.macro_block_3d_iter.by_ref() {
            if let Some(tap) = &mut self.macro_block_tap {
                let clone = macro_block.clone();
                tap.writer.send(clone)?;
            }

            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
                ..
            } = macro_block;

            let mut y_dct: TransformBlock3DDCT<YPixelComponentType> = y_components.into();
            let mut cb_dct: TransformBlock3DDCT<CbPixelComponentType> = cb_components.into();
            let mut cr_dct: TransformBlock3DDCT<CrPixelComponentType> = cr_components.into();

            let encode_signal = encode_signal(
                &mut y_dct,
                &mut cb_dct,
                &mut cr_dct,
                self.y_config,
                self.cb_config,
                self.cr_config,
                self.hadamard,
                self.ofdm,
            );

            for frame in encode_signal {
                count_symbols += OFDM_SYMBOL_LEN * frame.symbols.len();
                ofdm_symbol_writer.consume(frame.into_box_complex32_slice(), true)?;
                if abort_token.is_aborted() {
                    return Err("Encoder aborted.".into());
                }
            }
            eprintln!("Cumulative Symbols Transmitted: {}", count_symbols);
        }
        Ok(())
    }
}

fn encode_signal<'a>(
    y_dct: &'a mut TransformBlock3DDCT<YPixelComponentType>,
    cb_dct: &'a mut TransformBlock3DDCT<CbPixelComponentType>,
    cr_dct: &'a mut TransformBlock3DDCT<CrPixelComponentType>,
    y_config: PerPixelConfiguration,
    cb_config: PerPixelConfiguration,
    cr_config: PerPixelConfiguration,
    hadamard: bool,
    ofdm: bool,
) -> impl Iterator<Item = OFDMFrame> + 'a {
    let y_chunks: Box<_> = y_dct.chunks_iter(y_config.chunk_dimensions).collect();
    let cb_chunks: Box<_> = cb_dct.chunks_iter(cb_config.chunk_dimensions).collect();
    let cr_chunks: Box<_> = cr_dct.chunks_iter(cr_config.chunk_dimensions).collect();

    // metadata
    let y_mbitmap = MetadataBitmap::new(&y_chunks, y_config.compression_ratio);
    let cb_mbitmap = MetadataBitmap::new(&cb_chunks, cb_config.compression_ratio);
    let cr_mbitmap = MetadataBitmap::new(&cr_chunks, cr_config.compression_ratio);

    let metadata_signal = metadata_signal(
        (&y_mbitmap, &y_chunks),
        (&cb_mbitmap, &cb_chunks),
        (&cb_mbitmap, &cr_chunks),
    );

    // slices
    let y_slice_signal = slice_signal(y_chunks, y_mbitmap, hadamard);
    let cb_slice_signal = slice_signal(cb_chunks, cb_mbitmap, hadamard);
    let cr_slice_signal = slice_signal(cr_chunks, cr_mbitmap, hadamard);

    let clear_signal = metadata_signal
        .chain(y_slice_signal)
        .chain(cb_slice_signal)
        .chain(cr_slice_signal);

    // framing
    framed_signal(clear_signal, ofdm)
}

fn metadata_signal(
    y: (&MetadataBitmap, &[Chunk<YPixelComponentType>]),
    cb: (&MetadataBitmap, &[Chunk<CbPixelComponentType>]),
    cr: (&MetadataBitmap, &[Chunk<CrPixelComponentType>]),
) -> impl Iterator<Item = QuadratureSymbol> + use<> {
    let compressed_metadata = compress_metadata(
        (&y.0, y.1.metadata_iter()),
        (&cb.0, cb.1.metadata_iter()),
        (&cr.0, cr.1.metadata_iter()),
    )
    .expect("Compressing metadata failed.");
    let packetizer: Packetizer = compressed_metadata.into();
    let metadata_modulator: MetadataModulator<_> = packetizer.into();
    metadata_modulator.flatten()
}

fn slice_signal<PixelType: HasPixelComponentType>(
    chunks: Box<[Chunk<PixelType>]>,
    metadata_bitmap: MetadataBitmap,
    hadamard: bool,
) -> impl Iterator<Item = QuadratureSymbol> {
    let num_included_chunks = metadata_bitmap.values.count_ones();
    let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);
    let slice_modulator: SliceModulator<'_, _, _> = PowerScaler::new(compressor)
        .into_slice_iter(num_included_chunks, hadamard)
        .map(|slice_and_chunk_metadata| slice_and_chunk_metadata.slice)
        .into();

    slice_modulator
}

fn framed_signal<'a>(
    signal: impl Iterator<Item = QuadratureSymbol> + 'a,
    ofdm: bool,
) -> impl Iterator<Item = OFDMFrame> + 'a {
    // If whiten_len == 0, skip whitening.
    let Config {
        frame_length: _,
        whiten_length,
        whiten_rounds,
    } = Config::get();
    let iq_iter: Box<dyn Iterator<Item = QuadratureSymbol>> = if 0 != whiten_length {
        let whitener = Whitener::new(
            signal,
            NUM_SUBCARRIERS,
            (1 + whiten_length) / NUM_SUBCARRIERS,
            whiten_rounds,
            false,
        );
        Box::new(whitener)
    } else {
        Box::new(signal)
    };

    let framer: Box<dyn Iterator<Item = OFDMFrame>> = if ofdm {
        let ofdm_framer: OFDMFrameGenerator<_> = iq_iter.into();
        Box::new(ofdm_framer)
    } else {
        // cram raw iq symbols into an OFDMFrame iterator
        let mut iq_iter = iq_iter.peekable();
        let iter = std::iter::from_fn(move || {
            if iq_iter.peek().is_none() {
                return None;
            }
            let mut ofdm_symbol = OFDMSymbol::default();
            for iq in ofdm_symbol.time_domain_symbols.iter_mut() {
                *iq = iq_iter.next().unwrap_or_default().into();
            }
            Some(OFDMFrame {
                symbols: vec![ofdm_symbol; 1],
            })
        });
        Box::new(iter)
    };
    framer
}

trait ChunkMetadataIter {
    fn metadata_iter(&self) -> impl Iterator<Item = &ChunkMetadata>;
}
impl<PixelType: HasPixelComponentType> ChunkMetadataIter for &[Chunk<'_, PixelType>] {
    fn metadata_iter(&self) -> impl Iterator<Item = &ChunkMetadata> {
        self.iter().map(|chunk| &chunk.metadata)
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
