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

use crate::modulation::*;
use liquid_sys;
use num_complex::Complex32;

pub const NUM_SUBCARRIERS: usize = 64;
const CP_LEN: usize = 16;
const TAPER_LEN: usize = 4;
const FRAME_LEN: usize = NUM_SUBCARRIERS + CP_LEN;
pub const OFDM_SYMBOL_LEN: usize = FRAME_LEN;

static FFTW_PLANNER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug)]
pub struct OFDMSymbol {
    pub time_domain_symbols: Box<[Complex32]>,
}

impl Default for OFDMSymbol {
    fn default() -> Self {
        Self {
            time_domain_symbols: vec![Complex32::default(); FRAME_LEN].into(),
        }
    }
}

pub struct OFDMFrameGenerator<I: Iterator<Item = QuadratureSymbol>> {
    quadrature_symbol_iter: std::iter::Peekable<I>,
    ofdm_framegen: liquid_sys::ofdmframegen,
    state: OFDMFrameGeneratorState,
    subcarrier_allocation: Box<[u8]>,
}

enum OFDMFrameGeneratorState {
    S0a,
    S0b,
    S1,
    Data,
    Complete,
}

impl<I: Iterator<Item = QuadratureSymbol>> From<I> for OFDMFrameGenerator<I> {
    fn from(quadrature_symbol_iter: I) -> Self {
        let mut subcarrier_allocation = Box::new([0u8; NUM_SUBCARRIERS]);
        let status = unsafe {
            liquid_sys::ofdmframe_init_default_sctype(
                NUM_SUBCARRIERS as u32,
                subcarrier_allocation.as_mut_ptr(),
            )
        } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

        let ofdm_framegen = {
            // ofdmframesync calls FFTW_PLANNER, which is not thread safe.
            let _guard = FFTW_PLANNER_LOCK.lock().unwrap(); // drops at end of scope
            unsafe {
                liquid_sys::ofdmframegen_create(
                    NUM_SUBCARRIERS as u32,
                    CP_LEN as u32,
                    TAPER_LEN as u32,
                    std::ptr::null_mut(),
                )
            }
        };
        assert_ne!(std::ptr::null_mut(), ofdm_framegen);
        Self {
            quadrature_symbol_iter: quadrature_symbol_iter.peekable(),
            ofdm_framegen,
            state: OFDMFrameGeneratorState::S0a,
            subcarrier_allocation,
        }
    }
}

impl<I: Iterator<Item = QuadratureSymbol>> Iterator for OFDMFrameGenerator<I> {
    type Item = OFDMSymbol;

    fn next(&mut self) -> Option<Self::Item> {
        let mut symbol = OFDMSymbol::default();

        match self.state {
            OFDMFrameGeneratorState::S0a => unsafe {
                self.state = OFDMFrameGeneratorState::S0b;
                let status = liquid_sys::ofdmframegen_write_S0a(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S0b => unsafe {
                self.state = OFDMFrameGeneratorState::S1;
                let status = liquid_sys::ofdmframegen_write_S0b(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S1 => unsafe {
                self.state = OFDMFrameGeneratorState::Data;
                let status = liquid_sys::ofdmframegen_write_S1(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::Data => {
                if self.quadrature_symbol_iter.peek().is_none() {
                    self.state = OFDMFrameGeneratorState::Complete;
                    // write tail
                    let status = unsafe {
                        liquid_sys::ofdmframegen_writetail(
                            self.ofdm_framegen,
                            symbol.time_domain_symbols.as_mut_ptr(),
                        )
                    } as u32;
                    assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                    return Some(symbol);
                }

                let freq_domain: Box<_> = self
                    .subcarrier_allocation
                    .iter()
                    // Insert placeholders for null and pilot subcarriers.
                    .map(|subcarrier_type| match *subcarrier_type as u32 {
                        // Pad frame with zero values; don't drop data.
                        liquid_sys::OFDMFRAME_SCTYPE_DATA => {
                            self.quadrature_symbol_iter.next().unwrap_or_default()
                        }
                        _ => QuadratureSymbol::default(),
                    })
                    .collect();
                let time_domain = &mut symbol.time_domain_symbols;
                let status = unsafe {
                    liquid_sys::ofdmframegen_writesymbol(
                        self.ofdm_framegen,
                        freq_domain.as_ptr() as *mut Complex32,
                        time_domain.as_mut_ptr(),
                    )
                } as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            }
            OFDMFrameGeneratorState::Complete => None,
        }
    }
}

impl<I: Iterator<Item = QuadratureSymbol>> Drop for OFDMFrameGenerator<I> {
    fn drop(&mut self) {
        let status = unsafe { liquid_sys::ofdmframegen_destroy(self.ofdm_framegen) } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
    }
}

pub struct OFDMFrameSynchronizer<I: Iterator<Item = OFDMSymbol>> {
    ofdm_symbol_iter: I,
    ofdm_framesync: liquid_sys::ofdmframesync,
    callback_context: Box<CallbackContext>,
}

#[allow(non_snake_case)]
extern "C" fn ofdm_framesync_callback(
    _y: *mut Complex32,
    _p: *mut u8,
    _M: u32,
    _userdata: *mut core::ffi::c_void,
) -> i32 {
    let subcarrier_samples = unsafe { std::slice::from_raw_parts(_y, _M as usize) };
    let subcarrier_allocation = unsafe { std::slice::from_raw_parts(_p, _M as usize) };

    let context_ptr = _userdata as *mut CallbackContext;
    let context = unsafe { context_ptr.as_mut().expect("NULL context_ptr.") };

    context.ofdm_framesync_callback(subcarrier_samples, subcarrier_allocation);

    liquid_sys::liquid_error_code_LIQUID_OK as i32 // always ok
}

#[derive(Default)]
struct CallbackContext {
    time_domain_symbols: std::collections::VecDeque<QuadratureSymbol>,
}

impl CallbackContext {
    fn ofdm_framesync_callback(
        &mut self,
        subcarrier_samples: &[Complex32],
        subcarrier_allocation: &[u8],
    ) {
        let mut new_samples: std::collections::VecDeque<_> =
            std::collections::VecDeque::with_capacity(NUM_SUBCARRIERS);
        new_samples.extend(
            subcarrier_samples
                .iter()
                .enumerate()
                // ignore null and pilot subcarriers
                .filter(|(idx, _)| {
                    liquid_sys::OFDMFRAME_SCTYPE_DATA == subcarrier_allocation[*idx].into()
                })
                .map(|(_, sample)| QuadratureSymbol { value: *sample }),
        );
        self.time_domain_symbols.append(&mut new_samples);
    }
}

impl<I: Iterator<Item = OFDMSymbol>> From<I> for OFDMFrameSynchronizer<I> {
    fn from(ofdm_symbol_iter: I) -> Self {
        let mut callback_context_box = Box::new(CallbackContext::default());
        let callback_context_ptr: *mut CallbackContext = callback_context_box.as_mut();
        let callback_context_ptr = callback_context_ptr as *mut core::ffi::c_void;

        let ofdm_framesync = {
            // ofdmframesync calls FFTW_PLANNER, which is not thread safe.
            let _guard = FFTW_PLANNER_LOCK.lock().unwrap(); // drops at end of scope
            unsafe {
                liquid_sys::ofdmframesync_create(
                    NUM_SUBCARRIERS as u32,
                    CP_LEN as u32,
                    TAPER_LEN as u32,
                    std::ptr::null_mut(),
                    Some(ofdm_framesync_callback),
                    callback_context_ptr,
                )
            }
        };
        assert_ne!(std::ptr::null_mut(), ofdm_framesync);

        Self {
            ofdm_symbol_iter,
            ofdm_framesync,
            callback_context: callback_context_box,
        }
    }
}

impl<I: Iterator<Item = OFDMSymbol>> Iterator for OFDMFrameSynchronizer<I> {
    type Item = QuadratureSymbol;

    fn next(&mut self) -> Option<Self::Item> {
        while self.callback_context.time_domain_symbols.is_empty() {
            let ofdm_symbol = self.ofdm_symbol_iter.next()?; // breaks iteration
            let status = unsafe {
                // Pushes samples to self.time_domain_symbols via ofdm_framesync_callback.
                liquid_sys::ofdmframesync_execute(
                    self.ofdm_framesync,
                    ofdm_symbol.time_domain_symbols.as_ptr() as *mut Complex32,
                    FRAME_LEN as u32,
                )
            } as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
        }

        let q_symbol = self
            .callback_context
            .time_domain_symbols
            .pop_front()
            .expect("time_domain_symbols unexepectly empty.");
        Some(q_symbol)
    }
}

impl<I: Iterator<Item = OFDMSymbol>> Drop for OFDMFrameSynchronizer<I> {
    fn drop(&mut self) {
        let status = unsafe { liquid_sys::ofdmframesync_destroy(self.ofdm_framesync) } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ofdm_basic() {
        let mut quadrature_symbols = vec![
            QuadratureSymbol {
                value: Complex32::default()
            };
            1037
        ];
        for (idx, symbol) in quadrature_symbols.iter_mut().enumerate() {
            symbol.value.re = 0.01 * idx as f32;
            symbol.value.im = 0.01 * -(idx as f32);
        }
        let quadrature_symbols_clone: std::collections::VecDeque<_> =
            quadrature_symbols.clone().into();

        let ofdm_frame_generator: OFDMFrameGenerator<_> = quadrature_symbols.into_iter().into();
        let ofdm_symbols: Vec<OFDMSymbol> = ofdm_frame_generator.collect();
        eprintln!("{:?}", ofdm_symbols);

        let ofdm_frame_synchronizer: OFDMFrameSynchronizer<_> = ofdm_symbols.into_iter().into();

        let new_quadrature_symbols: Vec<_> = ofdm_frame_synchronizer.collect();

        // orig may be shorter than new, because of frame padding.
        assert!(quadrature_symbols_clone.len() <= new_quadrature_symbols.len());

        for (orig, new) in quadrature_symbols_clone
            .iter()
            .zip(new_quadrature_symbols.iter())
        {
            eprintln!("orig:{:?} new:{:?}", orig, new);
            assert!((orig.value.re - new.value.re).abs() < 0.0001);
            assert!((orig.value.im - new.value.im).abs() < 0.0001);
        }
    }

    #[test]
    fn test_ofdm_multiple_frames() {
        let mut quadrature_symbols = vec![
            QuadratureSymbol {
                value: Complex32::default()
            };
            1037
        ];
        for (idx, symbol) in quadrature_symbols.iter_mut().enumerate() {
            symbol.value.re = 0.01 * idx as f32;
            symbol.value.im = 0.01 * -(idx as f32);
        }
        let quadrature_symbols_clone: std::collections::VecDeque<_> =
            quadrature_symbols.clone().into();

        let ofdm_frame_generator: OFDMFrameGenerator<_> = quadrature_symbols.into_iter().into();
        let ofdm_symbols: Vec<OFDMSymbol> = ofdm_frame_generator.collect();
        eprintln!("{:?}", ofdm_symbols);

        let ofdm_frame_synchronizer: OFDMFrameSynchronizer<_> = ofdm_symbols.into_iter().into();

        let new_quadrature_symbols: Vec<_> = ofdm_frame_synchronizer.collect();

        // orig may be shorter than new, because of frame padding.
        assert!(quadrature_symbols_clone.len() <= new_quadrature_symbols.len());

        for (orig, new) in quadrature_symbols_clone
            .iter()
            .zip(new_quadrature_symbols.iter())
        {
            eprintln!("orig:{:?} new:{:?}", orig, new);
            assert!((orig.value.re - new.value.re).abs() < 0.0001);
            assert!((orig.value.im - new.value.im).abs() < 0.0001);
        }
    }

    use crate::asset_reader_writer::asset_reader::*;
    use crate::asset_reader_writer::*;
    use crate::channel_coding::slice::ChunkIterIntoExt;
    use crate::compressor::*;
    use crate::metadata_coding::packetizer::*;
    use crate::metadata_coding::*;
    use crate::modulation::metadata::*;
    use crate::modulation::slices::*;
    use crate::source_coding::chunk::*;

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_frame_inverse_to_packets_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");
        let mut y_dct = macro_block.y_components.into_dct();

        // Expensive.. can I defer?
        let y_slices_and_metadata: Box<_> = y_dct
            .chunks_iter((1, 30, 40))
            .into_slice_iter(LENGTH)
            .collect();

        let y_compressed_metadata: CompressedMetadata = y_slices_and_metadata
            .iter()
            .map(|slice| &slice.chunk_metadata)
            .into();
        let y_slices_iter = y_slices_and_metadata.into_iter().map(|slice| slice.slice);

        let packetizer: Packetizer = y_compressed_metadata.into();

        let orig_packets: Vec<_> = packetizer.collect();

        let metadata_modulator: MetadataModulator<_> = orig_packets.clone().into_iter().into();
        let slice_modulator: SliceModulator<'_, _, _> = y_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();

        let new_packets: Vec<_> = metadata_demodulator.take(orig_packets.len()).collect();
        assert_eq!(orig_packets, new_packets);
    }

    #[test]
    fn test_reader_to_frame_inverse_num_slices() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");
        let mut y_dct = macro_block.y_components.into_dct();

        // Expensive.. can I defer?
        let chunks: Vec<_> = y_dct.chunks_iter((1, 30, 40)).collect();

        let first_chunk = chunks.first().expect("No chunks.");
        let chunk_dim = first_chunk.values.dim();
        let chunks_per_gop =
            (LENGTH * asset_height * asset_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

        let y_compressed_metadata: CompressedMetadata =
            chunks.iter().map(|chunk| &chunk.metadata).into();
        let packetizer: Packetizer = y_compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let y_slices_and_metadata: Box<_> =
            chunks.into_iter().into_slice_iter(chunks_per_gop).collect();
        let y_slices_iter = y_slices_and_metadata.into_iter().map(|slice| slice.slice);
        let slice_modulator: SliceModulator<'_, _, _> = y_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
        let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

        let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
        let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
            .by_ref()
            .take(chunks_per_gop)
            .map(|r| r.unwrap())
            .collect();
        assert!(!chunk_metadatas.is_empty());
        assert_eq!(chunk_metadatas.len(), 6912);

        let num_slices = chunks_per_gop.next_power_of_two();
        let mut array3d_view: ndarray::Array3<f32> =
            ndarray::Array3::zeros((num_slices, chunk_dim.1, chunk_dim.2));

        let synchronizer: OFDMFrameSynchronizer<_> =
            metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing

        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; chunks_per_gop),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> =
            SliceDemodulator::new(chunk_dim, metadata_bitmap, synchronizer, &mut array3d_view);

        let slices: Vec<_> = slice_demodulator.collect();
        assert_eq!(slices.len(), 8192);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_frame_inverse_equality() {
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::*;
        use crate::source_coding::power_scaling::*;
        use crate::source_coding::transform_block_3d_dct::*;

        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: _,
            cr_components: _,
            ..
        } = macro_block.clone();

        let mut y_dct = macro_block.y_components.into_dct();
        //         let original_y_dct_components = y_dct.values.clone();

        // Expensive.. can I defer?
        let chunks: Vec<_> = y_dct.chunks_iter((1, 30, 40)).collect();

        let first_chunk = chunks.first().expect("No chunks.");
        let chunk_dim = first_chunk.values.dim();
        let chunks_per_gop =
            (LENGTH * asset_height * asset_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

        let y_compressed_metadata: CompressedMetadata =
            chunks.iter().map(|chunk| &chunk.metadata).into();

        let packetizer: Packetizer = y_compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let power_scaler = PowerScaler::new(chunks.into_iter());
        let y_slices_and_metadata: Box<_> = power_scaler.into_slice_iter(chunks_per_gop).collect();
        let y_slices_iter = y_slices_and_metadata.into_iter().map(|slice| slice.slice);

        let slice_modulator: SliceModulator<'_, _, _> = y_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
        let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

        let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
        let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
            .by_ref()
            .take(chunks_per_gop)
            .map(|r| r.unwrap())
            .collect();
        assert!(!chunk_metadatas.is_empty());
        assert_eq!(chunk_metadatas.len(), 6912);

        let num_slices = chunks_per_gop.next_power_of_two();

        let allocation_gop_length_with_padding =
            ((num_slices * chunk_dim.0 * chunk_dim.1 * chunk_dim.2) as f32
                / (asset_height * asset_width) as f32)
                .ceil() as usize;

        let mut array3d: ndarray::Array3<f32> = ndarray::Array3::zeros((
            allocation_gop_length_with_padding,
            asset_height,
            asset_width,
        ));

        let synchronizer: OFDMFrameSynchronizer<_> =
            metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing

        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; chunks_per_gop),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> =
            SliceDemodulator::new(chunk_dim, metadata_bitmap, synchronizer, &mut array3d);

        let mut slice_and_metadatas = vec![];
        let mut chunk_metadata_iter = chunk_metadatas.into_iter();

        for slice in slice_demodulator.take(num_slices) {
            // there will be more slices than chunk_metadatas
            let chunk_metadata = chunk_metadata_iter.next().unwrap_or_default();
            let slice_and_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
            slice_and_metadatas.push(slice_and_metadata);
        }
        let slice_and_chunk_metadata_iter = slice_and_metadatas.into_iter();

        let chunks_iter: ChunkIter<_, _> =
            slice_and_chunk_metadata_iter.into_chunks_iter(chunks_per_gop);

        let chunks: Box<_> = chunks_iter.take(chunks_per_gop).collect();
        let power_descaler = PowerScaler::inverse(chunks.into_iter());
        let chunk_metadatas_new: Box<_> = power_descaler
            .map(|chunk| chunk.metadata)
            .take(chunks_per_gop)
            .collect();

        let y_dct_components = TransformBlock3DDCT::from_chunks_owned(
            array3d,
            &chunk_metadatas_new,
            LENGTH,
            (asset_width, asset_height),
            chunk_dim,
        );
        //         let y_dct_components =
        //             TransformBlock3DDCT::from_chunks(&chunks, (asset_width, asset_height));

        //         assert_eq!(original_y_dct_components, y_dct_components.values);
        let new_y_components: TransformBlock3D<_> = y_dct_components.into();

        assert_eq!(original_y_components, new_y_components);
    }

    #[test]
    fn test_chunk_metadata_framing_decode() {
        let mut chunk_metadata = vec![ChunkMetadata::default(); 15];
        for (idx, cm) in chunk_metadata.iter_mut().enumerate() {
            cm.energy = idx as f32;
            cm.mean = -(idx as f32);
        }
        let compressed_metadata: CompressedMetadata = chunk_metadata.iter().into();
        let packetizer: Packetizer = compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();
        let ofdm_generator: OFDMFrameGenerator<_> = metadata_modulator.flatten().into();

        let ofdm_synchronizer: OFDMFrameSynchronizer<_> = ofdm_generator.into();
        let metadata_demodulator: MetadataDemodulator<_> = ofdm_synchronizer.into();
        let depacketizer: Depacketizer<_, ()> = metadata_demodulator.into();
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(depacketizer, chunk_metadata.len());
        let new_chunk_metatata: Vec<ChunkMetadata> =
            decompressor.map(|result| result.unwrap()).collect();

        assert_eq!(chunk_metadata, new_chunk_metatata);
    }

    #[test]
    #[cfg(false)] // too slow to run regularly
    fn test_reader_to_frame_to_writer() {
        use crate::asset_reader_writer::asset_writer::*;
        use crate::asset_reader_writer::pixel_buffer::*;
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::*;
        use crate::compressor::*;
        use crate::source_coding::chunk::*;
        use crate::source_coding::power_scaling::*;
        use crate::source_coding::transform_block_3d_dct::*;

        let input_path = "sample-media/bigbuck-7.5s.mov";
        let output_path = "/tmp/bigbuck-7.5s.mp4";
        //         let input_path = "sample-media/bipbop-1920x1080-5s.mp4";
        //         let output_path = "/tmp/bipbop-1920x1080-5s.mp4";

        let _ = std::fs::remove_file(output_path);
        let mut reader = AssetReader::new(input_path.into());
        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();
        let asset_resolution = (asset_width, asset_height);
        const GOP_LENGTH: usize = 90;

        let writer_settings = AssetWritterSettings {
            path: std::path::PathBuf::from(output_path),
            codec: Codec::H264,
            resolution: (asset_width as i32, asset_height as i32),
            frame_rate: reader.frame_rate().expect("Failed to get frame rate"),
        };
        let mut writer = AssetWriter::load_new(writer_settings).expect("Failed to load writer");
        writer.start_writing().expect("Failed to start writing");

        let macro_block_3d_iterator: MacroBlock3DIterator<_> = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator(GOP_LENGTH);

        fn framer<PixelType: HasPixelComponentType>(
            dct_components: &mut TransformBlock3DDCT<PixelType>,
        ) -> (impl Iterator<Item = OFDMSymbol>, (usize, usize, usize)) {
            let chunks: Box<_> = dct_components.chunks_iter((1, 30, 40)).collect();
            let first_chunk = chunks.first().expect("No chunks.");
            let chunk_dim = first_chunk.values.dim();

            let chunk_metadatas: Box<_> = chunks.iter().map(|chunk| chunk.metadata).collect();
            let metadata_bitmap = MetadataBitmap::new(&chunks, 0.0625);
            let compressed_metadata =
                CompressedMetadata::new(&metadata_bitmap, chunk_metadatas.iter());
            let num_included_chunks = metadata_bitmap.values.count_ones();
            let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);
            let power_scaler = PowerScaler::new(compressor);

            let packetizer: Packetizer = compressed_metadata.into();
            let metadata_modulator: MetadataModulator<_> = packetizer.into();

            let slices_and_metadata: Box<_> = power_scaler
                .into_iter()
                .into_slice_iter(num_included_chunks)
                .collect();
            let slices_iter = slices_and_metadata.into_iter().map(|slice| slice.slice);

            let slice_modulator: SliceModulator<'_, _, _> = slices_iter.into();
            let framer: OFDMFrameGenerator<_> =
                metadata_modulator.flatten().chain(slice_modulator).into();

            (framer, chunk_dim)
        }

        fn into_transform_block_3d_dct<
            PixelType: HasPixelComponentType,
            O: Iterator<Item = OFDMSymbol>,
        >(
            ofdm_symbol_iter: &mut O,
            asset_resolution: (usize, usize),
            chunk_dim: (usize, usize, usize),
        ) -> TransformBlock3DDCT<PixelType> {
            let (frame_width, frame_height) = (
                asset_resolution.0 / PixelType::TYPE.interleave_step(),
                asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
            );
            let chunks_per_gop = (GOP_LENGTH * frame_height * frame_width)
                / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

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
                GOP_LENGTH,
                asset_resolution,
                chunk_dim,
            )
        }

        fn slices_allocation<PixelType: HasPixelComponentType>(
            asset_resolution: (usize, usize),
            chunk_dim: (usize, usize, usize),
            num_padding_slices: usize,
        ) -> ndarray::Array3<f32> {
            let (frame_width, frame_height) = (
                asset_resolution.0 / PixelType::TYPE.interleave_step(),
                asset_resolution.1 / PixelType::TYPE.vertical_subsampling(),
            );
            let chunks_per_gop = (GOP_LENGTH * frame_height * frame_width)
                / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

            let allocation_gop_length_with_padding =
                (((chunks_per_gop + num_padding_slices) * chunk_dim.0 * chunk_dim.1 * chunk_dim.2)
                    as f64
                    / (frame_width * frame_height) as f64)
                    .ceil() as usize;

            ndarray::Array3::zeros((
                allocation_gop_length_with_padding,
                frame_height,
                frame_width,
            ))
        }

        for macro_block in macro_block_3d_iterator {
            // encoder
            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
                ..
            } = macro_block;

            let mut y_dct_in: TransformBlock3DDCT<_> = y_components.into();
            let (y_framer, y_chunk_dim) = framer(&mut y_dct_in);

            let mut cb_dct_in: TransformBlock3DDCT<_> = cb_components.into();
            let (cb_framer, cb_chunk_dim) = framer(&mut cb_dct_in);

            let mut cr_dct_in: TransformBlock3DDCT<_> = cr_components.into();
            let (cr_framer, cr_chunk_dim) = framer(&mut cr_dct_in);

            let mut encoder = y_framer.chain(cb_framer).chain(cr_framer);

            /*
            use rand::Rng;
            let mut encoder_plus_noise = encoder.map(|mut ofdmsymbol| {
                let mut rng = rand::rng();
                for iq in ofdmsymbol.time_domain_symbols.iter_mut() {
                    let distortion_power = 0.57;
                    let i_distortion = rng.random_range(-distortion_power..distortion_power);
                    let q_distortion = rng.random_range(-distortion_power..distortion_power);
                    iq.re += i_distortion;
                    iq.im += q_distortion;
                }
                ofdmsymbol
            });
            */

            //decoder
            let y_dct_out =
                into_transform_block_3d_dct(&mut encoder, asset_resolution, y_chunk_dim);
            let cb_dct_out =
                into_transform_block_3d_dct(&mut encoder, asset_resolution, cb_chunk_dim);
            let cr_dct_out =
                into_transform_block_3d_dct(&mut encoder, asset_resolution, cr_chunk_dim);

            let new_macro_block_3d = MacroBlock3D {
                y_components: y_dct_out.into(),
                cb_components: cb_dct_out.into(),
                cr_components: cr_dct_out.into(),
                gop_len: GOP_LENGTH,
            };

            let pixel_buffer_iterator: PixelBufferIterator<_> = new_macro_block_3d.into();

            for pixel_buffer in pixel_buffer_iterator {
                writer
                    .append_pixel_buffer(pixel_buffer)
                    .expect("Failed to append pixel buffer");
                writer
                    .wait_for_writer_to_be_ready()
                    .expect("Failed to become ready after writing some pixels.");
            }
        }
        writer.finish_writing().expect("Failed to finish writing.");
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_frame_inverse_mean_squared_error() {
        use crate::asset_reader_writer::pixel_buffer::*;
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::*;
        use crate::source_coding::power_scaling::*;
        use crate::source_coding::transform_block_3d_dct::*;
        use ndarray_stats::DeviationExt;

        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<_> =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: _,
            cr_components: _,
            ..
        } = macro_block.clone();

        let mut y_dct = macro_block.y_components.into_dct();
        //         let original_y_dct_components = y_dct.values.clone();

        // Expensive.. can I defer?
        let chunks: Vec<_> = y_dct.chunks_iter((1, 30, 40)).collect();

        let first_chunk = chunks.first().expect("No chunks.");
        let chunk_dim = first_chunk.values.dim();
        let chunks_per_gop =
            (LENGTH * asset_height * asset_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

        let y_compressed_metadata: CompressedMetadata =
            chunks.iter().map(|chunk| &chunk.metadata).into();

        let packetizer: Packetizer = y_compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let power_scaler = PowerScaler::new(chunks.into_iter());

        let y_slices_and_metadata: Box<_> = power_scaler.into_slice_iter(chunks_per_gop).collect();
        let y_slices_iter = y_slices_and_metadata.into_iter().map(|slice| slice.slice);

        let slice_modulator: SliceModulator<'_, _, _> = y_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
        let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

        let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
        let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
            .by_ref()
            .take(chunks_per_gop)
            .map(|r| r.unwrap())
            .collect();
        assert!(!chunk_metadatas.is_empty());
        assert_eq!(chunk_metadatas.len(), 6912);

        let num_slices = chunks_per_gop.next_power_of_two();

        let allocation_gop_length_with_padding =
            ((num_slices * chunk_dim.0 * chunk_dim.1 * chunk_dim.2) as f32
                / (asset_height * asset_width) as f32)
                .ceil() as usize;

        let mut array3d: ndarray::Array3<f32> = ndarray::Array3::zeros((
            allocation_gop_length_with_padding,
            asset_height,
            asset_width,
        ));

        let synchronizer: OFDMFrameSynchronizer<_> =
            metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing
        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; chunks_per_gop),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> =
            SliceDemodulator::new(chunk_dim, metadata_bitmap, synchronizer, &mut array3d);

        let mut slice_and_metadatas = vec![];
        let mut chunk_metadata_iter = chunk_metadatas.into_iter();

        for slice in slice_demodulator.take(num_slices) {
            // there will be more slices than chunk_metadatas
            let chunk_metadata = chunk_metadata_iter.next().unwrap_or_default();
            let slice_and_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
            slice_and_metadatas.push(slice_and_metadata);
        }
        let slice_and_chunk_metadata_iter = slice_and_metadatas.into_iter();

        let chunks_iter: ChunkIter<_, _> =
            slice_and_chunk_metadata_iter.into_chunks_iter(chunks_per_gop);

        let chunks: Box<_> = chunks_iter.take(chunks_per_gop).collect();
        let power_descaler = PowerScaler::inverse(chunks.into_iter());

        let chunk_metadatas_new: Box<_> = power_descaler
            .map(|chunk| chunk.metadata)
            .take(chunks_per_gop)
            .collect();

        let y_dct_components = TransformBlock3DDCT::from_chunks_owned(
            array3d,
            &chunk_metadatas_new,
            LENGTH,
            (asset_width, asset_height),
            chunk_dim,
        );
        //         let y_dct_components =
        //             TransformBlock3DDCT::from_chunks(&chunks, (asset_width, asset_height));

        //         assert_eq!(original_y_dct_components, y_dct_components.values);
        let new_y_components: TransformBlock3D<YPixelComponentType> = y_dct_components.into();

        let mean_sq_error = original_y_components
            .values()
            .mean_sq_err(new_y_components.values())
            .unwrap();
        assert_eq!(mean_sq_error, 0f64);
    }

    #[test]
    fn test_reader_to_frame_inverse_compression_mean_squared_error_y() {
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::*;
        use crate::compressor::*;
        use crate::source_coding::power_scaling::*;
        use crate::source_coding::transform_block_3d_dct::*;
        use ndarray_stats::DeviationExt;

        //         let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let path = "sample-media/bigbuck-7.5s.mov".into();
        let mut reader = AssetReader::new(path);

        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();

        const LENGTH: usize = 2;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: _,
            cr_components: _,
            ..
        } = macro_block.clone();

        let mut y_dct = macro_block.y_components.into_dct();

        let chunks: Box<_> = y_dct.chunks_iter((1, 30, 40)).collect();

        let first_chunk = chunks.first().expect("No chunks.");
        let chunk_dim = first_chunk.values.dim();
        let chunks_per_gop =
            (LENGTH * asset_height * asset_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

        let chunk_metadatas: Box<_> = chunks.iter().map(|chunk| chunk.metadata).collect();
        let compression_ratio = 0.125;
        let metadata_bitmap = MetadataBitmap::new(&chunks, compression_ratio);
        let y_compressed_metadata =
            CompressedMetadata::new(&metadata_bitmap, chunk_metadatas.iter());
        let num_included_chunks = metadata_bitmap.values.count_ones();
        let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);
        let power_scaler = PowerScaler::new(compressor);

        let packetizer: Packetizer = y_compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let y_slices_and_metadata: Box<_> =
            power_scaler.into_slice_iter(num_included_chunks).collect();
        let y_slices_iter = y_slices_and_metadata.into_iter().map(|slice| slice.slice);

        let slice_modulator: SliceModulator<'_, _, _> = y_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
        let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

        let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
        let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
            .by_ref()
            .take(chunks_per_gop)
            .map(|r| r.unwrap())
            .collect();
        assert!(!chunk_metadatas.is_empty());
        assert_eq!(chunk_metadatas.len(), 1728 * LENGTH);

        let metadata_bitmap = metadata_decompressor
            .take_metadata_bitmap()
            .expect("Failed to decode metadata bitmap");
        let included_chunk_metadatas: Box<_> = metadata_bitmap
            .values
            .iter_ones()
            .map(|idx| chunk_metadatas[idx])
            .collect();

        let num_all_chunks = chunk_metadatas.len();
        let num_included_chunks = metadata_bitmap.values.count_ones();
        let num_included_slices = num_included_chunks.next_power_of_two();
        let num_padding_slices = num_included_slices - num_included_chunks;

        let allocation_gop_length_with_padding =
            (((num_all_chunks + num_padding_slices) * chunk_dim.0 * chunk_dim.1 * chunk_dim.2)
                as f64
                / (asset_height * asset_width) as f64)
                .ceil() as usize;

        let mut array3d: ndarray::Array3<f32> = ndarray::Array3::zeros((
            allocation_gop_length_with_padding,
            asset_height,
            asset_width,
        ));

        let synchronizer: OFDMFrameSynchronizer<_> =
            metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> =
            SliceDemodulator::new(chunk_dim, metadata_bitmap, synchronizer, &mut array3d);

        let all_slices: Box<_> = slice_demodulator.take(num_included_slices).collect();

        let mut included_chunk_metadatas_iter = included_chunk_metadatas.into_iter();
        let mut slice_and_metadatas = vec![];
        for slice in all_slices.into_iter().take(num_included_slices) {
            let chunk_metadata = included_chunk_metadatas_iter.next().unwrap_or_default();
            let slice_and_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
            slice_and_metadatas.push(slice_and_metadata);
        }
        assert_eq!(num_included_slices, slice_and_metadatas.len());
        let slice_and_chunk_metadata_iter = slice_and_metadatas.into_iter();

        let chunks_iter = slice_and_chunk_metadata_iter
            .into_chunks_iter(num_included_chunks)
            .take(num_included_chunks);
        let power_descaler = PowerScaler::inverse(chunks_iter);
        let _chunks: Box<_> = power_descaler.collect(); // discard.. runs fwht

        let y_dct_components = TransformBlock3DDCT::from_chunks_owned(
            array3d,
            &chunk_metadatas,
            LENGTH,
            (asset_width, asset_height),
            chunk_dim,
        );

        let new_y_components: TransformBlock3D<YPixelComponentType> = y_dct_components.into();

        let mean_sq_error = original_y_components
            .values()
            .mean_sq_err(new_y_components.values())
            .unwrap();
        assert!(mean_sq_error < 3.0);
    }

    #[test]
    fn test_reader_to_frame_inverse_compression_mean_squared_error_cb() {
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::*;
        use crate::compressor::*;
        use crate::source_coding::power_scaling::*;
        use crate::source_coding::transform_block_3d_dct::*;
        use ndarray_stats::DeviationExt;

        let path = "sample-media/bigbuck-7.5s.mov".into();
        let mut reader = AssetReader::new(path);

        let (asset_width, asset_height) = reader.resolution().expect("failed to get resolution");
        let asset_width: usize = asset_width.try_into().unwrap();
        let asset_height: usize = asset_height.try_into().unwrap();
        let pixel_type = PixelComponentType::Cb;
        let (frame_width, frame_height) = (
            asset_width / pixel_type.interleave_step(),
            asset_height / pixel_type.vertical_subsampling(),
        );

        const LENGTH: usize = 2;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: _,
            cb_components: original_cb_components,
            cr_components: _,
            ..
        } = macro_block.clone();

        let mut cb_dct = macro_block.cb_components.into_dct();

        let chunks: Box<_> = cb_dct.chunks_iter((1, 30, 40)).collect();

        let first_chunk = chunks.first().expect("No chunks.");
        let chunk_dim = first_chunk.values.dim();
        let chunks_per_gop =
            (LENGTH * frame_height * frame_width) / (chunk_dim.0 * chunk_dim.1 * chunk_dim.2);

        let chunk_metadatas: Box<_> = chunks.iter().map(|chunk| chunk.metadata).collect();
        //         let compression_ratio = 0.234375; // has no error for bipbop
        let compression_ratio = 0.125;
        let metadata_bitmap = MetadataBitmap::new(&chunks, compression_ratio);
        let y_compressed_metadata =
            CompressedMetadata::new(&metadata_bitmap, chunk_metadatas.iter());
        let num_included_chunks = metadata_bitmap.values.count_ones();
        let compressor = Compressor::new(chunks.into_iter(), metadata_bitmap);

        let power_scaler = PowerScaler::new(compressor);

        assert_eq!(
            (chunks_per_gop as f64 * compression_ratio).floor() as usize,
            num_included_chunks
        );

        let packetizer: Packetizer = y_compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let cb_slices_and_metadata: Box<_> =
            power_scaler.into_slice_iter(num_included_chunks).collect();
        assert_eq!(
            ((chunks_per_gop as f64 * compression_ratio).floor() as usize).next_power_of_two(),
            cb_slices_and_metadata.len()
        );
        let cb_slices_iter = cb_slices_and_metadata.into_iter().map(|slice| slice.slice);

        let slice_modulator: SliceModulator<'_, _, _> = cb_slices_iter.into();
        let framer: OFDMFrameGenerator<_> =
            metadata_modulator.flatten().chain(slice_modulator).into();

        let synchronizer: OFDMFrameSynchronizer<_> = framer.into();

        let metadata_demodulator: MetadataDemodulator<_> = synchronizer.into();
        let depacketizer: Depacketizer<_, _> = metadata_demodulator.into();

        let mut metadata_decompressor = MetadataDecompressor::new(depacketizer, chunks_per_gop);
        let chunk_metadatas: Vec<ChunkMetadata> = metadata_decompressor
            .by_ref()
            .take(chunks_per_gop)
            .map(|r| r.unwrap())
            .collect();
        assert!(!chunk_metadatas.is_empty());
        assert_eq!(chunk_metadatas.len(), 432 * LENGTH);

        let metadata_bitmap = metadata_decompressor
            .take_metadata_bitmap()
            .expect("Failed to decode metadata bitmap");
        let included_chunk_metadatas: Box<_> = metadata_bitmap
            .values
            .iter_ones()
            .map(|idx| chunk_metadatas[idx])
            .collect();

        let num_all_chunks = chunk_metadatas.len();
        let num_included_chunks = metadata_bitmap.values.count_ones();
        let num_included_slices = num_included_chunks.next_power_of_two();

        let allocation_gop_length_with_padding = (((num_included_slices
            + (num_all_chunks as f64 * 1.0 - compression_ratio).ceil() as usize)
            * chunk_dim.0
            * chunk_dim.1
            * chunk_dim.2) as f64
            / (frame_height * frame_width) as f64)
            .ceil() as usize;

        let mut array3d: ndarray::Array3<f32> = ndarray::Array3::zeros((
            allocation_gop_length_with_padding,
            frame_height,
            frame_width,
        ));

        let synchronizer: OFDMFrameSynchronizer<_> =
            metadata_decompressor.into_inner_quadrature_symbol_iter(); // return quad_iter for slicing
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> =
            SliceDemodulator::new(chunk_dim, metadata_bitmap, synchronizer, &mut array3d);

        let mut included_chunk_metadatas_iter = included_chunk_metadatas.into_iter();
        let mut slice_and_metadatas = vec![];
        for slice in slice_demodulator.take(num_included_slices) {
            let chunk_metadata = included_chunk_metadatas_iter.next().unwrap_or_default();
            let slice_and_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
            slice_and_metadatas.push(slice_and_metadata);
        }
        assert_eq!(num_included_slices, slice_and_metadatas.len());
        let slice_and_chunk_metadata_iter = slice_and_metadatas.into_iter();

        let chunks_iter = slice_and_chunk_metadata_iter
            .into_chunks_iter(num_included_chunks)
            .take(num_included_chunks);
        let power_descaler = PowerScaler::inverse(chunks_iter);
        let _chunks: Box<_> = power_descaler.collect(); // discard.. runs fwht

        let cb_dct_components = TransformBlock3DDCT::from_chunks_owned(
            array3d,
            &chunk_metadatas,
            LENGTH,
            (asset_width, asset_height),
            chunk_dim,
        );

        let new_cb_components: TransformBlock3D<CbPixelComponentType> = cb_dct_components.into();
        //         assert_eq!(original_cb_components, new_cb_components);

        let mean_sq_error = original_cb_components
            .values()
            .mean_sq_err(new_cb_components.values())
            .unwrap();
        assert!(mean_sq_error < 1.0);
    }
}
