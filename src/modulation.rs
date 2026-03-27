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

use crate::compressor::*;
use crate::metadata_coding::packetizer::*;
use liquid_sys;
use num_complex::Complex32;

// TODO: Interleave metadata and slice symbols.

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct QuadratureSymbol {
    pub value: Complex32,
}
impl From<Complex32> for QuadratureSymbol {
    fn from(value: Complex32) -> Self {
        Self { value }
    }
}
trait FromByte {
    fn from_byte(byte: u8, modem: *mut liquid_sys::modemcf_s) -> Self;
}
type BPSKModulatedByte = [QuadratureSymbol; 8];
impl FromByte for BPSKModulatedByte {
    fn from_byte(byte: u8, modem: liquid_sys::modemcf) -> Self {
        let mut quadrature_symbols = BPSKModulatedByte::default();

        for (bitpos, q_symbol) in quadrature_symbols.iter_mut().enumerate().rev() {
            // MSB-first
            let bitval = 0 != (byte & (1 << bitpos));
            let symbol_ptr: *mut Complex32 = &mut q_symbol.value; // Complex32 is bincompat with C float complex
            let status =
                unsafe { liquid_sys::modemcf_modulate(modem, bitval.into(), symbol_ptr) } as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
        }

        quadrature_symbols
    }
}

trait ScaleBy {
    fn scale_by(&mut self, scale: f32);
}

impl ScaleBy for BPSKModulatedByte {
    fn scale_by(&mut self, scale: f32) {
        for q_symbol in self.iter_mut() {
            q_symbol.value *= scale;
        }
    }
}

pub mod metadata {
    use super::*;

    const MODULATION_SCHEME: u32 = liquid_sys::modulation_scheme_LIQUID_MODEM_BPSK;

    // TODO: consider replacing with qpacketmodem
    pub struct MetadataModulator<I: Iterator<Item = EncodedPacket>> {
        modemcf_wrapper: ModemCFWrapper,
        inner: I,
        working_packet: Option<EncodedPacket>,
        working_packet_pos: usize,
    }

    impl<I: Iterator<Item = EncodedPacket>> From<I> for MetadataModulator<I> {
        fn from(encoded_packet_iter: I) -> Self {
            let modemcf = unsafe { liquid_sys::modemcf_create(MODULATION_SCHEME) };
            assert_ne!(std::ptr::null_mut(), modemcf);
            let modemcf_wrapper = ModemCFWrapper { ptr: modemcf };

            MetadataModulator {
                modemcf_wrapper,
                inner: encoded_packet_iter,
                working_packet: None,
                working_packet_pos: 0,
            }
        }
    }

    impl<I: Iterator<Item = EncodedPacket>> Iterator for MetadataModulator<I> {
        // TODO: size hint

        type Item = BPSKModulatedByte;
        fn next(&mut self) -> Option<Self::Item> {
            if self.working_packet.is_none() {
                self.working_packet = self.inner.next();
            }
            let working_packet = self.working_packet.as_ref()?;
            let byte = working_packet.encoded_data[self.working_packet_pos];
            let mut quadrature_symbols =
                BPSKModulatedByte::from_byte(byte, self.modemcf_wrapper.ptr);

            self.working_packet_pos += 1;
            if working_packet.encoded_data.len() == self.working_packet_pos {
                self.working_packet = None;
                self.working_packet_pos = 0;
            }
            quadrature_symbols.scale_by(1.0); // TODO: factor out
            Some(quadrature_symbols)
        }
    }

    pub struct MetadataDemodulator<I: Iterator<Item = QuadratureSymbol>> {
        modemcf_wrapper: ModemCFWrapper,
        inner: I,
    }

    impl<I: Iterator<Item = QuadratureSymbol>> MetadataDemodulator<I> {
        pub fn into_inner(self) -> I {
            self.inner
        }
    }

    impl<I: Iterator<Item = QuadratureSymbol>> From<I> for MetadataDemodulator<I> {
        fn from(quadrature_symbol_iter: I) -> Self {
            let modemcf = unsafe { liquid_sys::modemcf_create(MODULATION_SCHEME) };
            assert_ne!(std::ptr::null_mut(), modemcf);
            let modemcf_wrapper = ModemCFWrapper { ptr: modemcf };

            MetadataDemodulator {
                modemcf_wrapper,
                inner: quadrature_symbol_iter,
            }
        }
    }
    impl<I: Iterator<Item = QuadratureSymbol>> Iterator for MetadataDemodulator<I> {
        // TODO: size hint

        type Item = EncodedPacket;
        fn next(&mut self) -> Option<Self::Item> {
            let mut packet_buf = [0u8; ENCODED_MESSAGE_LENGTH];
            for byte in &mut packet_buf {
                for bitpos in 0..8 {
                    // assume 0 bits if inner runs out, let CRC reject
                    if let Some(q_symbol) = self.inner.next() {
                        let mut bitval = 0u32;
                        let status = unsafe {
                            liquid_sys::modemcf_demodulate(
                                self.modemcf_wrapper.ptr,
                                q_symbol.value,
                                &mut bitval,
                            )
                        };
                        assert_eq!(status as u32, liquid_sys::liquid_error_code_LIQUID_OK);

                        let bitval = bitval as u8;
                        *byte |= bitval << bitpos;
                    } else if 0 == bitpos {
                        return None;
                    } else {
                        return Some(packet_buf.into());
                    }
                }
            }
            Some(packet_buf.into())
        }
    }

    impl<I: Iterator<Item = QuadratureSymbol>> IntoInnerQuadratureSymbolIter<I>
        for MetadataDemodulator<I>
    {
        fn into_inner_quadrature_symbol_iter(self) -> I {
            self.inner
        }
    }

    // Adds drop support to modemcf, necessary to work around rustc E0509.
    struct ModemCFWrapper {
        ptr: liquid_sys::modemcf,
    }
    impl Drop for ModemCFWrapper {
        fn drop(&mut self) {
            let status = unsafe { liquid_sys::modemcf_destroy(self.ptr) } as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
        }
    }
}

pub trait IntoInnerQuadratureSymbolIter<I: Iterator<Item = QuadratureSymbol>> {
    fn into_inner_quadrature_symbol_iter(self) -> I;
}

pub mod slices {
    use super::*;
    use crate::asset_reader_writer::HasPixelComponentType;
    use crate::channel_coding::fwht::ValuesProvider;
    use crate::channel_coding::slice::*;
    use ndarray;

    pub struct SliceModulator<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = Slice<'a, PixelType>>,
    > {
        slice_iter: I,
        working_slice: Option<Slice<'a, PixelType>>,
        working_idx: usize,
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Slice<'a, PixelType>>> From<I>
        for SliceModulator<'a, PixelType, I>
    {
        fn from(slice_iter: I) -> Self {
            Self {
                slice_iter,
                working_slice: None,
                working_idx: 0,
            }
        }
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Slice<'a, PixelType>>>
        SliceModulator<'a, PixelType, I>
    {
        fn next_real(&mut self) -> Option<f32> {
            if self.working_slice.is_none() {
                self.working_slice = self.slice_iter.next();
            }
            let working_slice = self.working_slice.as_ref()?; // ends iteration

            let values_len = working_slice.values_len();

            let real_value = working_slice.value_at(self.working_idx);
            self.working_idx += 1;
            self.working_idx %= values_len; // working_idx is indexed into a single slice.
            if 0 == self.working_idx {
                self.working_slice = None;
            }

            Some(real_value)
        }
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Slice<'a, PixelType>>> Iterator
        for SliceModulator<'a, PixelType, I>
    {
        type Item = QuadratureSymbol;

        fn next(&mut self) -> Option<Self::Item> {
            // TODO: use size hint for more thorough interleaving.
            let i_val = self.next_real()?;
            let q_val = self.next_real().unwrap_or_default(); // don't drop i_val

            Some(Complex32::new(i_val, q_val).into())
        }
    }

    pub struct SliceDemodulator<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = QuadratureSymbol>,
    > {
        quadrature_symbol_iter: I,
        exact_array3_chunks_iter: ndarray::iter::ExactChunksIterMut<'a, f32, ndarray::Ix3>,
        metadata_bitmap_iter: bitvec::boxed::IntoIter<u8>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = QuadratureSymbol>>
        SliceDemodulator<'a, PixelType, I>
    {
        pub fn new(
            slice_dimensions: (usize, usize, usize),
            metadata_bitmap: MetadataBitmap,
            quadrature_symbol_iter: I,
            array3: &'a mut ndarray::Array3<f32>,
        ) -> Self {
            Self {
                quadrature_symbol_iter,
                metadata_bitmap_iter: metadata_bitmap.values.into_iter(),
                exact_array3_chunks_iter: array3.exact_chunks_mut(slice_dimensions).into_iter(),
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = QuadratureSymbol>> Iterator
        for SliceDemodulator<'a, PixelType, I>
    {
        type Item = Slice<'a, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            let slices_to_skip = self
                .metadata_bitmap_iter
                .by_ref()
                .take_while(|bitval| !bitval)
                .count(); // returns 0 when metadata_bitmap_iter is exhausted

            // there will be padding slices beyond metadata_bitmap that are always included
            let slice_values = self.exact_array3_chunks_iter.by_ref().nth(slices_to_skip)?;

            let mut slice: Slice<'a, PixelType> = Slice::from_view(slice_values);

            let mut iq_iter = self
                .quadrature_symbol_iter
                .by_ref()
                .flat_map(|symbol| [symbol.value.re, symbol.value.im]);

            for dst in &mut slice.values_mut() {
                *dst = iq_iter // TODO: use mapv_inplace
                    .next()
                    .expect("Not enough values to complete slices.");
            }

            Some(slice)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_reader_writer::*;
    use crate::channel_coding::slice::*;
    use crate::modulation::slices::*;
    use metadata::*;

    #[test]
    fn test_modem_basic() {
        let mut encoded_packets: Vec<EncodedPacket> =
            vec![[0xbau8; ENCODED_MESSAGE_LENGTH].into(); 333];

        for (idx, packet) in &mut encoded_packets.iter_mut().enumerate() {
            packet.encoded_data[idx] = 0x11u8;
        }
        let cloned_encoded_packets = encoded_packets.clone();

        let modulator = MetadataModulator::from(encoded_packets.into_iter());
        let demodulator = MetadataDemodulator::from(modulator.flatten());

        let mut num_new_packets = 0;
        for (original_packet, new_packet) in cloned_encoded_packets.iter().zip(demodulator) {
            num_new_packets += 1;
            assert_eq!(original_packet.encoded_data, new_packet.encoded_data);
        }
        assert_eq!(cloned_encoded_packets.len(), num_new_packets);
    }

    #[test]
    fn test_modulate_one_slice() {
        let dim = (1, 30, 44);
        let mut array3_orig = ndarray::Array3::<f32>::zeros(dim);

        let mut val = 0f32;
        for dst in array3_orig.iter_mut() {
            *dst = val;
            val += 1f32;
        }

        let array3_orig_clone = array3_orig.clone();

        let slice_orig: Slice<'_, YPixelComponentType> = Slice::from_owned(array3_orig);
        let slices_orig = [slice_orig];

        let slice_modulator: SliceModulator<'_, _, _> = slices_orig.into_iter().into();
        let quadrature_symbols: Vec<QuadratureSymbol> = slice_modulator.collect();

        let mut array3_new = ndarray::Array3::<f32>::zeros(dim);

        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; 1),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> = SliceDemodulator::new(
            dim,
            metadata_bitmap,
            quadrature_symbols.into_iter(),
            &mut array3_new,
        );

        let slices_new: Vec<_> = slice_demodulator.collect();
        assert_eq!(slices_new.len(), 1);
        let slice_new = slices_new.first().expect("Failed to grab slice.");

        assert_eq!(array3_orig_clone, slice_new.values());
    }

    #[test]
    fn test_modulate_multiple_slices_1() {
        let dim = (1, 30, 44);
        let mut array3_orig = ndarray::Array3::<f32>::zeros((5, dim.1, dim.2)); // 5 slices

        let mut val = 0f32;
        for dst in array3_orig.iter_mut() {
            *dst = val;
            val += 1f32;
        }
        let array3_orig_clone = array3_orig.clone();

        let slices_orig: Vec<Slice<'_, YPixelComponentType>> = array3_orig
            .exact_chunks_mut(dim)
            .into_iter()
            .map(|view| Slice::from_view(view))
            .collect();
        let num_slices = slices_orig.len();

        let slice_modulator: SliceModulator<'_, _, _> = slices_orig.into_iter().into();
        let quadrature_symbols: Vec<QuadratureSymbol> = slice_modulator.collect();

        let mut array3_new = ndarray::Array3::<f32>::zeros((5, dim.1, dim.2));

        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; num_slices),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> = SliceDemodulator::new(
            dim,
            metadata_bitmap,
            quadrature_symbols.into_iter(),
            &mut array3_new,
        );

        let slices_new: Vec<_> = slice_demodulator.collect();
        assert_eq!(slices_new.len(), 5);

        for (slice_new, view_orig) in slices_new
            .iter()
            .zip(array3_orig_clone.exact_chunks(dim).into_iter())
        {
            assert_eq!(view_orig, slice_new.values());
        }
    }

    #[test]
    fn test_modulate_multiple_slices_2() {
        let dim = (1, 30, 44);
        // 500 slices
        let gop_dim = (dim.0 * 5, dim.1 * 10, dim.2 * 10);
        let mut array3_orig = ndarray::Array3::<f32>::zeros(gop_dim);

        let mut val = 0f32;
        for dst in array3_orig.iter_mut() {
            *dst = val;
            val += 1f32;
        }
        let array3_orig_clone = array3_orig.clone();

        let slices_orig: Vec<Slice<'_, YPixelComponentType>> = array3_orig
            .exact_chunks_mut(dim)
            .into_iter()
            .map(|view| Slice::from_view(view))
            .collect();
        let num_slices = slices_orig.len();

        let slice_modulator: SliceModulator<'_, _, _> = slices_orig.into_iter().into();
        let quadrature_symbols: Vec<QuadratureSymbol> = slice_modulator.collect();

        let mut array3_new = ndarray::Array3::<f32>::zeros(gop_dim);

        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; num_slices),
        };
        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> = SliceDemodulator::new(
            dim,
            metadata_bitmap,
            quadrature_symbols.into_iter(),
            &mut array3_new,
        );

        let slices_new: Vec<_> = slice_demodulator.collect();
        assert_eq!(slices_new.len(), 500);

        for (slice_new, view_orig) in slices_new
            .iter()
            .zip(array3_orig_clone.exact_chunks(dim).into_iter())
        {
            assert_eq!(view_orig, slice_new.values());
        }
    }

    #[test]
    fn test_skip_slices_1() {
        let dim = (1, 10, 10);
        let mut array3_orig = ndarray::Array3::<f32>::zeros((5, dim.1, dim.2)); // 5 slices

        let mut val = 0f32;
        for dst in array3_orig.iter_mut() {
            *dst = val;
            val += 1f32;
        }
        let array3_orig_clone = array3_orig.clone();

        let slices_orig: Vec<Slice<'_, YPixelComponentType>> = array3_orig
            .exact_chunks_mut(dim)
            .into_iter()
            .map(|view| Slice::from_view(view))
            .collect();
        let num_slices = slices_orig.len();

        let slice_modulator: SliceModulator<'_, _, _> = slices_orig.into_iter().into();
        let quadrature_symbols: Vec<QuadratureSymbol> = slice_modulator.collect();

        let mut array3_new = ndarray::Array3::<f32>::zeros((5, dim.1, dim.2));

        let mut metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox!(u8, bitvec::order::Lsb0; 1; num_slices),
        };

        metadata_bitmap.values.set(3, false);
        metadata_bitmap.values.set(2, false);

        let slice_demodulator: SliceDemodulator<'_, YPixelComponentType, _> = SliceDemodulator::new(
            dim,
            metadata_bitmap,
            quadrature_symbols.into_iter(),
            &mut array3_new,
        );

        let slices_new: Vec<_> = slice_demodulator.collect();
        assert_eq!(slices_new.len(), 3);
        drop(slices_new);

        let mut chunks_old_iter = array3_orig_clone.exact_chunks(dim).into_iter();
        for (chunk_idx, chunk_new) in array3_new.exact_chunks(dim).into_iter().enumerate() {
            if 3 == chunk_idx || 2 == chunk_idx {
                let zeros = ndarray::Array3::<f32>::zeros(dim);
                let _ = assert_eq!(zeros, chunk_new);
            } else {
                let chunk_old = chunks_old_iter.next().expect("ran out of chunks");
                assert_eq!(chunk_old, chunk_new);
            }
        }
    }

    use crate::metadata_coding::*;
    use crate::source_coding::chunk::*;

    #[test]
    fn test_chunk_metadata_modulation_values() {
        let mut chunk_metadata = vec![ChunkMetadata::default(); 15];
        for (idx, cm) in chunk_metadata.iter_mut().enumerate() {
            cm.energy = idx as f32;
            cm.mean = -(idx as f32);
        }
        let compressed_metadata: CompressedMetadata = chunk_metadata.iter().into();
        let packetizer: Packetizer = compressed_metadata.into();

        let orig_encoded_packets: Vec<_> = packetizer.collect();

        let metadata_modulator: MetadataModulator<_> =
            orig_encoded_packets.clone().into_iter().into();
        let metadata_demodulator: MetadataDemodulator<_> = metadata_modulator.flatten().into();

        let new_encoded_packets: Vec<_> = metadata_demodulator.collect();
        assert_eq!(orig_encoded_packets, new_encoded_packets);
    }

    #[test]
    fn test_chunk_metadata_modulation_decode_1() {
        let mut chunk_metadata = vec![ChunkMetadata::default(); 15];
        for (idx, cm) in chunk_metadata.iter_mut().enumerate() {
            cm.energy = idx as f32;
            cm.mean = -(idx as f32);
        }
        let compressed_metadata: CompressedMetadata = chunk_metadata.iter().into();
        let packetizer: Packetizer = compressed_metadata.into();

        let orig_encoded_packets: Vec<_> = packetizer.collect();

        let metadata_modulator: MetadataModulator<_> =
            orig_encoded_packets.clone().into_iter().into();
        let metadata_demodulator: MetadataDemodulator<_> = metadata_modulator.flatten().into();

        let depacketizer: Depacketizer<_, ()> = metadata_demodulator.into();
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(depacketizer, chunk_metadata.len());
        let new_chunk_metatata: Vec<ChunkMetadata> =
            decompressor.map(|result| result.unwrap()).collect();

        for (orig, new) in chunk_metadata.iter().zip(new_chunk_metatata.iter()) {
            assert!((orig.mean - new.mean).abs() < 0.5);
            assert!(orig.energy == new.energy || (1.0 - orig.energy / new.energy).abs() < 0.001);
        }
    }

    #[test]
    fn test_chunk_metadata_modulation_decode_2() {
        let mut chunk_metadata = vec![ChunkMetadata::default(); 15000];
        for (idx, cm) in chunk_metadata.iter_mut().enumerate() {
            cm.energy = idx as f32;
            cm.mean = -(idx as f32) % i8::MAX as f32;
        }
        let compressed_metadata: CompressedMetadata = chunk_metadata.iter().into();
        let packetizer: Packetizer = compressed_metadata.into();
        let metadata_modulator: MetadataModulator<_> = packetizer.into();

        let metadata_demodulator: MetadataDemodulator<_> = metadata_modulator.flatten().into();
        let depacketizer: Depacketizer<_, ()> = metadata_demodulator.into();
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(depacketizer, chunk_metadata.len());
        let new_chunk_metatata: Vec<ChunkMetadata> =
            decompressor.map(|result| result.unwrap()).collect();

        for (orig, new) in chunk_metadata.iter().zip(new_chunk_metatata.iter()) {
            assert!(
                (orig.mean - new.mean).abs() < 0.5,
                "{} ->  {}",
                orig.mean,
                new.mean
            );
            assert!(orig.energy == new.energy || (1.0 - orig.energy / new.energy).abs() < 0.001);
        }
    }
}
