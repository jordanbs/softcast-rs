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
use liquid_sys;
use num_complex::Complex32;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct QuadratureSymbol {
    pub value: Complex32,
}
impl From<Complex32> for QuadratureSymbol {
    fn from(c32: Complex32) -> Self {
        unsafe { std::mem::transmute(c32) }
    }
}
impl From<QuadratureSymbol> for Complex32 {
    fn from(iq: QuadratureSymbol) -> Self {
        unsafe { std::mem::transmute(iq) }
    }
}
pub trait FromU16 {
    fn from_u16(u: u16, modem: *mut liquid_sys::modemcf_s) -> Self;
}
pub struct U8QPacketModem {
    ptr: liquid_sys::qpacketmodem,
}
impl U8QPacketModem {
    pub const ENCODED_FRAME_LEN: usize = 80;

    pub fn new() -> Self {
        unsafe {
            let qpacket_modem = liquid_sys::qpacketmodem_create();
            let status = liquid_sys::qpacketmodem_configure(
                qpacket_modem,
                size_of::<u8>() as u32,
                liquid_sys::crc_scheme_LIQUID_CRC_24,
                liquid_sys::fec_scheme_LIQUID_FEC_CONV_V29,
                liquid_sys::fec_scheme_LIQUID_FEC_NONE,
                liquid_sys::modulation_scheme_LIQUID_MODEM_BPSK as i32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let frame_len = liquid_sys::qpacketmodem_get_frame_len(qpacket_modem) as usize;
            assert_eq!(Self::ENCODED_FRAME_LEN, frame_len);

            Self { ptr: qpacket_modem }
        }
    }
    pub fn encode(&mut self, u: u8) -> Box<[QuadratureSymbol]> {
        unsafe {
            let mut frame: Box<[Complex32]> = vec![Complex32::ZERO; Self::ENCODED_FRAME_LEN].into();
            let payload = u.to_be_bytes();
            let status =
                liquid_sys::qpacketmodem_encode(self.ptr, payload.as_ptr(), frame.as_mut_ptr())
                    as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            std::mem::transmute(frame)
        }
    }
    pub fn decode(&mut self, payload: &mut [QuadratureSymbol]) -> Result<u8, std::string::String> {
        assert_eq!(Self::ENCODED_FRAME_LEN, payload.len());

        let mut u8_be_bytes = [0u8; size_of::<u8>()];
        unsafe {
            let payload: &mut [Complex32] = std::mem::transmute(payload);
            let success = liquid_sys::qpacketmodem_decode_soft(
                self.ptr,
                payload.as_mut_ptr(),
                u8_be_bytes.as_mut_ptr(),
            );
            if 1 != success {
                return Err("frame header decode failed".to_string());
            }
        }
        Ok(u8::from_be_bytes(u8_be_bytes))
    }
}
impl Drop for U8QPacketModem {
    fn drop(&mut self) {
        let status = unsafe { liquid_sys::qpacketmodem_destroy(self.ptr) } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
    }
}

pub mod slices {
    use super::*;
    use crate::channel_coding::fwht_softcast::ValuesProvider;
    use crate::channel_coding::slice::*;
    use crate::pixel_buffer::HasPixelComponentType;
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
            // TODO: Add check to see that slice_dimensions is compatible with array3
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
    use crate::channel_coding::slice::*;
    use crate::modulation::slices::*;
    use crate::pixel_buffer::*;

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
}
