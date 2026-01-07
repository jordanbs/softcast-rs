// Copyright 2025-2026 Jordan Schneider
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

use crate::asset_reader_writer::*;
use crate::source_coding::chunked_dct_block::*;
use hadamard_block::*;

pub mod hadamard_block {
    use super::*;
    use fwht;

    pub enum ViewOrOwnedArray3<'a> {
        View(ndarray::ArrayViewMut3<'a, f32>),
        Owned(ndarray::Array3<f32>),
    }

    pub struct Slice<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ViewOrOwnedArray3<'a>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType>
        Slice<'a, GOP_LENGTH, PixelType>
    {
        pub fn new(values: ViewOrOwnedArray3<'a>) -> Self {
            Self {
                values,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct SliceIter<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>>,
    {
        chunked_dct_block_iter: std::iter::Peekable<I>,
        inner_slice_iter: std::vec::IntoIter<hadamard_block::Slice<'a, DCT_LENGTH, PixelType>>,
        chunks_per_gop: usize,
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        SliceIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>>,
    {
        pub fn new(chunked_dct_block_iter: I, chunks_per_gop: usize) -> Self {
            SliceIter {
                chunked_dct_block_iter: chunked_dct_block_iter.peekable(),
                inner_slice_iter: vec![].into_iter(),
                chunks_per_gop: chunks_per_gop,
            }
        }
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for SliceIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>>,
    {
        type Item = Slice<'a, DCT_LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                if let Some(slice) = self.inner_slice_iter.next() {
                    return Some(slice);
                }

                let chunks: Box<_> = self
                    .chunked_dct_block_iter
                    .by_ref()
                    .take(self.chunks_per_gop)
                    .collect();

                if chunks.is_empty() {
                    return None;
                }
                assert!(
                    chunks.len() == self.chunks_per_gop,
                    "Not enough chunks for a GOP."
                );

                let slices = fwht::fwht_chunks(chunks).expect("Failed to create slices.");
                self.inner_slice_iter = slices.into_iter();
            }
        }
    }
}

pub mod fwht {
    use super::*;

    pub(super) trait ValuesProvider {
        fn values(&self) -> ndarray::ArrayView3<'_, f32>;
        fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32>;
    }

    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> ValuesProvider
        for ChunkedDCTBlock<'_, DCT_LENGTH, PixelType>
    {
        fn values(&self) -> ndarray::ArrayView3<'_, f32> {
            self.values.view()
        }
        fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
            self.values.view_mut()
        }
    }

    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> ValuesProvider
        for Slice<'_, DCT_LENGTH, PixelType>
    {
        fn values(&self) -> ndarray::ArrayView3<'_, f32> {
            match &self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view(),
            }
        }
        fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
            match &mut self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view_mut(),
            }
        }
    }

    // MARK: could be performed more naturally with the entire 3d dct
    pub(super) fn fwht(
        data: &mut Box<[impl ValuesProvider]>,
        padding: &mut Box<[ndarray::Array3<f32>]>,
    ) {
        let first_chunk = data.first().expect("no data");
        let indicies_iter = ndarray::indices_of(&first_chunk.values());
        let hadamard_len = data.len() + padding.len();

        assert!(hadamard_len.is_power_of_two());

        for index_in_chunk in indicies_iter {
            let mut h = 1;
            while h < hadamard_len {
                for i in (0..hadamard_len).step_by(h * 2) {
                    for j in i..i + h {
                        let x = if j < data.len() {
                            data[j].values()[index_in_chunk]
                        } else {
                            padding[j - data.len()][index_in_chunk]
                        };

                        let y = if j + h < data.len() {
                            data[j + h].values()[index_in_chunk]
                        } else {
                            padding[j + h - data.len()][index_in_chunk]
                        };

                        if j < data.len() {
                            data[j].values_mut()[index_in_chunk] = x + y
                        } else {
                            padding[j - data.len()][index_in_chunk] = x + y
                        };

                        if j + h < data.len() {
                            data[j + h].values_mut()[index_in_chunk] = x - y
                        } else {
                            padding[j + h - data.len()][index_in_chunk] = x - y
                        };
                    }
                }
                h *= 2;
            }
        }
        let orthonormalization_factor = 1f32 / (hadamard_len as f32).sqrt();
        data.iter_mut()
            .for_each(|data_row| *data_row.values_mut() *= orthonormalization_factor);
        padding
            .iter_mut()
            .for_each(|padding_row| *padding_row *= orthonormalization_factor);
    }

    pub fn fwht_chunks<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
        chunks: Box<[ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>]>,
    ) -> Result<Box<[Slice<'a, DCT_LENGTH, PixelType>]>, &'static str> {
        // adapted from fwht crate, with the intention of avoiding copies
        let mut chunks = chunks;

        // add padding so each fwht is a power of two
        let hadamard_len = 2usize.pow(((chunks.len() as f32).log2()).ceil() as u32);

        let num_padding_rows = hadamard_len - chunks.len();
        let chunk_dim = chunks.first().expect("no data").values().raw_dim();
        let mut padding_chunks: Box<_> =
            vec![ndarray::Array3::<f32>::zeros(chunk_dim); num_padding_rows].into();

        fwht(&mut chunks, &mut padding_chunks);

        let mut slices = Vec::with_capacity(hadamard_len);
        for chunk in chunks {
            let slice = Slice::new(ViewOrOwnedArray3::View(chunk.values));
            slices.push(slice);
        }
        for padding_chunk in padding_chunks {
            let slice = Slice::new(ViewOrOwnedArray3::Owned(padding_chunk));
            slices.push(slice);
        }

        Ok(slices.into())
    }

    pub fn fwht_slices<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
        slices: Box<[Slice<'a, DCT_LENGTH, PixelType>]>,
        num_padding_rows: usize,
    ) -> Result<Box<[ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>]>, &'static str> {
        let mut slices = slices;

        fwht(&mut slices, &mut vec![].into());

        let mut chunks = Vec::with_capacity(slices.len() - num_padding_rows);
        for slice in slices {
            // consume slice.values
            let values = match slice.values {
                ViewOrOwnedArray3::View(view) => view,
                ViewOrOwnedArray3::Owned(_) => {
                    panic!("slice not expected to own its data.")
                }
            };

            let chunk: ChunkedDCTBlock<'a, DCT_LENGTH, PixelType> =
                ChunkedDCTBlock::new(values, 0f32, 0f32); // TODO: get mean and energy
            chunks.push(chunk);
        }

        Ok(chunks.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn ofdm_flexframe() {
        use liquid_dsp_sys;

        unsafe {
            let fg = liquid_dsp_sys::ofdmflexframegen_create(
                64,
                16,
                4,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert!(!fg.is_null());

            let header = [0u8; 8];
            let payload = [0u8; 20];
            let mut buf = [Complex32::ZERO; 120];

            let status = liquid_dsp_sys::ofdmflexframegen_assemble(
                fg,
                header.as_ptr(),
                payload.as_ptr(),
                payload.len().try_into().unwrap(),
            );
            assert_eq!(status, 0);

            let status = liquid_dsp_sys::ofdmflexframegen_print(fg);
            assert_eq!(status, 0);

            let mut frame_complete = 0;
            while 0 == frame_complete {
                frame_complete = liquid_dsp_sys::ofdmflexframegen_write(
                    fg,
                    buf.as_mut_ptr(),
                    buf.len().try_into().unwrap(),
                );

                eprintln!(
                    "ofdmflexframegen wrote to buffer{}",
                    match frame_complete {
                        0 => "",
                        _ => " (frame complete)",
                    },
                );
            }
        }
    }

    impl fwht::ValuesProvider for ndarray::Array3<f32> {
        fn values(&self) -> ndarray::ArrayView3<'_, f32> {
            self.view()
        }
        fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
            self.view_mut()
        }
    }

    #[test]
    fn test_fwht_basic() {
        let mut data: Box<[_]> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 4].into();

        data[0][(0, 0, 0)] = 1f32;
        data[1][(0, 0, 0)] = 2f32;
        data[2][(0, 0, 0)] = 3f32;
        data[3][(0, 0, 0)] = 4f32;

        data[0][(0, 0, 1)] = 5f32;
        data[1][(0, 0, 1)] = 6f32;
        data[2][(0, 0, 1)] = 7f32;
        data[3][(0, 0, 1)] = 8f32;

        fwht::fwht(&mut data, &mut vec![].into());

        assert_eq!(data[0][(0, 0, 0)], 5f32);
        assert_eq!(data[1][(0, 0, 0)], -1f32);
        assert_eq!(data[2][(0, 0, 0)], -2f32);
        assert_eq!(data[3][(0, 0, 0)], 0f32);

        assert_eq!(data[0][(0, 0, 1)], 13f32);
        assert_eq!(data[1][(0, 0, 1)], -1f32);
        assert_eq!(data[2][(0, 0, 1)], -2f32);
        assert_eq!(data[3][(0, 0, 1)], 0f32);
    }

    #[test]
    fn test_fwht_padding() {
        let mut data: Box<[_]> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 5].into();

        data[0][(0, 0, 0)] = 1f32;
        data[1][(0, 0, 0)] = 2f32;
        data[2][(0, 0, 0)] = 3f32;
        data[3][(0, 0, 0)] = 4f32;
        data[4][(0, 0, 0)] = 5f32;

        data[0][(0, 0, 1)] = 6f32;
        data[1][(0, 0, 1)] = 7f32;
        data[2][(0, 0, 1)] = 8f32;
        data[3][(0, 0, 1)] = 9f32;
        data[4][(0, 0, 1)] = 10f32;

        let mut padding: Box<_> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 3].into();
        fwht::fwht(&mut data, &mut padding);

        assert!((data[0][(0, 0, 0)] - 5.3033).abs() < 0.001);
        assert!((data[1][(0, 0, 0)] - 1.0607).abs() < 0.001);
        assert!((data[2][(0, 0, 0)] - 0.3536).abs() < 0.001);
        assert!((data[3][(0, 0, 0)] - 1.7678).abs() < 0.001);
        assert!((data[4][(0, 0, 0)] - 1.7678).abs() < 0.001);

        assert!((padding[0][(0, 0, 0)] - -2.4749).abs() < 0.001);
        assert!((padding[1][(0, 0, 0)] - -3.1820).abs() < 0.001);
        assert!((padding[2][(0, 0, 0)] - -1.7678).abs() < 0.001);

        assert!((data[0][(0, 0, 1)] - 14.1421).abs() < 0.001);
        assert!((data[1][(0, 0, 1)] - 2.8284).abs() < 0.001);
        assert!((data[2][(0, 0, 1)] - 2.1213).abs() < 0.001);
        assert!((data[3][(0, 0, 1)] - 3.5355).abs() < 0.001);
        assert!((data[4][(0, 0, 1)] - 7.0711).abs() < 0.001);

        assert!((padding[0][(0, 0, 1)] - -4.2426).abs() < 0.001);
        assert!((padding[1][(0, 0, 1)] - -4.9497).abs() < 0.001);
        assert!((padding[2][(0, 0, 1)] - -3.5355).abs() < 0.001);
    }

    #[test]
    fn test_fwht_inverse() {
        let mut data: Box<[_]> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 4].into();

        data[0][(0, 0, 0)] = 1f32;
        data[1][(0, 0, 0)] = 2f32;
        data[2][(0, 0, 0)] = 3f32;
        data[3][(0, 0, 0)] = 4f32;

        data[0][(0, 0, 1)] = 5f32;
        data[1][(0, 0, 1)] = 6f32;
        data[2][(0, 0, 1)] = 7f32;
        data[3][(0, 0, 1)] = 8f32;

        fwht::fwht(&mut data, &mut vec![].into());
        fwht::fwht(&mut data, &mut vec![].into());

        assert_eq!(data[0][(0, 0, 0)], 1f32);
        assert_eq!(data[1][(0, 0, 0)], 2f32);
        assert_eq!(data[2][(0, 0, 0)], 3f32);
        assert_eq!(data[3][(0, 0, 0)], 4f32);

        assert_eq!(data[0][(0, 0, 1)], 5f32);
        assert_eq!(data[1][(0, 0, 1)], 6f32);
        assert_eq!(data[2][(0, 0, 1)], 7f32);
        assert_eq!(data[3][(0, 0, 1)], 8f32);
    }

    #[test]
    fn test_fwht_inverse_padding() {
        let mut data: Box<[_]> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 5].into();

        data[0][(0, 0, 0)] = 1f32;
        data[1][(0, 0, 0)] = 2f32;
        data[2][(0, 0, 0)] = 3f32;
        data[3][(0, 0, 0)] = 4f32;
        data[4][(0, 0, 0)] = 5f32;

        data[0][(0, 0, 1)] = 6f32;
        data[1][(0, 0, 1)] = 7f32;
        data[2][(0, 0, 1)] = 8f32;
        data[3][(0, 0, 1)] = 9f32;
        data[4][(0, 0, 1)] = 10f32;

        let mut padding: Box<_> = vec![ndarray::Array3::<f32>::zeros((1, 1, 2)); 3].into();
        fwht::fwht(&mut data, &mut padding);
        fwht::fwht(&mut data, &mut padding);

        assert!((data[0][(0, 0, 0)] - 1f32).abs() < 0.001);
        assert!((data[1][(0, 0, 0)] - 2f32).abs() < 0.001);
        assert!((data[2][(0, 0, 0)] - 3f32).abs() < 0.001);
        assert!((data[3][(0, 0, 0)] - 4f32).abs() < 0.001);
        assert!((data[4][(0, 0, 0)] - 5f32).abs() < 0.001);

        assert!((padding[0][(0, 0, 0)] - 0f32).abs() < 0.001);
        assert!((padding[1][(0, 0, 0)] - 0f32).abs() < 0.001);
        assert!((padding[2][(0, 0, 0)] - 0f32).abs() < 0.001);

        assert!((data[0][(0, 0, 1)] - 6f32).abs() < 0.001);
        assert!((data[1][(0, 0, 1)] - 7f32).abs() < 0.001);
        assert!((data[2][(0, 0, 1)] - 8f32).abs() < 0.001);
        assert!((data[3][(0, 0, 1)] - 9f32).abs() < 0.001);
        assert!((data[4][(0, 0, 1)] - 10f32).abs() < 0.001);

        assert!((padding[0][(0, 0, 1)] - 0f32).abs() < 0.001);
        assert!((padding[1][(0, 0, 1)] - 0f32).abs() < 0.001);
        assert!((padding[2][(0, 0, 1)] - 0f32).abs() < 0.001);
    }
}
