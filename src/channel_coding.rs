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

                let slices = fwht_chunks(chunks).expect("Failed to create slices.");
                self.inner_slice_iter = slices.into_iter();
            }
        }
    }
}

fn fwht_chunks<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
    chunks: Box<[ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>]>,
) -> Result<Box<[Slice<'a, DCT_LENGTH, PixelType>]>, &'static str> {
    // adapted from fwht crate, with the intention of avoiding copies
    let mut chunks = chunks;
    let first_chunk = chunks.first().expect("no chunks");

    // add padding so each fwht is a power of two
    let hadamard_len = 2usize.pow(((chunks.len() as f32).log2()).ceil() as u32);

    let num_padding_rows = hadamard_len - chunks.len();
    let mut padding_chunks =
        vec![ndarray::Array3::<f32>::zeros(first_chunk.values.raw_dim()); num_padding_rows];

    for index_in_chunk in ndarray::indices_of(&first_chunk.values) {
        let mut h = 1;
        while h < hadamard_len {
            for i in (0..hadamard_len).step_by(h * 2) {
                for j in i..i + h {
                    let x = if j < chunks.len() {
                        chunks[j].values[index_in_chunk]
                    } else {
                        padding_chunks[j - chunks.len()][index_in_chunk]
                    };

                    let y = if j + h < chunks.len() {
                        chunks[j + h].values[index_in_chunk]
                    } else {
                        padding_chunks[j + h - chunks.len()][index_in_chunk]
                    };

                    if j < chunks.len() {
                        chunks[j].values[index_in_chunk] = x + y
                    } else {
                        padding_chunks[j - chunks.len()][index_in_chunk] = x + y
                    };

                    if j + h < chunks.len() {
                        chunks[j + h].values[index_in_chunk] = x - y
                    } else {
                        padding_chunks[j + h - chunks.len()][index_in_chunk] = x - y
                    };
                }
            }
            h *= 2;
        }
    }
    let orthonormalization_factor = 1f32 / (hadamard_len as f32).sqrt();
    chunks
        .iter_mut()
        .for_each(|chunk| *chunk.values *= orthonormalization_factor);
    padding_chunks
        .iter_mut()
        .for_each(|padding_chunk| *padding_chunk *= orthonormalization_factor);

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

#[cfg(test)]
mod tests {
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
}
