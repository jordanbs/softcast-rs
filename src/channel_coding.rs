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

    #[derive(Debug)]
    pub struct Slice<const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array1<f32>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const GOP_LENGTH: usize, PixelType: HasPixelComponentType> Slice<GOP_LENGTH, PixelType> {
        pub fn new(values: ndarray::Array1<f32>) -> Self {
            Self {
                values,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub enum ViewOrOwnedArray3<'a> {
        View(ndarray::ArrayViewMut3<'a, f32>),
        Owned(ndarray::Array3<f32>),
    }

    pub struct Slice2<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ViewOrOwnedArray3<'a>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType>
        Slice2<'a, GOP_LENGTH, PixelType>
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
                chunks_per_gop: chunks_per_gop,
            }
        }
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for SliceIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>>,
    {
        type Item = Box<[Slice<DCT_LENGTH, PixelType>]>;

        // TODO: This should all be done without copying (currently has two copies!), likely involving a
        // custom implementation of fwht and mutable access to chunks.
        fn next(&mut self) -> Option<Self::Item> {
            let hadamard_len = 2usize.pow(((self.chunks_per_gop as f32).log2()).ceil() as u32);

            let first_chunk = self.chunked_dct_block_iter.peek().unwrap(); // TODO: don't unwrap

            let chunk_len = first_chunk.values.len();
            let mut hadamard_rvalues = ndarray::Array2::zeros((chunk_len, hadamard_len));

            // copy each column to apply a hadamard to smear the energy across chunks
            for (row, index_in_chunk) in ndarray::indices_of(&first_chunk.values)
                .into_iter()
                .enumerate()
            {
                // copy
                self.chunked_dct_block_iter
                    .by_ref()
                    .take(self.chunks_per_gop)
                    .map(|chunk| chunk.values[index_in_chunk])
                    .enumerate()
                    .for_each(|(col, value)| hadamard_rvalues[(row, col)] = value);
            }

            // now compute the hadamards
            for col in 0..self.chunks_per_gop {
                let mut hadamard_rvalues_column =
                    hadamard_rvalues.index_axis_mut(ndarray::Axis(0), col);
                fwht::fwht_slice(
                    hadamard_rvalues_column
                        .as_slice_mut()
                        .expect("not contiguous"),
                )
                .expect("fwht failed.");

                // scale by 1/√n
                hadamard_rvalues_column
                    .iter_mut()
                    .for_each(|value| *value *= 1f32 / (self.chunks_per_gop as f32).sqrt());
            }

            let slices: Vec<Slice<DCT_LENGTH, PixelType>> = hadamard_rvalues
                .outer_iter()
                .map(|hadamard_rrow| hadamard_rrow.to_owned()) // copy
                .map(|hadamard_rrow_owned| Slice::new(hadamard_rrow_owned))
                .collect();

            Some(slices.into())
        }
    }
}

fn fwht_chunks<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
    chunks: Box<[ChunkedDCTBlock<'a, DCT_LENGTH, PixelType>]>,
) -> Result<Box<[Slice2<'a, DCT_LENGTH, PixelType>]>, &'static str> {
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

    let mut slices = Vec::with_capacity(hadamard_len);
    for chunk in chunks {
        let slice = Slice2::new(ViewOrOwnedArray3::View(chunk.values));
        slices.push(slice);
    }
    for padding_chunk in padding_chunks {
        let slice = Slice2::new(ViewOrOwnedArray3::Owned(padding_chunk));
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
