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
use crate::source_coding::chunk::*;
use slice::*;

pub mod slice {
    use super::*;
    use fwht;

    pub struct SliceAndChunkMetadata<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
        pub slice: Slice<'a, GOP_LENGTH, PixelType>,
        pub chunk_metadata: ChunkMetadata,
    }
    impl<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType>
        SliceAndChunkMetadata<'a, GOP_LENGTH, PixelType>
    {
        pub fn new(slice: Slice<'a, GOP_LENGTH, PixelType>, chunk_metadata: ChunkMetadata) -> Self {
            Self {
                slice,
                chunk_metadata,
            }
        }
    }

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

        pub fn from_view(view: ndarray::ArrayViewMut3<'a, f32>) -> Self {
            Self::new(ViewOrOwnedArray3::View(view))
        }
        pub fn from_owned(owned: ndarray::Array3<f32>) -> Self {
            Self::new(ViewOrOwnedArray3::Owned(owned))
        }

        pub fn values(&self) -> ndarray::ArrayView3<'_, f32> {
            match &self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view(),
            }
        }

        pub fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
            match &mut self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view_mut(),
            }
        }
    }

    pub struct SliceIter<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        chunk_iter: std::iter::Peekable<I>,
        inner_slice_iter: std::vec::IntoIter<SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>>,
        chunks_per_gop: usize,
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        SliceIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        pub fn new(chunk_iter: I, chunks_per_gop: usize) -> Self {
            SliceIter {
                chunk_iter: chunk_iter.peekable(),
                inner_slice_iter: vec![].into_iter(),
                chunks_per_gop: chunks_per_gop,
            }
        }
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for SliceIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        type Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                if let Some(slice) = self.inner_slice_iter.next() {
                    return Some(slice);
                }

                let chunks: Box<_> = self.chunk_iter.by_ref().take(self.chunks_per_gop).collect();

                if chunks.is_empty() {
                    return None;
                }
                assert_eq!(
                    chunks.len(),
                    self.chunks_per_gop,
                    "Not enough chunks for a GOP."
                );

                let slices = fwht::fwht_chunks(chunks).expect("Failed to create slices.");
                self.inner_slice_iter = slices.into_iter();
            }
        }
    }

    pub struct ChunkIter<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
    where
        I: Iterator<Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>>,
    {
        slice_iter: std::iter::Peekable<I>,
        inner_chunk_iter: std::vec::IntoIter<Chunk<'a, DCT_LENGTH, PixelType>>,
        chunks_per_gop: usize,
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        ChunkIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>>,
    {
        pub fn new(slice_iter: I, chunks_per_gop: usize) -> Self {
            ChunkIter {
                slice_iter: slice_iter.peekable(),
                inner_chunk_iter: vec![].into_iter(),
                chunks_per_gop: chunks_per_gop,
            }
        }
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for ChunkIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>>,
    {
        type Item = Chunk<'a, DCT_LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                if let Some(chunk) = self.inner_chunk_iter.next() {
                    return Some(chunk);
                }

                let hadamard_len = self.chunks_per_gop.next_power_of_two();

                let slices: Box<_> = self.slice_iter.by_ref().take(hadamard_len).collect();

                if slices.is_empty() {
                    return None;
                }
                assert_eq!(slices.len(), hadamard_len, "Not enough slices.");

                let chunks = fwht::fwht_slices(slices, hadamard_len - self.chunks_per_gop)
                    .expect("Failed to create chunks.");
                self.inner_chunk_iter = chunks.into_iter();
            }
        }
    }

    pub trait ChunkIterIntoExt<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>:
        Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>> + Sized
    {
        fn into_slice_iter(
            self,
            chunks_per_gop: usize,
        ) -> SliceIter<'a, DCT_LENGTH, PixelType, Self>;
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        ChunkIterIntoExt<'a, DCT_LENGTH, PixelType> for I
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        fn into_slice_iter(
            self,
            chunks_per_gop: usize,
        ) -> SliceIter<'a, DCT_LENGTH, PixelType, Self> {
            SliceIter::new(self, chunks_per_gop)
        }
    }

    pub trait SliceIterExt<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>:
        Iterator<Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>> + Sized
    {
        fn into_chunks_iter(
            self,
            chunks_per_gop: usize,
        ) -> ChunkIter<'a, DCT_LENGTH, PixelType, Self>;
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        SliceIterExt<'a, DCT_LENGTH, PixelType> for I
    where
        I: Iterator<Item = SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>>,
    {
        fn into_chunks_iter(
            self,
            chunks_per_gop: usize,
        ) -> ChunkIter<'a, DCT_LENGTH, PixelType, Self> {
            ChunkIter::new(self, chunks_per_gop)
        }
    }
}

pub mod fwht {
    use super::*;
    use rayon::prelude::*;

    pub trait ValuesProvider {
        fn value_at(&self, idx: usize) -> f32;
        fn ptr_at(&self, idx: usize) -> *mut f32;
        fn values_len(&self) -> usize;
    }

    trait To3Dim {
        fn to_3dim_index(self, dim: (usize, usize, usize)) -> (usize, usize, usize);
    }
    impl To3Dim for usize {
        fn to_3dim_index(self, dim: (usize, usize, usize)) -> (usize, usize, usize) {
            let i = self / (dim.1 * dim.2);
            let j = (self % (dim.1 * dim.2)) / dim.2;
            let k = (self % (dim.1 * dim.2)) % dim.2;
            (i, j, k)
        }
    }

    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> ValuesProvider
        for Chunk<'_, DCT_LENGTH, PixelType>
    {
        fn value_at(&self, idx: usize) -> f32 {
            let idx = idx.to_3dim_index(self.values.dim());
            self.values[idx]
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            let idx = idx.to_3dim_index(self.values.dim());
            let ptr: *const f32 = &self.values[idx];
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.values.len()
        }
    }
    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> ValuesProvider
        for Slice<'_, DCT_LENGTH, PixelType>
    {
        fn value_at(&self, idx: usize) -> f32 {
            let idx = idx.to_3dim_index(self.values().dim());
            self.values()[idx]
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            let idx = idx.to_3dim_index(self.values().dim());
            let value = match &self.values {
                ViewOrOwnedArray3::View(view) => &view[idx],
                ViewOrOwnedArray3::Owned(owned) => &owned[idx],
            };
            let ptr: *const f32 = value;
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.values().len()
        }
    }
    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> ValuesProvider
        for SliceAndChunkMetadata<'_, DCT_LENGTH, PixelType>
    {
        fn value_at(&self, idx: usize) -> f32 {
            self.slice.value_at(idx)
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            self.slice.ptr_at(idx)
        }
        fn values_len(&self) -> usize {
            self.slice.values_len()
        }
    }
    impl ValuesProvider for ndarray::Array3<f32> {
        fn value_at(&self, idx: usize) -> f32 {
            let idx = idx.to_3dim_index(self.dim());
            self[idx]
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            let idx = idx.to_3dim_index(self.dim());
            let ptr: *const f32 = &self[idx];
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.len()
        }
    }
    impl ValuesProvider for [f32] {
        fn value_at(&self, idx: usize) -> f32 {
            self[idx]
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            let ptr: *const f32 = &self[idx];
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.len()
        }
    }

    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> std::ops::MulAssign<f32>
        for Chunk<'_, DCT_LENGTH, PixelType>
    {
        fn mul_assign(&mut self, rhs: f32) {
            self.values.mul_assign(rhs);
        }
    }
    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> std::ops::MulAssign<f32>
        for Slice<'_, DCT_LENGTH, PixelType>
    {
        fn mul_assign(&mut self, rhs: f32) {
            self.values_mut().mul_assign(rhs);
        }
    }
    impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> std::ops::MulAssign<f32>
        for SliceAndChunkMetadata<'_, DCT_LENGTH, PixelType>
    {
        fn mul_assign(&mut self, rhs: f32) {
            self.slice.mul_assign(rhs);
        }
    }

    unsafe impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> Send
        for Chunk<'_, DCT_LENGTH, PixelType>
    {
    }
    unsafe impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> Send
        for Slice<'_, DCT_LENGTH, PixelType>
    {
    }

    unsafe impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> Sync
        for Chunk<'_, DCT_LENGTH, PixelType>
    {
    }
    unsafe impl<const DCT_LENGTH: usize, PixelType: HasPixelComponentType> Sync
        for Slice<'_, DCT_LENGTH, PixelType>
    {
    }

    pub(super) fn fwht(
        data: &mut Box<[impl ValuesProvider + std::ops::MulAssign<f32> + Send + Sync]>,
        padding: &mut Box<[impl ValuesProvider + std::ops::MulAssign<f32> + Send + Sync]>,
    ) {
        let num_columns = data.first().expect("no_data").values_len();
        let hadamard_len = data.len() + padding.len();

        assert!(hadamard_len.is_power_of_two());

        (0..num_columns).into_par_iter().for_each(|index_in_chunk| {
            let mut h = 1;
            while h < hadamard_len {
                for i in (0..hadamard_len).step_by(h * 2) {
                    for j in i..i + h {
                        let x = if j < data.len() {
                            data[j].value_at(index_in_chunk)
                        } else {
                            padding[j - data.len()].value_at(index_in_chunk)
                        };

                        let y = if j + h < data.len() {
                            data[j + h].value_at(index_in_chunk)
                        } else {
                            padding[j + h - data.len()].value_at(index_in_chunk)
                        };

                        let ptr = if j < data.len() {
                            data[j].ptr_at(index_in_chunk)
                        } else {
                            padding[j - data.len()].ptr_at(index_in_chunk)
                        };
                        unsafe {
                            *ptr = x + y;
                        }

                        let ptr = if j + h < data.len() {
                            data[j + h].ptr_at(index_in_chunk)
                        } else {
                            padding[j + h - data.len()].ptr_at(index_in_chunk)
                        };
                        unsafe {
                            *ptr = x - y;
                        }
                    }
                }
                h *= 2;
            }
        });
        let orthonormalization_factor = 1f32 / (hadamard_len as f32).sqrt();
        data.iter_mut()
            .for_each(|data_row| *data_row *= orthonormalization_factor);
        padding
            .iter_mut()
            .for_each(|padding_row| *padding_row *= orthonormalization_factor);
    }

    pub fn fwht_chunks<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
        chunks: Box<[Chunk<'a, DCT_LENGTH, PixelType>]>,
    ) -> Result<Box<[SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>]>, &'static str> {
        // adapted from fwht crate, with the intention of avoiding copies
        let mut chunks = chunks;

        // add padding so each fwht is a power of two
        let hadamard_len = chunks.len().next_power_of_two();

        let num_padding_rows = hadamard_len - chunks.len();
        let chunk_dim = chunks.first().expect("no data").values.raw_dim();
        let mut padding_chunks: Box<_> =
            vec![ndarray::Array3::<f32>::zeros(chunk_dim); num_padding_rows].into();

        fwht(&mut chunks, &mut padding_chunks);

        // metadata

        let mut slices = Vec::with_capacity(hadamard_len);
        for chunk in chunks {
            let slice = Slice::from_view(chunk.values);
            let slice = SliceAndChunkMetadata::new(slice, chunk.metadata);
            slices.push(slice);
        }
        for padding_chunk in padding_chunks {
            let slice = Slice::from_owned(padding_chunk);
            let slice = SliceAndChunkMetadata::new(slice, ChunkMetadata::default() /* zero */);
            slices.push(slice);
        }

        Ok(slices.into())
    }

    pub fn fwht_slices<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
        slices: Box<[SliceAndChunkMetadata<'a, DCT_LENGTH, PixelType>]>,
        num_padding_rows: usize,
    ) -> Result<Box<[Chunk<'a, DCT_LENGTH, PixelType>]>, &'static str> {
        let mut slices = slices;

        let mut empty: Box<[ndarray::Array3<f32>]> = vec![].into();
        fwht(&mut slices, &mut empty);

        let mut chunks = Vec::with_capacity(slices.len() - num_padding_rows);
        for slice in slices {
            // consume slice.values
            let values = match slice.slice.values {
                ViewOrOwnedArray3::View(view) => view,
                ViewOrOwnedArray3::Owned(_) => {
                    // TODO: This assumption might not be true in the testing loopback.
                    panic!("slice not expected to own its data in decode.")
                }
            };

            let chunk: Chunk<'a, DCT_LENGTH, PixelType> = Chunk::new(values, slice.chunk_metadata);
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
        use liquid_sys;

        unsafe {
            let fg = liquid_sys::ofdmflexframegen_create(
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

            let status = liquid_sys::ofdmflexframegen_assemble(
                fg,
                header.as_ptr(),
                payload.as_ptr(),
                payload.len().try_into().unwrap(),
            );
            assert_eq!(status, 0);

            let status = liquid_sys::ofdmflexframegen_print(fg);
            assert_eq!(status, 0);

            let mut frame_complete = 0;
            while 0 == frame_complete {
                frame_complete = liquid_sys::ofdmflexframegen_write(
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

    #[test]
    fn link_libfec() {
        use liquid_sys;
        unsafe {
            let ptr = liquid_sys::fec_create(
                liquid_sys::fec_scheme_LIQUID_FEC_RS_M8,
                core::ptr::null_mut(),
            );
            assert_ne!(ptr, std::ptr::null_mut());
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

        let mut empty: Box<[ndarray::Array3<f32>]> = vec![].into();
        fwht::fwht(&mut data, &mut empty);

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

        let mut empty: Box<[ndarray::Array3<f32>]> = vec![].into();
        fwht::fwht(&mut data, &mut empty);
        fwht::fwht(&mut data, &mut empty);

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

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_slice_inverse_equality() {
        use crate::asset_reader_writer::pixel_buffer::*;
        use crate::asset_reader_writer::transform_block_3d::*;
        use crate::channel_coding::slice::{ChunkIterIntoExt, SliceIterExt};
        use asset_reader::*; // idk why this only works here..

        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let frame_resolution = reader.resolution().expect("Failed to get resolution.");
        let frame_resolution = (frame_resolution.0 as usize, frame_resolution.1 as usize);

        const LENGTH: usize = 8;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: original_cb_components,
            cr_components: original_cr_components,
        } = macro_block.clone();

        let mut y_dct = macro_block.y_components.into_dct();
        let mut cb_dct = macro_block.cb_components.into_dct();
        let mut cr_dct = macro_block.cr_components.into_dct();

        //         let original_y_dct = y_dct.clone();
        let y_slices: Box<_> = y_dct.chunks_iter().into_slice_iter(LENGTH).collect();
        let new_y_dct = y_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, std::iter::empty())
            .next()
            .expect("Failed to produce a Y 3D DCT");

        //         assert_eq!(original_y_dct, new_y_dct);

        let new_y_components = new_y_dct.into();

        let cb_slices: Box<_> = cb_dct.chunks_iter().into_slice_iter(LENGTH).collect();
        let new_cb_components = cb_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, std::iter::empty())
            .next()
            .expect("Failed to produce a Cb 3D DCT")
            .into();

        let cr_slices: Box<_> = cr_dct.chunks_iter().into_slice_iter(LENGTH).collect();
        let new_cr_components = cr_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, std::iter::empty())
            .next()
            .expect("Failed to produce a Cr 3D DCT")
            .into();

        // check the original pixel values, which will have floating point errors rounded
        assert_eq!(original_y_components, new_y_components);
        assert_eq!(original_cb_components, new_cb_components);
        assert_eq!(original_cr_components, new_cr_components);
    }
}
