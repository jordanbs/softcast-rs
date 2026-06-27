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

use crate::pixel_buffer::*;
use crate::source_coding::chunk::*;
use crate::utils::*;
use fwht;
use slice::*;

pub mod slice {
    use super::*;

    // TODO: there is no reason for slices to carry chunk metadata
    pub struct SliceAndChunkMetadata<'a, PixelType: HasPixelComponentType> {
        pub slice: Slice<'a, PixelType>,
        pub chunk_metadata: ChunkMetadata,
    }
    impl<'a, PixelType: HasPixelComponentType> SliceAndChunkMetadata<'a, PixelType> {
        pub fn new(slice: Slice<'a, PixelType>, chunk_metadata: ChunkMetadata) -> Self {
            Self {
                slice,
                chunk_metadata,
            }
        }
    }

    pub struct Slice<'a, PixelType: HasPixelComponentType> {
        pub values: ViewOrOwnedArray3<'a>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, PixelType: HasPixelComponentType> Slice<'a, PixelType> {
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
        pub fn from_owned_arc(owned_arc: ndarray::ArcArray<f32, ndarray::Ix3>) -> Self {
            Self::new(ViewOrOwnedArray3::OwnedArc(owned_arc))
        }

        pub fn values(&self) -> ndarray::ArrayView3<'_, f32> {
            match &self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view(),
                ViewOrOwnedArray3::OwnedArc(owned) => owned.view(),
            }
        }

        pub fn values_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
            match &mut self.values {
                ViewOrOwnedArray3::View(view) => view.into(),
                ViewOrOwnedArray3::Owned(owned) => owned.view_mut(),
                ViewOrOwnedArray3::OwnedArc(owned) => owned.view_mut(),
            }
        }
    }

    pub struct SliceIter<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = Chunk<'a, PixelType>>,
    > {
        chunk_iter: std::iter::Peekable<I>,
        inner_slice_iter: std::vec::IntoIter<SliceAndChunkMetadata<'a, PixelType>>,
        chunks_per_gop: usize,
    }

    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Chunk<'a, PixelType>>>
        SliceIter<'a, PixelType, I>
    {
        pub fn new(chunk_iter: I, chunks_per_gop: usize) -> Self {
            SliceIter {
                chunk_iter: chunk_iter.peekable(),
                inner_slice_iter: vec![].into_iter(),
                chunks_per_gop,
            }
        }
    }
    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Chunk<'a, PixelType>>> Iterator
        for SliceIter<'a, PixelType, I>
    {
        type Item = SliceAndChunkMetadata<'a, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                if let Some(slice) = self.inner_slice_iter.next() {
                    return Some(slice);
                }

                let chunks: Box<_> = self.chunk_iter.by_ref().take(self.chunks_per_gop).collect();

                if chunks.is_empty() {
                    return None;
                }

                let slices =
                    fwht_softcast::fwht_chunks_copy(chunks).expect("Failed to create slices.");
                self.inner_slice_iter = slices.into_iter();
            }
        }
    }

    pub struct ChunkIter<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = SliceAndChunkMetadata<'a, PixelType>>,
    > {
        slice_iter: std::iter::Peekable<I>,
        inner_chunk_iter: std::vec::IntoIter<Chunk<'a, PixelType>>,
        chunks_per_gop: usize,
    }

    impl<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = SliceAndChunkMetadata<'a, PixelType>>,
    > ChunkIter<'a, PixelType, I>
    {
        pub fn new(slice_iter: I, chunks_per_gop: usize) -> Self {
            ChunkIter {
                slice_iter: slice_iter.peekable(),
                inner_chunk_iter: vec![].into_iter(),
                chunks_per_gop,
            }
        }
    }

    impl<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = SliceAndChunkMetadata<'a, PixelType>>,
    > Iterator for ChunkIter<'a, PixelType, I>
    {
        type Item = Chunk<'a, PixelType>;

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

                let chunks = fwht_softcast::fwht_slices(slices, hadamard_len - self.chunks_per_gop)
                    .expect("Failed to create chunks.");
                self.inner_chunk_iter = chunks.into_iter();
            }
        }
    }

    pub trait ChunkIterIntoExt<'a, PixelType: HasPixelComponentType>:
        Iterator<Item = Chunk<'a, PixelType>> + Sized
    {
        fn into_slice_iter(self, chunks_per_gop: usize) -> SliceIter<'a, PixelType, Self>;
    }
    impl<'a, PixelType: HasPixelComponentType, I: Iterator<Item = Chunk<'a, PixelType>>>
        ChunkIterIntoExt<'a, PixelType> for I
    {
        fn into_slice_iter(self, chunks_per_gop: usize) -> SliceIter<'a, PixelType, Self> {
            SliceIter::new(self, chunks_per_gop)
        }
    }

    pub trait SliceIterExt<'a, PixelType: HasPixelComponentType>:
        Iterator<Item = SliceAndChunkMetadata<'a, PixelType>> + Sized
    {
        fn into_chunks_iter(self, chunks_per_gop: usize) -> ChunkIter<'a, PixelType, Self>;
    }
    impl<
        'a,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = SliceAndChunkMetadata<'a, PixelType>>,
    > SliceIterExt<'a, PixelType> for I
    {
        fn into_chunks_iter(self, chunks_per_gop: usize) -> ChunkIter<'a, PixelType, Self> {
            ChunkIter::new(self, chunks_per_gop)
        }
    }
}

pub mod fwht_softcast {
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

    impl<PixelType: HasPixelComponentType> ValuesProvider for Chunk<'_, PixelType> {
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
    impl<PixelType: HasPixelComponentType> ValuesProvider for Slice<'_, PixelType> {
        fn value_at(&self, idx: usize) -> f32 {
            let idx = idx.to_3dim_index(self.values().dim());
            self.values()[idx]
        }
        fn ptr_at(&self, idx: usize) -> *mut f32 {
            let idx = idx.to_3dim_index(self.values().dim());
            let value = match &self.values {
                ViewOrOwnedArray3::View(view) => &view[idx],
                ViewOrOwnedArray3::Owned(owned) => &owned[idx],
                ViewOrOwnedArray3::OwnedArc(owned_arc) => &owned_arc[idx],
            };
            let ptr: *const f32 = value;
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.values().len()
        }
    }
    impl<PixelType: HasPixelComponentType> ValuesProvider for SliceAndChunkMetadata<'_, PixelType> {
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
            let ptr = &raw const self[idx];
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
            let ptr = &raw const self[idx];
            ptr as *mut f32
        }
        fn values_len(&self) -> usize {
            self.len()
        }
    }

    impl<PixelType: HasPixelComponentType> std::ops::MulAssign<f32> for Chunk<'_, PixelType> {
        fn mul_assign(&mut self, rhs: f32) {
            self.values.mul_assign(rhs);
        }
    }
    impl<PixelType: HasPixelComponentType> std::ops::MulAssign<f32> for Slice<'_, PixelType> {
        fn mul_assign(&mut self, rhs: f32) {
            self.values_mut().mul_assign(rhs);
        }
    }
    impl<PixelType: HasPixelComponentType> std::ops::MulAssign<f32>
        for SliceAndChunkMetadata<'_, PixelType>
    {
        fn mul_assign(&mut self, rhs: f32) {
            self.slice.mul_assign(rhs);
        }
    }

    unsafe impl<PixelType: HasPixelComponentType> Send for Chunk<'_, PixelType> {}
    unsafe impl<PixelType: HasPixelComponentType> Send for Slice<'_, PixelType> {}

    unsafe impl<PixelType: HasPixelComponentType> Sync for Chunk<'_, PixelType> {}
    unsafe impl<PixelType: HasPixelComponentType> Sync for Slice<'_, PixelType> {}

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

    pub fn fwht_chunks<PixelType: HasPixelComponentType>(
        chunks: Box<[Chunk<'_, PixelType>]>,
    ) -> Result<Box<[SliceAndChunkMetadata<'_, PixelType>]>, &'static str> {
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

    pub fn fwht_chunks_copy<PixelType: HasPixelComponentType>(
        chunks: Box<[Chunk<'_, PixelType>]>,
    ) -> Result<Box<[SliceAndChunkMetadata<'_, PixelType>]>, &'static str> {
        // adapted from fwht crate, with the intention of making copies
        // add padding so each fwht is a power of two
        let hadamard_len = chunks.len().next_power_of_two();

        // each chunk is spread on the major axis, such that fwht is performed on the minor axis.
        let chunk_dim = chunks.first().expect("no data").values.raw_dim();
        let num_pixels_per_chunk = chunk_dim[0] * chunk_dim[1] * chunk_dim[2];
        let mut new_alloc = ndarray::Array2::<f32>::zeros((num_pixels_per_chunk, hadamard_len));
        for (col, chunk) in chunks.iter().enumerate() {
            for (mut dst, src) in new_alloc
                .axis_iter_mut(ndarray::Axis(0))
                .zip(chunk.values.iter())
            {
                dst[col] = *src;
            }
        }

        for mut row_view in new_alloc.outer_iter_mut() {
            let mut memory_slice = row_view.as_slice_mut().expect("Not in memory order.");
            fwht::fwht_slice(&mut memory_slice)?;
            let orthonormalization_factor = 1f32 / (hadamard_len as f32).sqrt();
            memory_slice
                .iter_mut()
                .for_each(|elm| *elm *= orthonormalization_factor);
        }

        // transpose the axis for slices to be arranged in memory order
        let new_alloc = new_alloc.t().as_standard_layout().to_owned().into_shared();
        assert!(new_alloc.is_standard_layout());

        let mut slices = Vec::with_capacity(hadamard_len);
        let mut chunks_iter = chunks.iter();
        for slice_idx in 0..hadamard_len {
            let row_view = new_alloc
                .clone() // refcount++
                .index_axis_move(ndarray::Axis(0), slice_idx);
            assert!(row_view.is_standard_layout());

            let reshaped_view = row_view
                .into_shape_with_order(chunk_dim)
                .expect("Reshape failed.");
            let slice: Slice<'_, PixelType> = Slice::from_owned_arc(reshaped_view);
            let chunk_metadata = chunks_iter
                .next()
                .map(|chunk| chunk.metadata)
                .unwrap_or_default();
            let slice_and_chunk_metadata = SliceAndChunkMetadata::new(slice, chunk_metadata);
            slices.push(slice_and_chunk_metadata);
        }

        return Ok(slices.into());
    }

    // no copy version atm
    pub fn fwht_slices<'a, PixelType: HasPixelComponentType>(
        slices: Box<[SliceAndChunkMetadata<'a, PixelType>]>,
        num_padding_rows: usize,
    ) -> Result<Box<[Chunk<'a, PixelType>]>, &'static str> {
        let mut slices = slices;

        let mut empty: Box<[ndarray::Array3<f32>]> = vec![].into();
        fwht(&mut slices, &mut empty);

        let mut chunks = Vec::with_capacity(slices.len() - num_padding_rows);
        for slice in slices {
            // consume slice.values
            let chunk: Chunk<'a, PixelType> = match slice.slice.values {
                ViewOrOwnedArray3::View(view) => Chunk::new(view, slice.chunk_metadata),
                ViewOrOwnedArray3::Owned(_) => {
                    // TODO: This assumption might not be true in the testing loopback.
                    panic!("slice not expected to own its data in decode.")
                }
                ViewOrOwnedArray3::OwnedArc(owned_arc) => {
                    Chunk::with_owned_arc(owned_arc, slice.chunk_metadata)
                }
            };

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
        fwht_softcast::fwht(&mut data, &mut empty);

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
        fwht_softcast::fwht(&mut data, &mut padding);

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
        fwht_softcast::fwht(&mut data, &mut empty);
        fwht_softcast::fwht(&mut data, &mut empty);

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
        fwht_softcast::fwht(&mut data, &mut padding);
        fwht_softcast::fwht(&mut data, &mut padding);

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
    fn test_fwht_chunks_copy() {
        let num_chunks = 9;
        let chunk_len = 5;
        let mut all_values_1 = ndarray::Array3::zeros((num_chunks, 1, chunk_len));
        for mut subview in all_values_1.outer_iter_mut() {
            for (idx, elm) in subview.iter_mut().enumerate() {
                *elm = idx as f32;
            }
        }
        let mut all_values_2 = all_values_1.clone();

        let chunks_1: Box<_> = all_values_1
            .exact_chunks_mut((1, 1, chunk_len))
            .into_iter()
            .map(|chunk_values| {
                Chunk::<'_, YPixelComponentType>::new(chunk_values, ChunkMetadata::default())
            })
            .collect();
        let chunks_2: Box<_> = all_values_2
            .exact_chunks_mut((1, 1, chunk_len))
            .into_iter()
            .map(|chunk_values| {
                Chunk::<'_, YPixelComponentType>::new(chunk_values, ChunkMetadata::default())
            })
            .collect();

        let slices_1 = fwht_softcast::fwht_chunks(chunks_1).expect("fwht_chunks failed");
        let slices_2 = fwht_softcast::fwht_chunks_copy(chunks_2).expect("fwht_chunks_copy failed");

        for (slice_1, slice_2) in slices_1.iter().zip(slices_2.iter()) {
            assert_eq!(
                slice_1.slice.values(),
                slice_2.slice.values(),
                "slices_1: {:?}\n slices_2: {:?}",
                slices_1
                    .iter()
                    .map(|slice| slice.slice.values())
                    .collect::<Vec<_>>(),
                slices_2
                    .iter()
                    .map(|slice| slice.slice.values())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn test_reader_to_slice_inverse_equality() {
        use crate::asset_reader_writer::asset_reader::*;
        use crate::channel_coding::slice::{ChunkIterIntoExt, SliceIterExt};
        use crate::pixel_buffer::transform_block_3d::*;
        use crate::pixel_buffer::*;

        let path = "sample-media/bipbop-768x432-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        let frame_resolution = reader.resolution().expect("Failed to get resolution.");
        let frame_resolution = (frame_resolution.0 as usize, frame_resolution.1 as usize);

        const LENGTH: usize = 2;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<_, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: original_cb_components,
            cr_components: original_cr_components,
            ..
        } = macro_block.clone();

        let mut y_dct = macro_block.y_components.into_dct();
        let mut cb_dct = macro_block.cb_components.into_dct();
        let mut cr_dct = macro_block.cr_components.into_dct();

        //         let original_y_dct = y_dct.clone();
        let y_slices: Box<_> = y_dct
            .chunks_iter((1, 36, 48))
            .into_slice_iter(LENGTH)
            .collect();
        let new_y_dct = y_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, LENGTH)
            .next()
            .expect("Failed to produce a Y 3D DCT");

        //         assert_eq!(original_y_dct, new_y_dct);

        let new_y_components = new_y_dct.into();

        let cb_slices: Box<_> = cb_dct
            .chunks_iter((1, 27, 32))
            .into_slice_iter(LENGTH)
            .collect();
        let new_cb_components = cb_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, LENGTH)
            .next()
            .expect("Failed to produce a Cb 3D DCT")
            .into();

        let cr_slices: Box<_> = cr_dct
            .chunks_iter((1, 27, 32))
            .into_slice_iter(LENGTH)
            .collect();
        let new_cr_components = cr_slices
            .into_iter()
            .into_chunks_iter(LENGTH)
            .into_transform_block_3d_dct_iter(frame_resolution, LENGTH)
            .next()
            .expect("Failed to produce a Cr 3D DCT")
            .into();

        // check the original pixel values, which will have floating point errors rounded
        assert_eq!(original_y_components, new_y_components);
        assert_eq!(original_cb_components, new_cb_components);
        assert_eq!(original_cr_components, new_cr_components);
    }
}
