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

use crate::asset_reader_writer::{transform_block_3d::*, *};
use ndarray;
use ndrustfft;

pub mod transform_block_3d_dct {
    use super::*;
    use chunk::*;

    #[derive(Debug)]
    pub struct TransformBlock3DDCT<const LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array3<f32>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    fn forwards_scale_factor(idx: usize, axis_len: usize) -> f32 {
        let n = axis_len as f32;
        match idx {
            0 => 1.0 / (2.0 * n.sqrt()),
            _ => 1.0 / (2.0 * n).sqrt(),
        }
    }

    fn backwards_scale_factor(idx: usize, axis_len: usize) -> f32 {
        1.0 / forwards_scale_factor(idx, axis_len)
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType>
        From<TransformBlock3D<LENGTH, PixelType>> for TransformBlock3DDCT<LENGTH, PixelType>
    {
        fn from(transform_block: TransformBlock3D<LENGTH, PixelType>) -> Self {
            let mut component_values = transform_block.consume_values();
            let (length, height, width) = component_values.dim();
            assert_eq!(length, LENGTH);

            for (axis_idx, axis_len) in [(0, length), (1, height), (2, width)] {
                let handler = ndrustfft::DctHandler::new(axis_len)
                    .normalization(ndrustfft::Normalization::None);
                ndrustfft::nddct2_inplace_par(&mut component_values, &handler, axis_idx);
            }
            let mut dct_values = component_values; // move

            // manual orthonormal normalization
            let i_scales: Box<[f32]> = (0..length)
                .map(|i| forwards_scale_factor(i, length))
                .collect();
            let j_scales: Box<[f32]> = (0..height)
                .map(|j| forwards_scale_factor(j, height))
                .collect();
            let k_scales: Box<[f32]> = (0..width)
                .map(|i| forwards_scale_factor(i, width))
                .collect();
            dct_values
                .indexed_iter_mut()
                .for_each(|((i, j, k), value)| *value *= i_scales[i] * j_scales[j] * k_scales[k]);

            TransformBlock3DDCT::<LENGTH, PixelType> {
                values: dct_values,
                _marker: std::marker::PhantomData,
            }
        }
    }

    fn max_factor_at_or_below(limit: usize, value: usize) -> usize {
        assert!(limit > 0);
        (1..=limit).rev().find(|i| value % i == 0).unwrap()
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType>
        transform_block_3d::TransformBlock3D<LENGTH, PixelType>
    {
        pub fn into_dct(self) -> TransformBlock3DDCT<LENGTH, PixelType> {
            TransformBlock3DDCT::from(self)
        }
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType>
        From<TransformBlock3DDCT<LENGTH, PixelType>>
        for transform_block_3d::TransformBlock3D<LENGTH, PixelType>
    {
        fn from(transform_block_dct: TransformBlock3DDCT<LENGTH, PixelType>) -> Self {
            let mut dct_values = transform_block_dct.consume_values();
            let (length, height, width) = dct_values.dim();
            assert_eq!(length, LENGTH);

            let i_scales: Box<[f32]> = (0..length)
                .map(|i| backwards_scale_factor(i, length))
                .collect();
            let j_scales: Box<[f32]> = (0..height)
                .map(|j| backwards_scale_factor(j, height))
                .collect();
            let k_scales: Box<[f32]> = (0..width)
                .map(|i| backwards_scale_factor(i, width))
                .collect();
            dct_values
                .indexed_iter_mut()
                .for_each(|((i, j, k), value)| *value *= i_scales[i] * j_scales[j] * k_scales[k]);

            for (axis_idx, axis_len) in [(0, length), (1, height), (2, width)] {
                let handler = ndrustfft::DctHandler::new(axis_len)
                    .normalization(ndrustfft::Normalization::None);
                ndrustfft::nddct3_inplace_par(&mut dct_values, &handler, axis_idx);
            }
            let mut component_values = dct_values; // move

            // rustdct applies a 2n scale factor per dimension
            let scale = 0.125 * (length * width * height) as f32;
            component_values.mapv_inplace(|value| value / scale);
            component_values.mapv_inplace(|value| value.round());

            transform_block_3d::TransformBlock3D::with_values(component_values)
        }
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType> TransformBlock3DDCT<LENGTH, PixelType> {
        pub fn consume_values(self) -> ndarray::Array3<f32> {
            self.values
        }

        fn power_scale(
            chunk_dimensions: (usize, usize, usize),
            chunk_energy: f32,
            chunk_energies: &[f32],
            sqrt_p_over_sum_sqrt_energies: &std::cell::OnceCell<f32>,
        ) -> f32 {
            // must check that return value is normal

            let num_chunks = chunk_energies.len();

            // average transmit power per chunk = 1
            let power_budget = num_chunks as f32
                / (chunk_dimensions.0 * chunk_dimensions.1 * chunk_dimensions.2) as f32;

            // skip math if energy is 0
            if chunk_energy.abs() < f32::EPSILON {
                return f32::NAN;
            }

            // scale each chunk by g_i = λ_i^(-0.25)*√(P/Σ_i(√λ_i))
            let sqrt_p_over_sum_sqrt_energies = sqrt_p_over_sum_sqrt_energies.get_or_init(|| {
                let sum_sqrt_energies: f32 = chunk_energies.iter().map(|&λ| λ.sqrt()).sum();
                (power_budget / sum_sqrt_energies).sqrt()
            });

            chunk_energy.powf(-0.25) * sqrt_p_over_sum_sqrt_energies
        }

        pub fn chunks_iter(&mut self) -> impl Iterator<Item = Chunk<'_, LENGTH, PixelType>> {
            const SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS: (usize, usize, usize) = (1, 30, 44);
            let (length, height, width) = self.values.dim();
            let chunk_length =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.0, length);
            let chunk_width =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.1, height);
            let chunk_height =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.2, width);

            let chunk_size = chunk_width * chunk_height * chunk_length;
            assert!(chunk_size > 0);
            assert_eq!((length * width * height) % chunk_size, 0);

            let chunk_dimensions = (chunk_length, chunk_width, chunk_height);

            // must preflight mutatation of the 3D DCT, because power scaling requires all energies.
            let means: Box<[f32]> = self
                .values
                .exact_chunks(chunk_dimensions)
                .into_iter()
                // .par_bridge() <-- parallelism doesn't seem to help for O(n) operations.
                .map(|chunk| chunk.mean().unwrap()) // chunk should always be nonempty
                .collect();

            // Softcast specifies a zero-mean distribution
            self.values
                .exact_chunks_mut(chunk_dimensions)
                .into_iter()
                .zip(means.iter())
                // .par_bridge() <-- parallelism doesn't seem to help for O(n) operations.
                .for_each(|(mut chunk, &mean)| *chunk -= mean);

            // compute energy/variance after mean has been substracted, since we have to take the mean anyway
            let energies: Box<[f32]> = self
                .values
                .exact_chunks(chunk_dimensions)
                .into_iter()
                //.par_bridge() <-- makes test_reader_to_chunked_dct_inverse_equality fail for some reason
                .map(|chunk| chunk.pow2().sum() / chunk.len() as f32)
                .collect();

            let compute_cache = std::cell::OnceCell::new();
            energies
                .iter()
                .zip(self.values.exact_chunks_mut(chunk_dimensions))
                .for_each(|(&energy, mut chunk)| {
                    let power_scale =
                        Self::power_scale(chunk_dimensions, energy, &energies, &compute_cache);
                    // needs a comment
                    if power_scale.is_normal() {
                        chunk *= power_scale;
                    }
                });

            //          TODO: Make distortion iterator
            //             {
            //                 use rand::Rng;
            //                 let mut rng = rand::rng();
            //                 for value in self.values.iter_mut() {
            //                     let distortion = rng.random_range(-10.0..10.1);
            //                     *value += distortion;
            //                 }
            //             }

            // two passes necessary due to the impossibility to coerce 'chunks' in the previous loop to an immutable borrow
            let chunked_transform_blocks = self
                .values
                .exact_chunks_mut(chunk_dimensions)
                .into_iter()
                .zip(means.into_iter())
                .zip(energies.into_iter())
                .map(|((chunk, mean), energy)| Chunk::new(chunk, ChunkMetadata { mean, energy }));

            // TODO: Add option to sort chunks by energy, for compression

            chunked_transform_blocks
        }

        pub(super) fn from_chunks(
            chunks: &[Chunk<'_, LENGTH, PixelType>],
            frame_resolution: (usize, usize),
        ) -> Self {
            let (dct_length, dct_height, dct_width) =
                (LENGTH, frame_resolution.1, frame_resolution.0);
            let mut values = ndarray::Array3::zeros((dct_length, dct_height, dct_width)); // TODO: eliminiate copy
            let chunk_dimensions = chunks.first().expect("chunks is empty").values.dim();

            assert_eq!(
                dct_length * dct_height * dct_width,
                chunks.len() * chunk_dimensions.0 * chunk_dimensions.1 * chunk_dimensions.2
            );

            let chunk_energies: Box<[f32]> =
                chunks.iter().map(|chunk| chunk.metadata.energy).collect();
            let compute_cache = std::cell::OnceCell::new();

            // Could be optimized by iterating over dst or src in memory order.
            values
                .exact_chunks_mut(chunk_dimensions)
                .into_iter()
                .zip(chunks)
                // .par_bridge() <-- parallelism doesn't seem to help for O(n) operations.
                .for_each(|(mut dst, src)| {
                    let power_scale = Self::power_scale(
                        chunk_dimensions,
                        src.metadata.energy,
                        &chunk_energies,
                        &compute_cache,
                    );
                    if power_scale.is_normal() {
                        dst.assign(&src.values);
                        dst /= power_scale;
                    } // else assume all zeros

                    dst += src.metadata.mean;
                });

            TransformBlock3DDCT::<LENGTH, PixelType> {
                values: values,
                _marker: std::marker::PhantomData,
            }
        }
    }
}

pub mod chunk {
    use super::transform_block_3d_dct::*;
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Default)]
    pub struct ChunkMetadata {
        pub mean: f32,
        pub energy: f32,
    }

    pub struct Chunk<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::ArrayViewMut3<'a, f32>,
        pub metadata: ChunkMetadata,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>
        Chunk<'a, DCT_LENGTH, PixelType>
    {
        pub fn new(values: ndarray::ArrayViewMut3<'a, f32>, metadata: ChunkMetadata) -> Self {
            Chunk {
                values,
                metadata,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct TransformBlock3DDCTIter<
        'a,
        const DCT_LENGTH: usize,
        PixelType: HasPixelComponentType,
        I,
    >
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        chunk_iter: I,
        frame_resolution: (usize, usize),
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        TransformBlock3DDCTIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        fn new(chunk_iter: I, frame_resolution: (usize, usize)) -> Self {
            let (frame_width, frame_height) = frame_resolution;
            let pixel_type = PixelType::TYPE;
            let component_frame_resolution = (
                frame_width / pixel_type.interleave_step(),
                frame_height / pixel_type.vertical_subsampling(),
            );
            TransformBlock3DDCTIter {
                chunk_iter: chunk_iter,
                frame_resolution: component_frame_resolution,
            }
        }
    }

    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for TransformBlock3DDCTIter<'a, DCT_LENGTH, PixelType, I>
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        type Item = TransformBlock3DDCT<DCT_LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            use std::cell::OnceCell;

            let num_transform_block_3d_dct_values =
                DCT_LENGTH * self.frame_resolution.0 * self.frame_resolution.1;
            let chunks_needed = OnceCell::new();
            let chunk_dim = OnceCell::new();
            let mut chunks_to_consume = OnceCell::new();
            let mut chunk_iter_is_empty = true;
            loop {
                let chunk = self.chunk_iter.next();

                if chunk.is_none() {
                    return match chunk_iter_is_empty {
                        true => None,
                        false => panic!("Not enough chunks to form a TransformBlock3DDCT."),
                    };
                }
                chunk_iter_is_empty = false;
                let chunk = chunk.unwrap();

                let chunk_dim = chunk_dim.get_or_init(|| chunk.values.dim());
                assert_eq!(*chunk_dim, chunk.values.dim());

                let chunks_needed = chunks_needed.get_or_init(|| {
                    let num_chunk_values =
                        chunk.values.dim().0 * chunk.values.dim().1 * chunk.values.dim().2;
                    assert_eq!(DCT_LENGTH % chunk.values.dim().0, 0);
                    assert_eq!(self.frame_resolution.0 % chunk.values.dim().2, 0); // width
                    assert_eq!(self.frame_resolution.1 % chunk.values.dim().1, 0); // height
                    assert_eq!(num_transform_block_3d_dct_values % num_chunk_values, 0);
                    num_transform_block_3d_dct_values / num_chunk_values
                });
                let _ = chunks_to_consume.get_or_init(|| Vec::with_capacity(*chunks_needed));
                let chunks_to_consume = chunks_to_consume.get_mut().unwrap();

                chunks_to_consume.push(chunk);

                if *chunks_needed == chunks_to_consume.len() {
                    let transform_block_3d_dct =
                        TransformBlock3DDCT::from_chunks(chunks_to_consume, self.frame_resolution);
                    return Some(transform_block_3d_dct);
                }
            }
        }
    }

    pub trait ChunkIterFromExt<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>:
        Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>> + Sized
    {
        fn into_transform_block_3d_dct_iter(
            self,
            frame_resolution: (usize, usize),
        ) -> TransformBlock3DDCTIter<'a, DCT_LENGTH, PixelType, Self>;
    }
    impl<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType, I>
        ChunkIterFromExt<'a, DCT_LENGTH, PixelType> for I
    where
        I: Iterator<Item = Chunk<'a, DCT_LENGTH, PixelType>>,
    {
        fn into_transform_block_3d_dct_iter(
            self,
            frame_resolution: (usize, usize),
        ) -> TransformBlock3DDCTIter<'a, DCT_LENGTH, PixelType, Self> {
            TransformBlock3DDCTIter::new(self, frame_resolution)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_reader::*;
    #[cfg(not(debug_assertions))]
    use asset_writer::*;
    use chunk::*;
    use pixel_buffer::*;
    use std::fs;
    #[cfg(not(debug_assertions))]
    use std::path;
    use transform_block_3d_dct::*;

    #[test]
    fn test_print_3d_dct() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);
        const LENGTH: usize = 18;

        let transform_block_3d_dct: TransformBlock3DDCT<LENGTH, YPixelComponentType> = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator()
            .map(|macro_block| macro_block.y_components.into_dct())
            .next()
            .expect("No DCT performed.");

        eprintln!("Got transform_block_3d_dct: {:?}", transform_block_3d_dct);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_print_chunk_means() {
        let path = "sample-media/sample-5s.mp4";
        let mut reader = AssetReader::new(path);

        let mut transform_block_3d_dct: TransformBlock3DDCT<4, YPixelComponentType> = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator()
            .map(|macro_block| macro_block.y_components.into_dct())
            .next()
            .expect("No DCT performed.");

        for Chunk {
            values: _,
            metadata: ChunkMetadata { mean, .. },
            ..
        } in transform_block_3d_dct.chunks_iter()
        {
            eprintln!("mean:{}", mean);
        }
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_transform_block_3d_dct_to_writer() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let output_path = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = fs::remove_file(output_path);
        let writer_settings = AssetWritterSettings {
            path: path::PathBuf::from(output_path),
            codec: Codec::H264,
            resolution: reader.resolution().expect("Failed to get resolution."),
            frame_rate: reader.frame_rate().expect("Failed to get frame rate"),
        };
        let mut writer = AssetWriter::load_new(writer_settings).expect("Failed to load writer");
        writer.start_writing().expect("Failed to start writing");

        const LENGTH: usize = 8;
        let macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let mut pixel_buffers_consumed = 0;

        for macro_block in macro_block_3d_iterator {
            let y_dct = macro_block.y_components.into_dct();
            let cb_dct = macro_block.cb_components.into_dct();
            let cr_dct = macro_block.cr_components.into_dct();

            let y_components: TransformBlock3D<LENGTH, YPixelComponentType> = y_dct.into();
            let cb_components: TransformBlock3D<LENGTH, CbPixelComponentType> = cb_dct.into();
            let cr_components: TransformBlock3D<LENGTH, CrPixelComponentType> = cr_dct.into();

            let new_macro_block = MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
            };

            let pixel_buffer_iterator = [new_macro_block].into_iter().pixel_buffer_iter();

            for pixel_buffer in pixel_buffer_iterator {
                pixel_buffers_consumed += 1;

                writer
                    .append_pixel_buffer(pixel_buffer)
                    .expect("Failed to append pixel buffer");
                writer
                    .wait_for_writer_to_be_ready()
                    .expect("Failed to become ready after writing some pixels.");
            }
        }
        writer.finish_writing().expect("Failed to finish writing.");
        assert_eq!(pixel_buffers_consumed, 301 + LENGTH - (301 % LENGTH)); // TODO: No mechanism to signal empty frames.
    }

    #[test]
    fn test_reader_to_dct2_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let output_path = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = fs::remove_file(output_path);

        const LENGTH: usize = 2;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");
        let MacroBlock3D {
            y_components: original_y_components,
            cb_components: original_cb_components,
            cr_components: original_cr_components,
        } = macro_block.clone();

        let y_dct = macro_block.y_components.into_dct();
        let cb_dct = macro_block.cb_components.into_dct();
        let cr_dct = macro_block.cr_components.into_dct();

        eprintln!(
            "max:{} {:?}",
            y_dct
                .values
                .map(|v| v.abs())
                .iter()
                .copied()
                .reduce(f32::max)
                .unwrap(),
            y_dct
        );

        let new_y_components: TransformBlock3D<LENGTH, YPixelComponentType> = y_dct.into();
        let new_cb_components: TransformBlock3D<LENGTH, CbPixelComponentType> = cb_dct.into();
        let new_cr_components: TransformBlock3D<LENGTH, CrPixelComponentType> = cr_dct.into();

        assert_eq!(original_y_components, new_y_components);
        assert_eq!(original_cb_components, new_cb_components);
        assert_eq!(original_cr_components, new_cr_components);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_chunks_to_writer() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let output_path = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = fs::remove_file(output_path);
        let frame_resolution = reader.resolution().expect("Failed to get resolution.");
        let writer_settings = AssetWritterSettings {
            path: path::PathBuf::from(output_path),
            codec: Codec::H264,
            resolution: frame_resolution,
            frame_rate: reader.frame_rate().expect("Failed to get frame rate"),
        };
        let mut writer = AssetWriter::load_new(writer_settings).expect("Failed to load writer");
        writer.start_writing().expect("Failed to start writing");

        const LENGTH: usize = 8;
        let macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let mut pixel_buffers_consumed = 0;

        for macro_block in macro_block_3d_iterator {
            // and back again

            let frame_resolution = (frame_resolution.0 as usize, frame_resolution.1 as usize);

            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
            } = macro_block;

            let mut y_dct = y_components.into_dct();
            let mut cb_dct = cb_components.into_dct();
            let mut cr_dct = cr_components.into_dct();

            let mut y_dct_iter = y_dct
                .chunks_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let y_dct = y_dct_iter.next().expect("Failed to recreate Y 3D DCT.");
            assert!(y_dct_iter.next().is_none());

            let mut cb_dct_iter = cb_dct
                .chunks_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let cb_dct = cb_dct_iter.next().expect("Failed to recreate Cb 3D DCT.");
            assert!(cb_dct_iter.next().is_none());

            let mut cr_dct_iter = cr_dct
                .chunks_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let cr_dct = cr_dct_iter.next().expect("Failed to recreate Cr 3D DCT.");
            assert!(cr_dct_iter.next().is_none());

            let new_macro_block = MacroBlock3D::<LENGTH> {
                y_components: y_dct.into(),
                cb_components: cb_dct.into(),
                cr_components: cr_dct.into(),
            };

            let pixel_buffer_iterator = [new_macro_block].into_iter().pixel_buffer_iter();

            for pixel_buffer in pixel_buffer_iterator {
                pixel_buffers_consumed += 1;

                writer
                    .append_pixel_buffer(pixel_buffer)
                    .expect("Failed to append pixel buffer");
                writer
                    .wait_for_writer_to_be_ready()
                    .expect("Failed to become ready after writing some pixels.");
            }
        }
        writer.finish_writing().expect("Failed to finish writing.");
        assert_eq!(pixel_buffers_consumed, 301 + LENGTH - (301 % LENGTH)); // TODO: No mechanism to signal empty frames.
    }

    #[test]
    fn test_reader_to_chunked_dct_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let frame_resolution = reader.resolution().expect("Failed to get resolution.");
        let frame_resolution = (frame_resolution.0 as usize, frame_resolution.1 as usize);

        let output_path = "/tmp/bipbop-1920x1080-5s.mp4";
        let _ = fs::remove_file(output_path);

        const LENGTH: usize = 2;
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

        let new_y_dct = y_dct
            .chunks_iter()
            .into_transform_block_3d_dct_iter(frame_resolution)
            .next()
            .expect("Failed to produce a Y 3D DCT");

        //         assert_eq!(original_y_dct, new_y_dct);

        let new_y_components = new_y_dct.into();

        let new_cb_components = cb_dct
            .chunks_iter()
            .into_transform_block_3d_dct_iter(frame_resolution)
            .next()
            .expect("Failed to produce a Cb 3D DCT")
            .into();

        let new_cr_components = cr_dct
            .chunks_iter()
            .into_transform_block_3d_dct_iter(frame_resolution)
            .next()
            .expect("Failed to produce a Cr 3D DCT")
            .into();

        // check the original pixel values, which will have floating point errors rounded
        assert_eq!(original_y_components, new_y_components);
        assert_eq!(original_cb_components, new_cb_components);
        assert_eq!(original_cr_components, new_cr_components);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_count_zero_valued_chunk() {
        let path1 = "sample-media/bipbop-1920x1080-5s.mp4";
        let path2 = "sample-media/sample-5s.mp4";

        fn print_zero_values(path: &'static str) {
            let mut reader = AssetReader::new(path);

            const LENGTH: usize = 30;
            let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
                reader.pixel_buffer_iter().macro_block_3d_iterator();

            let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

            let MacroBlock3D {
                y_components,
                cb_components,
                cr_components,
            } = macro_block;

            fn count_zero_values<PixelType: HasPixelComponentType>(
                transform_block_3d_dct: &mut TransformBlock3DDCT<LENGTH, PixelType>,
            ) -> (usize, usize, f32, f32) {
                transform_block_3d_dct.chunks_iter().fold(
                    (0, 0, 0f32, 0f32),
                    |(zero_values, total_values, max_variance, max_value), chunk| {
                        let zero_values = zero_values
                            + match chunk.values.sum() {
                                value if value.abs() < 0.01 => 1, // TODO: RMS of energy to figure out a decent threshold.
                                _ => 0,
                            };
                        let total_values = total_values + 1;
                        let max_variance = max_variance.max(chunk.values.var(3.0));
                        let max_value =
                            max_value.max(chunk.values.iter().copied().reduce(f32::max).unwrap());
                        (zero_values, total_values, max_variance, max_value)
                    },
                )
            }

            let mut y_components_dct = y_components.into_dct();
            let mut cb_components_dct = cb_components.into_dct();
            let mut cr_components_dct = cr_components.into_dct();

            let (y_zero_values, y_total_values, y_max_variance, y_max_value) =
                count_zero_values(&mut y_components_dct);
            let (cb_zero_values, cb_total_values, cb_max_variance, cb_max_value) =
                count_zero_values(&mut cb_components_dct);
            let (cr_zero_values, cr_total_values, cr_max_variance, cr_max_value) =
                count_zero_values(&mut cr_components_dct);

            eprintln!(
                "y_zero_values:{} total_values:{} {:.3}% max_σ:{} max_value:{}",
                y_zero_values,
                y_total_values,
                100.0 * y_zero_values as f64 / y_total_values as f64,
                f32::sqrt(y_max_variance),
                y_max_value
            );
            eprintln!(
                "cb_zero_values:{} total_values:{} {:.3}% max_σ:{} max_value:{}",
                cb_zero_values,
                cb_total_values,
                100.0 * cb_zero_values as f64 / cb_total_values as f64,
                f32::sqrt(cb_max_variance),
                cb_max_value
            );
            eprintln!(
                "cr_zero_values:{} total_values:{} {:.3}% max_σ:{} max_value:{}",
                cr_zero_values,
                cr_total_values,
                100.0 * cr_zero_values as f64 / cr_total_values as f64,
                f32::sqrt(cr_max_variance),
                cr_max_value
            );
        }

        eprintln!("BipBop");
        print_zero_values(path1);
        eprintln!();
        eprintln!("sample-5s");
        print_zero_values(path2);
    }
}
