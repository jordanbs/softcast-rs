// Copyright 2025 Jordan Schneider
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

#[derive(Debug)]
pub struct TransformBlock3DDCT {
    pub values: ndarray::Array3<f32>,
}

impl<const LENGTH: usize, PixelType: HasPixelComponentType>
    From<TransformBlock3D<LENGTH, PixelType>> for TransformBlock3DDCT
{
    fn from(transform_block: TransformBlock3D<LENGTH, PixelType>) -> Self {
        let input_values = transform_block.consume_values();
        let (length, width, height) = input_values.dim();
        assert_eq!(length, LENGTH);

        let mut output_a = input_values;
        let mut output_b = ndarray::Array3::zeros(output_a.raw_dim());

        for (axis_idx, axis_len) in [(0, length), (1, width), (2, height)] {
            let handler = ndrustfft::DctHandler::new(axis_len);
            ndrustfft::nddct2(&output_a, &mut output_b, &handler, axis_idx);

            std::mem::swap(&mut output_a, &mut output_b);
        }

        TransformBlock3DDCT { values: output_a }
    }
}

fn max_factor_at_or_below(limit: usize, value: usize) -> usize {
    assert!(limit > 0);
    (1..=limit).rev().find(|i| value % i == 0).unwrap()
}

impl TransformBlock3DDCT {
    pub fn chunks_iter(&mut self) -> impl Iterator<Item = ChunkedTransformBlock3D<'_>> {
        const SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS: (usize, usize, usize) = (1, 44, 30);
        let (length, width, height) = self.values.dim();
        let chunk_length = max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.0, length);
        let chunk_width = max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.1, width);
        let chunk_height = max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.2, height);

        let chunk_size = chunk_width * chunk_height * chunk_length;
        assert!(chunk_size > 0);
        assert_eq!((length * width * height) % chunk_size, 0);
        let num_chunks = (length * width * height) / chunk_size;

        let chunk_dimensions = (chunk_length, chunk_width, chunk_height);

        // must preflight mutatation of the 3D DCT, because we are going to be giving out immutable borrows.
        let mut chunked_transform_blocks = Vec::with_capacity(num_chunks);
        let mut means = Vec::with_capacity(num_chunks);

        for mut chunk in self.values.exact_chunks_mut(chunk_dimensions) {
            let mean = chunk.mean().unwrap(); // chunk should always be nonempty

            chunk.iter_mut().for_each(|value| *value -= mean); // Softcast specifies a zero-mean distribution

            means.push(mean);
        }

        // two passes necessary due to the impossibility to coerce 'chunks' in the previous loop to an immutable borrow
        for (chunk, mean) in self
            .values
            .exact_chunks(chunk_dimensions)
            .into_iter()
            .zip(means.into_iter())
        {
            chunked_transform_blocks.push(ChunkedTransformBlock3D::new(chunk, mean));
        }
        chunked_transform_blocks.into_iter()
    }
}

pub struct ChunkedTransformBlock3D<'a> {
    values: ndarray::ArrayView3<'a, f32>,
    mean: f32,
}

impl<'a> ChunkedTransformBlock3D<'a> {
    fn new(values: ndarray::ArrayView3<'a, f32>, mean: f32) -> Self {
        ChunkedTransformBlock3D { values, mean }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_reader::*;

    #[test]
    //     #[cfg(not(debug_assertions))] // too slow on debug
    fn test_print_3d_dct() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let transform_block_3d_dct: TransformBlock3DDCT = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator::<18>()
            .map(|macro_block| macro_block.y_components)
            .map(|transform_block| TransformBlock3DDCT::from(transform_block))
            .next()
            .expect("No DCT performed.");

        eprintln!("Got transform_block_3d_dct: {:?}", transform_block_3d_dct);
    }

    #[test]
    fn test_print_chunk_means() {
        let path = "sample-media/sample-5s.mp4";
        let mut reader = AssetReader::new(path);

        let mut transform_block_3d_dct: TransformBlock3DDCT = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator::<4>()
            .map(|macro_block| macro_block.y_components)
            .map(|transform_block| TransformBlock3DDCT::from(transform_block))
            .next()
            .expect("No DCT performed.");

        for ChunkedTransformBlock3D { values: _, mean } in transform_block_3d_dct.chunks_iter() {
            eprintln!("mean:{}", mean);
        }
    }
}
