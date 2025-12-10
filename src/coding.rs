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
}
