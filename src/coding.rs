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

pub mod transform_block_3d_dct {
    use super::*;
    use chunked_dct_block::*;

    #[derive(Debug)]
    pub struct TransformBlock3DDCT<const LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array3<f32>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType>
        From<TransformBlock3D<LENGTH, PixelType>> for TransformBlock3DDCT<LENGTH, PixelType>
    {
        fn from(transform_block: TransformBlock3D<LENGTH, PixelType>) -> Self {
            let input_values = transform_block.consume_values();
            let (length, width, height) = input_values.dim();
            assert_eq!(length, LENGTH);

            let mut output_a = input_values;
            let mut output_b = ndarray::Array3::zeros(output_a.raw_dim());

            for (axis_idx, axis_len) in [(0, length), (1, width), (2, height)] {
                let handler = ndrustfft::DctHandler::new(axis_len)
                    .normalization(ndrustfft::Normalization::None);
                ndrustfft::nddct2_par(&output_a, &mut output_b, &handler, axis_idx);

                std::mem::swap(&mut output_a, &mut output_b);
            }

            TransformBlock3DDCT::<LENGTH, PixelType> {
                values: output_a,
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
            let dct_values = transform_block_dct.consume_values();
            let (length, width, height) = dct_values.dim();
            assert_eq!(length, LENGTH);

            let mut output_a = dct_values;
            let mut output_b = ndarray::Array3::zeros(output_a.raw_dim());

            for (axis_idx, axis_len) in [(0, length), (1, width), (2, height)] {
                let handler = ndrustfft::DctHandler::new(axis_len)
                    .normalization(ndrustfft::Normalization::None);
                ndrustfft::nddct3_par(&output_a, &mut output_b, &handler, axis_idx); // dct3 is the inverse of dct2

                std::mem::swap(&mut output_a, &mut output_b);
            }

            let scale = 0.125 * (length * width * height) as f32; //  dimensions / (2^num_dimensions)
            output_a.mapv_inplace(|value| value / scale);

            transform_block_3d::TransformBlock3D::with_values(output_a)
        }
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType> TransformBlock3DDCT<LENGTH, PixelType> {
        pub fn consume_values(self) -> ndarray::Array3<f32> {
            self.values
        }

        pub fn chunks_iter(&mut self) -> impl Iterator<Item = ChunkedDCTBlock<'_, PixelType>> {
            const SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS: (usize, usize, usize) = (1, 44, 30);
            let (length, width, height) = self.values.dim();
            let chunk_length =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.0, length);
            let chunk_width =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.1, width);
            let chunk_height =
                max_factor_at_or_below(SOFTCAST_RECOMMENDED_CHUNK_DIMENSIONS.2, height);

            let chunk_size = chunk_width * chunk_height * chunk_length;
            assert!(chunk_size > 0);
            assert_eq!((length * width * height) % chunk_size, 0);
            let num_chunks = (length * width * height) / chunk_size;

            let chunk_dimensions = (chunk_length, chunk_width, chunk_height);

            // must preflight mutatation of the 3D DCT, because we are going to be giving out immutable borrows.
            let mut means = Vec::with_capacity(num_chunks);

            for mut chunk in self.values.exact_chunks_mut(chunk_dimensions) {
                let mean = chunk.mean().unwrap(); // chunk should always be nonempty

                chunk.iter_mut().for_each(|value| *value -= mean); // Softcast specifies a zero-mean distribution

                means.push(mean);
            }

            // two passes necessary due to the impossibility to coerce 'chunks' in the previous loop to an immutable borrow
            let chunked_transform_blocks = self
                .values
                .exact_chunks(chunk_dimensions)
                .into_iter()
                .zip(means.into_iter())
                .map(|(chunk, mean)| ChunkedDCTBlock::new(chunk, mean));

            chunked_transform_blocks.into_iter()
        }

        pub(super) fn from_chunked_dct_blocks(
            chunks: &[ChunkedDCTBlock<'_, PixelType>],
            dct_length: usize,
            frame_resolution: (usize, usize),
        ) -> Self {
            let mut values =
                ndarray::Array3::zeros((dct_length, frame_resolution.0, frame_resolution.1));
            let chunk_dimensions = chunks.first().expect("chunks is empty").values.dim();

            assert_eq!(
                dct_length * frame_resolution.0 * frame_resolution.1,
                chunks.len() * chunk_dimensions.0 * chunk_dimensions.1 * chunk_dimensions.2
            );

            for (mut dst, src) in values
                .exact_chunks_mut(chunk_dimensions)
                .into_iter()
                .zip(chunks)
            {
                dst.assign(&src.values);
            }

            TransformBlock3DDCT::<LENGTH, PixelType> {
                values: values,
                _marker: std::marker::PhantomData,
            }
        }
    }
}

pub mod chunked_dct_block {
    use super::transform_block_3d_dct::*;
    use super::*;

    pub struct ChunkedDCTBlock<'a, PixelType: HasPixelComponentType> {
        pub values: ndarray::ArrayView3<'a, f32>,
        pub mean: f32,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, PixelType: HasPixelComponentType> ChunkedDCTBlock<'a, PixelType> {
        pub(super) fn new(values: ndarray::ArrayView3<'a, f32>, mean: f32) -> Self {
            ChunkedDCTBlock {
                values,
                mean,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct TransformBlock3DDCTIter<'a, const LENGTH: usize, PixelType: HasPixelComponentType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, PixelType>>,
    {
        chunked_dct_block_iter: I,
        frame_resolution: (usize, usize),
    }
    impl<'a, const LENGTH: usize, PixelType: HasPixelComponentType, I>
        TransformBlock3DDCTIter<'a, LENGTH, PixelType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, PixelType>>,
    {
        fn new(chunked_dct_block_iter: I, frame_resolution: (usize, usize)) -> Self {
            let pixel_type = PixelType::TYPE;
            let component_frame_resolution = (
                frame_resolution.0 / pixel_type.interleave_step(),
                frame_resolution.1 / pixel_type.vertical_subsampling(),
            );
            TransformBlock3DDCTIter {
                chunked_dct_block_iter: chunked_dct_block_iter,
                frame_resolution: component_frame_resolution,
            }
        }
    }

    impl<'a, const LENGTH: usize, PixelType: HasPixelComponentType, I> Iterator
        for TransformBlock3DDCTIter<'a, LENGTH, PixelType, I>
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, PixelType>>,
    {
        type Item = TransformBlock3DDCT<LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            use std::cell::OnceCell;

            let num_transform_block_3d_dct_values =
                LENGTH * self.frame_resolution.0 * self.frame_resolution.1;
            let chunks_needed = OnceCell::new();
            let chunk_dim = OnceCell::new();
            let mut chunks_to_consume = OnceCell::new();
            let mut chunked_dct_block_iter_is_empty = true;
            loop {
                let chunk = self.chunked_dct_block_iter.next();

                if chunk.is_none() {
                    return match chunked_dct_block_iter_is_empty {
                        true => None,
                        false => panic!("Not enough chunks to form a TransformBlock3DDCT."),
                    };
                }
                chunked_dct_block_iter_is_empty = false;
                let chunk = chunk.unwrap();

                let chunk_dim = chunk_dim.get_or_init(|| chunk.values.dim());
                assert_eq!(*chunk_dim, chunk.values.dim());

                let chunks_needed = chunks_needed.get_or_init(|| {
                    let num_chunk_values =
                        chunk.values.dim().0 * chunk.values.dim().1 * chunk.values.dim().2;
                    assert_eq!(LENGTH % chunk.values.dim().0, 0);
                    assert_eq!(self.frame_resolution.0 % chunk.values.dim().1, 0);
                    assert_eq!(self.frame_resolution.1 % chunk.values.dim().2, 0);
                    assert_eq!(num_transform_block_3d_dct_values % num_chunk_values, 0);
                    num_transform_block_3d_dct_values / num_chunk_values
                });
                let _ = chunks_to_consume.get_or_init(|| Vec::with_capacity(*chunks_needed));
                let chunks_to_consume = chunks_to_consume.get_mut().unwrap();

                chunks_to_consume.push(chunk);

                if *chunks_needed == chunks_to_consume.len() {
                    let transform_block_3d_dct = TransformBlock3DDCT::from_chunked_dct_blocks(
                        chunks_to_consume,
                        LENGTH,
                        self.frame_resolution,
                    );
                    return Some(transform_block_3d_dct);
                }
            }
        }
    }

    pub trait ChunkedDCTBlockIterExt<
        'a,
        const MACRO_BLOCK_LEN: usize,
        PixelType: HasPixelComponentType,
    >: Iterator<Item = ChunkedDCTBlock<'a, PixelType>> + Sized
    {
        fn into_transform_block_3d_dct_iter(
            self,
            frame_resolution: (usize, usize),
        ) -> TransformBlock3DDCTIter<'a, MACRO_BLOCK_LEN, PixelType, Self>;
    }
    impl<'a, const MACRO_BLOCK_LEN: usize, PixelType: HasPixelComponentType, I>
        ChunkedDCTBlockIterExt<'a, MACRO_BLOCK_LEN, PixelType> for I
    where
        I: Iterator<Item = ChunkedDCTBlock<'a, PixelType>>,
    {
        fn into_transform_block_3d_dct_iter(
            self,
            frame_resolution: (usize, usize),
        ) -> TransformBlock3DDCTIter<'a, MACRO_BLOCK_LEN, PixelType, Self> {
            TransformBlock3DDCTIter::new(self, frame_resolution)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_reader::*;
    use asset_writer::*;
    use chunked_dct_block::*;
    use pixel_buffer::*;
    use std::fs;
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

        for ChunkedDCTBlock {
            values: _, mean, ..
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

        let mut new_y_components: TransformBlock3D<LENGTH, YPixelComponentType> = y_dct.into();
        let mut new_cb_components: TransformBlock3D<LENGTH, CbPixelComponentType> = cb_dct.into();
        let mut new_cr_components: TransformBlock3D<LENGTH, CrPixelComponentType> = cr_dct.into();

        // get rid of floating point errors. For radio-derived values, we do not want to round.
        new_y_components = TransformBlock3D::<LENGTH, YPixelComponentType>::with_values(
            new_y_components.values().mapv(|value| value.round()),
        );
        new_cb_components = TransformBlock3D::<LENGTH, CbPixelComponentType>::with_values(
            new_cb_components.values().mapv(|value| value.round()),
        );
        new_cr_components = TransformBlock3D::<LENGTH, CrPixelComponentType>::with_values(
            new_cr_components.values().mapv(|value| value.round()),
        );

        assert_eq!(original_y_components, new_y_components);
        assert_eq!(original_cb_components, new_cb_components);
        assert_eq!(original_cr_components, new_cr_components);
    }

    #[test]
    //     #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_chunked_dct_blocks_to_writer() {
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
            let mut y_dct_in = macro_block.y_components.into_dct();
            let y_dct_chunks: Vec<_> = y_dct_in.chunks_iter().collect();
            let mut cb_dct_in = macro_block.cb_components.into_dct();
            let cb_dct_chunks: Vec<_> = cb_dct_in.chunks_iter().collect();
            let mut cr_dct_in = macro_block.cr_components.into_dct();
            let cr_dct_chunks: Vec<_> = cr_dct_in.chunks_iter().collect();

            // and back again

            let frame_resolution = (frame_resolution.0 as usize, frame_resolution.1 as usize);
            let mut y_dct_iter: chunked_dct_block::TransformBlock3DDCTIter<
                '_,
                LENGTH,
                YPixelComponentType,
                std::vec::IntoIter<ChunkedDCTBlock<'_, YPixelComponentType>>, // lame that I have to type this
            > = y_dct_chunks
                .into_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let y_dct = y_dct_iter.next().expect("Failed to recreate Y 3D DCT.");
            assert!(y_dct_iter.next().is_none());

            let mut cb_dct_iter: chunked_dct_block::TransformBlock3DDCTIter<
                '_,
                LENGTH,
                CbPixelComponentType,
                std::vec::IntoIter<ChunkedDCTBlock<'_, CbPixelComponentType>>,
            > = cb_dct_chunks
                .into_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let cb_dct = cb_dct_iter.next().expect("Failed to recreate Y 3D DCT.");
            assert!(cb_dct_iter.next().is_none());

            let mut cr_dct_iter: chunked_dct_block::TransformBlock3DDCTIter<
                '_,
                LENGTH,
                CrPixelComponentType,
                std::vec::IntoIter<ChunkedDCTBlock<'_, CrPixelComponentType>>,
            > = cr_dct_chunks
                .into_iter()
                .into_transform_block_3d_dct_iter(frame_resolution);
            let cr_dct = cr_dct_iter.next().expect("Failed to recreate Y 3D DCT.");
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
}
