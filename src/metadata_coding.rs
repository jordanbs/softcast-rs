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

use crate::asset_reader_writer::HasPixelComponentType;
use crate::channel_coding::slice::*;
use zstd;

// TODO: compress bitmap of discarded chunks with RLE and huffman
// TODO: consider using protobuf or similar for metadata binary format

// No iterator here, since we don't want to hold a borrow on the slices or any
// intermediary arrays.
pub fn compress_metadata<'a, const DCT_LENGTH: usize, PixelType: HasPixelComponentType>(
    slices: &[Slice<'_, DCT_LENGTH, PixelType>],
) -> Result<Box<[u8]>, Box<dyn std::error::Error>> {
    let mut metadata = Vec::with_capacity(2 * slices.len());
    slices
        .iter()
        .map(|slice| slice.chunk_mean.to_be_bytes())
        .flatten()
        .for_each(|byte| metadata.push(byte));
    slices
        .iter()
        .map(|slice| slice.chunk_energy.to_be_bytes())
        .flatten()
        .for_each(|byte| metadata.push(byte));

    let compressed_metadata = zstd::stream::encode_all(std::io::Cursor::new(metadata), 0)?;

    Ok(compressed_metadata.into())
}

pub struct Metadata {
    pub mean: f32,
    pub energy: f32,
}

pub fn decompress_metadata(
    compressed_metadata: Box<[u8]>,
) -> Result<Box<[Metadata]>, Box<dyn std::error::Error>> {
    let data = zstd::stream::decode_all(std::io::Cursor::new(compressed_metadata))?;
    assert_eq!(data.len() % size_of::<f32>(), 0); // divisible by 4

    // could get rid of this copy
    let floats: Vec<f32> = data
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();

    let metadata_len = floats.len() / 2;

    let (means, energies) = floats.split_at(metadata_len);
    let metadata = means
        .iter()
        .zip(energies.iter())
        .map(|(&mean, &energy)| Metadata { mean, energy })
        .collect();

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_reader_writer::asset_reader::*;
    use crate::asset_reader_writer::pixel_buffer::*;

    #[test]
    fn test_reader_to_slice_metadata_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 8;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_slices: Box<_> = y_dct.chunks_iter().into_slice_iter(LENGTH).collect();
        let y_compressed_metadata = compress_metadata(&y_slices).expect("y_metadata failed");
        eprintln!(
            "orig_size:{} compressed size:{}",
            y_slices.len() * 2 * 4,
            y_compressed_metadata.len()
        );
        let y_decompressed_metadata =
            decompress_metadata(y_compressed_metadata).expect("y_metadata decompression failed");
        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert_eq!(y_slice.chunk_mean, y_metadata.mean);
            assert_eq!(y_slice.chunk_energy, y_metadata.energy);
        }
    }
}
