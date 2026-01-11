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

use crate::source_coding::chunk::*;
use std::io::Read;
use zstd;

// TODO: compress bitmap of discarded chunks with RLE and huffman
// TODO: consider using protobuf or similar for metadata binary format

trait ToBytes {
    fn to_bytes(&self) -> [u8; 8];
}

trait ConstantByteSize {
    const CONSTANT_BYTE_SIZE: usize;
}

impl ConstantByteSize for ChunkMetadata {
    const CONSTANT_BYTE_SIZE: usize = 8;
}

impl From<&ChunkMetadata> for [u8; ChunkMetadata::CONSTANT_BYTE_SIZE] {
    fn from(chunk: &ChunkMetadata) -> Self {
        let mut bytes = [0u8; ChunkMetadata::CONSTANT_BYTE_SIZE];
        bytes[0..4].copy_from_slice(&chunk.mean.to_be_bytes());
        bytes[4..8].copy_from_slice(&chunk.energy.to_be_bytes());
        bytes
    }
}

impl From<&[u8; ChunkMetadata::CONSTANT_BYTE_SIZE]> for ChunkMetadata {
    fn from(bytes: &[u8; ChunkMetadata::CONSTANT_BYTE_SIZE]) -> Self {
        ChunkMetadata {
            mean: f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            energy: f32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        }
    }
}

impl ToBytes for ChunkMetadata {
    fn to_bytes(&self) -> [u8; 8] {
        self.into()
    }
}

pub fn compress_metadata<'a, I: Iterator<Item = &'a ChunkMetadata>>(
    chunk_metadata_iter: I,
) -> Result<Box<[u8]>, Box<dyn std::error::Error>> {
    let binary_metadata: Box<[u8]> = chunk_metadata_iter
        .flat_map(|metadata| metadata.to_bytes())
        .collect();

    // Could stream this, but this buffer should only be on the order of 1-2MB.
    let compressed_metadata = zstd::stream::encode_all(std::io::Cursor::new(binary_metadata), 0)?;

    Ok(compressed_metadata.into())
}

pub fn decompress_metadata(
    compressed_metadata: &[u8],
) -> Result<impl Iterator<Item = ChunkMetadata>, Box<dyn std::error::Error>> {
    let cursor = std::io::Cursor::new(compressed_metadata);
    let mut decoder = zstd::stream::read::Decoder::new(cursor)?;

    // TODO: has no size hint.
    let iter = std::iter::from_fn(move || {
        let mut buf = [0u8; ChunkMetadata::CONSTANT_BYTE_SIZE];
        decoder.read_exact(&mut buf).ok()?; // handles EOF

        let meta = ChunkMetadata::from(&buf);
        Some(meta)
    });
    Ok(iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_reader_writer::asset_reader::*;
    use crate::asset_reader_writer::pixel_buffer::*;
    use crate::channel_coding::slice::ChunkIterExt;

    #[test]
    fn test_reader_to_slice_metadata_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_slices: Box<_> = y_dct.chunks_iter().into_slice_iter(LENGTH).collect();
        let y_metadata_iter = y_slices.iter().map(|slice| &slice.chunk_metadata);
        let y_compressed_metadata = compress_metadata(y_metadata_iter).expect("y_metadata failed");
        eprintln!(
            "orig_size:{} compressed size:{}",
            y_slices.len() * 2 * 4,
            y_compressed_metadata.len()
        );
        let y_decompressed_metadata: Box<[ChunkMetadata]> =
            decompress_metadata(&y_compressed_metadata)
                .expect("y_metadata decompression failed")
                .collect();
        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert_eq!(y_slice.chunk_metadata.mean, y_metadata.mean);
            assert_eq!(y_slice.chunk_metadata.energy, y_metadata.energy);
        }
    }
}
