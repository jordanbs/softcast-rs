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
use liquid_sys;
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

pub fn compress_metadata<'a>(
    chunk_metadata: &[&ChunkMetadata],
) -> Result<CompressedMetadata, Box<dyn std::error::Error>> {
    let binary_metadata: Box<[u8]> = chunk_metadata
        .iter()
        .flat_map(|metadata| metadata.to_bytes())
        .collect();

    // Could stream this, but this buffer should only be on the order of 1-2MB.
    let compressed_metadata = zstd::stream::encode_all(std::io::Cursor::new(binary_metadata), 0)?;

    Ok(CompressedMetadata {
        data: compressed_metadata.into(),
    })
}

pub fn decompress_metadata(
    compressed_metadata: CompressedMetadata,
) -> Result<impl Iterator<Item = ChunkMetadata>, Box<dyn std::error::Error>> {
    let cursor = std::io::Cursor::new(compressed_metadata.data);
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

#[derive(Debug)]
pub struct CompressedMetadata {
    data: Box<[u8]>,
}

impl From<&[&ChunkMetadata]> for CompressedMetadata {
    fn from(chunk_metadata: &[&ChunkMetadata]) -> Self {
        compress_metadata(chunk_metadata).expect("compress_metadata failed.")
    }
}

impl CompressedMetadata {
    pub fn crc(&self) -> u32 {
        unsafe {
            let crc_scheme = CompressedMetadataAndCRC::CRC_SCHEME;
            liquid_sys::crc_generate_key(
                crc_scheme,
                self.data.as_ptr() as *mut u8,
                self.data.len() as u32,
            )
        }
    }
}

#[derive(Debug)]
pub struct CompressedMetadataAndCRC {
    compressed_metadata: CompressedMetadata,
    crc: u32,
}

impl From<CompressedMetadata> for CompressedMetadataAndCRC {
    fn from(compressed_metadata: CompressedMetadata) -> Self {
        let crc = compressed_metadata.crc();
        CompressedMetadataAndCRC {
            compressed_metadata,
            crc,
        }
    }
}

impl CompressedMetadataAndCRC {
    const CRC_SCHEME: liquid_sys::crc_scheme = liquid_sys::crc_scheme_LIQUID_CRC_32;

    pub fn into_valid_data(self) -> Result<CompressedMetadata, &'static str> {
        unsafe {
            let success: bool = liquid_sys::crc_validate_message(
                Self::CRC_SCHEME,
                self.compressed_metadata.data.as_ptr() as *mut u8,
                self.compressed_metadata.data.len() as u32,
                self.crc,
            ) != 0;
            match success {
                true => Ok(self.compressed_metadata),
                _ => Err("crc_validate_message failed."),
            }
        }
    }

    pub fn into_bytes(self) -> Box<[u8]> {
        let data_and_crc_len = self.compressed_metadata.data.len() + size_of_val(&self.crc);
        let mut bytes: Box<[u8]> = vec![0u8; data_and_crc_len].into();
        bytes[..self.compressed_metadata.data.len()]
            .copy_from_slice(&self.compressed_metadata.data);
        bytes[self.compressed_metadata.data.len()..]
            .copy_from_slice(&self.compressed_metadata.crc().to_be_bytes());
        bytes
    }

    pub fn from_bytes(mut bytes: Vec<u8>) -> Result<Self, Box<dyn std::error::Error>> {
        let crc_len = size_of::<u32>();
        if bytes.len() < crc_len {
            return Err("bytes.len() < crc_len".into());
        }
        let crc_bytes: [u8; 4] = bytes[bytes.len() - crc_len..].try_into()?;
        let crc = u32::from_be_bytes(crc_bytes);
        bytes.truncate(bytes.len() - crc_len);

        let data_and_crc = Self {
            compressed_metadata: CompressedMetadata { data: bytes.into() },
            crc,
        };
        Ok(data_and_crc)
    }
}

pub struct CompressedMetadataAndCRCAndRS {
    data: Box<[u8]>,
    decoded_data_len: usize,
}
impl CompressedMetadataAndCRCAndRS {
    const FEC_SCHEME: liquid_sys::fec_scheme = liquid_sys::fec_scheme_LIQUID_FEC_RS_M8;

    pub fn into_recovered_data(
        self,
    ) -> Result<CompressedMetadataAndCRC, Box<dyn std::error::Error>> {
        let mut recovered_data = vec![0u8; self.decoded_data_len];
        unsafe {
            let fec = liquid_sys::fec_create(Self::FEC_SCHEME, std::ptr::null_mut());
            let status = liquid_sys::fec_decode(
                fec,
                recovered_data.len() as u32,
                self.data.as_ptr() as *mut u8,
                recovered_data.as_mut_ptr(),
            ) as u32;
            // should never fail
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let data_and_crc = CompressedMetadataAndCRC::from_bytes(recovered_data)?;
            Ok(data_and_crc)
        }
    }
}

impl From<CompressedMetadataAndCRC> for CompressedMetadataAndCRCAndRS {
    fn from(compressed_metadata_and_crc: CompressedMetadataAndCRC) -> Self {
        let decoded_data = compressed_metadata_and_crc.into_bytes();

        unsafe {
            let encoded_data_len =
                liquid_sys::fec_get_enc_msg_length(Self::FEC_SCHEME, decoded_data.len() as u32)
                    as usize;
            let mut encoded_data: Box<[u8]> = vec![0u8; encoded_data_len].into();
            let fec = liquid_sys::fec_create(Self::FEC_SCHEME, std::ptr::null_mut());

            let success = liquid_sys::fec_encode(
                fec,
                decoded_data.len() as u32,
                decoded_data.as_ptr() as *mut u8,
                encoded_data.as_mut_ptr(),
            ) as u32;
            // should never fail
            assert_eq!(success, liquid_sys::liquid_error_code_LIQUID_OK);

            CompressedMetadataAndCRCAndRS {
                data: encoded_data,
                decoded_data_len: decoded_data.len(),
            }
        }
    }
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
        let y_metadata: Box<_> = y_slices.iter().map(|slice| &slice.chunk_metadata).collect();
        let y_compressed_metadata = compress_metadata(&y_metadata).expect("y_metadata failed");
        eprintln!(
            "orig_size:{} compressed size:{}",
            y_slices.len() * 2 * 4,
            y_compressed_metadata.data.len()
        );
        let y_decompressed_metadata: Box<[ChunkMetadata]> =
            decompress_metadata(y_compressed_metadata)
                .expect("y_metadata decompression failed")
                .collect();
        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert_eq!(y_slice.chunk_metadata.mean, y_metadata.mean);
            assert_eq!(y_slice.chunk_metadata.energy, y_metadata.energy);
        }
    }

    #[test]
    fn test_metatada_crc_valid() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_metadata: CompressedMetadata = metadata_refs.as_ref().into();
        let orig_compressed_data = compressed_metadata.data.clone();
        let compressed_metadata_crc: CompressedMetadataAndCRC = compressed_metadata.into();

        let validated_compressed_metadata = compressed_metadata_crc
            .into_valid_data()
            .expect("validation failed");
        assert_eq!(orig_compressed_data, validated_compressed_metadata.data);
    }

    #[test]
    fn test_metatada_crc_fail() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_metadata: CompressedMetadata = metadata_refs.as_ref().into();
        let mut compressed_metadata_crc: CompressedMetadataAndCRC = compressed_metadata.into();

        compressed_metadata_crc.compressed_metadata.data[5] -= 1;
        let _ = compressed_metadata_crc
            .into_valid_data()
            .expect_err("validation unexpectly passed");
    }

    #[test]
    fn test_metatada_rs_recovery_no_corruption() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_data: CompressedMetadata = metadata_refs.as_ref().into();
        let orig_compressed_data = compressed_data.data.clone();
        let crc_data_in: CompressedMetadataAndCRC = compressed_data.into();

        let rs_crc_data: CompressedMetadataAndCRCAndRS = crc_data_in.into();

        let crc_data_out = rs_crc_data
            .into_recovered_data()
            .expect("rs recovery failed");
        let valid_compressed_data = crc_data_out
            .into_valid_data()
            .expect("crc validation failed");

        assert_eq!(orig_compressed_data, valid_compressed_data.data);
    }

    #[test]
    fn test_metatada_rs_recovery_bit_flip() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_data: CompressedMetadata = metadata_refs.as_ref().into();
        let orig_compressed_data = compressed_data.data.clone();
        let crc_data_in: CompressedMetadataAndCRC = compressed_data.into();

        let mut rs_crc_data: CompressedMetadataAndCRCAndRS = crc_data_in.into();

        rs_crc_data.data[5] ^= 0x1u8;

        let crc_data_out = rs_crc_data
            .into_recovered_data()
            .expect("rs recovery failed");
        let valid_compressed_data = crc_data_out
            .into_valid_data()
            .expect("crc validation failed");

        assert_eq!(orig_compressed_data, valid_compressed_data.data);
    }
    #[test]
    fn test_metatada_rs_recovery_byte_flip() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_data: CompressedMetadata = metadata_refs.as_ref().into();
        let orig_compressed_data = compressed_data.data.clone();
        let crc_data_in: CompressedMetadataAndCRC = compressed_data.into();

        let mut rs_crc_data: CompressedMetadataAndCRCAndRS = crc_data_in.into();

        eprintln!("data_in: {:x?}", rs_crc_data.data);
        rs_crc_data.data[5] ^= 0xffu8;
        eprintln!("data_out: {:x?}", rs_crc_data.data);

        let crc_data_out = rs_crc_data
            .into_recovered_data()
            .expect("rs recovery failed");
        let valid_compressed_data = crc_data_out
            .into_valid_data()
            .expect("crc validation failed");

        assert_eq!(orig_compressed_data, valid_compressed_data.data);
    }

    #[test]
    fn test_metatada_rs_recovery_flip_16_bytes() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_data: CompressedMetadata = metadata_refs.as_ref().into();
        let orig_compressed_data = compressed_data.data.clone();
        let crc_data_in: CompressedMetadataAndCRC = compressed_data.into();

        let mut rs_crc_data: CompressedMetadataAndCRCAndRS = crc_data_in.into();

        eprintln!("data_in: {:x?}", rs_crc_data.data);
        for idx in 5..21 {
            rs_crc_data.data[idx] ^= 0xffu8;
        }
        eprintln!("data_out: {:x?}", rs_crc_data.data);

        let crc_data_out = rs_crc_data
            .into_recovered_data()
            .expect("rs recovery failed");
        let valid_compressed_data = crc_data_out
            .into_valid_data()
            .expect("crc validation failed");

        assert_eq!(orig_compressed_data, valid_compressed_data.data);
    }

    #[test]
    fn test_metatada_rs_recovery_fail_flip_40_bytes() {
        let metadata = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let metadata_refs: Box<_> = metadata.iter().collect();
        let compressed_data: CompressedMetadata = metadata_refs.as_ref().into();
        let crc_data_in: CompressedMetadataAndCRC = compressed_data.into();

        eprintln!("crc_data_in : {:x?}", crc_data_in.compressed_metadata.data);

        let mut rs_crc_data: CompressedMetadataAndCRCAndRS = crc_data_in.into();

        // eprintln!("data_in : {:x?}", rs_crc_data.data);
        for idx in 5..45 {
            rs_crc_data.data[idx] ^= 0xffu8;
        }
        // eprintln!("data_out: {:x?}", rs_crc_data.data);

        let crc_data_out = rs_crc_data
            .into_recovered_data()
            .expect("rs recovery failed");

        eprintln!("crc_data_out: {:x?}", crc_data_out.compressed_metadata.data);

        let _ = crc_data_out
            .into_valid_data()
            .expect_err("crc validation unexpectedly passed");
    }
}
