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
use std::io::{Read, Write};
use zstd;

// TODO: compress bitmap of discarded chunks with RLE and huffman
// TODO: consider using protobuf or similar for metadata binary format

trait ToBytes<const SERIALIZED_SIZE: usize> {
    fn to_bytes(&self) -> [u8; SERIALIZED_SIZE];
}

trait SerializedSize {
    const SERIALIZED_SIZE: usize;
}

impl SerializedSize for ChunkMetadata {
    const SERIALIZED_SIZE: usize = 8;
}

impl From<&ChunkMetadata> for [u8; ChunkMetadata::SERIALIZED_SIZE] {
    fn from(chunk: &ChunkMetadata) -> Self {
        let mut bytes = [0u8; ChunkMetadata::SERIALIZED_SIZE];
        bytes[0..4].copy_from_slice(&chunk.mean.to_be_bytes());
        bytes[4..8].copy_from_slice(&chunk.energy.to_be_bytes());
        bytes
    }
}

impl From<&[u8; ChunkMetadata::SERIALIZED_SIZE]> for ChunkMetadata {
    fn from(bytes: &[u8; ChunkMetadata::SERIALIZED_SIZE]) -> Self {
        ChunkMetadata {
            mean: f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            energy: f32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        }
    }
}

impl ToBytes<{ Self::SERIALIZED_SIZE }> for ChunkMetadata {
    fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        self.into()
    }
}

fn compress_metadata<'a, I>(
    chunk_metadata_iter: I,
) -> Result<CompressedMetadata, Box<dyn std::error::Error>>
where
    I: Iterator<Item = &'a ChunkMetadata>,
{
    let (_, max_chunks) = chunk_metadata_iter.size_hint();
    let write_buf = Vec::with_capacity(max_chunks.unwrap_or_default());
    let cursor = std::io::Cursor::new(write_buf);

    let mut encoder = zstd::stream::Encoder::new(cursor, 0)?;
    for metadata in chunk_metadata_iter {
        encoder.write_all(&metadata.to_bytes())?
    }
    let compressed_bytes = encoder.finish()?.into_inner();

    Ok(CompressedMetadata {
        data: compressed_bytes.into(),
    })
}

fn decompress_metadata(
    compressed_metadata: CompressedMetadata,
) -> Result<impl Iterator<Item = ChunkMetadata>, Box<dyn std::error::Error>> {
    let cursor = std::io::Cursor::new(compressed_metadata.data);
    let mut decoder = zstd::stream::read::Decoder::new(cursor)?;

    // TODO: has no size hint.
    let iter = std::iter::from_fn(move || {
        let mut buf = [0u8; ChunkMetadata::SERIALIZED_SIZE];
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

impl<'a, I> From<I> for CompressedMetadata
where
    I: Iterator<Item = &'a ChunkMetadata>,
{
    fn from(chunk_metadata_iter: I) -> Self {
        compress_metadata(chunk_metadata_iter).expect("Compressing metadata failed.")
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

    pub fn into_crc_frame(self) -> CompressedMetadataAndCRC {
        self.into()
    }

    pub fn into_chunk_metadata_iter(
        self,
    ) -> Result<impl Iterator<Item = ChunkMetadata>, Box<dyn std::error::Error>> {
        decompress_metadata(self)
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

    pub fn into_rs_frame(self) -> CompressedMetadataAndCRCAndRS {
        self.into()
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

pub mod packetizer {
    use super::*;

    pub struct DecodedPacket {
        pub decoded_data: Vec<u8>,
        pub compressed_metadata_len: Option<usize>, // only present in the first packet of a compressed metadata
    }

    pub struct EncodedPacket {
        pub encoded_data: Box<[u8]>,
    }

    const DECODED_MESSAGE_LENGTH: usize = 223 * 4; // liquid uses {255, 223}-rs
    const ENCODED_MESSAGE_LENGTH: usize = 1060;
    const CRC_SCHEME: liquid_sys::crc_scheme = liquid_sys::crc_scheme_LIQUID_CRC_32;
    const FEC_SCHEME_1: liquid_sys::fec_scheme = liquid_sys::fec_scheme_LIQUID_FEC_RS_M8;
    const FEC_SCHEME_2: liquid_sys::fec_scheme = liquid_sys::fec_scheme_LIQUID_FEC_NONE;

    pub struct Packetizer {
        packetizer: *mut liquid_sys::packetizer_s,
        payload_cursor: std::io::Cursor<Box<[u8]>>,
        encoded_payload_len: usize,
    }

    fn new_packetizer() -> *mut liquid_sys::packetizer_s {
        unsafe {
            liquid_sys::packetizer_create(
                DECODED_MESSAGE_LENGTH as u32,
                CRC_SCHEME as i32,
                FEC_SCHEME_1 as i32,
                FEC_SCHEME_2 as i32,
            )
        }
    }

    // TODO: Packetizer currently uses 32 parity bits for 223 bytes of data.
    // This is a 14% redundancy rate. Softcast specifies 50%.
    impl Packetizer {
        pub(super) fn new(data: Box<[u8]>) -> Self {
            let packetizer = new_packetizer();
            let encoded_payload_len = unsafe {
                liquid_sys::packetizer_compute_enc_msg_len(
                    DECODED_MESSAGE_LENGTH as u32,
                    CRC_SCHEME as i32,
                    FEC_SCHEME_1 as i32,
                    FEC_SCHEME_2 as i32,
                )
            } as usize;

            Packetizer {
                packetizer,
                payload_cursor: std::io::Cursor::new(data),
                encoded_payload_len,
            }
        }

        fn encode_packet(&self, decoded_data: &[u8]) -> Box<[u8]> {
            assert_eq!(decoded_data.len(), DECODED_MESSAGE_LENGTH);
            let mut encoded_data: Box<[u8]> = vec![0u8; self.encoded_payload_len].into();
            unsafe {
                let status = liquid_sys::packetizer_encode(
                    self.packetizer,
                    decoded_data.as_ptr() as *mut u8,
                    encoded_data.as_mut_ptr(),
                );
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK as i32);
            }

            encoded_data
        }
    }

    impl From<CompressedMetadata> for Packetizer {
        fn from(compressed_metadata: CompressedMetadata) -> Self {
            Self::new(compressed_metadata.data)
        }
    }

    impl Iterator for Packetizer {
        type Item = EncodedPacket;

        fn next(&mut self) -> Option<Self::Item> {
            // send payload len as footer
            const PACKET_LEN_FOOTER_LEN: usize = size_of::<u16>();

            let mut buf = [0u8; DECODED_MESSAGE_LENGTH];
            let mut dst_pos = 0;

            // If this is the start of this message, append four bytes for payload_len before padding
            let payload_len = if self.payload_cursor.position() == 0 {
                let payload_len = self.payload_cursor.get_ref().len() as u32;
                Some(payload_len)
            } else {
                None
            };
            let payload_len_footer_len = match payload_len {
                Some(payload_len) => size_of_val(&payload_len),
                None => 0,
            };

            while dst_pos + payload_len_footer_len + PACKET_LEN_FOOTER_LEN < DECODED_MESSAGE_LENGTH
            {
                let end_pos =
                    DECODED_MESSAGE_LENGTH - payload_len_footer_len - PACKET_LEN_FOOTER_LEN;
                let dst = &mut buf[dst_pos..end_pos];
                let bytes_written = self
                    .payload_cursor
                    .read(dst)
                    .expect("Failed to write bytes.");
                dst_pos += bytes_written;

                if bytes_written == 0 {
                    break; // EOF
                }
            }

            // Four bytes on the payload_len_footer for the payload len of this packet
            if let Some(payload_len) = payload_len {
                assert!(dst_pos + payload_len_footer_len <= DECODED_MESSAGE_LENGTH);
                let end_pos = dst_pos + payload_len_footer_len;
                let dst = &mut buf[dst_pos..end_pos];
                dst.copy_from_slice(&payload_len.to_be_bytes());
                dst_pos += payload_len_footer_len;
            }

            // Two bytes in the footer for the packet len of this packet
            match dst_pos {
                0 => None, // Last packet was EOF
                _ => {
                    // U16_MAX == 65,536
                    assert!(dst_pos + PACKET_LEN_FOOTER_LEN <= DECODED_MESSAGE_LENGTH);
                    let footer = &mut buf[DECODED_MESSAGE_LENGTH - PACKET_LEN_FOOTER_LEN..];

                    let decoded_packet_len: u16 =
                        dst_pos.try_into().expect("decoded_packet_len > U16_MAX");
                    footer.copy_from_slice(&decoded_packet_len.to_be_bytes());
                    let encoded_data = self.encode_packet(&buf);

                    Some(EncodedPacket { encoded_data })
                }
            }
        }
    }

    pub struct Depacketizer<I: Iterator<Item = EncodedPacket>> {
        packetizer: *mut liquid_sys::packetizer_s,
        packet_iter: I,
    }

    impl<I: Iterator<Item = EncodedPacket>> Depacketizer<I> {
        pub fn new(packet_iter: I) -> Self {
            let packetizer = new_packetizer();

            Self {
                packetizer,
                packet_iter,
            }
        }

        fn decode_packet(
            &self,
            packet: EncodedPacket,
            is_first: bool,
        ) -> Result<DecodedPacket, Box<dyn std::error::Error>> {
            let encoded_data = packet.encoded_data;

            if encoded_data.len() != ENCODED_MESSAGE_LENGTH {
                return Err("Unexpected encoded_data len.".into());
            }
            let mut decoded_data = vec![0u8; DECODED_MESSAGE_LENGTH];
            let success = unsafe {
                liquid_sys::packetizer_decode(
                    self.packetizer,
                    encoded_data.as_ptr() as *mut u8,
                    decoded_data.as_mut_ptr(),
                )
            };
            if success != 1 {
                return Err("packetizer_decode failed.".into());
            }

            // decode packet_len
            {
                const FOOTER_LEN: usize = size_of::<u16>();
                let footer = &decoded_data[decoded_data.len() - FOOTER_LEN..];
                let mut footer_bytes = [0u8; FOOTER_LEN];
                footer_bytes.copy_from_slice(footer);
                let packet_len = u16::from_be_bytes(footer_bytes);
                if packet_len as usize + FOOTER_LEN > DECODED_MESSAGE_LENGTH {
                    return Err("packet_len > DECODED_MESSAGE_LENGTH".into());
                }
                decoded_data.truncate(packet_len.into()); // truncates padding and packet_len
            }

            // decode payload_len
            let payload_len = if is_first {
                const FOOTER_LEN: usize = size_of::<u32>();
                let footer = &decoded_data[decoded_data.len() - FOOTER_LEN..];
                let mut footer_bytes = [0u8; FOOTER_LEN];
                footer_bytes.copy_from_slice(footer);
                let payload_len = u32::from_be_bytes(footer_bytes) as usize;

                // TODO: any upper bound for payload_len?

                decoded_data.truncate(decoded_data.len() - FOOTER_LEN); // truncates payload_len
                Some(payload_len)
            } else {
                None
            };

            Ok(DecodedPacket {
                decoded_data: decoded_data.into(),
                compressed_metadata_len: payload_len,
            })
        }
    }

    impl<I: Iterator<Item = EncodedPacket>> Iterator for Depacketizer<I> {
        type Item = Result<CompressedMetadata, Box<dyn std::error::Error>>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut is_first = true;

            let mut compressed_metadata = vec![];
            let mut compressed_metadata_len = 0;

            while let Some(encoded_packet) = self.packet_iter.next() {
                let mut decoded_packet = match self.decode_packet(encoded_packet, is_first) {
                    Ok(decoded_packet) => decoded_packet,
                    Err(err) => return Some(Err(err.into())),
                };
                if let Some(len) = decoded_packet.compressed_metadata_len {
                    assert!(is_first);
                    compressed_metadata = Vec::with_capacity(len);
                    compressed_metadata_len = len;
                }
                is_first = false;

                compressed_metadata.append(&mut decoded_packet.decoded_data);
            }

            assert_eq!(
                compressed_metadata.len(),
                compressed_metadata_len,
                "paylod_len from footer does not match compressed_metadata.len()"
            );
            let compressed_metadata = CompressedMetadata {
                data: compressed_metadata.into(),
            };
            Some(Ok(compressed_metadata))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_reader_writer::asset_reader::*;
    use crate::asset_reader_writer::pixel_buffer::*;
    use crate::channel_coding::slice::ChunkIterExt;
    use packetizer::*;

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
        let y_compressed_metadata =
            compress_metadata(y_slices.iter().map(|slice| &slice.chunk_metadata))
                .expect("y_metadata failed");
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
    fn test_reader_to_framed_metadata_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator: MacroBlock3DIterator<LENGTH, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_slices: Box<_> = y_dct.chunks_iter().into_slice_iter(LENGTH).collect();

        let y_compressed_metadata: CompressedMetadata =
            y_slices.iter().map(|slice| &slice.chunk_metadata).into();

        let y_framed_metadata = y_compressed_metadata.into_crc_frame().into_rs_frame();

        let y_decompressed_metadata: Vec<ChunkMetadata> = y_framed_metadata
            .into_recovered_data()
            .expect("Reed Soloman failed")
            .into_valid_data()
            .expect("CRC failed")
            .into_chunk_metadata_iter()
            .expect("Decompress failed")
            .collect();

        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert_eq!(y_slice.chunk_metadata, *y_metadata);
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
        let compressed_metadata: CompressedMetadata = metadata.iter().into();
        let orig_compressed_data = compressed_metadata.data.clone();
        let compressed_metadata_crc: CompressedMetadataAndCRC = compressed_metadata.into();

        let validated_compressed_metadata = compressed_metadata_crc
            .into_valid_data()
            .expect("validation failed");
        assert_eq!(orig_compressed_data, validated_compressed_metadata.data);
    }
    #[test]
    fn test_metatada_decompression() {
        let metadata_in = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            1024
        ];
        let compressed_metadata: CompressedMetadata = metadata_in.iter().into();
        let medatadata_out: Vec<ChunkMetadata> = compressed_metadata
            .into_chunk_metadata_iter()
            .expect("Failed to decompress")
            .collect();

        assert_eq!(metadata_in, medatadata_out);
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
        let compressed_metadata: CompressedMetadata = metadata.iter().into();
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
        let compressed_data: CompressedMetadata = metadata.iter().into();
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
        let compressed_data: CompressedMetadata = metadata.iter().into();
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
        let compressed_data: CompressedMetadata = metadata.iter().into();
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
        let compressed_data: CompressedMetadata = metadata.iter().into();
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
        let compressed_data: CompressedMetadata = metadata.iter().into();
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

    #[test]
    fn test_packetizer() {
        let data = vec![0xbau8; 2056];
        let packetizer = Packetizer::new(data.into());

        for encoded_data in packetizer {
            assert_eq!(encoded_data.encoded_data.len(), 1060);
        }
    }
    #[test]
    fn test_depacketizer() {
        let data = vec![0xbau8; 2056];
        let compressed_metadata = CompressedMetadata {
            data: data.clone().into(),
        };
        let packetizer = Packetizer::from(compressed_metadata);

        let depacketizer = Depacketizer::new(packetizer);
        let new_data: CompressedMetadata = depacketizer
            .map(|result| result.expect("depacketizing failed."))
            .next()
            .expect("No compressed metadatas produced.");

        assert_eq!(data.into_boxed_slice(), new_data.data);
    }

    #[test]
    fn test_depacketizer_odd_boundary() {
        let mut data = vec![0xbau8; 1777];
        for (idx, byte) in data.iter_mut().enumerate() {
            if idx % 7 == 0 {
                *byte ^= 0xff;
            }
        }

        let compressed_metadata = CompressedMetadata {
            data: data.clone().into(),
        };
        let packetizer = Packetizer::from(compressed_metadata);

        let depacketizer = Depacketizer::new(packetizer);
        let new_data: CompressedMetadata = depacketizer
            .map(|result| result.expect("depacketizing failed."))
            .next()
            .expect("No compressed metadatas produced.");

        assert_eq!(data.into_boxed_slice(), new_data.data);
    }
}
