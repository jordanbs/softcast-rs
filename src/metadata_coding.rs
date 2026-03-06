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

use crate::compressor::*;
use crate::modulation::IntoInnerQuadratureSymbolIter;
use crate::modulation::QuadratureSymbol;
use crate::source_coding::chunk::*;
use half::f16;
use liquid_sys;
use std::io::{Read, Write};
use zstd;

// TODO: compress bitmap of discarded chunks with RLE and huffman
// TODO: consider using protobuf or similar for metadata binary format

fn compress_metadata<'a, I>(
    metadata_bitmap: &MetadataBitmap,
    chunk_metadata_iter: I,
) -> Result<CompressedMetadata, Box<dyn std::error::Error>>
where
    I: Iterator<Item = &'a ChunkMetadata>,
{
    let (_, max_chunks) = chunk_metadata_iter.size_hint();
    let write_buf = Vec::with_capacity(max_chunks.unwrap_or_default());
    let cursor = std::io::Cursor::new(write_buf);

    let mut encoder = zstd::stream::Encoder::new(cursor, 0)?;

    // [bitmap (len == num_chunks)]
    // [chunk.mean; (chunk.energy if bitmap==1)]

    encoder.write_all(metadata_bitmap.values.as_raw_slice())?;

    // compress chunks
    for (idx, chunk_metadata) in chunk_metadata_iter.enumerate() {
        let mean_i8 = chunk_metadata.mean.round() as i8;
        encoder.write_all(&mean_i8.to_be_bytes())?;
        if metadata_bitmap.values[idx] {
            let energy_f16 = f16::from_f32(chunk_metadata.energy.sqrt());
            encoder.write_all(&energy_f16.to_be_bytes())?;
        }
    }
    let compressed_bytes = encoder.finish()?.into_inner();

    Ok(CompressedMetadata {
        data: compressed_bytes.into(),
    })
}

pub struct MetadataDecompressor<QI, R: Read> {
    reader: Option<R>,
    decoder: std::cell::OnceCell<zstd::stream::read::Decoder<'static, std::io::BufReader<R>>>,
    error: Option<std::rc::Rc<dyn std::error::Error>>,
    num_chunks: usize,
    chunk_idx: usize,
    metadata_bitmap: Option<MetadataBitmap>,
    has_decoded_metadata_bitmap: bool,
    _marker: std::marker::PhantomData<QI>,
}

impl<QI, R: Read> MetadataDecompressor<QI, R> {
    pub fn new(reader: R, num_chunks: usize) -> Self {
        Self {
            reader: Some(reader),
            decoder: std::cell::OnceCell::new(),
            error: None,
            num_chunks,
            chunk_idx: 0,
            metadata_bitmap: None,
            has_decoded_metadata_bitmap: false,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn metadata_bitmap(
        &mut self,
    ) -> Result<&MetadataBitmap, std::rc::Rc<dyn std::error::Error>> {
        self.ensure_metadata_bitmap()?;
        Ok(self.metadata_bitmap.as_ref().unwrap())
    }

    pub fn take_metadata_bitmap(
        &mut self,
    ) -> Result<MetadataBitmap, std::rc::Rc<dyn std::error::Error>> {
        self.ensure_metadata_bitmap()?;
        Ok(self.metadata_bitmap.take().unwrap())
    }

    fn ensure_metadata_bitmap(&mut self) -> Result<(), std::rc::Rc<dyn std::error::Error>> {
        if let Some(err) = self.error.as_ref() {
            return Err(err.clone());
        }

        if self.metadata_bitmap.is_none() {
            assert!(
                !self.has_decoded_metadata_bitmap,
                "metadata_bitmap already taken"
            );

            self.ensure_decoder()?;
            let decoder = self.decoder.get_mut().unwrap();
            let mut bitmap = bitvec::bitbox!(u8, bitvec::order::Lsb0; 0; self.num_chunks);
            if let Some(err) = decoder.read_exact(bitmap.as_raw_mut_slice()).err() {
                return Err(self.set_err(err));
            }
            self.has_decoded_metadata_bitmap = true;
            self.metadata_bitmap = Some(MetadataBitmap { values: bitmap });
        }
        Ok(())
    }

    fn ensure_decoder(&mut self) -> Result<(), std::rc::Rc<dyn std::error::Error>> {
        if let Some(err) = self.error.as_ref() {
            return Err(err.clone());
        }

        // zstd will return an error when parsing the dictionary failed.
        if self.decoder.get().is_none() {
            let reader = self.reader.take().unwrap(); // move into decoder
            let decoder = match zstd::stream::read::Decoder::new(reader) {
                Ok(decoder) => decoder,
                Err(err) => return Err(self.set_err(err)),
            };
            let _ = self.decoder.set(decoder);
        }
        Ok(())
    }

    fn set_err<E: std::error::Error + 'static>(
        &mut self,
        err: E,
    ) -> std::rc::Rc<dyn std::error::Error> {
        let rc_error = std::rc::Rc::new(err);
        self.error = Some(rc_error.clone());
        rc_error
    }
}

impl<QI, R: Read> Iterator for MetadataDecompressor<QI, R> {
    type Item = Result<ChunkMetadata, std::rc::Rc<dyn std::error::Error>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.error.is_some() {
            return None;
        }
        if self.chunk_idx == self.num_chunks {
            return None;
        }

        if let Err(err) = self.ensure_decoder() {
            return Some(Err(err));
        }
        if let Err(err) = self.ensure_metadata_bitmap() {
            return Some(Err(err));
        }
        let decoder = self.decoder.get_mut().unwrap();
        let metadata_bitmap = self.metadata_bitmap.as_ref().unwrap();

        let mut mean_buf = [0u8; size_of::<i8>()];
        match decoder.read_exact(&mut mean_buf) {
            Ok(()) => {
                let mean = i8::from_be_bytes(mean_buf) as f32;
                if !mean.is_finite() {
                    let err = std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("mean idx:{} is not finite", self.chunk_idx),
                    );
                    return Some(Err(self.set_err(err)));
                }

                let energy = if metadata_bitmap.values[self.chunk_idx] {
                    let mut energy_buf = [0u8; size_of::<f16>()];
                    match decoder.read_exact(&mut energy_buf) {
                        Ok(()) => {
                            let energy = f16::from_be_bytes(energy_buf).to_f32().powi(2);
                            if !energy.is_finite() {
                                let err = std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!("energy idx:{} is not finite", self.chunk_idx),
                                );
                                return Some(Err(self.set_err(err)));
                            }
                            energy
                        }
                        Err(err) => return Some(Err(self.set_err(err))),
                    }
                } else {
                    0f32
                };
                let meta = ChunkMetadata { mean, energy };
                self.chunk_idx += 1;
                Some(Ok(meta))
            }
            Err(err) => {
                match err.kind() {
                    std::io::ErrorKind::UnexpectedEof => None, // expected EoF, no more metadata
                    _ => Some(Err(self.set_err(err))),
                }
            }
        }
    }
}

impl<QI: Iterator<Item = QuadratureSymbol>, R: Read + IntoInnerQuadratureSymbolIter<QI>>
    IntoInnerQuadratureSymbolIter<QI> for MetadataDecompressor<QI, R>
{
    fn into_inner_quadrature_symbol_iter(self) -> QI {
        let reader = match self.reader {
            Some(reader) => reader,
            None => {
                let decoder = self.decoder.into_inner().unwrap();
                decoder.finish().into_inner()
            }
        };
        reader.into_inner_quadrature_symbol_iter()
    }
}

#[derive(Debug)]
pub struct CompressedMetadata {
    data: Box<[u8]>,
}

impl CompressedMetadata {
    pub fn new<'a, I: Iterator<Item = &'a ChunkMetadata>>(
        metadata_bitmap: &MetadataBitmap,
        chunk_metadata_iter: I,
    ) -> Self {
        compress_metadata(metadata_bitmap, chunk_metadata_iter)
            .expect("Compressing metadata failed.")
    }
}

impl<'a, I> From<I> for CompressedMetadata
where
    I: Iterator<Item = &'a ChunkMetadata>,
{
    fn from(chunk_metadata_iter: I) -> Self {
        // keeping for legacy tests
        let chunk_metadatas: Box<_> = chunk_metadata_iter.collect();
        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; chunk_metadatas.len()],
        };
        let chunk_metadata_iter = chunk_metadatas.into_iter();
        compress_metadata(&metadata_bitmap, chunk_metadata_iter)
            .expect("Compressing metadata failed.")
    }
}

pub mod packetizer {
    use super::*;

    pub struct DecodedPacket {
        pub decoded_data: Box<[u8]>,
        pub compressed_metadata_len: Option<usize>, // only present in the first packet of a compressed metadata
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct EncodedPacket {
        pub encoded_data: [u8; ENCODED_MESSAGE_LENGTH],
    }
    impl From<[u8; ENCODED_MESSAGE_LENGTH]> for EncodedPacket {
        fn from(encoded_data: [u8; ENCODED_MESSAGE_LENGTH]) -> Self {
            EncodedPacket { encoded_data }
        }
    }

    const DECODED_MESSAGE_LENGTH: usize = 223 * 4; // liquid uses {255, 223}-rs
    pub const ENCODED_MESSAGE_LENGTH: usize = 1060;
    const CRC_SCHEME: liquid_sys::crc_scheme = liquid_sys::crc_scheme_LIQUID_CRC_32;
    const FEC_SCHEME_1: liquid_sys::fec_scheme = liquid_sys::fec_scheme_LIQUID_FEC_RS_M8;
    const FEC_SCHEME_2: liquid_sys::fec_scheme = liquid_sys::fec_scheme_LIQUID_FEC_NONE;

    pub struct Packetizer {
        packetizer: *mut liquid_sys::packetizer_s,
        payload_cursor: std::io::Cursor<Box<[u8]>>,
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
            assert_eq!(encoded_payload_len, ENCODED_MESSAGE_LENGTH);

            Packetizer {
                packetizer,
                payload_cursor: std::io::Cursor::new(data),
            }
        }

        fn encode_packet(&self, decoded_data: &[u8]) -> [u8; ENCODED_MESSAGE_LENGTH] {
            assert_eq!(decoded_data.len(), DECODED_MESSAGE_LENGTH);
            let mut encoded_data = [0u8; ENCODED_MESSAGE_LENGTH];
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

                    Some(EncodedPacket::from(encoded_data))
                }
            }
        }
    }

    pub struct Depacketizer<I: Iterator<Item = EncodedPacket>, QI> {
        packetizer: *mut liquid_sys::packetizer_s,
        packet_iter: I,
        working_cursor: std::io::Cursor<Box<[u8]>>,
        has_read_first_packet: bool,
        payload_len: std::cell::OnceCell<usize>,
        bytes_read: usize,
        _marker: std::marker::PhantomData<QI>,
    }

    impl<I: Iterator<Item = EncodedPacket>, QI> From<I> for Depacketizer<I, QI> {
        fn from(packet_iter: I) -> Self {
            let packetizer = new_packetizer();

            Self {
                packetizer,
                packet_iter,
                working_cursor: std::io::Cursor::new(vec![].into()),
                has_read_first_packet: false,
                payload_len: std::cell::OnceCell::new(),
                bytes_read: 0,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<
        I: Iterator<Item = EncodedPacket> + IntoInnerQuadratureSymbolIter<QI>,
        QI: Iterator<Item = QuadratureSymbol>,
    > IntoInnerQuadratureSymbolIter<QI> for Depacketizer<I, QI>
    {
        fn into_inner_quadrature_symbol_iter(self) -> QI {
            self.packet_iter.into_inner_quadrature_symbol_iter()
        }
    }

    impl<I: Iterator<Item = EncodedPacket>, QI> Depacketizer<I, QI> {
        pub fn into_inner(self) -> I {
            self.packet_iter
        }

        fn decode_packet(
            &self,
            packet: &EncodedPacket,
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
            eprintln!("packet integrity check succeeded {}", decoded_data.len());

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
                eprintln!("payload_len:{payload_len}");

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

    impl<I: Iterator<Item = EncodedPacket>, QI> Read for Depacketizer<I, QI> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let mut buf = buf;
            let mut bytes_in_this_read = 0;

            loop {
                if let Some(&payload_len) = self.payload_len.get() {
                    let remaining_in_payload = payload_len - (self.bytes_read + bytes_in_this_read);
                    if 0 == remaining_in_payload {
                        self.bytes_read += bytes_in_this_read;
                        return Ok(bytes_in_this_read); // EOF when bytes_in_this_read == 0
                    }
                }

                // Check if cursor is at the end its underlying buffer
                if self.working_cursor.position() as usize == self.working_cursor.get_ref().len() {
                    let encoded_packet = self.packet_iter.next();
                    if encoded_packet.is_none() {
                        // No more packets
                        self.bytes_read += bytes_in_this_read;

                        let payload_len = self.payload_len.get().copied().unwrap_or_default();
                        if payload_len != self.bytes_read {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "EOF does not match payload_len.",
                            ));
                        }
                        // return EOF
                        return Ok(bytes_in_this_read);
                    }
                    let encoded_packet = encoded_packet.unwrap();
                    let decoded_packet =
                        self.decode_packet(&encoded_packet, !self.has_read_first_packet);
                    if decoded_packet.is_err() {
                        let decode_err = decoded_packet.err().unwrap();
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            decode_err.to_string(),
                        ));
                    }
                    let decoded_packet = decoded_packet.unwrap();
                    self.has_read_first_packet = true;
                    let _ = self
                        .payload_len
                        .get_or_init(|| decoded_packet.compressed_metadata_len.unwrap());
                    self.working_cursor = std::io::Cursor::new(decoded_packet.decoded_data);
                }

                // short circuit read if it is asking for more than the payload
                let payload_len = *self
                    .payload_len
                    .get()
                    .expect("Payload unexpectedly not initialized.");
                let remaining_in_payload = payload_len - (self.bytes_read + bytes_in_this_read);
                if remaining_in_payload < buf.len() {
                    buf = &mut buf[..remaining_in_payload];
                }

                let remaining_in_cursor =
                    self.working_cursor.get_ref().len() - self.working_cursor.position() as usize;

                if remaining_in_cursor < buf.len() {
                    // read is asking for more than what is in cursor.
                    // read remainder of cursor and loop
                    bytes_in_this_read +=
                        self.working_cursor.read(&mut buf[..remaining_in_cursor])?;
                    buf = &mut buf[remaining_in_cursor..];
                } else {
                    // working cursor has enough buffer to fully satisfy this read.
                    bytes_in_this_read += self.working_cursor.read(buf)?;
                    self.bytes_read += bytes_in_this_read;
                    return Ok(bytes_in_this_read);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_reader_writer::asset_reader::*;
    use crate::channel_coding::slice::ChunkIterIntoExt;
    use packetizer::*;

    #[test]
    fn test_reader_to_slice_metadata_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_chunks: Box<_> = y_dct.chunks_iter((1, 30, 40)).collect();
        let num_chunks = y_chunks.len();
        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH).collect();
        let y_compressed_metadata: CompressedMetadata =
            y_slices.iter().map(|slice| &slice.chunk_metadata).into();

        eprintln!(
            "orig_size:{} compressed size:{}",
            y_slices.len() * 2 * 4,
            y_compressed_metadata.data.len()
        );
        let reader = std::io::Cursor::new(y_compressed_metadata.data);
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(reader, num_chunks);
        let y_decompressed_metadata: Box<[ChunkMetadata]> =
            decompressor.map(|r| r.unwrap()).collect();

        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert!((y_slice.chunk_metadata.mean - y_metadata.mean).abs() < 0.5);
            assert!(
                (1.0 - y_slice.chunk_metadata.energy / y_metadata.energy).abs() < 0.001,
                "{} -> {}",
                y_slice.chunk_metadata.energy,
                y_metadata.energy
            );
        }
    }

    #[test]
    fn test_metatada_decompression() {
        let num_chunks = 1024;
        let metadata_in = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            num_chunks
        ];
        let compressed_metadata: CompressedMetadata = metadata_in.iter().into();
        let reader = std::io::Cursor::new(compressed_metadata.data);
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(reader, num_chunks);

        let medatadata_out: Vec<ChunkMetadata> = decompressor.map(|r| r.unwrap()).collect();

        for (orig, new) in metadata_in.iter().zip(medatadata_out) {
            assert!((orig.mean - new.mean).abs() < 0.01);
            assert!((orig.energy - new.energy).abs() < 0.01);
        }
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

        let mut depacketizer: Depacketizer<_, ()> = packetizer.into();
        let mut new_data = vec![];
        let read_bytes = depacketizer
            .read_to_end(&mut new_data)
            .expect("failed to read to end.");

        assert_eq!(read_bytes, 2056);

        assert_eq!(data, new_data);
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

        let mut depacketizer: Depacketizer<_, ()> = packetizer.into();
        let mut new_data = vec![];
        let read_bytes = depacketizer
            .read_to_end(&mut new_data)
            .expect("failed to read to end.");

        assert_eq!(read_bytes, 1777);

        assert_eq!(data, new_data);
    }

    #[test]
    fn test_depacketizer_even_boundary() {
        let mut data = vec![0xbau8; (223 * 4 - 2) * 8 - 4];
        for (idx, byte) in data.iter_mut().enumerate() {
            if idx % 7 == 0 {
                *byte ^= 0xff;
            }
        }

        let compressed_metadata = CompressedMetadata {
            data: data.clone().into(),
        };
        let packetizer = Packetizer::from(compressed_metadata);

        let mut depacketizer: Depacketizer<_, ()> = packetizer.into();
        let mut new_data = vec![];
        let read_bytes = depacketizer
            .read_to_end(&mut new_data)
            .expect("failed to read to end.");

        assert_eq!(read_bytes, (223 * 4 - 2) * 8 - 4);

        assert_eq!(data, new_data);
    }

    #[test]
    fn test_depacketizer_extra_data_in_iterator() {
        let data = vec![0xbau8; 8];
        let compressed_metadata = CompressedMetadata {
            data: data.clone().into(),
        };
        let packetizer = Packetizer::from(compressed_metadata);

        let zeros = [0u8; 1060];
        let zeros_encoded_tail = EncodedPacket {
            encoded_data: zeros,
        };

        let mut depacketizer: Depacketizer<_, ()> =
            packetizer.chain([zeros_encoded_tail].into_iter()).into();

        let mut new_data = vec![];
        let read_bytes = depacketizer
            .read_to_end(&mut new_data)
            .expect("failed to read to end.");

        assert_eq!(read_bytes, 8);

        assert_eq!(data, new_data);
    }

    #[test]
    fn test_reader_to_packet_inverse_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_chunks: Box<_> = y_dct.chunks_iter((1, 30, 40)).collect();
        let num_chunks = y_chunks.len();
        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH).collect();
        let y_compressed_metadata: CompressedMetadata =
            y_slices.iter().map(|slice| &slice.chunk_metadata).into();

        let packetizer: Packetizer = y_compressed_metadata.into();
        let encoded_packets: Box<[EncodedPacket]> = packetizer.collect();
        let depacketizer: Depacketizer<_, ()> = Depacketizer::from(encoded_packets.into_iter());
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(depacketizer, num_chunks);

        let y_decompressed_metadata: Box<[ChunkMetadata]> =
            decompressor.map(|r| r.unwrap()).collect();

        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert!((y_slice.chunk_metadata.mean - y_metadata.mean).abs() < 0.5);
            assert!(
                (1.0 - y_slice.chunk_metadata.energy / y_metadata.energy).abs() < 0.001,
                "{} -> {}",
                y_slice.chunk_metadata.energy,
                y_metadata.energy
            );
        }
    }

    #[test]
    fn test_reader_to_packet_inverse_equality_reader() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        const LENGTH: usize = 4;
        let mut macro_block_3d_iterator =
            reader.pixel_buffer_iter().macro_block_3d_iterator(LENGTH);

        let macro_block = macro_block_3d_iterator.next().expect("No macro blocks");

        let mut y_dct = macro_block.y_components.into_dct();

        let y_chunks: Box<_> = y_dct.chunks_iter((1, 30, 40)).collect();
        let num_chunks = y_chunks.len();
        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH).collect();
        let y_compressed_metadata: CompressedMetadata =
            y_slices.iter().map(|slice| &slice.chunk_metadata).into();

        let packetizer: Packetizer = y_compressed_metadata.into();
        let encoded_packets: Box<[EncodedPacket]> = packetizer.collect();
        let depacketizer: Depacketizer<_, ()> = encoded_packets.into_iter().into();
        let decompressor: MetadataDecompressor<(), _> =
            MetadataDecompressor::new(depacketizer, num_chunks);

        let y_decompressed_metadata: Box<[ChunkMetadata]> =
            decompressor.map(|r| r.unwrap()).collect();

        assert_eq!(y_slices.len(), y_decompressed_metadata.len());

        for (y_slice, y_metadata) in y_slices.iter().zip(y_decompressed_metadata.iter()) {
            assert!((y_slice.chunk_metadata.mean - y_metadata.mean).abs() < 0.5);
            assert!(
                (1.0 - y_slice.chunk_metadata.energy / y_metadata.energy).abs() < 0.001,
                "{} -> {}",
                y_slice.chunk_metadata.energy,
                y_metadata.energy
            );
        }
    }
}
