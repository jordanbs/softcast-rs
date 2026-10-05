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
use crate::modulation::QuadratureSymbol;
use crate::source_coding::chunk::*;
use half::f16;
use liquid_sys;
use num_complex::Complex32;
use std::io::{Read, Write};
use zstd;

// TODO: consider using protobuf or similar for metadata binary format

pub fn compress_metadata<
    'a,
    Y: Iterator<Item = &'a ChunkMetadata>,
    B: Iterator<Item = &'a ChunkMetadata>,
    R: Iterator<Item = &'a ChunkMetadata>,
>(
    y: (&MetadataBitmap, Y),
    cb: (&MetadataBitmap, B),
    cr: (&MetadataBitmap, R),
) -> Result<CompressedMetadata2, Box<dyn std::error::Error>> {
    // 12 bytes per chunk upper limit
    let size_hint = y.1.size_hint().1.unwrap_or_default()
        + cb.1.size_hint().1.unwrap_or_default()
        + cr.1.size_hint().1.unwrap_or_default();
    let write_buf_len = 12 * size_hint;
    let write_buf = Vec::with_capacity(write_buf_len);
    let cursor = std::io::Cursor::new(write_buf);

    let mut encoder = zstd::stream::Encoder::new(cursor, 22)?; // 22 is max compression

    fn compress<'a, W: Write, M: Iterator<Item = &'a ChunkMetadata>>(
        encoder: &mut zstd::stream::Encoder<W>,
        metadata_bitmap: &MetadataBitmap,
        chunk_metadatas: M,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // compress bitmaps
        encoder.write_all(metadata_bitmap.values.as_raw_slice())?;

        // compress chunks
        for (idx, metadata) in chunk_metadatas.enumerate() {
            let mean_i8 = metadata.mean.round() as i8;
            encoder.write_all(&mean_i8.to_be_bytes())?;
            if metadata_bitmap.values[idx] {
                let energy_f16 = f16::from_f32(metadata.energy.sqrt());
                encoder.write_all(&energy_f16.to_be_bytes())?;
            }
        }
        Ok(())
    }
    compress(&mut encoder, y.0, y.1)?;
    compress(&mut encoder, cb.0, cb.1)?;
    compress(&mut encoder, cb.0, cr.1)?;
    let compressed_bytes = encoder.finish()?.into_inner();

    Ok(CompressedMetadata2(compressed_bytes.into()))
}

pub struct MetadataDecompressor<R: Read> {
    reader: Option<R>,
    decoder: std::cell::OnceCell<zstd::stream::read::Decoder<'static, std::io::BufReader<R>>>,
    error: Option<std::rc::Rc<dyn std::error::Error>>,
    num_chunks: usize,
    chunk_idx: usize,
    metadata_bitmap: Option<MetadataBitmap>,
    has_decoded_metadata_bitmap: bool,
}

impl<R: Read> MetadataDecompressor<R> {
    fn default() -> Self {
        Self {
            reader: None,
            error: None,
            decoder: std::cell::OnceCell::new(),
            num_chunks: 0,
            chunk_idx: 0,
            metadata_bitmap: None,
            has_decoded_metadata_bitmap: false,
        }
    }
    pub fn new(reader: R, num_chunks: usize) -> Self {
        let mut new_decompressor = Self::default();
        new_decompressor.reader = Some(reader);
        new_decompressor.num_chunks = num_chunks;
        new_decompressor
    }
    pub fn into_next(self, num_chunks: usize) -> Self {
        assert!(self.reader.is_none());
        assert!(self.decoder.get().is_some());
        assert!(self.error.is_none());
        assert_eq!(self.num_chunks, self.chunk_idx);
        assert!(self.has_decoded_metadata_bitmap);

        let mut new_decompressor = Self::default();
        new_decompressor.decoder = self.decoder;
        new_decompressor.num_chunks = num_chunks;
        new_decompressor
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

impl<R: Read> Iterator for MetadataDecompressor<R> {
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

pub struct CompressedMetadata2(Box<[u8]>);
impl CompressedMetadata2 {
    pub fn data(&self) -> &[u8] {
        &self.0
    }
}

pub mod packet_modem {
    use super::*;

    const PACKET_LEN: usize = 1023;
    const FRAME_LEN: usize = 33864;
    const HEADER_LEN: usize = size_of::<u32>();

    fn qpacketmodem_new() -> *mut liquid_sys::qpacketmodem_s {
        unsafe {
            let qpacketmodem = liquid_sys::qpacketmodem_create();
            // TODO: Apply CRC check for entire metadata rather than for each packet.
            let status = liquid_sys::qpacketmodem_configure(
                qpacketmodem,
                PACKET_LEN as u32,
                liquid_sys::crc_scheme_LIQUID_CRC_32,
                liquid_sys::fec_scheme_LIQUID_FEC_CONV_V27,
                liquid_sys::fec_scheme_LIQUID_FEC_RS_M8_50,
                liquid_sys::modulation_scheme_LIQUID_MODEM_BPSK as i32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let frame_len = liquid_sys::qpacketmodem_get_frame_len(qpacketmodem);
            assert_eq!(FRAME_LEN, frame_len as usize);
            qpacketmodem
        }
    }

    pub struct PacketModulator {
        qpacketmodem: *mut liquid_sys::qpacketmodem_s,
        payload_reader: std::io::BufReader<std::io::Cursor<Box<[u8]>>>,
        payload_len: u32,
        needs_header: bool,
        finished: bool,
    }

    impl From<CompressedMetadata2> for PacketModulator {
        fn from(compressed_metadata: CompressedMetadata2) -> Self {
            let qpacketmodem = qpacketmodem_new();
            let payload_len = compressed_metadata.data().len() as u32;
            let payload_reader =
                std::io::BufReader::new(std::io::Cursor::new(compressed_metadata.0));
            Self {
                qpacketmodem,
                payload_reader,
                payload_len,
                needs_header: true,
                finished: false,
            }
        }
    }
    impl PacketModulator {
        fn write_header<W: std::io::Write>(&mut self, packet_writer: &mut W) {
            let header_bytes = self.payload_len.to_be_bytes();
            let mut header_reader = std::io::BufReader::new(&header_bytes[..]);
            std::io::copy(&mut header_reader, packet_writer).expect("Failed to write header.");
        }
    }
    impl Iterator for PacketModulator {
        type Item = Box<[QuadratureSymbol]>;

        fn next(&mut self) -> Option<Box<[QuadratureSymbol]>> {
            if self.finished {
                return None;
            }

            let mut packet_buf = [0u8; PACKET_LEN];
            let mut packet_writer = std::io::BufWriter::new(&mut packet_buf[..]);

            let mut bytes_needed = PACKET_LEN;
            if self.needs_header {
                self.write_header(&mut packet_writer);
                bytes_needed -= HEADER_LEN;
                self.needs_header = false;
            };

            let mut payload_reader = self.payload_reader.by_ref().take(bytes_needed as u64); // to prevent over-reads
            let bytes_written = std::io::copy(&mut payload_reader, &mut packet_writer)
                .expect("Failed to write payload.") as usize;
            drop(packet_writer); // give borrow back to packet_buf

            if 0 == bytes_written {
                return None;
            }
            if bytes_written < bytes_needed {
                self.finished = true;
            }

            let mut encoded_packet = vec![Complex32::ZERO; FRAME_LEN];
            unsafe {
                let status = liquid_sys::qpacketmodem_encode(
                    self.qpacketmodem,
                    packet_buf.as_ptr(),
                    encoded_packet.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
            }
            Some(unsafe { std::mem::transmute(encoded_packet.into_boxed_slice()) })
        }
    }
    impl Drop for PacketModulator {
        fn drop(&mut self) {
            unsafe {
                liquid_sys::qpacketmodem_destroy(self.qpacketmodem);
            }
        }
    }

    pub struct PacketDemodulator<I: Iterator<Item = QuadratureSymbol>> {
        qpacketmodem: *mut liquid_sys::qpacketmodem_s,
        inner: I,
        payload_reader: Option<std::io::BufReader<std::io::Cursor<Box<[u8]>>>>,
    }
    impl<I: Iterator<Item = QuadratureSymbol>> From<I> for PacketDemodulator<I> {
        fn from(inner: I) -> Self {
            Self {
                qpacketmodem: qpacketmodem_new(),
                inner,
                payload_reader: None,
            }
        }
    }
    impl<I: Iterator<Item = QuadratureSymbol>> PacketDemodulator<I> {
        fn decode_next_packet(&mut self, buf: &mut [u8]) -> Result<(), std::io::Error> {
            let mut encoded_packet = Vec::with_capacity(FRAME_LEN);
            encoded_packet.extend(
                self.inner
                    .by_ref()
                    .chain([QuadratureSymbol::default()].into_iter().cycle()) // pad with 0s if necessary
                    .take(FRAME_LEN),
            );

            let crc_pass = unsafe {
                0 != liquid_sys::qpacketmodem_decode_soft(
                    self.qpacketmodem,
                    encoded_packet.as_mut_ptr() as *mut Complex32,
                    buf.as_mut_ptr(),
                )
            };
            if !crc_pass {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Metadata decode failed.",
                ));
            }
            Ok(())
        }
        fn decode(&mut self) -> Result<Box<[u8]>, std::io::Error> {
            let mut first_packet_buf = [0u8; PACKET_LEN];
            self.decode_next_packet(&mut first_packet_buf)?;

            let mut header_buf = [0u8; HEADER_LEN];
            header_buf.copy_from_slice(&first_packet_buf[..HEADER_LEN]);
            let payload_len = u32::from_be_bytes(header_buf) as usize;

            let num_packets = 1 + (HEADER_LEN + payload_len) / PACKET_LEN;
            let mut payload = Vec::with_capacity(num_packets * PACKET_LEN);
            payload.extend(&first_packet_buf[HEADER_LEN..]);

            while payload.len() < payload_len {
                let start = payload.len();
                unsafe {
                    payload.set_len(start + PACKET_LEN);
                    self.decode_next_packet(&mut payload[start..])?;
                }
            }
            payload.truncate(payload_len);

            Ok(payload.into())
        }
        fn payload_reader(&mut self) -> Result<impl Read, std::io::Error> {
            if self.payload_reader.is_none() {
                let decoded_packet = self.decode()?;
                let reader = std::io::BufReader::new(std::io::Cursor::new(decoded_packet));
                self.payload_reader = Some(reader);
            }
            Ok(self.payload_reader.as_mut().unwrap())
        }
    }
    impl<I: Iterator<Item = QuadratureSymbol>> Read for PacketDemodulator<I> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.payload_reader()?.read(buf)
        }
        fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
            self.payload_reader()?.read_exact(buf)
        }
        fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize> {
            self.payload_reader()?.read_to_end(buf)
        }
        fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize> {
            self.payload_reader()?.read_to_string(buf)
        }
        fn read_vectored(
            &mut self,
            bufs: &mut [std::io::IoSliceMut<'_>],
        ) -> std::io::Result<usize> {
            self.payload_reader()?.read_vectored(bufs)
        }
    }
    impl<I: Iterator<Item = QuadratureSymbol>> Drop for PacketDemodulator<I> {
        fn drop(&mut self) {
            unsafe {
                liquid_sys::qpacketmodem_destroy(self.qpacketmodem);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_vendor = "apple")]
    use crate::asset_reader_writer::asset_reader::*;
    use crate::channel_coding::slice::ChunkIterIntoExt;
    use packet_modem::*;

    #[test]
    #[cfg(target_vendor = "apple")]
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
        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; num_chunks],
        };
        let y_compressed_metadata = compress_metadata(
            (&metadata_bitmap, y_chunks.iter().map(|c| &c.metadata)),
            (&metadata_bitmap, std::iter::empty()),
            (&metadata_bitmap, std::iter::empty()),
        )
        .expect("Failed to compress");

        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH, true).collect();

        eprintln!(
            "orig_size:{} compressed size:{}",
            y_slices.len() * 2 * 4,
            y_compressed_metadata.0.len()
        );
        let reader = std::io::Cursor::new(y_compressed_metadata.0);
        let decompressor: MetadataDecompressor<_> = MetadataDecompressor::new(reader, num_chunks);
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
    fn test_metatada_decompression_multiple() {
        const NUM_CHUNKS_0: usize = 1024;
        let metadata_in_0 = vec![
            ChunkMetadata {
                mean: 5f32,
                energy: 6f32,
            };
            NUM_CHUNKS_0
        ];
        const NUM_CHUNKS_1: usize = 256;
        let metadata_in_1 = vec![
            ChunkMetadata {
                mean: 7f32,
                energy: 8f32,
            };
            NUM_CHUNKS_1
        ];
        let metadata_bitmap_0 = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; metadata_in_0.len()],
        };
        let metadata_bitmap_1 = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; metadata_in_1.len()],
        };
        let compressed_metadata: CompressedMetadata2 = compress_metadata(
            (&metadata_bitmap_0, metadata_in_0.iter()),
            (&metadata_bitmap_1, metadata_in_1.iter()),
            (&metadata_bitmap_1, metadata_in_1.iter()),
        )
        .expect("CompresMetadata failed");
        let uncompressed_len = 2 * 16 * (NUM_CHUNKS_0 + 2 * NUM_CHUNKS_1)
            + (metadata_bitmap_0.values.len() + 2 * metadata_bitmap_1.values.len());
        let compressed_len = compressed_metadata.data().len();
        eprintln!("{} -> {}", uncompressed_len, compressed_len);

        let reader = std::io::Cursor::new(compressed_metadata.data());
        let mut decompressor_0: MetadataDecompressor<_> =
            MetadataDecompressor::new(reader, NUM_CHUNKS_0);
        let metadata_out_0: Vec<ChunkMetadata> =
            decompressor_0.by_ref().map(|r| r.unwrap()).collect();
        let mut decompressor_1: MetadataDecompressor<_> = decompressor_0.into_next(NUM_CHUNKS_1);
        let metadata_out_1: Vec<ChunkMetadata> =
            decompressor_1.by_ref().map(|r| r.unwrap()).collect();
        let decompressor_2: MetadataDecompressor<_> = decompressor_1.into_next(NUM_CHUNKS_1);
        let metadata_out_2: Vec<ChunkMetadata> = decompressor_2.map(|r| r.unwrap()).collect();

        for (orig, new) in metadata_in_0
            .iter()
            .chain(metadata_in_1.iter())
            .chain(metadata_in_1.iter())
            .zip(
                metadata_out_0
                    .iter()
                    .chain(metadata_out_1.iter())
                    .chain(metadata_out_2.iter()),
            )
        {
            assert!((orig.mean - new.mean).abs() < 0.01);
            assert!((orig.energy - new.energy).abs() < 0.01);
        }
    }

    #[test]
    fn test_depacketizer_odd_boundary() {
        let mut data = vec![0xbau8; 1777];
        for (idx, byte) in data.iter_mut().enumerate() {
            if idx % 7 == 0 {
                *byte ^= 0xff;
            }
        }

        let compressed_metadata = CompressedMetadata2(data.clone().into());
        let packetizer = PacketModulator::from(compressed_metadata);

        let mut depacketizer: PacketDemodulator<_> = packetizer.flatten().into();
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

        let compressed_metadata = CompressedMetadata2(data.clone().into());
        let packetizer = PacketModulator::from(compressed_metadata);

        let mut depacketizer: PacketDemodulator<_> = packetizer.flatten().into();
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
        let compressed_metadata = CompressedMetadata2(data.clone().into());
        let packetizer = PacketModulator::from(compressed_metadata);

        let zeros = vec![QuadratureSymbol::default(); 5000];

        let mut depacketizer: PacketDemodulator<_> =
            packetizer.flatten().chain(zeros.into_iter()).into();

        let mut new_data = vec![];
        let read_bytes = depacketizer
            .read_to_end(&mut new_data)
            .expect("failed to read to end.");

        assert_eq!(read_bytes, 8);

        assert_eq!(data, new_data);
    }

    #[test]
    fn test_packet_modem() {
        let data_in = vec![0xbau8; 5000];
        let compressed_metadata = CompressedMetadata2(data_in.clone().into());

        let modulator: PacketModulator = compressed_metadata.into();
        let iqs: Vec<QuadratureSymbol> = modulator.flatten().collect();
        let mut data_out = vec![];
        let mut demodulator: PacketDemodulator<_> = iqs.into_iter().into();
        demodulator
            .read_to_end(&mut data_out)
            .expect("Failed to read data to end.");

        assert_eq!(data_in, data_out);
    }

    #[test]
    #[cfg(target_vendor = "apple")]
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
        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; num_chunks],
        };
        let y_compressed_metadata = compress_metadata(
            (&metadata_bitmap, y_chunks.iter().map(|c| &c.metadata)),
            (&metadata_bitmap, std::iter::empty()),
            (&metadata_bitmap, std::iter::empty()),
        )
        .expect("Failed to compress");
        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH, true).collect();

        let packetizer: PacketModulator = y_compressed_metadata.into();
        let depacketizer: PacketDemodulator<_> = packetizer.flatten().into();
        let decompressor: MetadataDecompressor<_> =
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
    #[cfg(target_vendor = "apple")]
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
        let metadata_bitmap = MetadataBitmap {
            values: bitvec::bitbox![u8, bitvec::order::Lsb0; 1; num_chunks],
        };
        let y_compressed_metadata = compress_metadata(
            (&metadata_bitmap, y_chunks.iter().map(|c| &c.metadata)),
            (&metadata_bitmap, std::iter::empty()),
            (&metadata_bitmap, std::iter::empty()),
        )
        .expect("Failed to compress");
        let y_slices: Box<_> = y_chunks.into_iter().into_slice_iter(LENGTH, true).collect();

        let packetizer: PacketModulator = y_compressed_metadata.into();
        let depacketizer: PacketDemodulator<_> = packetizer.flatten().into();
        let decompressor: MetadataDecompressor<_> =
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
