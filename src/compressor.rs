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

use crate::asset_reader_writer::*;
use crate::source_coding::chunk::*;
use std::io::{Read, Write};

pub struct MetadataBitmap {
    pub rle_encoded_bytes: Box<[u8]>,
}

pub struct RunLengthBitmapEncoder<I: Iterator<Item = bool>> {
    bit_iter: std::iter::Peekable<I>,
    has_written_starting_bit: bool,
    iter_is_finished: bool,
}

impl<I: Iterator<Item = bool>> From<I> for RunLengthBitmapEncoder<I> {
    fn from(bit_iter: I) -> Self {
        RunLengthBitmapEncoder {
            bit_iter: bit_iter.peekable(),
            has_written_starting_bit: false,
            iter_is_finished: false,
        }
    }
}

pub struct RunLengthBitmapDecoder<R: Read> {
    buf_reader: R,
    last_value: Option<bool>,
}

impl<R: Read> From<R> for RunLengthBitmapDecoder<R> {
    fn from(buf_reader: R) -> Self {
        Self {
            buf_reader,
            last_value: None,
        }
    }
}

impl<R: Read> RunLengthBitmapDecoder<R> {
    pub fn into_inner(self) -> R {
        self.buf_reader
    }
}

impl<R: Read> Iterator for RunLengthBitmapDecoder<R> {
    type Item = Result<(bool, u32), std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        // get the next value to vend
        let current_value = match self.last_value {
            Some(last_value) => !last_value, // new value, flip
            None => {
                // first val, read it from buffer
                let mut byte_buf = [0u8; size_of::<u8>()];
                if let Some(err) = self.buf_reader.read_exact(&mut byte_buf).err() {
                    return Some(Err(err));
                }
                let first_byte = u8::from_be_bytes(byte_buf);
                let new_value = 0 != first_byte;
                self.last_value = Some(new_value);
                new_value
            }
        };
        let mut run_count_bytes = [0u8; size_of::<u32>()];

        if let Some(err) = self.buf_reader.read_exact(&mut run_count_bytes).err() {
            // returns EoF if buffer is emptied
            return Some(Err(err));
        }
        let run_count = u32::from_be_bytes(run_count_bytes);
        Some(Ok((current_value, run_count)))
    }
}

impl<I: Iterator<Item = bool>> Read for RunLengthBitmapEncoder<I> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if 0 == buf.len() {
            return Ok(0);
        }
        if self.iter_is_finished {
            return Ok(0);
        }
        let buf_len = buf.len();
        let mut buf_cursor = std::io::Cursor::new(buf);

        let mut bytes_read = 0usize;
        loop {
            let remaining = buf_len - buf_cursor.position() as usize;
            if remaining < size_of::<u32>() {
                assert_ne!(0, bytes_read, "Buf too small for a single read");
                return Ok(bytes_read);
            }

            let bit = self.bit_iter.next();
            if bit.is_none() {
                self.iter_is_finished = true;
                return Ok(bytes_read);
            }
            let bit = bit.unwrap();

            if !self.has_written_starting_bit {
                assert!(
                    1 + size_of::<u32>() <= remaining,
                    "buf too small for a single read."
                );

                // keep this buffer byte aligned for zstd performance
                let bit_buf = (bit as u8).to_be_bytes();
                buf_cursor.write_all(&bit_buf)?;
                bytes_read += 1;
                self.has_written_starting_bit = true;
            }

            let mut run_length = 1u32;
            while let Some(&next_bit) = self.bit_iter.peek()
                && next_bit == bit
            {
                let _ = self.bit_iter.next();
                run_length += 1;
            }

            let run_length_bytes = run_length.to_be_bytes();
            buf_cursor.write_all(&run_length_bytes)?;
            bytes_read += run_length_bytes.len();
        }
    }
}

pub struct Compressor<
    'a,
    const GOP_LENGTH: usize,
    PixelType: HasPixelComponentType,
    I: Iterator<Item = Chunk<'a, GOP_LENGTH, PixelType>>,
> {
    chunk_iter: Option<I>, // for input
    compression_ratio: f64,
    loader: std::cell::OnceCell<LoadedCompressor<'a, GOP_LENGTH, PixelType>>,
}

struct LoadedCompressor<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
    filter_predicate: Box<dyn Fn(&Chunk<'a, GOP_LENGTH, PixelType>) -> bool>,
    chunks_iter: std::vec::IntoIter<Chunk<'a, GOP_LENGTH, PixelType>>,
    metadata_bitmap: Option<MetadataBitmap>, // can be consumed
}

impl<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType>
    LoadedCompressor<'a, GOP_LENGTH, PixelType>
{
    pub fn new(
        filter_predicate: Box<dyn Fn(&Chunk<'a, GOP_LENGTH, PixelType>) -> bool>,
        chunks_iter: std::vec::IntoIter<Chunk<'a, GOP_LENGTH, PixelType>>,
        metadata_bitmap: MetadataBitmap,
    ) -> Self {
        Self {
            filter_predicate,
            chunks_iter,
            metadata_bitmap: Some(metadata_bitmap),
        }
    }
}

impl<
    'a,
    const GOP_LENGTH: usize,
    PixelType: HasPixelComponentType,
    I: Iterator<Item = Chunk<'a, GOP_LENGTH, PixelType>>,
> Compressor<'a, GOP_LENGTH, PixelType, I>
{
    pub fn new(chunk_iter: I, compression_ratio: f64) -> Self {
        assert!(compression_ratio > 0f64);
        assert!(compression_ratio <= 1f64);
        Self {
            chunk_iter: Some(chunk_iter),
            compression_ratio,
            loader: std::cell::OnceCell::new(),
        }
    }

    pub fn take_metadata_bitmap(&mut self) -> MetadataBitmap {
        let loader = self.load_mut();
        loader
            .metadata_bitmap
            .take()
            .expect("Metadata bitmap already taken.")
    }

    fn load(&mut self) -> &LoadedCompressor<'a, GOP_LENGTH, PixelType> {
        self.loader.get_or_init(|| {
            // consume self.chunk_iter, share it between LoadedCompressor and MetadataBitmap
            let chunks: Box<_> = self.chunk_iter.take().unwrap().collect();
            // if this invariant changes, must not consume chunks_iter
            assert_eq!(GOP_LENGTH, chunks.len());

            let cutoff_energy = {
                let mut chunks_clone: Box<_> = chunks.iter().collect();

                let cutoff_idx =
                    ((1f64 - self.compression_ratio) * chunks.len() as f64).floor() as usize; // round down
                let (_, cutoff, _) = chunks_clone.select_nth_unstable_by(cutoff_idx, |c1, c2| {
                    c1.metadata
                        .energy
                        .partial_cmp(&c2.metadata.energy)
                        .expect("Unexpected NaN")
                });
                cutoff.metadata.energy
                // drop chunks_clone, because order is unstable.
            };
            let filter_predicate = move |chunk: &Chunk<'a, GOP_LENGTH, PixelType>| {
                cutoff_energy <= chunk.metadata.energy
            };
            let filter_predicate = Box::new(filter_predicate);

            // RLE the metadata
            // avoid storing the expanded bitmap, to limit peak memory usage
            let mut rle_compressor: RunLengthBitmapEncoder<_> =
                chunks.iter().map(&filter_predicate).into();
            let mut rle_encoded_metadata = vec![];
            let _ = rle_compressor
                .read_to_end(&mut rle_encoded_metadata)
                .expect("Failed to run-length-encode metadata.");

            let metadata_bitmap = MetadataBitmap {
                rle_encoded_bytes: rle_encoded_metadata.into(),
            };
            LoadedCompressor::new(filter_predicate, chunks.into_iter(), metadata_bitmap)
        })
    }
    fn load_mut(&mut self) -> &mut LoadedCompressor<'a, GOP_LENGTH, PixelType> {
        let _ = self.load();
        self.loader.get_mut().unwrap()
    }
}

impl<
    'a,
    const GOP_LENGTH: usize,
    PixelType: HasPixelComponentType,
    I: Iterator<Item = Chunk<'a, GOP_LENGTH, PixelType>>,
> Iterator for Compressor<'a, GOP_LENGTH, PixelType, I>
{
    type Item = Chunk<'a, GOP_LENGTH, PixelType>;
    fn next(&mut self) -> Option<Self::Item> {
        let LoadedCompressor {
            filter_predicate,
            chunks_iter,
            ..
        } = self.load_mut();

        chunks_iter.filter(filter_predicate).next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_rle_compressor_basic() {
        let bytes = vec![0x00, 0x00u8, 0x00u8, 0x01u8] // 0x1F zeros followed by one 1
            .iter()
            .map(|n| n.to_be_bytes())
            .flatten()
            .collect();
        let bits: BitVec<u8, Msb0> = BitVec::from_vec(bytes);
        let mut rle_compressor: RunLengthBitmapEncoder<_> = bits.into_iter().into();

        let mut actual = vec![];
        let bytes_read = rle_compressor
            .read_to_end(&mut actual)
            .expect("Failed to read to end.");

        let mut expected: Vec<u8> = vec![0x00; 1]; // first value
        let mut run_lengths: Vec<u8> = [0x1Fu32, 0x01u32]
            .iter()
            .map(|n| n.to_be_bytes())
            .flatten()
            .collect();
        expected.append(&mut run_lengths);

        assert_eq!(expected.len(), bytes_read);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_rle_decoder_basic() {
        let mut rle_bytes: Vec<u8> = vec![0x00; 1]; // first value
        let mut run_lengths: Vec<u8> = [0x1Fu32, 0x01u32]
            .iter()
            .map(|n| n.to_be_bytes())
            .flatten()
            .collect();
        rle_bytes.append(&mut run_lengths);
        let rle_cursor = std::io::Cursor::new(rle_bytes);

        let num_values_to_take = 0x20;

        let rle_decompressor: RunLengthBitmapDecoder<_> = rle_cursor.into();
        let mut values_taken = 0;
        let mut decompressed_bits: BitVec<u8, Msb0> = BitVec::new();
        for (value, run_length) in
            rle_decompressor.map(|r| r.expect("Error reading RLE-compressed bytes."))
        {
            decompressed_bits.extend([value].repeat(run_length as usize).iter());
            values_taken += run_length;
            if values_taken >= num_values_to_take {
                break;
            }
        }

        let decompressed_bytes: Vec<u8> = decompressed_bits.into();

        let expected_bytes: Vec<u8> = vec![0x00, 0x00u8, 0x00u8, 0x01u8] // 0x1F zeros followed by one 1
            .iter()
            .map(|n| n.to_be_bytes())
            .flatten()
            .collect();

        assert_eq!(expected_bytes, decompressed_bytes);
    }
}
