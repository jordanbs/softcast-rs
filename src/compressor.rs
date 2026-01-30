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

#[derive(Default)]
pub struct MetadataBitmap {
    bitmap: bitvec::boxed::BitBox<usize, bitvec::order::Lsb0>,
}

pub struct RunLengthBitmapCompressor {
    bitmap_iter: std::iter::Peekable<bitvec::boxed::IntoIter<usize, bitvec::order::Lsb0>>,
    has_written_starting_bit: bool,
    iter_is_finished: bool,
}

impl From<MetadataBitmap> for RunLengthBitmapCompressor {
    fn from(metadata_bitmap: MetadataBitmap) -> Self {
        RunLengthBitmapCompressor {
            bitmap_iter: metadata_bitmap.bitmap.into_iter().peekable(),
            has_written_starting_bit: false,
            iter_is_finished: false,
        }
    }
}

impl Read for RunLengthBitmapCompressor {
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

            let bit = self.bitmap_iter.next();
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

                buf_cursor.write_all(&[bit as u8])?; // keep this buffer byte aligned for zstd performance
                bytes_read += 1;
                self.has_written_starting_bit = true;
            }

            let mut run_length = 1u32;
            while let Some(&next_bit) = self.bitmap_iter.peek()
                && next_bit == bit
            {
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
    chunk_iter: Option<I>,
    compression_ratio: f64,
    loader: std::cell::OnceCell<LoadedCompressor<'a, GOP_LENGTH, PixelType>>,
}

struct LoadedCompressor<'a, const GOP_LENGTH: usize, PixelType: HasPixelComponentType> {
    filter_predicate: Box<dyn Fn(&Chunk<'a, GOP_LENGTH, PixelType>) -> bool>,
    chunks_cloned_iter: std::vec::IntoIter<Chunk<'a, GOP_LENGTH, PixelType>>,
    metadata_bitmap: Option<MetadataBitmap>,
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
            // consume self.chunk_iter
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

            let metadata_bitmap = MetadataBitmap {
                bitmap: chunks.iter().map(&filter_predicate).collect(),
            };

            let chunks_cloned_iter = chunks.into_iter();

            LoadedCompressor {
                filter_predicate,
                chunks_cloned_iter,
                metadata_bitmap: Some(metadata_bitmap),
            }
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
            chunks_cloned_iter,
            ..
        } = self.load_mut();

        chunks_cloned_iter.filter(filter_predicate).next()
    }
}
