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

use num_complex::Complex32;

pub trait Complex32Reader {
    fn into_iter(self) -> impl Iterator<Item = Box<[Complex32]>>;
}

pub trait Complex32Consumer {
    // consumes buf, so it can be sent without copies
    fn consume(
        &mut self,
        buf: Box<[Complex32]>,
        flush: bool,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct Complex32IterReadWrapper<R: std::io::Read> {
    reader: R,
    finished: bool,
}
impl<R: std::io::Read> From<R> for Complex32IterReadWrapper<R> {
    fn from(reader: R) -> Self {
        Self {
            reader,
            finished: false,
        }
    }
}
impl<R: std::io::Read> Iterator for Complex32IterReadWrapper<R> {
    type Item = Box<[Complex32]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        const BUF_SIZE: usize = 0x1000;
        let mut buf = vec![0u8; BUF_SIZE];
        let bytes_read = self.reader.read(&mut buf).ok()?;
        self.finished = 0 != bytes_read % 8; // file has rough ending; don't read any more
        buf.truncate(bytes_read);
        if buf.is_empty() {
            return None;
        }
        let c32_buf: Box<[Complex32]> = buf
            .chunks_exact(size_of::<Complex32>())
            .map(|be_bytes| {
                const F32_SIZE: usize = size_of::<f32>();
                let re_bytes: [u8; F32_SIZE] = be_bytes[..F32_SIZE].try_into().unwrap();
                let im_bytes: [u8; F32_SIZE] = be_bytes[F32_SIZE..].try_into().unwrap();
                let re = f32::from_be_bytes(re_bytes);
                let im = f32::from_be_bytes(im_bytes);
                Complex32::new(re, im)
            })
            .collect();
        Some(c32_buf)
    }
}

impl<R: std::io::Read> Complex32Reader for Complex32IterReadWrapper<R> {
    fn into_iter(self) -> impl Iterator<Item = Box<[Complex32]>> {
        self
    }
}
