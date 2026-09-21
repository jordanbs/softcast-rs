// Copyright 2025-2026 Jordan Schneider
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

#[derive(Debug)]
pub enum ViewOrOwnedArray3<'a> {
    View(ndarray::ArrayViewMut3<'a, f32>),
    Owned(ndarray::Array3<f32>),
    OwnedArc(ndarray::ArcArray<f32, ndarray::Ix3>),
}

pub trait PSNR {
    fn psnr(&self, peak: Self) -> Self;
}
impl PSNR for f64 {
    fn psnr(&self, peak: Self) -> Self {
        10.0 * (peak.powi(2) / self).log10()
    }
}
pub trait DbToAWGNPower {
    fn db_to_awgn_power(&self) -> Self;
}
impl DbToAWGNPower for f32 {
    fn db_to_awgn_power(&self) -> Self {
        10f32.powf(-self / 10f32)
    }
}

pub mod dump_file {
    use num_complex::Complex32;
    use std::io::Write;

    pub fn create_dump_file(is_write: bool) -> std::fs::File {
        let mut idx = 0;
        loop {
            let rw = if is_write { "w" } else { "r" };
            let try_path = format!("/tmp/dump{}_{:03}", rw, idx);
            if let Ok(file) = std::fs::File::create_new(try_path) {
                return file;
            }
            idx += 1;
        }
    }

    pub fn write_complex32_symbols(
        file: &mut std::fs::File,
        symbols: &[Complex32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for iq in symbols {
            file.write_all(&iq.re.to_be_bytes())?;
            file.write_all(&iq.im.to_be_bytes())?;
        }
        Ok(())
    }
}

pub trait InvertDimensions {
    fn invert_dimensions(&mut self);
}
impl InvertDimensions for (usize, usize, usize) {
    fn invert_dimensions(&mut self) {
        std::mem::swap(&mut self.0, &mut self.2);
    }
}
