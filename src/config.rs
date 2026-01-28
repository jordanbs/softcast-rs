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

pub struct Config {
    chunk_dimensions: (usize, usize, usize),
    asset_resolution: (usize, usize),
    gop_length: usize,
}

impl Config {
    pub fn chunks_per_gop(&self, pixel_type: PixelComponentType) -> usize {
        self.gop_length
            * self.chunk_dimensions.0
            * self.chunk_dimensions.1
            * self.chunk_dimensions.2
            / (pixel_type.interleave_step() * pixel_type.vertical_subsampling())
    }
}

pub trait ConfigProvider {
    fn config(&self) -> Config;
}
