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

#![cfg(not(target_vendor = "apple"))]

use crate::pixel_buffer::*;

struct Camera {}

impl Iterator for Camera {
    type Item = NV12PixelBuffer;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl Camera {
    pub fn start(&mut self) {}
}
