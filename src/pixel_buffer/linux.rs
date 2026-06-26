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

use crate::pixel_buffer::*;

pub struct NV12PixelBuffer {}

impl PixelBuffer for NV12PixelBuffer {
    fn plane_row_len(&self, _pixel_component_type: PixelComponentType) -> usize {
        todo!();
    }
    fn plane_height(&self, _pixel_component_type: PixelComponentType) -> usize {
        todo!();
    }
    fn from_frame_view(
        _y_components: &transform_block_3d::FrameComponentView<YPixelComponentType>,
        _cb_components: &transform_block_3d::FrameComponentView<CbPixelComponentType>,
        _cr_components: &transform_block_3d::FrameComponentView<CrPixelComponentType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        todo!();
    }
    fn access_guard<'a>(&'a self) -> Box<dyn PixelBufferAccessGuard<Self> + 'a> {
        todo!();
    }
    fn access_guard_mut<'a>(&'a mut self) -> Box<dyn PixelBufferAccessGuardMut<Self> + 'a> {
        todo!();
    }
}
