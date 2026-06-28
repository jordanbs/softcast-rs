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

use crate::camera::FrameRequest;
use crate::pixel_buffer::*;

pub struct NV12PixelBuffer {
    request: FrameRequest,
}
impl From<FrameRequest> for NV12PixelBuffer {
    fn from(request: FrameRequest) -> Self {
        Self { request }
    }
}

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
        let guard = NV12PixelBufferAccessGuard::from(self);
        Box::new(guard)
    }
    fn access_guard_mut<'a>(&'a mut self) -> Box<dyn PixelBufferAccessGuardMut<Self> + 'a> {
        todo!();
    }
}

struct NV12PixelBufferAccessGuard<'a> {
    pixel_buffer: &'a NV12PixelBuffer,
}
impl<'a> From<&'a NV12PixelBuffer> for NV12PixelBufferAccessGuard<'a> {
    fn from(pixel_buffer: &'a NV12PixelBuffer) -> Self {
        Self { pixel_buffer }
    }
}
impl<'a> PixelBufferAccessGuard<NV12PixelBuffer> for NV12PixelBufferAccessGuard<'a> {
    fn pixel_buffer(&self) -> &NV12PixelBuffer {
        &self.pixel_buffer
    }
    fn get_ptr(&self, plane_index: usize) -> *const u8 {
        let frame_request = &self.pixel_buffer.request;
        let frame_buffer: &libcamera::framebuffer_map::MemoryMappedFrameBuffer<
            libcamera::framebuffer_allocator::FrameBuffer,
        > = frame_request
            .request
            .as_ref()
            .expect("No request")
            .buffer(&frame_request.stream)
            .expect("no frame buffer");
        let planes = frame_buffer.data();
        planes.get(plane_index).expect("No plane at index").as_ptr()
    }
}
