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

use super::transform_block_3d::*;
use super::*;

use std::ptr::NonNull;

use objc2_core_foundation::{CFRetained, kCFAllocatorDefault};
use objc2_core_video::*;

#[derive(Debug)]
pub struct CVPixelBufferWrapper {
    pub cv_image_buffer: CFRetained<CVImageBuffer>,
}

impl CVPixelBufferWrapper {
    pub fn new(cv_image_buffer: CFRetained<CVImageBuffer>) -> Self {
        assert!(CVPixelBufferIsPlanar(&cv_image_buffer));

        Self { cv_image_buffer }
    }

    fn new_cv_pixel_buffer(
        width: usize,
        height: usize,
    ) -> Result<CFRetained<CVPixelBuffer>, Box<dyn std::error::Error>> {
        unsafe {
            let mut cv_pixel_buffer: *mut CVPixelBuffer = std::ptr::null_mut();
            let pixel_buffer_out: NonNull<*mut CVPixelBuffer> = NonNull::from(&mut cv_pixel_buffer);
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                None,
                pixel_buffer_out,
            );
            if status != 0 {
                return Err(format!("Failed to create CVPixelBuffer with error {}", status).into());
            }

            let cv_pixel_buffer = CFRetained::from_raw(
                // coerce the mutable ptr into an immutable ptr
                NonNull::new(cv_pixel_buffer).ok_or("CVPixelBuffer is NULL")?,
            );
            if !CVPixelBufferIsPlanar(&cv_pixel_buffer) {
                return Err("New CVPixelBuffer is not planar.".into());
            }
            Ok(cv_pixel_buffer)
        }
    }

    pub fn lock_base_address(&self, read_only: bool) {
        let flags = match read_only {
            true => CVPixelBufferLockFlags::ReadOnly,
            false => CVPixelBufferLockFlags::empty(),
        };
        unsafe {
            CVPixelBufferLockBaseAddress(&self.cv_image_buffer, flags);
        }
    }
    pub fn unlock_base_address(&self, read_only: bool) {
        let flags = match read_only {
            true => CVPixelBufferLockFlags::ReadOnly,
            false => CVPixelBufferLockFlags::empty(),
        };
        unsafe {
            CVPixelBufferUnlockBaseAddress(&self.cv_image_buffer, flags);
        }
    }
    pub fn resolution(&self) -> (usize, usize) {
        let width = CVPixelBufferGetWidth(&self.cv_image_buffer);
        let height = CVPixelBufferGetHeight(&self.cv_image_buffer);
        (width, height)
    }

    pub fn dump_file(&self, prefix: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::slice;
        use std::sync::atomic;

        unsafe {
            let flags = CVPixelBufferLockFlags::ReadOnly;
            CVPixelBufferLockBaseAddress(&self.cv_image_buffer, flags);

            let y_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.cv_image_buffer,
                PixelComponentType::Y.plane_index(),
            ) as *const u8;
            let y_bytes_per_row = self.plane_row_len(PixelComponentType::Y);
            let y_height = self.plane_height(PixelComponentType::Y);
            let y_bytes: &[u8] = slice::from_raw_parts(y_ptr, y_bytes_per_row * y_height);

            let cbcr_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.cv_image_buffer,
                PixelComponentType::Cb.plane_index(),
            ) as *const u8;
            let cbcr_bytes_per_row = self.plane_row_len(PixelComponentType::Cb);
            let cbcr_height = self.plane_height(PixelComponentType::Cb);
            let cbcr_bytes: &[u8] =
                slice::from_raw_parts(cbcr_ptr, cbcr_bytes_per_row * cbcr_height);

            static Y_COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
            static CBCR_COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(0);

            let y_path = format!(
                "/tmp/{}_Y_{:04}.out",
                prefix,
                Y_COUNTER.fetch_add(1, atomic::Ordering::Relaxed)
            );
            let cbcr_path = format!(
                "/tmp/{}_CbCr_{:04}.out",
                prefix,
                CBCR_COUNTER.fetch_add(1, atomic::Ordering::Relaxed)
            );

            fs::write(y_path, y_bytes)?;
            fs::write(cbcr_path, cbcr_bytes)?;

            CVPixelBufferUnlockBaseAddress(&self.cv_image_buffer, flags);
        }

        Ok(())
    }
}

struct CVPixelBufferAccessGuard<'a> {
    pixel_buffer: &'a CVPixelBufferWrapper,
}
struct CVPixelBufferAccessGuardMut<'a> {
    pixel_buffer: &'a mut CVPixelBufferWrapper,
}
impl PixelBufferAccessGuard for CVPixelBufferAccessGuard<'_> {}
impl PixelBufferAccessGuard for CVPixelBufferAccessGuardMut<'_> {}
impl<'a> PixelBufferAccessGuardMut<CVPixelBufferWrapper> for CVPixelBufferAccessGuardMut<'a> {
    fn pixel_buffer_mut(&mut self) -> &mut CVPixelBufferWrapper {
        self.pixel_buffer
    }
}
impl<'a> CVPixelBufferAccessGuard<'a> {
    fn new(pixel_buffer: &'a CVPixelBufferWrapper) -> Self {
        pixel_buffer.lock_base_address(true);
        Self { pixel_buffer }
    }
}
impl<'a> CVPixelBufferAccessGuardMut<'a> {
    fn new(pixel_buffer: &'a mut CVPixelBufferWrapper) -> Self {
        pixel_buffer.lock_base_address(false);
        Self { pixel_buffer }
    }
}
impl Drop for CVPixelBufferAccessGuard<'_> {
    fn drop(&mut self) {
        self.pixel_buffer.unlock_base_address(true);
    }
}
impl Drop for CVPixelBufferAccessGuardMut<'_> {
    fn drop(&mut self) {
        self.pixel_buffer.unlock_base_address(false);
    }
}

impl PixelBuffer for CVPixelBufferWrapper {
    fn get_ptr(&self, plane_index: usize) -> *const u8 {
        CVPixelBufferGetBaseAddressOfPlane(&self.cv_image_buffer, plane_index) as *const u8
    }
    fn get_ptr_mut(&mut self, plane_index: usize) -> *mut u8 {
        CVPixelBufferGetBaseAddressOfPlane(&self.cv_image_buffer, plane_index) as *mut u8
    }
    fn access_guard<'a>(&'a self) -> Box<dyn PixelBufferAccessGuard + 'a> {
        Box::new(CVPixelBufferAccessGuard::new(self))
    }
    fn access_guard_mut<'a>(&'a mut self) -> Box<dyn PixelBufferAccessGuardMut<Self> + 'a> {
        Box::new(CVPixelBufferAccessGuardMut::new(self))
    }
    // The following functions are safe to call without locking the base address of CVPixelBuffer.
    fn plane_row_len(&self, pixel_component_type: PixelComponentType) -> usize {
        let bytes_per_row = CVPixelBufferGetBytesPerRowOfPlane(
            &self.cv_image_buffer,
            pixel_component_type.plane_index(),
        );

        let (asset_width, _) = self.resolution();
        let expected_bytes_per_row = asset_width;
        assert_eq!(
            expected_bytes_per_row, bytes_per_row,
            "This asset has extra bytes per row of each plane \
                         (expected: {} vs actual: {}); currently not supported.",
            expected_bytes_per_row, bytes_per_row
        );
        bytes_per_row
    }
    fn plane_height(&self, pixel_component_type: PixelComponentType) -> usize {
        CVPixelBufferGetHeightOfPlane(&self.cv_image_buffer, pixel_component_type.plane_index())
    }

    fn from_frame_view(
        y_components: &FrameComponentView<YPixelComponentType>,
        cb_components: &FrameComponentView<CbPixelComponentType>,
        cr_components: &FrameComponentView<CrPixelComponentType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (height, width) = y_components.values.dim();
        let cv_pixel_buffer = Self::new_cv_pixel_buffer(width, height)?;
        let mut wrapper = Self {
            cv_image_buffer: cv_pixel_buffer,
        };
        {
            let mut guard = CVPixelBufferAccessGuardMut::new(&mut wrapper);
            assign_values(&y_components, &mut guard)?;
            assign_values(&cb_components, &mut guard)?;
            assign_values(&cr_components, &mut guard)?;
        }

        Ok(wrapper)
    }
}

impl PartialEq for CVPixelBufferWrapper {
    // A deep comparison. Useful for testing. Bad idea? Should I hide?
    fn eq(&self, other: &Self) -> bool {
        // cheap comparisons first
        if self.cv_image_buffer == other.cv_image_buffer {
            return true;
        }

        let l_cv_y_pixel_buffer_len =
            self.plane_row_len(PixelComponentType::Y) * self.plane_height(PixelComponentType::Y);
        let r_cv_y_pixel_buffer_len =
            other.plane_row_len(PixelComponentType::Y) * other.plane_height(PixelComponentType::Y);
        if l_cv_y_pixel_buffer_len != r_cv_y_pixel_buffer_len {
            return false;
        }

        let l_cv_cbcr_pixel_buffer_len =
            self.plane_row_len(PixelComponentType::Cb) * self.plane_height(PixelComponentType::Cb);
        let r_cv_cbcr_pixel_buffer_len = other.plane_row_len(PixelComponentType::Cb)
            * other.plane_height(PixelComponentType::Cb);
        if l_cv_cbcr_pixel_buffer_len != r_cv_cbcr_pixel_buffer_len {
            return false;
        }
        unsafe {
            let flags = CVPixelBufferLockFlags::ReadOnly;
            CVPixelBufferLockBaseAddress(&self.cv_image_buffer, flags);
            CVPixelBufferLockBaseAddress(&other.cv_image_buffer, flags);

            // TODO: factor into a shared fn.

            let l_cv_y_pixel_buffer_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.cv_image_buffer,
                PixelComponentType::Y.plane_index(),
            ) as *const u8;
            let r_cv_y_pixel_buffer_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &other.cv_image_buffer,
                PixelComponentType::Y.plane_index(),
            ) as *const u8;
            let l_cv_cbcr_pixel_buffer_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.cv_image_buffer,
                PixelComponentType::Cb.plane_index(),
            ) as *const u8;
            let r_cv_cbcr_pixel_buffer_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &other.cv_image_buffer,
                PixelComponentType::Cb.plane_index(),
            ) as *const u8;

            let l_y_slice =
                std::slice::from_raw_parts(l_cv_y_pixel_buffer_ptr, l_cv_y_pixel_buffer_len);
            let r_y_slice =
                std::slice::from_raw_parts(r_cv_y_pixel_buffer_ptr, r_cv_y_pixel_buffer_len);
            let l_cbcr_slice =
                std::slice::from_raw_parts(l_cv_cbcr_pixel_buffer_ptr, l_cv_cbcr_pixel_buffer_len);
            let r_cbcr_slice =
                std::slice::from_raw_parts(r_cv_cbcr_pixel_buffer_ptr, r_cv_cbcr_pixel_buffer_len);

            if l_y_slice.cmp(r_y_slice) != std::cmp::Ordering::Equal {
                CVPixelBufferUnlockBaseAddress(&other.cv_image_buffer, flags);
                CVPixelBufferUnlockBaseAddress(&self.cv_image_buffer, flags);
                return false;
            }
            if l_cbcr_slice.cmp(r_cbcr_slice) != std::cmp::Ordering::Equal {
                CVPixelBufferUnlockBaseAddress(&other.cv_image_buffer, flags);
                CVPixelBufferUnlockBaseAddress(&self.cv_image_buffer, flags);
                return false;
            }

            CVPixelBufferUnlockBaseAddress(&other.cv_image_buffer, flags);
            CVPixelBufferUnlockBaseAddress(&self.cv_image_buffer, flags);
            true
        }
    }
}
