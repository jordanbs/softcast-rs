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

use transform_block_3d::*;

use ndarray;
use std::ptr::NonNull;

use objc2_core_foundation::{CFRetained, kCFAllocatorDefault};
use objc2_core_video::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PixelComponentType {
    Y,
    Cb,
    Cr,
}
impl std::fmt::Display for PixelComponentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Y => "Y",
                Self::Cb => "Cb",
                Self::Cr => "Cr",
            }
        )
    }
}
impl PixelComponentType {
    fn plane_index(&self) -> usize {
        match self {
            Self::Y => 0,
            Self::Cb | Self::Cr => 1,
        }
    }
    fn interleave_offset(self) -> usize {
        match self {
            PixelComponentType::Y | PixelComponentType::Cb => 0,
            PixelComponentType::Cr => 1,
        }
    }
    pub(super) fn interleave_step(self) -> usize {
        match self {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        }
    }
    pub(super) fn vertical_subsampling(self) -> usize {
        match self {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        }
    }
}

pub trait HasPixelComponentType: std::fmt::Debug {
    const TYPE: PixelComponentType;
}

#[derive(Debug, Clone, PartialEq)]
pub struct YPixelComponentType;
#[derive(Debug, Clone, PartialEq)]
pub struct CbPixelComponentType;
#[derive(Debug, Clone, PartialEq)]
pub struct CrPixelComponentType;

impl HasPixelComponentType for YPixelComponentType {
    const TYPE: PixelComponentType = PixelComponentType::Y;
}
impl HasPixelComponentType for CbPixelComponentType {
    const TYPE: PixelComponentType = PixelComponentType::Cb;
}
impl HasPixelComponentType for CrPixelComponentType {
    const TYPE: PixelComponentType = PixelComponentType::Cr;
}

pub trait DomainShiftedAs<Dst>
where
    Dst: Copy,
{
    fn domain_shifted_as_(&self) -> Dst;
}

impl DomainShiftedAs<u8> for f32 {
    fn domain_shifted_as_(&self) -> u8 {
        self.uncenter_u8_domain().round() as u8
    }
}

impl DomainShiftedAs<f32> for u8 {
    fn domain_shifted_as_(&self) -> f32 {
        (*self as f32).center_u8_domain()
    }
}

trait ShiftU8Domain {
    const CENTER_SHIFT: f32 = 127.5;

    fn center_u8_domain(&self) -> Self;
    fn uncenter_u8_domain(&self) -> Self;
}
impl ShiftU8Domain for f32 {
    fn center_u8_domain(&self) -> Self {
        self - Self::CENTER_SHIFT
    }
    fn uncenter_u8_domain(&self) -> Self {
        self + Self::CENTER_SHIFT
    }
}

pub trait GOPLen {
    fn gop_len(&self) -> usize;
}

// Holds a single frame
#[derive(Debug)]
pub struct PixelBuffer {
    pub(super) cv_image_buffer: CFRetained<CVImageBuffer>,
}

impl PixelBuffer {
    pub fn new(cv_image_buffer: CFRetained<CVImageBuffer>) -> Self {
        assert!(CVPixelBufferIsPlanar(&cv_image_buffer));

        PixelBuffer { cv_image_buffer }
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

    fn from_frame_view(
        y_components: &FrameComponentView<YPixelComponentType>,
        cb_components: &FrameComponentView<CbPixelComponentType>,
        cr_components: &FrameComponentView<CrPixelComponentType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        fn assign_values<PixelType: HasPixelComponentType>(
            src: &FrameComponentView<PixelType>,
            dst: &CVPixelBuffer,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let src_ptr = src
                .values
                .get_ptr([0, 0])
                .ok_or("Could not get TransformBlock ptr.")?;
            let src_len = src.values.len();

            let pixel_type = PixelType::TYPE;
            let interleave_step = pixel_type.interleave_step();
            let interleave_offset = pixel_type.interleave_offset();
            let plane_index = pixel_type.plane_index();

            let dst_ptr = CVPixelBufferGetBaseAddressOfPlane(dst, plane_index) as *mut u8;
            let dst_len = CVPixelBufferGetBytesPerRowOfPlane(dst, plane_index)
                * CVPixelBufferGetHeightOfPlane(dst, plane_index);
            assert_eq!(src_len * interleave_step, dst_len);

            PixelBuffer::copy_frame(
                src_ptr,
                dst_ptr,
                dst_len,
                false,
                interleave_offset,
                interleave_step,
            );
            Ok(())
        }

        let (height, width) = y_components.values.dim();
        let cv_pixel_buffer = Self::new_cv_pixel_buffer(width, height)?;

        unsafe {
            let flags = CVPixelBufferLockFlags::empty(); // empty means write
            CVPixelBufferLockBaseAddress(&cv_pixel_buffer, flags);
        };

        assign_values(&y_components, &cv_pixel_buffer)?; // TODO: implement Drop for guarding cleanup
        assign_values(&cb_components, &cv_pixel_buffer)?;
        assign_values(&cr_components, &cv_pixel_buffer)?;

        unsafe {
            let flags = CVPixelBufferLockFlags::empty(); // empty means write
            CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);
        };

        Ok(Self {
            cv_image_buffer: cv_pixel_buffer,
        })
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

    pub(super) fn copy_frame<SrcType, DstType>(
        src_ptr: *const SrcType,
        dst_ptr: *mut DstType,
        dst_len: usize,
        interleave_src: bool,
        interleave_offset: usize,
        interleave_step: usize,
    ) where
        DstType: Copy,
        SrcType: DomainShiftedAs<DstType>,
    {
        unsafe {
            let mut src_ptr = src_ptr;
            let mut dst_ptr = dst_ptr;

            if interleave_src {
                src_ptr = src_ptr.add(interleave_offset);
            } else {
                dst_ptr = dst_ptr.add(interleave_offset);
            }

            let dst_ptr_end = dst_ptr.add(dst_len);
            while dst_ptr < dst_ptr_end {
                *dst_ptr = (*src_ptr).domain_shifted_as_();

                if interleave_src {
                    src_ptr = src_ptr.add(interleave_step);
                    dst_ptr = dst_ptr.add(1);
                } else {
                    src_ptr = src_ptr.add(1);
                    dst_ptr = dst_ptr.add(interleave_step);
                }
            }
        }
    }

    // The following functions are safe to call without locking the base address of CVPixelBuffer.

    pub fn plane_row_len(&self, pixel_component_type: PixelComponentType) -> usize {
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
    pub fn plane_height(&self, pixel_component_type: PixelComponentType) -> usize {
        CVPixelBufferGetHeightOfPlane(&self.cv_image_buffer, pixel_component_type.plane_index())
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

impl PartialEq for PixelBuffer {
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

pub struct MacroBlock3DIterator<I: Iterator<Item = PixelBuffer>> {
    pixel_buffer_iter: I,
    gop_len: usize,
}
impl<I: Iterator<Item = PixelBuffer>> MacroBlock3DIterator<I> {
    pub fn new(pixel_buffer_iter: I, gop_len: usize) -> Self {
        MacroBlock3DIterator {
            pixel_buffer_iter,
            gop_len,
        }
    }
    pub fn pixel_buffer_iter(self) -> PixelBufferIterator<Self> {
        PixelBufferIterator::from(self)
    }
}
impl<I: Iterator<Item = PixelBuffer>> GOPLen for MacroBlock3DIterator<I> {
    fn gop_len(&self) -> usize {
        self.gop_len
    }
}
impl<I: Iterator<Item = PixelBuffer>> Iterator for MacroBlock3DIterator<I> {
    // Output all three TransformBlocks at once to linearly process frames
    type Item = MacroBlock3D;

    fn next(&mut self) -> Option<Self::Item> {
        let MacroBlock3D {
            y_components: mut y_block,
            cb_components: mut cb_block,
            cr_components: mut cr_block,
            ..
        } = MacroBlock3D::new(self.gop_len);

        let mut pixel_buffer_iterator_is_empty = true;
        for pixel_buffer in self.pixel_buffer_iter.by_ref().take(self.gop_len) {
            pixel_buffer.lock_base_address(true);

            y_block
                .populate_next_frame(&pixel_buffer)
                .expect("Populating Y block failed.");
            cb_block
                .populate_next_frame(&pixel_buffer)
                .expect("Populating Cb block failed.");
            cr_block
                .populate_next_frame(&pixel_buffer)
                .expect("Populating Cr block failed.");

            pixel_buffer.unlock_base_address(true);
            pixel_buffer_iterator_is_empty = false;
        }
        if pixel_buffer_iterator_is_empty {
            return None;
        }

        Some(MacroBlock3D {
            y_components: y_block,
            cb_components: cb_block,
            cr_components: cr_block,
            gop_len: self.gop_len,
        })
    }
}

pub mod transform_block_3d {
    use super::*;
    use std::cell::OnceCell;

    #[derive(Debug, Clone)]
    pub struct TransformBlock3D<PixelType: HasPixelComponentType> {
        pub values_cell: OnceCell<ndarray::Array3<f32>>,
        pub populated_frames_len: usize,
        gop_len: usize,
        _marker: std::marker::PhantomData<PixelType>,
    }
    impl<PixelType: HasPixelComponentType> GOPLen for TransformBlock3D<PixelType> {
        fn gop_len(&self) -> usize {
            self.gop_len
        }
    }
    impl<PixelType: HasPixelComponentType> TransformBlock3D<PixelType> {
        pub fn new(gop_len: usize) -> Self {
            Self {
                values_cell: OnceCell::new(),
                populated_frames_len: 0,
                gop_len,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn with_values(values: ndarray::Array3<f32>) -> Self {
            let gop_len = values.dim().0;

            let once_cell = OnceCell::new();
            once_cell.set(values).expect("Failed to set once_cell");

            Self {
                values_cell: once_cell,
                populated_frames_len: gop_len,
                gop_len,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn values(&self) -> &ndarray::Array3<f32> {
            self.values_cell
                .get()
                .expect("Values not initialized. Must call populate_next_frame first.")
        }

        pub fn consume_values(self) -> ndarray::Array3<f32> {
            self.values_cell
                .into_inner()
                .expect("Values not initialized. Must call populate_next_frame first.")
        }

        pub(super) fn populate_next_frame(
            &mut self,
            pixel_buffer: &PixelBuffer,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let frame_idx = self.populated_frames_len;
            self.populated_frames_len += 1;

            let _ = self.values_cell.get_or_init(|| {
                let block_width =
                    pixel_buffer.plane_row_len(PixelType::TYPE) / PixelType::TYPE.interleave_step();
                let block_height = pixel_buffer.plane_height(PixelType::TYPE);
                // length, height, width to match the memory layout of CVPixelBuffer
                ndarray::Array3::zeros((self.gop_len, block_height, block_width))
            });
            let values = self.values_cell.get_mut().unwrap(); // get_mut_or_init is nightly-only.

            // Axis(0) is the length/depth dimension
            let mut values_2d = values.index_axis_mut(ndarray::Axis(0), frame_idx);
            assert!(values_2d.is_standard_layout()); // standard_layout = contiguous memory layout

            let pixel_type = PixelType::TYPE;

            let (dst_height, dst_width) = values_2d.dim();
            let dst_len = dst_width * dst_height;

            pixel_buffer.lock_base_address(true);

            let plane_index = pixel_type.plane_index();

            let src_ptr =
                CVPixelBufferGetBaseAddressOfPlane(&pixel_buffer.cv_image_buffer, plane_index)
                    as *const u8;

            let dst_ptr = values_2d
                .get_mut_ptr([0, 0])
                .ok_or("Failed to get mut_ptr.")?;
            PixelBuffer::copy_frame(
                src_ptr,
                dst_ptr,
                dst_len,
                true,
                pixel_type.interleave_offset(),
                pixel_type.interleave_step(),
            );

            pixel_buffer.unlock_base_address(true);

            Ok(())
        }

        pub(super) fn frame_view(
            &self,
            frame_idx: usize,
        ) -> Result<FrameComponentView<'_, PixelType>, Box<dyn std::error::Error>> {
            let arr = self.values().index_axis(ndarray::Axis(0), frame_idx);
            Ok(FrameComponentView::new(arr))
        }
    }
    impl<PixelType: HasPixelComponentType> PartialEq for TransformBlock3D<PixelType> {
        fn eq(&self, other: &Self) -> bool {
            if self.values().len() != other.values().len() {
                return false;
            }
            if self.values().shape() != other.values().shape() {
                return false;
            }
            for (self_v, other_v) in self.values().iter().zip(other.values().iter()) {
                // f32::EPSILON is too small
                if (self_v - other_v).abs() > 0.001 {
                    return false;
                }
            }
            true
        }
    }

    // 4:2:0
    #[derive(Clone, Debug)]
    pub struct MacroBlock3D {
        pub y_components: TransformBlock3D<YPixelComponentType>,
        pub cb_components: TransformBlock3D<CbPixelComponentType>,
        pub cr_components: TransformBlock3D<CrPixelComponentType>,
        pub gop_len: usize,
    }
    impl MacroBlock3D {
        pub fn new(gop_len: usize) -> Self {
            Self {
                y_components: TransformBlock3D::new(gop_len),
                cb_components: TransformBlock3D::new(gop_len),
                cr_components: TransformBlock3D::new(gop_len),
                gop_len,
            }
        }
    }
    impl GOPLen for MacroBlock3D {
        fn gop_len(&self) -> usize {
            self.gop_len
        }
    }

    pub struct PixelBufferIterator<I: Iterator<Item = MacroBlock3D>> {
        macro_block_3d_iterator: I,
        current_macro_block: Option<MacroBlock3D>,
        frame_index: usize,
        gop_length: usize,
    }

    impl<I: Iterator<Item = MacroBlock3D>> PixelBufferIterator<I> {
        pub fn new(macro_block_3d_iterator: I, gop_length: usize) -> Self {
            Self {
                macro_block_3d_iterator,
                current_macro_block: None,
                frame_index: 0,
                gop_length,
            }
        }
    }

    impl<I: Iterator<Item = MacroBlock3D>> Iterator for PixelBufferIterator<I> {
        type Item = PixelBuffer;
        fn next(&mut self) -> Option<Self::Item> {
            let macro_block_3d = match self.current_macro_block {
                Some(ref macro_block_3d) => macro_block_3d,
                None => self
                    .current_macro_block
                    .insert(self.macro_block_3d_iterator.next()?),
            };
            // A MacroBlock3D can be shorter than it's populated_frames_len
            if self.frame_index == macro_block_3d.y_components.populated_frames_len {
                self.frame_index = 0;
                self.current_macro_block = None;
                return self.next();
            }

            let y_components = macro_block_3d
                .y_components
                .frame_view(self.frame_index)
                .expect("Failed to get Y components.");
            let cb_components = macro_block_3d
                .cb_components
                .frame_view(self.frame_index)
                .expect("Failed to get Cb components.");
            let cr_components = macro_block_3d
                .cr_components
                .frame_view(self.frame_index)
                .expect("Failed to get Cr components.");

            let pixel_buffer =
                PixelBuffer::from_frame_view(&y_components, &cb_components, &cr_components)
                    .expect("Failed to create pixel buffer.");

            self.frame_index += 1;
            if self.frame_index == self.gop_length {
                self.frame_index = 0;
                self.current_macro_block = None;
            }

            Some(pixel_buffer)
        }
    }

    impl<I: Iterator<Item = MacroBlock3D> + GOPLen> From<I> for PixelBufferIterator<I> {
        fn from(macro_block_iter: I) -> Self {
            let gop_len = macro_block_iter.gop_len();
            Self::new(macro_block_iter, gop_len)
        }
    }
    impl From<MacroBlock3D> for PixelBufferIterator<std::array::IntoIter<MacroBlock3D, 1>> {
        fn from(macro_block_3d: MacroBlock3D) -> Self {
            let gop_len = macro_block_3d.gop_len();
            let macro_block_iter = [macro_block_3d].into_iter();
            Self::new(macro_block_iter, gop_len)
        }
    }

    pub(super) struct FrameComponentView<'a, PixelType: HasPixelComponentType> {
        pub(super) values: ndarray::ArrayView2<'a, f32>,
        _marker: std::marker::PhantomData<PixelType>,
    }
    impl<'a, PixelType: HasPixelComponentType> FrameComponentView<'a, PixelType> {
        fn new(values: ndarray::ArrayView2<'a, f32>) -> Self {
            assert!(values.is_standard_layout());

            FrameComponentView {
                values,
                _marker: std::marker::PhantomData,
            }
        }
    }
}
