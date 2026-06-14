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

use ndarray;
use transform_block_3d::*;

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
    pub fn plane_index(&self) -> usize {
        match self {
            Self::Y => 0,
            Self::Cb | Self::Cr => 1,
        }
    }
    pub fn interleave_offset(self) -> usize {
        match self {
            PixelComponentType::Y | PixelComponentType::Cb => 0,
            PixelComponentType::Cr => 1,
        }
    }
    pub fn interleave_step(self) -> usize {
        match self {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        }
    }
    pub fn vertical_subsampling(self) -> usize {
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
pub trait PixelBuffer: Sized {
    fn plane_row_len(&self, pixel_component_type: PixelComponentType) -> usize;
    fn plane_height(&self, pixel_component_type: PixelComponentType) -> usize;
    fn from_frame_view(
        y_components: &FrameComponentView<YPixelComponentType>,
        cb_components: &FrameComponentView<CbPixelComponentType>,
        cr_components: &FrameComponentView<CrPixelComponentType>,
    ) -> Result<Self, Box<dyn std::error::Error>>;

    fn access_guard<'a>(&'a self) -> Box<dyn PixelBufferAccessGuard<Self> + 'a>;
    fn access_guard_mut<'a>(&'a mut self) -> Box<dyn PixelBufferAccessGuardMut<Self> + 'a>;
}
pub trait PixelBufferAccessGuard<PB: PixelBuffer> {
    fn pixel_buffer(&self) -> &PB;
    fn get_ptr(&self, plane_index: usize) -> *const u8;
}
pub trait PixelBufferAccessGuardMut<PB: PixelBuffer>: PixelBufferAccessGuard<PB> {
    fn pixel_buffer_mut(&mut self) -> &mut PB; // exclusive borrow
    fn get_ptr_mut(&mut self, plane_index: usize) -> *mut u8;
}

pub struct MacroBlock3DIterator<I: Iterator<Item = PB>, PB: PixelBuffer> {
    pixel_buffer_iter: I,
    gop_len: usize,
    _marker: std::marker::PhantomData<PB>,
}
impl<I: Iterator<Item = PB>, PB: PixelBuffer> MacroBlock3DIterator<I, PB> {
    pub fn new(pixel_buffer_iter: I, gop_len: usize) -> Self {
        MacroBlock3DIterator {
            pixel_buffer_iter,
            gop_len,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn pixel_buffer_iter(self) -> PixelBufferIterator<Self, PB> {
        PixelBufferIterator::from(self)
    }
}
impl<I: Iterator<Item = PB>, PB: PixelBuffer> GOPLen for MacroBlock3DIterator<I, PB> {
    fn gop_len(&self) -> usize {
        self.gop_len
    }
}
impl<I: Iterator<Item = PB>, PB: PixelBuffer> Iterator for MacroBlock3DIterator<I, PB> {
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
            let access_guard = pixel_buffer.access_guard();
            let access_guard = &*access_guard;

            y_block
                .populate_next_frame(access_guard)
                .expect("Populating Y block failed.");
            cb_block
                .populate_next_frame(access_guard)
                .expect("Populating Cb block failed.");
            cr_block
                .populate_next_frame(access_guard)
                .expect("Populating Cr block failed.");

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

pub fn copy_frame<SrcType, DstType>(
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
pub fn assign_values<
    PixelType: HasPixelComponentType,
    PB: PixelBuffer,
    Guard: PixelBufferAccessGuardMut<PB>,
>(
    src: &FrameComponentView<PixelType>,
    dst_guard: &mut Guard,
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

    let dst = dst_guard.pixel_buffer_mut();
    let dst_len = dst.plane_row_len(pixel_type) * dst.plane_height(pixel_type);
    assert_eq!(src_len * interleave_step, dst_len);

    let dst_ptr = dst_guard.get_ptr_mut(plane_index);

    copy_frame(
        src_ptr,
        dst_ptr,
        dst_len,
        false,
        interleave_offset,
        interleave_step,
    );
    Ok(())
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

        pub(super) fn populate_next_frame<PB: PixelBuffer>(
            &mut self,
            pixel_buffer_access_guard: &dyn PixelBufferAccessGuard<PB>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let pixel_buffer = pixel_buffer_access_guard.pixel_buffer();
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

            let plane_index = pixel_type.plane_index();
            let src_ptr = pixel_buffer_access_guard.get_ptr(plane_index);

            let dst_ptr = values_2d
                .get_mut_ptr([0, 0])
                .ok_or("Failed to get mut_ptr.")?;
            copy_frame(
                src_ptr,
                dst_ptr,
                dst_len,
                true,
                pixel_type.interleave_offset(),
                pixel_type.interleave_step(),
            );

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

    pub struct PixelBufferIterator<I: Iterator<Item = MacroBlock3D>, PB: PixelBuffer> {
        macro_block_3d_iterator: I,
        current_macro_block: Option<MacroBlock3D>,
        frame_index: usize,
        gop_length: usize,
        _marker: std::marker::PhantomData<PB>,
    }

    impl<I: Iterator<Item = MacroBlock3D>, PB: PixelBuffer> PixelBufferIterator<I, PB> {
        pub fn new(macro_block_3d_iterator: I, gop_length: usize) -> Self {
            Self {
                macro_block_3d_iterator,
                current_macro_block: None,
                frame_index: 0,
                gop_length,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<I: Iterator<Item = MacroBlock3D>, PB: PixelBuffer> Iterator for PixelBufferIterator<I, PB> {
        type Item = PB;
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

            let pixel_buffer = PB::from_frame_view(&y_components, &cb_components, &cr_components)
                .expect("Failed to create pixel buffer.");

            self.frame_index += 1;
            if self.frame_index == self.gop_length {
                self.frame_index = 0;
                self.current_macro_block = None;
            }

            Some(pixel_buffer)
        }
    }

    impl<I: Iterator<Item = MacroBlock3D> + GOPLen, PB: PixelBuffer> From<I>
        for PixelBufferIterator<I, PB>
    {
        fn from(macro_block_iter: I) -> Self {
            let gop_len = macro_block_iter.gop_len();
            Self::new(macro_block_iter, gop_len)
        }
    }
    impl<PB: PixelBuffer> From<MacroBlock3D>
        for PixelBufferIterator<std::array::IntoIter<MacroBlock3D, 1>, PB>
    {
        fn from(macro_block_3d: MacroBlock3D) -> Self {
            let gop_len = macro_block_3d.gop_len();
            let macro_block_iter = [macro_block_3d].into_iter();
            Self::new(macro_block_iter, gop_len)
        }
    }

    pub struct FrameComponentView<'a, PixelType: HasPixelComponentType> {
        pub values: ndarray::ArrayView2<'a, f32>,
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
