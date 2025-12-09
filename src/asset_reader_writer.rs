// Copyright 2025 Jordan Schneider
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
use std::{error, path, ptr::NonNull};

use objc2_foundation::{NSArray, NSDictionary, NSMutableDictionary, NSNumber, NSString, NSURL};

use objc2::{rc::Retained, runtime::AnyObject};
use objc2_av_foundation::{
    AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput,
    AVAssetTrack, AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor,
    AVAssetWriterStatus, AVFileTypeMPEG4, AVMediaTypeVideo, AVURLAsset, AVVideoCodecKey,
    AVVideoCodecTypeH264, AVVideoHeightKey, AVVideoWidthKey,
};

use objc2_core_foundation::{kCFAllocatorDefault, CFRetained, CFString};

use objc2_core_media::CMTime;

use objc2_core_video::{
    kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, CVImageBuffer, CVPixelBuffer,
    CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
    CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetWidth,
    CVPixelBufferIsPlanar, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};

use std::thread::sleep;
use std::time::{Duration, SystemTime};

pub mod asset_reader {
    use super::*;
    use pixel_buffer::*;

    pub struct AssetReader {
        path: path::PathBuf,
        loaded_reader: Option<LoadedAssetReader>,
    }

    impl AssetReader {
        pub fn new(file_path: &str) -> Self {
            AssetReader {
                path: path::PathBuf::from(file_path),
                loaded_reader: None,
            }
        }

        fn loaded_reader(&mut self) -> Result<LoadedAssetReader, Box<dyn error::Error>> {
            if self.loaded_reader.is_none() {
                self.loaded_reader = Some(LoadedAssetReader::load(self.path.as_path())?)
            }
            Ok(self.loaded_reader.as_ref().unwrap().clone())
        }

        fn av_asset_reader(&mut self) -> Result<Retained<AVAssetReader>, Box<dyn error::Error>> {
            Ok(self.loaded_reader()?.av_asset_reader)
        }

        fn av_asset_output(
            &mut self,
        ) -> Result<Retained<AVAssetReaderTrackOutput>, Box<dyn error::Error>> {
            Ok(self.loaded_reader()?.av_asset_output)
        }

        #[allow(deprecated)] // blocking i/o is expected here
        pub(super) fn get_next_pixel_buffer(
            &mut self,
        ) -> Result<Option<PixelBuffer>, Box<dyn error::Error>> {
            let av_reader = self.av_asset_reader()?;
            let av_output = self.av_asset_output()?;

            unsafe {
                // copyNextSampleBuffer returns the next CMSampleBufferRef or nil at EOF.
                if let Some(sample_buffer) = av_output.copyNextSampleBuffer() {
                    let cv_pixel_buffer = sample_buffer
                        .image_buffer()
                        .ok_or("Failed to get CVPixelBuffer.")?;

                    let pixel_buffer = PixelBuffer::new(cv_pixel_buffer);
                    return Ok(Some(pixel_buffer));
                }
                // No sample buffer, see if we've reached the end of file.
                match av_reader.status() {
                    AVAssetReaderStatus::Completed => Ok(None),
                    status => Err(format!("Reader stopped with status {:?}", status).into()),
                }
            }
        }

        pub fn pixel_buffer_iter(&mut self) -> PixelBufferIterator<'_> {
            PixelBufferIterator::new(self)
        }

        pub fn resolution(&mut self) -> Result<(i32, i32), Box<dyn error::Error>> {
            self.loaded_reader()?.resolution()
        }
        pub fn frame_rate(&mut self) -> Result<f64, Box<dyn error::Error>> {
            self.loaded_reader()?.frame_rate()
        }
    }

    #[derive(Clone)]
    struct LoadedAssetReader {
        av_asset_reader: Retained<AVAssetReader>,
        av_asset_output: Retained<AVAssetReaderTrackOutput>,
        av_asset_track: Retained<AVAssetTrack>,
    }

    impl LoadedAssetReader {
        #[allow(deprecated)] // blocking i/o is expected here
        fn load(path: &path::Path) -> Result<Self, Box<dyn error::Error>> {
            let path_bytes = path.as_os_str().as_encoded_bytes();
            let path_str = std::str::from_utf8(path_bytes)?;
            let ns_path = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath_isDirectory(&ns_path, false);

            unsafe {
                // Reader config.
                let pixel_format_key: &NSString =
                    &*(kCVPixelBufferPixelFormatTypeKey as *const CFString as *const NSString);
                let pixel_format_value =
                    NSNumber::new_u32(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);

                let video_settings: Retained<NSDictionary<NSString, AnyObject>> =
                    NSDictionary::from_slices::<NSString>(
                        &[pixel_format_key.as_ref()],
                        &[pixel_format_value.as_ref()],
                    );

                // Asset / track / reader / output setup.
                let asset: Retained<AVURLAsset> = AVURLAsset::assetWithURL(&url);

                // Get all video tracks.
                let tracks: Retained<NSArray<AVAssetTrack>> =
                    asset.tracksWithMediaType(&AVMediaTypeVideo.unwrap());

                let track: Retained<AVAssetTrack> =
                    tracks.firstObject().ok_or("File has no video tracks.")?;

                let reader: Retained<AVAssetReader> =
                    AVAssetReader::assetReaderWithAsset_error(&asset as &AVAsset)?;

                // Attach a track output that will give us CVPixelBuffer-backed CMSampleBuffers.
                let output: Retained<AVAssetReaderTrackOutput> =
                    AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(
                        &track,
                        Some(&video_settings),
                    );

                reader.addOutput(&output as &AVAssetReaderOutput);

                if !reader.startReading() {
                    return Err("startReading() failed".into());
                }

                Ok(LoadedAssetReader {
                    av_asset_reader: reader,
                    av_asset_output: output,
                    av_asset_track: track,
                })
            }
        }
        pub fn resolution(&mut self) -> Result<(i32, i32), Box<dyn error::Error>> {
            unsafe {
                let natural_size = self.av_asset_track.naturalSize();
                Ok((natural_size.width as i32, natural_size.height as i32))
            }
        }
        pub fn frame_rate(&mut self) -> Result<f64, Box<dyn error::Error>> {
            unsafe {
                let frame_rate: f64 = self.av_asset_track.nominalFrameRate().into();
                Ok(frame_rate)
            }
        }
    }

    pub struct PixelBufferIterator<'a> {
        asset_reader: &'a mut AssetReader,
    }

    impl<'a> PixelBufferIterator<'a> {
        fn new(asset_reader: &'a mut AssetReader) -> Self {
            PixelBufferIterator {
                asset_reader: asset_reader,
            }
        }
        pub fn macro_block_3d_iterator<const GOP_LENGTH: usize>(
            self,
        ) -> MacroBlock3DIterator<GOP_LENGTH, Self> {
            pixel_buffer::MacroBlock3DIterator::new(self)
        }
    }

    impl<'a> Iterator for PixelBufferIterator<'a> {
        type Item = PixelBuffer;
        fn next(&mut self) -> Option<Self::Item> {
            self.asset_reader
                .get_next_pixel_buffer()
                .expect("Failed to get next pixel buffer.")
        }
    }
}

pub mod asset_writer {
    use super::*;
    use pixel_buffer::*;

    pub struct AssetWritterSettings {
        pub path: path::PathBuf,
        pub codec: Codec,
        pub resolution: (i32, i32),
        pub frame_rate: f64,
    }

    pub struct AssetWriter {
        //         settings: AssetWritterSettings,
        av_asset_writer: Retained<AVAssetWriter>,
        av_asset_writer_input: Retained<AVAssetWriterInput>,
        av_asset_writer_input_pixel_buffer_adaptor: Retained<AVAssetWriterInputPixelBufferAdaptor>,

        frame_index: i64,
        started_writing: bool,
        timescale: i32,
    }

    impl AssetWriter {
        fn new(
            settings: AssetWritterSettings,
            av_asset_writer: Retained<AVAssetWriter>,
            av_asset_writer_input: Retained<AVAssetWriterInput>,
            av_asset_writer_input_pixel_buffer_adaptor: Retained<
                AVAssetWriterInputPixelBufferAdaptor,
            >,
        ) -> Self {
            let tiemscale = settings.frame_rate as i32;
            AssetWriter {
                //                 settings: settings,
                av_asset_writer: av_asset_writer,
                av_asset_writer_input: av_asset_writer_input,
                av_asset_writer_input_pixel_buffer_adaptor:
                    av_asset_writer_input_pixel_buffer_adaptor,
                frame_index: 0,
                started_writing: false,
                timescale: tiemscale as i32,
            }
        }

        pub fn load_new(settings: AssetWritterSettings) -> Result<Self, Box<dyn error::Error>> {
            let path_bytes = settings.path.as_path().as_os_str().as_encoded_bytes();
            let ns_path = NSString::from_str(std::str::from_utf8(path_bytes)?);
            let url = NSURL::fileURLWithPath_isDirectory(&ns_path, false);

            unsafe {
                let writer = AVAssetWriter::assetWriterWithURL_fileType_error(
                    &url,
                    &AVFileTypeMPEG4.unwrap(),
                )?;

                let codec_value = NSString::from_str(&settings.codec.as_string());
                let width_value = NSNumber::new_i32(settings.resolution.0);
                let height_value = NSNumber::new_i32(settings.resolution.1);

                let input_settings_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                    NSMutableDictionary::new();
                input_settings_dict.insert(AVVideoCodecKey.unwrap(), &codec_value);
                input_settings_dict.insert(AVVideoWidthKey.unwrap(), &width_value);
                input_settings_dict.insert(AVVideoHeightKey.unwrap(), &height_value);

                let input = AVAssetWriterInput::assetWriterInputWithMediaType_outputSettings(
                    &AVMediaTypeVideo.unwrap(),
                    Some(&input_settings_dict),
                );

                let pixel_buffer_settings_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                    NSMutableDictionary::new();

                let pixel_format = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;

                let pixel_format_num = NSNumber::new_u32(pixel_format);

                pixel_buffer_settings_dict.insert(
                    kCVPixelBufferPixelFormatTypeKey.as_nsstring(),
                    &pixel_format_num,
                );
                pixel_buffer_settings_dict
                    .insert(kCVPixelBufferWidthKey.as_nsstring(), &width_value);
                pixel_buffer_settings_dict
                    .insert(kCVPixelBufferHeightKey.as_nsstring(), &height_value);

                let adaptor =
                AVAssetWriterInputPixelBufferAdaptor::
                    assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes(
                        &input,
                        Some(&pixel_buffer_settings_dict)
                    );

                input.setExpectsMediaDataInRealTime(false);
                writer.addInput(&input);

                let writer = AssetWriter::new(settings, writer, input, adaptor);

                Ok(writer)
            }
        }

        pub fn is_ready_for_more_media_data(&self) -> bool {
            unsafe { self.av_asset_writer_input.isReadyForMoreMediaData() }
        }

        pub fn start_writing(&mut self) -> Result<(), Box<dyn error::Error>> {
            self.ensure_started_writing()
        }

        pub fn wait_for_writer_to_be_ready(&self) -> Result<(), Box<dyn error::Error>> {
            // TODO: very lame to use sleep here... should be using KVO to monitor this property
            // TODO: use -requestMediaDataWhenReadyOnQueue:usingBlock:
            unsafe {
                const TIMEOUT: Duration = Duration::from_secs(1);
                const WAIT_INTERVAL: Duration = Duration::from_millis(16); // a 60fps frame
                let start = SystemTime::now();
                while (self.av_asset_writer.status() == AVAssetWriterStatus::Unknown
                    || !self.is_ready_for_more_media_data())
                    && start + TIMEOUT > SystemTime::now()
                {
                    sleep(WAIT_INTERVAL);
                }
                if !self.is_ready_for_more_media_data() {
                    return Err("Did not become ready for more media data.".into());
                }
                match self.av_asset_writer.status() {
                    AVAssetWriterStatus::Writing => Ok(()),
                    _ => Err("Failed to become ready to write.".into()),
                }
            }
        }

        fn ensure_started_writing(&mut self) -> Result<(), Box<dyn error::Error>> {
            unsafe {
                if !self.started_writing {
                    self.av_asset_writer.startWriting();
                    let start_pts = CMTime::new(0, self.timescale);
                    //                 eprintln!("start_pts {:?}", start_pts);
                    self.av_asset_writer.startSessionAtSourceTime(start_pts);
                    self.started_writing = true;
                }
                Ok(())
            }
        }

        pub fn append_pixel_buffer(
            &mut self,
            pixel_buffer: PixelBuffer,
        ) -> Result<(), Box<dyn error::Error>> {
            unsafe {
                self.ensure_started_writing()?;
                //             assert!(self.is_ready_for_more_media_data());

                let pts = CMTime::new(self.frame_index, self.timescale);
                //             eprintln!("pts {:?}", pts);
                self.av_asset_writer_input_pixel_buffer_adaptor
                    .appendPixelBuffer_withPresentationTime(&pixel_buffer.cv_image_buffer, pts);
                self.frame_index += 1;
                Ok(())
            }
        }

        #[allow(deprecated)] // allow synchronous version of finishWriting()
        pub fn finish_writing(&self) -> Result<(), Box<dyn error::Error>> {
            unsafe {
                self.av_asset_writer_input.markAsFinished();
                match self.av_asset_writer.finishWriting() {
                    true => Ok(()),
                    false => Err("Failed to finish writing.".into()),
                }
            }
        }
    }
}

trait AsNSString: AsRef<AnyObject> {
    fn as_nsstring(&self) -> &NSString {
        let any: &AnyObject = self.as_ref();
        any.downcast_ref::<NSString>()
            .expect("Failed to toll-free bridge to NSString.")
    }
}

impl AsNSString for CFString {}

pub enum Codec {
    H264,
}

impl Codec {
    fn as_string(&self) -> String {
        unsafe {
            match self {
                Codec::H264 => AVVideoCodecTypeH264.unwrap().to_string(),
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PixelComponentType {
    Y,
    Cb,
    Cr,
}

impl PixelComponentType {
    fn plane_index(&self) -> usize {
        match self {
            Self::Y => 0,
            Self::Cb => 1,
            Self::Cr => 1,
        }
    }
    fn interleave_offset(&self) -> usize {
        match self {
            PixelComponentType::Y | PixelComponentType::Cb => 0,
            PixelComponentType::Cr => 1,
        }
    }
    fn interleave_step(&self) -> usize {
        match self {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        }
    }
}

pub trait HasPixelComponentType {
    const TYPE: PixelComponentType;
}

pub struct YPixelComponentType;
pub struct CbPixelComponentType;
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

pub mod pixel_buffer {
    use super::*;
    use transform_block_3d::*;

    // Holds a single frame
    #[derive(Debug)]
    pub struct PixelBuffer {
        pub(super) cv_image_buffer: CFRetained<CVImageBuffer>,
    }

    impl PixelBuffer {
        pub fn new(cv_image_buffer: CFRetained<CVImageBuffer>) -> Self {
            assert!(CVPixelBufferIsPlanar(&cv_image_buffer));

            PixelBuffer {
                cv_image_buffer: cv_image_buffer,
            }
        }

        pub fn block_index_to_cv_pixel_buffer_plane_offset(
            block_index: usize,
            block_row_index: usize,
            block_len: usize,
            plane_bytes_per_row: usize,
            pixel_component_type: PixelComponentType,
        ) -> usize {
            Self::block_index_to_cv_pixel_buffer_plane_offset_rectangular(
                block_index,
                block_row_index,
                block_len,
                block_len,
                plane_bytes_per_row,
                pixel_component_type,
            )
        }
        pub fn block_index_to_cv_pixel_buffer_plane_offset_rectangular(
            block_index: usize,
            block_row_index: usize,
            block_width: usize,
            block_height: usize,
            plane_bytes_per_row: usize,
            pixel_component_type: PixelComponentType,
        ) -> usize {
            let interleave_offset = pixel_component_type.interleave_offset();
            let interleave_step = pixel_component_type.interleave_step();

            let cv_row_index = block_height * ((block_width * block_index) / plane_bytes_per_row);
            let cv_row_offset = plane_bytes_per_row * (cv_row_index + block_row_index);

            let cv_column_index = (block_width * block_index * interleave_step + interleave_offset)
                % plane_bytes_per_row;
            let cv_column_offset = cv_column_index;

            cv_row_offset + cv_column_offset
        }

        fn new_cv_pixel_buffer(
            width: usize,
            height: usize,
        ) -> Result<CFRetained<CVPixelBuffer>, Box<dyn std::error::Error>> {
            unsafe {
                let mut cv_pixel_buffer: *mut CVPixelBuffer = std::ptr::null_mut();
                let pixel_buffer_out: NonNull<*mut CVPixelBuffer> =
                    NonNull::from(&mut cv_pixel_buffer);
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    width,
                    height,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    None,
                    pixel_buffer_out,
                );
                if status != 0 {
                    return Err(
                        format!("Failed to create CVPixelBuffer with error {}", status).into(),
                    );
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

        pub(super) fn from_frame_view(
            y_components: FrameComponentView<YPixelComponentType>,
            cb_components: FrameComponentView<CbPixelComponentType>,
            cr_components: FrameComponentView<CrPixelComponentType>,
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

            let (width, height) = y_components.values.dim();
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

        pub(super) fn copy_frame(
            src_ptr: *const u8,
            dst_ptr: *mut u8,
            dst_len: usize,
            interleave_src: bool,
            interleave_offset: usize,
            interleave_step: usize,
        ) {
            unsafe {
                match interleave_step {
                    1 => std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, dst_len),
                    _ => {
                        let mut src_ptr = src_ptr;
                        let mut dst_ptr = dst_ptr;

                        if interleave_src {
                            src_ptr = src_ptr.add(interleave_offset);
                        } else {
                            dst_ptr = dst_ptr.add(interleave_offset);
                        }

                        let dst_ptr_end = dst_ptr.add(dst_len);
                        while dst_ptr < dst_ptr_end {
                            *dst_ptr = *src_ptr;

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
            }
        }

        // The following functions are safe to call without locking the base address of CVPixelBuffer.

        pub fn plane_row_len(&self, pixel_component_type: PixelComponentType) -> usize {
            CVPixelBufferGetBytesPerRowOfPlane(
                &self.cv_image_buffer,
                pixel_component_type.plane_index(),
            ) as usize
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

            let l_cv_y_pixel_buffer_len = self.plane_row_len(PixelComponentType::Y)
                * self.plane_height(PixelComponentType::Y);
            let r_cv_y_pixel_buffer_len = other.plane_row_len(PixelComponentType::Y)
                * other.plane_height(PixelComponentType::Y);
            if l_cv_y_pixel_buffer_len != r_cv_y_pixel_buffer_len {
                return false;
            }

            let l_cv_cbcr_pixel_buffer_len = self.plane_row_len(PixelComponentType::Cb)
                * self.plane_height(PixelComponentType::Cb);
            let r_cv_cbcr_pixel_buffer_len = other.plane_row_len(PixelComponentType::Cb)
                * other.plane_height(PixelComponentType::Cb);
            if l_cv_cbcr_pixel_buffer_len != r_cv_cbcr_pixel_buffer_len {
                return false;
            }
            unsafe {
                let flags = CVPixelBufferLockFlags::ReadOnly;
                CVPixelBufferLockBaseAddress(&self.cv_image_buffer, flags);
                CVPixelBufferLockBaseAddress(&other.cv_image_buffer, flags);

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
                let l_cbcr_slice = std::slice::from_raw_parts(
                    l_cv_cbcr_pixel_buffer_ptr,
                    l_cv_cbcr_pixel_buffer_len,
                );
                let r_cbcr_slice = std::slice::from_raw_parts(
                    r_cv_cbcr_pixel_buffer_ptr,
                    r_cv_cbcr_pixel_buffer_len,
                );

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

    pub struct MacroBlock3DIterator<const LENGTH: usize, PixelBufferIterator>
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        pixel_buffer_iterator: PixelBufferIterator,
    }
    impl<const LENGTH: usize, PixelBufferIterator> MacroBlock3DIterator<LENGTH, PixelBufferIterator>
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        pub(super) fn new(pixel_buffer_iterator: PixelBufferIterator) -> Self {
            MacroBlock3DIterator {
                pixel_buffer_iterator: pixel_buffer_iterator,
            }
        }

        pub fn pixel_buffer_iter(self) -> transform_block_3d::PixelBufferIterator<LENGTH, Self> {
            transform_block_3d::PixelBufferIterator::new(self)
        }
    }

    impl<const LENGTH: usize, PixelBufferIterator> Iterator
        for MacroBlock3DIterator<LENGTH, PixelBufferIterator>
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        // Output all three TransformBlocks at once to linearly process frames
        type Item = MacroBlock3D<LENGTH>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut y_block = TransformBlock3D::new();
            let mut cb_block = TransformBlock3D::new();
            let mut cr_block = TransformBlock3D::new();

            let mut pixel_buffer_iterator_is_empty = true;
            for pixel_buffer in self.pixel_buffer_iterator.by_ref().take(LENGTH) {
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
            })
        }
    }
}

pub mod transform_block_3d {
    use super::*;
    use pixel_buffer::*;
    use std::cell::OnceCell;

    pub struct TransformBlock3D<const LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values_cell: OnceCell<ndarray::Array3<u8>>,
        pub(super) len: usize,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType> TransformBlock3D<LENGTH, PixelType> {
        pub(super) fn new() -> Self {
            TransformBlock3D {
                values_cell: OnceCell::new(),
                len: 0,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn values(&self) -> &ndarray::Array3<u8> {
            self.values_cell
                .get()
                .expect("Values not initialized. Must call populate_next_frame first.")
        }

        pub(super) fn populate_next_frame(
            &mut self,
            pixel_buffer: &PixelBuffer,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let frame_idx = self.len;
            self.len += 1;

            let _ = self.values_cell.get_or_init(|| {
                let block_width =
                    pixel_buffer.plane_row_len(PixelType::TYPE) / PixelType::TYPE.interleave_step();
                let block_height = pixel_buffer.plane_height(PixelType::TYPE);
                ndarray::Array3::zeros((LENGTH, block_width, block_height))
            });
            let values = self.values_cell.get_mut().unwrap(); // get_mut_or_init is nightly-only.

            // Axis(0) is the length/depth dimension
            let mut values_2d = values.index_axis_mut(ndarray::Axis(0), frame_idx);
            assert!(values_2d.is_standard_layout()); // standard_layout = contiguous memory layout

            let pixel_type = PixelType::TYPE;

            let (dst_width, dst_height) = values_2d.dim();
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

    // 4:2:0
    pub struct MacroBlock3D<const LENGTH: usize> {
        pub y_components: TransformBlock3D<LENGTH, YPixelComponentType>,
        pub cb_components: TransformBlock3D<LENGTH, CbPixelComponentType>,
        pub cr_components: TransformBlock3D<LENGTH, CrPixelComponentType>,
    }

    pub struct PixelBufferIterator<const MACRO_BLOCK_LEN: usize, MacroBlock3DIterator>
    where
        MacroBlock3DIterator: Iterator<Item = MacroBlock3D<MACRO_BLOCK_LEN>>,
    {
        macro_block_3d_iterator: MacroBlock3DIterator,
        current_macro_block: Option<MacroBlock3D<MACRO_BLOCK_LEN>>,
        frame_index: usize,
    }

    impl<const MACRO_BLOCK_LEN: usize, MacroBlock3DIterator>
        PixelBufferIterator<MACRO_BLOCK_LEN, MacroBlock3DIterator>
    where
        MacroBlock3DIterator: Iterator<Item = MacroBlock3D<MACRO_BLOCK_LEN>>,
    {
        pub(super) fn new(macro_block_3d_iterator: MacroBlock3DIterator) -> Self {
            PixelBufferIterator {
                macro_block_3d_iterator: macro_block_3d_iterator,
                current_macro_block: None,
                frame_index: 0,
            }
        }
    }

    impl<const MACRO_BLOCK_LEN: usize, MacroBlock3DIterator> Iterator
        for PixelBufferIterator<MACRO_BLOCK_LEN, MacroBlock3DIterator>
    where
        MacroBlock3DIterator: Iterator<Item = MacroBlock3D<MACRO_BLOCK_LEN>>,
    {
        type Item = PixelBuffer;
        fn next(&mut self) -> Option<Self::Item> {
            let macro_block_3d = match self.current_macro_block {
                Some(ref macro_block_3d) => macro_block_3d,
                None => self
                    .current_macro_block
                    .insert(self.macro_block_3d_iterator.next()?),
            };
            // A MacroBlock3D can be shorter than it's LENGTH
            if self.frame_index == macro_block_3d.y_components.len {
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
                PixelBuffer::from_frame_view(y_components, cb_components, cr_components)
                    .expect("Failed to create pixel buffer.");

            self.frame_index += 1;
            if self.frame_index == MACRO_BLOCK_LEN {
                self.frame_index = 0;
                self.current_macro_block = None;
            }

            Some(pixel_buffer)
        }
    }

    pub(super) struct FrameComponentView<'a, PixelType: HasPixelComponentType> {
        pub(super) values: ndarray::ArrayView2<'a, u8>,
        _marker: std::marker::PhantomData<PixelType>,
    }
    impl<'a, PixelType: HasPixelComponentType> FrameComponentView<'a, PixelType> {
        fn new(values: ndarray::ArrayView2<'a, u8>) -> Self {
            assert!(values.is_standard_layout());

            FrameComponentView {
                values: values,
                _marker: std::marker::PhantomData,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::asset_reader::*;
    use super::asset_writer::*;
    use super::pixel_buffer::*;
    use super::transform_block_3d::*;
    use super::*;
    use std::fs;

    #[test]
    fn test_reader_to_writer_0() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let output_file = "/tmp/sample-5s-0.mp4";

        let writer_settings = AssetWritterSettings {
            path: path::PathBuf::from(output_file),
            codec: Codec::H264,
            resolution: reader.resolution().expect("Failed to get resolution."),
            frame_rate: reader.frame_rate().expect("Failed to get frame rate"),
        };
        let _ = fs::remove_file(output_file);
        let mut writer = AssetWriter::load_new(writer_settings).expect("Failed to load writer");
        writer.start_writing().expect("Failed to start writing");

        writer
            .wait_for_writer_to_be_ready()
            .expect("Failed to become ready before writing.");

        for pixel_buffer in reader.pixel_buffer_iter() {
            writer
                .append_pixel_buffer(pixel_buffer)
                .expect("Failed to append pixel buffer");
            writer
                .wait_for_writer_to_be_ready()
                .expect("Failed to become ready after writing some pixels.");
        }
        writer.finish_writing().expect("Failed to finish writing.");
    }

    #[test]
    fn test_get_transform_blocks_3d() {
        const GOP_SIZE: usize = 30;
        let mut reader = AssetReader::new("sample-media/bipbop-1920x1080-5s.mp4"); // 301 frames long

        let num_frames_processed = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator::<GOP_SIZE>()
            .fold(0, |acc, macro_block| {
                acc + macro_block.y_components.len
                    + macro_block.cb_components.len
                    + macro_block.cr_components.len
            });
        let num_frames_expected = 3 + (300 * 3);
        assert_eq!(num_frames_processed, num_frames_expected);
    }

    #[test]
    fn test_macro_block_3d_move() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        const MACRO_BLOCK_LEN: usize = 60;

        let macro_block_3d: MacroBlock3D<MACRO_BLOCK_LEN> = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator()
            .next()
            .expect("Failed to get a MacroBlock3D");

        let MacroBlock3D {
            y_components,
            cb_components,
            cr_components,
        } = macro_block_3d; // demonstrating moving the components

        assert_ne!(y_components.values().len(), 0);
        assert_ne!(cb_components.values().len(), 0);
        assert_ne!(cr_components.values().len(), 0);
    }

    #[test]
    fn test_reader_to_transform_block_3d_to_pb_exact_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader_1 = AssetReader::new(path);
        let mut reader_2 = AssetReader::new(path);

        // reader -> pixel buffer -> macro_block_3d (3x TransformBlock3D) -> PixelBuffer
        let pb1 = reader_1
            .pixel_buffer_iter()
            .macro_block_3d_iterator::<20>()
            .pixel_buffer_iter()
            .next()
            .expect("Failed to get pb1");

        //         pb1.dump_file("o").expect("first dump file failed");

        let pb2 = reader_2
            .pixel_buffer_iter()
            .next()
            .expect("Failed to get pb2");

        //         pb2.dump_file("i").expect("second dump file failed");

        assert_eq!(pb1, pb2);
    }

    #[test]
    fn test_reader_to_transform_block_3d_to_writer() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader = AssetReader::new(path);

        let output_path = "/tmp/bipbop-1920x1080-3d-5s.mp4";
        let _ = fs::remove_file(output_path);
        let writer_settings = AssetWritterSettings {
            path: path::PathBuf::from(output_path),
            codec: Codec::H264,
            resolution: reader.resolution().expect("Failed to get resolution."),
            frame_rate: reader.frame_rate().expect("Failed to get frame rate"),
        };
        let mut writer = AssetWriter::load_new(writer_settings).expect("Failed to load writer");
        writer.start_writing().expect("Failed to start writing");

        let macro_block_3d_iterator: MacroBlock3DIterator<90, _> =
            reader.pixel_buffer_iter().macro_block_3d_iterator();

        let pixel_buffer_iterator = macro_block_3d_iterator.pixel_buffer_iter();

        let mut pixel_buffers_consumed = 0;
        for pixel_buffer in pixel_buffer_iterator {
            pixel_buffers_consumed += 1;

            writer
                .append_pixel_buffer(pixel_buffer)
                .expect("Failed to append pixel buffer");
            writer
                .wait_for_writer_to_be_ready()
                .expect("Failed to become ready after writing some pixels.");
        }
        writer.finish_writing().expect("Failed to finish writing.");

        assert_eq!(pixel_buffers_consumed, 301);
    }
}
