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
use std::{error, marker::PhantomData, path, ptr::NonNull};

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
    CVPixelBufferGetDataSize, CVPixelBufferGetHeightOfPlane, CVPixelBufferIsPlanar,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
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

        pub fn pixel_buffer_iter(&mut self) -> PixelBufferIterator {
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

#[derive(Clone, Copy)]
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
}

pub trait HasPixelComponentType {
    const TYPE: PixelComponentType;
}

struct YPixelComponentType;
struct CbPixelComponentType;
struct CrPixelComponentType;

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
    use transform_block::*;

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

        pub(super) fn from<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>(
            transform_blocks: &[TransformBlock<BLOCK_LEN, PixelType>],
            resolution: (usize, usize),
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let mut cv_pixel_buffer: *mut CVPixelBuffer = std::ptr::null_mut();
            let pixel_buffer_out: NonNull<*mut CVPixelBuffer> = NonNull::from(&mut cv_pixel_buffer);
            unsafe {
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    resolution.0,
                    resolution.1,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    None,
                    pixel_buffer_out,
                );
                if status == 0 {
                    return Err(
                        format!("Failed to create CVPixelBuffer with error {}", status).into(),
                    );
                }

                let cv_pixel_buffer = CFRetained::from_raw(
                    // coerce the mutable ptr into an immutable ptr
                    NonNull::new(cv_pixel_buffer).ok_or("CVPixelBuffer is NULL")?,
                );
                assert!(CVPixelBufferIsPlanar(&cv_pixel_buffer));

                let flags = CVPixelBufferLockFlags::empty(); // empty means write
                CVPixelBufferLockBaseAddress(&cv_pixel_buffer, flags);

                CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);

                let pixel_buffer = Self {
                    cv_image_buffer: cv_pixel_buffer,
                };
                Ok(pixel_buffer)
            }
        }

        // The following functions are safe to call without locking the base address of CVPixelBuffer.

        fn plane_row_len(&self, pixel_component_type: PixelComponentType) -> usize {
            CVPixelBufferGetBytesPerRowOfPlane(
                &self.cv_image_buffer,
                pixel_component_type.plane_index(),
            ) as usize
        }
        fn plane_height(&self, pixel_component_type: PixelComponentType) -> usize {
            CVPixelBufferGetHeightOfPlane(&self.cv_image_buffer, pixel_component_type.plane_index())
        }
        fn plane_data_len(&self) -> usize {
            CVPixelBufferGetDataSize(&self.cv_image_buffer)
        }
    }

    // only square blocks supported
    pub struct TransformBlockIterator<const BLOCK_LEN: usize, PixelType: HasPixelComponentType> {
        pixel_buffer: PixelBuffer,
        current_block_index: usize,
        locked_pixel_buffer_memory: bool,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>
        TransformBlockIterator<BLOCK_LEN, PixelType>
    {
        // only supports 4:2:0 YCbCr
        pub fn new(pixel_buffer: PixelBuffer) -> Self {
            Self {
                pixel_buffer: pixel_buffer,
                current_block_index: 0,
                locked_pixel_buffer_memory: false,
                _marker: std::marker::PhantomData,
            }
        }

        fn new_transform_block(
            &self,
            block_index: usize,
        ) -> Result<TransformBlock<BLOCK_LEN, PixelType>, Box<dyn error::Error>> {
            let mut block_ndarray = ndarray::Array2::zeros((BLOCK_LEN, BLOCK_LEN));

            let plane_row_len = self.pixel_buffer.plane_row_len(PixelType::TYPE);
            let plane_height = self.pixel_buffer.plane_height(PixelType::TYPE);
            let plane_len = self.pixel_buffer.plane_data_len();

            assert!(plane_row_len * plane_height <= plane_len);
            assert_eq!(plane_row_len % BLOCK_LEN, 0);
            assert_eq!(plane_height % BLOCK_LEN, 0);

            // CbCr samples are interleaved.
            let interleave_offset = self.interleave_offset();
            let interleave_step = self.interleave_step();

            // always include pixels in the plane beyond width
            let plane_row_samples = plane_row_len / interleave_step;
            let column_start = interleave_offset + (block_index * BLOCK_LEN) % plane_row_samples;
            let column_end = column_start + BLOCK_LEN * interleave_step;
            let row_start = BLOCK_LEN * ((block_index * BLOCK_LEN) / plane_row_samples);
            let row_end = row_start + BLOCK_LEN;

            //         eprintln!(
            //             "{}x{} - {}x{}",
            //             column_start, row_start, column_end, row_end
            //         );

            assert!(column_end <= plane_row_len);
            assert!(row_end <= plane_height);

            unsafe {
                let plane_ptr = CVPixelBufferGetBaseAddressOfPlane(
                    &self.pixel_buffer.cv_image_buffer,
                    PixelType::TYPE.plane_index(),
                ) as *const u8;

                if plane_ptr.is_null() {
                    return Err("CVPixelBufferGetBaseAddressOfPlane returned NULL.".into());
                }

                let plane_slice = std::slice::from_raw_parts(plane_ptr, plane_len);

                for plane_row in row_start..row_end {
                    for plane_column in (column_start..column_end).step_by(interleave_step) {
                        let plane_idx = plane_column + plane_row * plane_row_len;
                        let block_column = (plane_column - column_start) / interleave_step;
                        let block_row = plane_row - row_start;
                        block_ndarray[(block_column, block_row)] = plane_slice[plane_idx];
                    }
                }
            }
            let transform_block = TransformBlock::<BLOCK_LEN, PixelType>::new(block_ndarray);
            Ok(transform_block)
        }
        fn ensure_pixel_buffer_address_is_locked(&mut self) {
            if !self.locked_pixel_buffer_memory {
                unsafe {
                    let flags = CVPixelBufferLockFlags::ReadOnly;
                    CVPixelBufferLockBaseAddress(&self.pixel_buffer.cv_image_buffer, flags);
                    self.locked_pixel_buffer_memory = true;
                }
            }
        }
        fn ensure_pixel_buffer_address_is_unlocked(&mut self) {
            if self.locked_pixel_buffer_memory {
                unsafe {
                    let flags = CVPixelBufferLockFlags::ReadOnly;
                    CVPixelBufferUnlockBaseAddress(&self.pixel_buffer.cv_image_buffer, flags);
                }
                self.locked_pixel_buffer_memory = false;
            }
        }

        // the following could be static fns
        fn interleave_offset(&self) -> usize {
            match PixelType::TYPE {
                PixelComponentType::Y | PixelComponentType::Cb => 0,
                PixelComponentType::Cr => 1,
            }
        }
        fn interleave_step(&self) -> usize {
            match PixelType::TYPE {
                PixelComponentType::Y => 1,
                PixelComponentType::Cb | PixelComponentType::Cr => 2,
            }
        }

        fn plane_indexes_per_block(&self) -> usize {
            BLOCK_LEN * BLOCK_LEN * self.interleave_step()
        }
        fn plane_indexes_per_pixel_buffer(&self) -> usize {
            self.pixel_buffer.plane_height(PixelType::TYPE)
                * self.pixel_buffer.plane_row_len(PixelType::TYPE)
        }
    }

    impl<const BLOCK_LEN: usize, PixelType: HasPixelComponentType> Iterator
        for TransformBlockIterator<BLOCK_LEN, PixelType>
    {
        type Item = TransformBlock<BLOCK_LEN, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            let next_plane_index = self.current_block_index * self.plane_indexes_per_block();
            if next_plane_index >= self.plane_indexes_per_pixel_buffer() {
                // Completed this pixel buffer.
                self.ensure_pixel_buffer_address_is_unlocked();
                return None;
            }

            //         eprintln!("Getting block {}", self.current_block_index);

            self.ensure_pixel_buffer_address_is_locked();

            let next_block = self
                .new_transform_block(self.current_block_index)
                .expect("Failed to make new transform block."); // Internal error, should crash rather than returning None.
            self.current_block_index += 1;
            Some(next_block)
        }
    }
}

pub mod transform_block {
    use super::*;
    use pixel_buffer::*;

    pub struct TransformBlock<const BLOCK_LEN: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array2<u8>,
        resolution: (usize, usize),
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>
        TransformBlock<BLOCK_LEN, PixelType>
    {
        pub(super) fn new(values: ndarray::Array2<u8>) -> Self {
            assert_eq!(BLOCK_LEN, values.dim().0);
            assert_eq!(BLOCK_LEN, values.dim().1);

            // take ownership of values with a move
            let resolution = values.dim();
            TransformBlock {
                values: values,
                resolution: resolution,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct PixelBufferIterator<I, const BLOCK_LEN: usize, PixelType: HasPixelComponentType> {
        inner: I,
        pixel_buffer_resolution: (usize, usize),
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<I, const BLOCK_LEN: usize, PixelType: HasPixelComponentType>
        PixelBufferIterator<I, BLOCK_LEN, PixelType>
    where
        I: Iterator<Item = TransformBlock<BLOCK_LEN, PixelType>>,
    {
        fn new(inner: I, pixel_buffer_resolution: (usize, usize)) -> Self {
            assert_eq!(pixel_buffer_resolution.0 % BLOCK_LEN, 0);
            assert_eq!(pixel_buffer_resolution.1 % BLOCK_LEN, 0);
            PixelBufferIterator {
                inner: inner,
                pixel_buffer_resolution: pixel_buffer_resolution,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<I, const BLOCK_LEN: usize, PixelType: HasPixelComponentType> Iterator
        for PixelBufferIterator<I, BLOCK_LEN, PixelType>
    where
        I: Iterator<Item = TransformBlock<BLOCK_LEN, PixelType>>,
    {
        type Item = PixelBuffer;
        fn next(&mut self) -> Option<Self::Item> {
            let num_macro_blocks = BLOCK_LEN * BLOCK_LEN
                / (self.pixel_buffer_resolution.0 * self.pixel_buffer_resolution.1);
            let macro_blocks = self.inner.by_ref().take(num_macro_blocks);

            assert_eq!(macro_blocks.count(), num_macro_blocks);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::asset_reader::*;
    use super::asset_writer::*;
    use super::pixel_buffer::*;
    use super::transform_block::*;
    use super::*;
    use std::fs;

    #[test]
    fn test_get_transform_block_0() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let mut iter = TransformBlockIterator::<8, YPixelComponentType>::new(pixel_buffer);
        let block = iter.next().expect("No transform blocks produced.");

        assert_eq!(block.values.len(), 64);
    }

    #[test]
    fn test_get_transform_block_1() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<8, YPixelComponentType>::new(pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }

    #[test]
    fn test_get_transform_block_2() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<4, CbPixelComponentType>::new(pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }

    #[test]
    fn test_get_transform_block_3() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<4, CrPixelComponentType>::new(pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_get_transform_block_4() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");

        let mut count = 0;
        for pixel_buffer in reader.pixel_buffer_iter() {
            let iter = TransformBlockIterator::<8>::new(pixel_buffer, PixelComponentType::Y);
            count = iter.fold(count, |acc, _| acc + 1);
        }

        assert_eq!(count, 5540400);
    }

    #[test]
    fn test_reader_to_writer_0() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let output_file = "/tmp/sample-5s.mp4";

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
}
