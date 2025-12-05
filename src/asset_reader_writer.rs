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
    use transform_block::*;
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

        pub(super) fn from<const BLOCK_LEN: usize>(
            y_pixel_components: &mut TransformBlockIterator<BLOCK_LEN, YPixelComponentType>,
            cb_pixel_components: &mut TransformBlockIterator<BLOCK_LEN, CbPixelComponentType>,
            cr_pixel_components: &mut TransformBlockIterator<BLOCK_LEN, CrPixelComponentType>,
            resolution: (usize, usize),
        ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
            // Takes each type of TransformBlock and assigns it to a CVPixelBuffer
            // Returns true if the pixel buffer has been completely filled.
            // Returns false if the the iterators have been drained before attempting to fill a pixel buffer.
            fn assign_values<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>(
                src: &mut TransformBlockIterator<BLOCK_LEN, PixelType>,
                dst: &CVPixelBuffer,
            ) -> Result<bool, Box<dyn std::error::Error>> {
                let plane_index = PixelType::TYPE.plane_index();

                let plane_ptr = CVPixelBufferGetBaseAddressOfPlane(&dst, plane_index) as *mut u8;
                let plane_bytes_per_row = CVPixelBufferGetBytesPerRowOfPlane(&dst, plane_index);
                let plane_height = CVPixelBufferGetHeightOfPlane(&dst, plane_index);
                let plane_data_len = plane_bytes_per_row * plane_height;

                if plane_ptr.is_null() {
                    return Err("CVPixelBufferGetBaseAddressOfPlane returned NULL.".into());
                }

                if plane_bytes_per_row % BLOCK_LEN != 0 {
                    return Err(format!(
                        "plane_bytes_per_row:{} % BLOCK_LEN:{} != 0. pixel_type:{:?}",
                        plane_bytes_per_row,
                        BLOCK_LEN,
                        PixelType::TYPE
                    )
                    .into());
                }
                if plane_height % BLOCK_LEN != 0 {
                    return Err(format!(
                        "plane_height:{} % BLOCK_LEN:{} != 0. pixel_type:{:?}",
                        plane_height,
                        BLOCK_LEN,
                        PixelType::TYPE
                    )
                    .into());
                }

                let mut block_index = 0;
                let mut prev_block_index = block_index;
                let mut block_row_index = 0;
                let mut dst_offset = PixelBuffer::block_index_to_cv_pixel_buffer_plane_offset(
                    block_index,
                    0,
                    BLOCK_LEN,
                    plane_bytes_per_row,
                    PixelType::TYPE,
                );

                let mut block = match src.next() {
                    Some(block) => block,
                    None => {
                        return Ok(false); // if no bytes were processed, the iterator is correctly exhausted
                    }
                };

                while dst_offset < plane_data_len {
                    if prev_block_index != block_index {
                        block = match src.next() {
                            Some(block) => block,
                            None => {
                                return Err(
                                    format!("Ran out of {:?} TransformBlocks. Expected {} bytes. Got {} bytes. ",
                                        PixelType::TYPE, plane_data_len, dst_offset).into()
                                    );
                            }
                        };
                    }

                    let src_ptr_start = block
                        .values
                        .get_ptr([block_row_index, 0])
                        .ok_or("Could not get TransformBlock ptr.")?;

                    unsafe {
                        let dst_ptr_start = plane_ptr.add(dst_offset);
                        PixelBuffer::copy_row(
                            src_ptr_start,
                            dst_ptr_start,
                            false,
                            BLOCK_LEN,
                            PixelType::TYPE,
                        );
                    }

                    prev_block_index = block_index;
                    block_row_index += 1;
                    block_index += block_row_index / BLOCK_LEN;
                    block_row_index %= BLOCK_LEN;

                    dst_offset = PixelBuffer::block_index_to_cv_pixel_buffer_plane_offset(
                        block_index,
                        block_row_index,
                        BLOCK_LEN,
                        plane_bytes_per_row,
                        PixelType::TYPE,
                    );
                }

                Ok(true)
            }

            unsafe {
                let mut cv_pixel_buffer: *mut CVPixelBuffer = std::ptr::null_mut();
                let pixel_buffer_out: NonNull<*mut CVPixelBuffer> =
                    NonNull::from(&mut cv_pixel_buffer);
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    resolution.0,
                    resolution.1,
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
                assert!(CVPixelBufferIsPlanar(&cv_pixel_buffer));

                let flags = CVPixelBufferLockFlags::empty(); // empty means write
                CVPixelBufferLockBaseAddress(&cv_pixel_buffer, flags);

                if !assign_values(y_pixel_components, &cv_pixel_buffer)? {
                    CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);
                    return Ok(None);
                }
                if !assign_values(cb_pixel_components, &cv_pixel_buffer)? {
                    CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);
                    return Err(
                        "Consumed Y pixel components but exhausted all Cb pixel components.".into(),
                    );
                }
                if !assign_values(cr_pixel_components, &cv_pixel_buffer)? {
                    CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);
                    return Err(
                        "Consumed Y and Cb pixel components but exhausted all Cr pixel components."
                            .into(),
                    );
                }

                CVPixelBufferUnlockBaseAddress(&cv_pixel_buffer, flags);

                Ok(Some(Self {
                    cv_image_buffer: cv_pixel_buffer,
                }))
            }
        }

        // CVPixelBuffer MUST be locked.
        fn to_transform_block<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>(
            &self,
            block_index: usize,
        ) -> Result<TransformBlock<BLOCK_LEN, PixelType>, Box<dyn std::error::Error>> {
            let mut arr = ndarray::Array2::zeros((BLOCK_LEN, BLOCK_LEN));
            self.populate_2d_transform_block_array::<PixelType, _>(
                block_index,
                (BLOCK_LEN, BLOCK_LEN),
                &mut arr,
            )?;
            let transform_block = TransformBlock::<BLOCK_LEN, PixelType>::new(arr);
            Ok(transform_block)
        }

        fn populate_2d_transform_block_array<PixelType: HasPixelComponentType, ArrayStorage>(
            &self,
            block_index: usize,
            (block_width, block_height): (usize, usize),
            dst: &mut ndarray::ArrayBase<ArrayStorage, ndarray::Ix2>,
        ) -> Result<(), Box<dyn std::error::Error>>
        where
            ArrayStorage: ndarray::DataMut<Elem = u8>,
        {
            let plane_index = PixelType::TYPE.plane_index();
            let plane_row_len = self.plane_row_len(PixelType::TYPE);
            let plane_height = self.plane_height(PixelType::TYPE);

            assert_eq!(plane_row_len % block_width, 0);
            assert_eq!(plane_height % block_height, 0);

            unsafe {
                let plane_ptr =
                    CVPixelBufferGetBaseAddressOfPlane(&self.cv_image_buffer, plane_index)
                        as *mut u8;

                if plane_ptr.is_null() {
                    return Err("CVPixelBufferGetBaseAddressOfPlane returned NULL.".into());
                }

                for block_row_index in 0..block_height {
                    let src_offset_start =
                        PixelBuffer::block_index_to_cv_pixel_buffer_plane_offset_rectangular(
                            block_index,
                            block_row_index,
                            block_width,
                            block_height,
                            plane_row_len,
                            PixelType::TYPE,
                        );
                    let src_ptr_start = plane_ptr.add(src_offset_start);
                    let dst_ptr_start = dst
                        .get_mut_ptr([block_row_index, 0])
                        .ok_or("Could not get mut ptr of ndarray.")?;

                    PixelBuffer::copy_row(
                        src_ptr_start,
                        dst_ptr_start,
                        true,
                        block_width,
                        PixelType::TYPE,
                    );
                }
            }
            Ok(())
        }
        fn copy_row(
            src_ptr: *const u8,
            dst_ptr: *mut u8,
            interleaving_src: bool, // false to interleave dst
            block_len: usize,
            pixel_component_type: PixelComponentType,
        ) {
            let src_ptr_start = src_ptr;
            let dst_ptr_start = dst_ptr;

            unsafe {
                match pixel_component_type.interleave_step() {
                    1 => {
                        std::ptr::copy_nonoverlapping(src_ptr_start, dst_ptr_start, block_len);
                    }
                    interleave_step => {
                        let mut src_ptr = src_ptr_start; // shadow
                        let mut dst_ptr = dst_ptr_start; // shadow

                        // offset is applied in block_index_to_cv_pixel_buffer_plane_offset
                        for _src_col_index in 0..block_len {
                            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1);

                            if interleaving_src {
                                src_ptr = src_ptr.add(interleave_step);
                                dst_ptr = dst_ptr.add(1);
                            } else {
                                src_ptr = src_ptr.add(1);
                                dst_ptr = dst_ptr.add(interleave_step);
                            }
                        }
                    }
                };
            }
        }

        pub fn transform_block_iter<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>(
            &self,
        ) -> Result<TransformBlockIterator<'_, BLOCK_LEN, PixelType>, Box<dyn std::error::Error>>
        {
            Ok(TransformBlockIterator::<BLOCK_LEN, PixelType>::new(self))
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

    // only square blocks supported
    pub struct TransformBlockIterator<'a, const BLOCK_LEN: usize, PixelType: HasPixelComponentType> {
        pixel_buffer: &'a PixelBuffer,
        current_block_index: usize,
        locked_pixel_buffer_memory: bool,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<'a, const BLOCK_LEN: usize, PixelType: HasPixelComponentType>
        TransformBlockIterator<'a, BLOCK_LEN, PixelType>
    {
        // only supports 4:2:0 YCbCr
        pub fn new(pixel_buffer: &'a PixelBuffer) -> Self {
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
            self.pixel_buffer.to_transform_block(block_index)
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
    }

    impl<'a, const BLOCK_LEN: usize, PixelType: HasPixelComponentType> Iterator
        for TransformBlockIterator<'a, BLOCK_LEN, PixelType>
    {
        type Item = TransformBlock<BLOCK_LEN, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            let plane_bytes_per_row = self.pixel_buffer.plane_row_len(PixelType::TYPE);
            let next_plane_ptr_offset = PixelBuffer::block_index_to_cv_pixel_buffer_plane_offset(
                self.current_block_index,
                0,
                BLOCK_LEN,
                plane_bytes_per_row,
                PixelType::TYPE,
            );
            let bytes_per_plane =
                self.pixel_buffer.plane_height(PixelType::TYPE) * plane_bytes_per_row;

            if next_plane_ptr_offset >= bytes_per_plane {
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

    pub struct TransformBlock3DIterator<
        const LENGTH: usize,
        PixelType: HasPixelComponentType,
        PixelBufferIterator,
    >
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        pixel_buffer_iterator: PixelBufferIterator,
        resolution: (usize, usize),
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType, PixelBufferIterator>
        TransformBlock3DIterator<LENGTH, PixelType, PixelBufferIterator>
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        fn new(pixel_buffer_iterator: PixelBufferIterator, resolution: (usize, usize)) -> Self {
            TransformBlock3DIterator {
                pixel_buffer_iterator: pixel_buffer_iterator,
                resolution: resolution,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType, PixelBufferIterator> Iterator
        for TransformBlock3DIterator<LENGTH, PixelType, PixelBufferIterator>
    where
        PixelBufferIterator: Iterator<Item = PixelBuffer>,
    {
        type Item = TransformBlock3D<LENGTH, PixelType>;

        fn next(&mut self) -> Option<Self::Item> {
            let (width, height) = self.resolution;

            // Length must be the first dimension to afford contiguous memory regions per-frame.
            let mut values_3d: ndarray::Array3<u8> =
                ndarray::Array3::zeros((LENGTH, width, height));
            while let Some((frame_idx, pixel_buffer)) = self
                .pixel_buffer_iterator
                .by_ref()
                .take(LENGTH)
                .enumerate()
                .next()
            {
                let mut values_2d = values_3d.index_axis_mut(ndarray::Axis(0), frame_idx);
                assert!(values_2d.is_standard_layout()); // standard_layout = contiguous memory layout

                pixel_buffer
                    .populate_2d_transform_block_array::<PixelType, _>(
                        0,
                        pixel_buffer.resolution(),
                        &mut values_2d,
                    )
                    .expect("Failed to populate frame of 3D Transform Block.");
            }

            None
        }
    }
}

pub mod transform_block {
    use super::*;
    use pixel_buffer::*;

    pub struct TransformBlock<const BLOCK_LEN: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array2<u8>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const BLOCK_LEN: usize, PixelType: HasPixelComponentType>
        TransformBlock<BLOCK_LEN, PixelType>
    {
        pub(super) fn new(values: ndarray::Array2<u8>) -> Self {
            assert_eq!(BLOCK_LEN, values.dim().0);
            assert_eq!(BLOCK_LEN, values.dim().1);

            // take ownership of values with a move
            TransformBlock {
                values: values,
                _marker: std::marker::PhantomData,
            }
        }
    }

    pub struct PixelBufferIterator<'a, const BLOCK_LEN: usize> {
        y_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, YPixelComponentType>,
        cb_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, CbPixelComponentType>,
        cr_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, CrPixelComponentType>,
        pixel_buffer_resolution: (usize, usize),
    }

    impl<'a, const BLOCK_LEN: usize> PixelBufferIterator<'a, BLOCK_LEN> {
        pub fn new(
            y_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, YPixelComponentType>,
            cb_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, CbPixelComponentType>,
            cr_pixel_components: TransformBlockIterator<'a, BLOCK_LEN, CrPixelComponentType>,
            pixel_buffer_resolution: (usize, usize),
        ) -> Self {
            assert_eq!(pixel_buffer_resolution.0 % BLOCK_LEN, 0);
            assert_eq!(pixel_buffer_resolution.1 % BLOCK_LEN, 0);

            PixelBufferIterator {
                y_pixel_components: y_pixel_components,
                cb_pixel_components: cb_pixel_components,
                cr_pixel_components: cr_pixel_components,
                pixel_buffer_resolution: pixel_buffer_resolution,
            }
        }
    }

    impl<'a, const BLOCK_LEN: usize> Iterator for PixelBufferIterator<'a, BLOCK_LEN> {
        type Item = PixelBuffer;
        fn next(&mut self) -> Option<Self::Item> {
            // will return optional if the iterators are exhausted.
            PixelBuffer::from(
                &mut self.y_pixel_components,
                &mut self.cb_pixel_components,
                &mut self.cr_pixel_components,
                self.pixel_buffer_resolution,
            )
            .expect("Failed to create pixel buffer.")
        }
    }
}

mod transform_block_3d {
    use super::*;

    pub struct TransformBlock3D<const LENGTH: usize, PixelType: HasPixelComponentType> {
        pub values: ndarray::Array3<u8>,
        _marker: std::marker::PhantomData<PixelType>,
    }

    impl<const LENGTH: usize, PixelType: HasPixelComponentType> TransformBlock3D<LENGTH, PixelType> {
        pub(super) fn new(values: ndarray::Array3<u8>, resolution: (usize, usize)) -> Self {
            assert_eq!(LENGTH, values.dim().0);
            assert_eq!(resolution.0, values.dim().1);
            assert_eq!(resolution.1, values.dim().2);

            // take ownership of values with a move
            TransformBlock3D {
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
    //     use super::transform_block::*;
    use super::*;
    use std::fs;

    #[test]
    fn test_get_transform_block_0() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let mut iter = TransformBlockIterator::<8, YPixelComponentType>::new(&pixel_buffer);
        let block = iter.next().expect("No transform blocks produced.");

        assert_eq!(block.values.len(), 64);
    }

    #[test]
    fn test_get_transform_block_1() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<8, YPixelComponentType>::new(&pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }

    #[test]
    fn test_get_transform_block_2() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<4, CbPixelComponentType>::new(&pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 64800);
    }

    #[test]
    fn test_get_transform_block_3() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let iter = TransformBlockIterator::<4, CrPixelComponentType>::new(&pixel_buffer);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 64800);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_get_transform_block_4() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");

        let mut count = 0;
        for pixel_buffer in reader.pixel_buffer_iter() {
            let iter = TransformBlockIterator::<8, YPixelComponentType>::new(&pixel_buffer);
            count = iter.fold(count, |acc, _| acc + 1);
        }

        assert_eq!(count, 5540400);
    }

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
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_writer_1() {
        let mut reader = AssetReader::new("sample-media/bipbop-1920x1080-5s.mp4");
        let output_file = "/tmp/bipbop-1920x1080-5s.mp4";

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
            const BLOCK_LEN: usize = 60;

            // PixelBuffer -> TransformBlock
            let y_components: TransformBlockIterator<BLOCK_LEN, YPixelComponentType> = pixel_buffer
                .transform_block_iter()
                .expect("Failed to get Y components.");
            let cb_components: TransformBlockIterator<BLOCK_LEN, CbPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cb components.");
            let cr_components: TransformBlockIterator<BLOCK_LEN, CrPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cr components.");

            // TransformBlock -> PixelBuffer
            let mut pixel_buffer_iter = transform_block::PixelBufferIterator::new(
                y_components,
                cb_components,
                cr_components,
                pixel_buffer.resolution(),
            );

            let new_pixel_buffer = pixel_buffer_iter
                .next()
                .expect("No pixel buffers generated.");

            assert!(
                pixel_buffer_iter.next().is_none(),
                "More than one pixel buffer generated."
            );

            //             pixel_buffer.dump_file("i").expect("first dump file failed");
            //             new_pixel_buffer
            //                 .dump_file("o")
            //                 .expect("second dump file failed");

            writer
                .append_pixel_buffer(new_pixel_buffer)
                .expect("Failed to append pixel buffer");
            writer
                .wait_for_writer_to_be_ready()
                .expect("Failed to become ready after writing some pixels.");
        }
        writer.finish_writing().expect("Failed to finish writing.");
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_writer_2() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4");
        let output_file = "/tmp/sample-5s-1.mp4";

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
            const BLOCK_LEN: usize = 12;

            // PixelBuffer -> TransformBlock
            let y_components: TransformBlockIterator<BLOCK_LEN, YPixelComponentType> = pixel_buffer
                .transform_block_iter()
                .expect("Failed to get Y components.");
            let cb_components: TransformBlockIterator<BLOCK_LEN, CbPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cb components.");
            let cr_components: TransformBlockIterator<BLOCK_LEN, CrPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cr components.");

            // TransformBlock -> PixelBuffer
            let mut pixel_buffer_iter = transform_block::PixelBufferIterator::new(
                y_components,
                cb_components,
                cr_components,
                pixel_buffer.resolution(),
            );

            let new_pixel_buffer = pixel_buffer_iter
                .next()
                .expect("No pixel buffers generated.");

            assert!(
                pixel_buffer_iter.next().is_none(),
                "More than one pixel buffer generated."
            );

            //             pixel_buffer.dump_file("i").expect("first dump file failed");
            //             new_pixel_buffer
            //                 .dump_file("o")
            //                 .expect("second dump file failed");

            writer
                .append_pixel_buffer(new_pixel_buffer)
                .expect("Failed to append pixel buffer");
            writer
                .wait_for_writer_to_be_ready()
                .expect("Failed to become ready after writing some pixels.");
        }
        writer.finish_writing().expect("Failed to finish writing.");
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_reader_to_transform_block_to_pb_exact_equality() {
        let mut reader = AssetReader::new("sample-media/bipbop-1920x1080-5s.mp4");

        for pixel_buffer in reader.pixel_buffer_iter() {
            const BLOCK_LEN: usize = 60;

            // PixelBuffer -> TransformBlock
            let y_components: TransformBlockIterator<BLOCK_LEN, YPixelComponentType> = pixel_buffer
                .transform_block_iter()
                .expect("Failed to get Y components.");
            let cb_components: TransformBlockIterator<BLOCK_LEN, CbPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cb components.");
            let cr_components: TransformBlockIterator<BLOCK_LEN, CrPixelComponentType> =
                pixel_buffer
                    .transform_block_iter()
                    .expect("Failed to get Cr components.");

            // TransformBlock -> PixelBuffer
            let mut pixel_buffer_iter = transform_block::PixelBufferIterator::new(
                y_components,
                cb_components,
                cr_components,
                pixel_buffer.resolution(),
            );

            let new_pixel_buffer = pixel_buffer_iter
                .next()
                .expect("No pixel buffers generated.");

            assert!(
                pixel_buffer_iter.next().is_none(),
                "More than one pixel buffer generated."
            );

            assert_eq!(pixel_buffer, new_pixel_buffer);
        }
    }

    #[test]
    fn test_reader_to_transform_block_to_pb_exact_equality_ok_for_debug() {
        let mut reader = AssetReader::new("sample-media/bipbop-1920x1080-5s.mp4");

        let pixel_buffer = reader
            .pixel_buffer_iter()
            .next()
            .expect("Failed to get a pixel buffer");
        const BLOCK_LEN: usize = 60;

        // PixelBuffer -> TransformBlock
        let y_components: TransformBlockIterator<BLOCK_LEN, YPixelComponentType> = pixel_buffer
            .transform_block_iter()
            .expect("Failed to get Y components.");
        let cb_components: TransformBlockIterator<BLOCK_LEN, CbPixelComponentType> = pixel_buffer
            .transform_block_iter()
            .expect("Failed to get Cb components.");
        let cr_components: TransformBlockIterator<BLOCK_LEN, CrPixelComponentType> = pixel_buffer
            .transform_block_iter()
            .expect("Failed to get Cr components.");

        // TransformBlock -> PixelBuffer
        let mut pixel_buffer_iter = transform_block::PixelBufferIterator::new(
            y_components,
            cb_components,
            cr_components,
            pixel_buffer.resolution(),
        );

        let new_pixel_buffer = pixel_buffer_iter
            .next()
            .expect("No pixel buffers generated.");

        assert!(
            pixel_buffer_iter.next().is_none(),
            "More than one pixel buffer generated."
        );

        assert_eq!(pixel_buffer, new_pixel_buffer);
    }
}
