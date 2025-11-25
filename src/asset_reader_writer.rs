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
use std::{error, path};

use objc2_foundation::{
    ns_string, NSArray, NSDictionary, NSMutableDictionary, NSNumber, NSObject, NSString, NSURL,
};

use objc2::{rc::Retained, runtime::AnyObject};
use objc2_av_foundation::{
    AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput,
    AVAssetTrack, AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor,
    AVFileTypeMPEG4, AVMediaTypeVideo, AVURLAsset, AVVideoCodecKey, AVVideoCodecTypeH264,
    AVVideoHeightKey, AVVideoWidthKey,
};

use objc2_core_foundation::{CFDictionary, CFDictionaryCreateMutable, CFRetained, CFString};

use objc2_core_video::{
    kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, CVImageBuffer,
    CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
    CVPixelBufferGetDataSize, CVPixelBufferGetHeightOfPlane, CVPixelBufferIsPlanar,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};

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

    fn av_asset_reader(&mut self) -> Result<Retained<AVAssetReader>, Box<dyn error::Error>> {
        if self.loaded_reader.is_none() {
            self.loaded_reader = Some(LoadedAssetReader::load(self.path.as_path())?)
        }

        Ok(self
            .loaded_reader
            .as_ref()
            .cloned()
            .ok_or("No reader")?
            .av_asset_reader)
    }

    fn av_asset_output(
        &mut self,
    ) -> Result<Retained<AVAssetReaderTrackOutput>, Box<dyn error::Error>> {
        if self.loaded_reader.is_none() {
            self.loaded_reader = Some(LoadedAssetReader::load(self.path.as_path())?)
        }

        Ok(self
            .loaded_reader
            .as_ref()
            .cloned()
            .ok_or("No reader")?
            .av_asset_output)
    }

    #[allow(deprecated)] // blocking i/o is expected here
    fn get_next_pixel_buffer(&mut self) -> Result<Option<PixelBuffer>, Box<dyn error::Error>> {
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
}

#[derive(Clone)]
struct LoadedAssetReader {
    av_asset_reader: Retained<AVAssetReader>,
    av_asset_output: Retained<AVAssetReaderTrackOutput>,
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
            })
        }
    }
}

pub struct AssetWriter {
    av_asset_writer: Retained<AVAssetWriter>,
    av_asset_writer_input: Retained<AVAssetWriterInput>,
    av_asset_writer_input_pixel_buffer_adaptor: Retained<AVAssetWriterInputPixelBufferAdaptor>,
}

impl AssetWriter {
    fn new(
        av_asset_writer: Retained<AVAssetWriter>,
        av_asset_writer_input: Retained<AVAssetWriterInput>,
        av_asset_writer_input_pixel_buffer_adaptor: Retained<AVAssetWriterInputPixelBufferAdaptor>,
    ) -> Self {
        AssetWriter {
            av_asset_writer: av_asset_writer,
            av_asset_writer_input: av_asset_writer_input,
            av_asset_writer_input_pixel_buffer_adaptor: av_asset_writer_input_pixel_buffer_adaptor,
        }
    }

    pub fn load_new(settings: AssetWritterSettings) -> Result<Self, Box<dyn error::Error>> {
        let path_bytes = settings.path.as_path().as_os_str().as_encoded_bytes();
        let ns_path = NSString::from_str(std::str::from_utf8(path_bytes)?);
        let url = NSURL::fileURLWithPath_isDirectory(&ns_path, false);

        unsafe {
            let writer =
                AVAssetWriter::assetWriterWithURL_fileType_error(&url, &AVFileTypeMPEG4.unwrap())?;

            let codec_value = NSString::from_str(&settings.codec);
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
                cfstring_as_nsstring(kCVPixelBufferPixelFormatTypeKey),
                &pixel_format_num,
            );
            pixel_buffer_settings_dict
                .insert(cfstring_as_nsstring(kCVPixelBufferWidthKey), &width_value);
            pixel_buffer_settings_dict
                .insert(cfstring_as_nsstring(kCVPixelBufferHeightKey), &height_value);

            let adaptor =
                AVAssetWriterInputPixelBufferAdaptor::
                    assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes(
                        &input,
                        Some(&pixel_buffer_settings_dict)
                    );

            let writer = AssetWriter::new(writer, input, adaptor);
            Ok(writer)
        }
    }
}

fn cfstring_as_nsstring<'a>(cf: &'a CFString) -> &'a NSString {
    // CFString implements AsRef<AnyObject>
    let any: &AnyObject = cf.as_ref();

    // Downcast via Objective-C's isKindOfClass:
    any.downcast_ref::<NSString>().unwrap()
}

pub struct AssetWritterSettings {
    path: path::PathBuf,
    codec: String,
    resolution: (i32, i32),
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

// Holds a single frame
#[derive(Debug)]
pub struct PixelBuffer {
    cv_image_buffer: CFRetained<CVImageBuffer>,
}

impl PixelBuffer {
    pub fn new(cv_image_buffer: CFRetained<CVImageBuffer>) -> Self {
        assert!(CVPixelBufferIsPlanar(&cv_image_buffer));

        PixelBuffer {
            cv_image_buffer: cv_image_buffer,
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

pub struct TransformBlock {
    pixel_component_type: PixelComponentType,
    values: ndarray::Array2<u8>,
}

impl TransformBlock {}

pub struct TransformBlockIterator {
    pixel_buffer: PixelBuffer,
    block_size: usize,
    pixel_component_type: PixelComponentType,
    current_block_index: usize,
    locked_pixel_buffer_memory: bool,
}

impl TransformBlockIterator {
    // only supports 4:2:0 YCbCr
    pub fn new(
        pixel_buffer: PixelBuffer,
        block_size: usize,
        pixel_component_type: PixelComponentType,
    ) -> Self {
        Self {
            block_size: block_size,
            pixel_buffer: pixel_buffer,
            pixel_component_type: pixel_component_type,
            current_block_index: 0,
            locked_pixel_buffer_memory: false,
        }
    }

    fn new_transform_block(
        &self,
        block_size: usize, // only square blocks supported atm
        block_index: usize,
    ) -> Result<TransformBlock, Box<dyn error::Error>> {
        let mut block_ndarray = ndarray::Array2::zeros((block_size, block_size));

        let plane_row_len = self.pixel_buffer.plane_row_len(self.pixel_component_type);
        let plane_height = self.pixel_buffer.plane_height(self.pixel_component_type);
        let plane_len = self.pixel_buffer.plane_data_len();

        assert!(plane_row_len * plane_height <= plane_len);
        assert_eq!(plane_row_len % block_size, 0);
        assert_eq!(plane_height % block_size, 0);

        // CbCr samples are interleaved.
        let interleave_offset = self.interleave_offset();
        let interleave_step = self.interleave_step();

        // always include pixels in the plane beyond width
        let plane_row_samples = plane_row_len / interleave_step;
        let column_start = interleave_offset + (block_index * block_size) % plane_row_samples;
        let column_end = column_start + block_size * interleave_step;
        let row_start = block_size * ((block_index * block_size) / plane_row_samples);
        let row_end = row_start + block_size;

        //         eprintln!(
        //             "{}x{} - {}x{}",
        //             column_start, row_start, column_end, row_end
        //         );

        assert!(column_end <= plane_row_len);
        assert!(row_end <= plane_height);

        unsafe {
            let plane_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.pixel_buffer.cv_image_buffer,
                self.pixel_component_type.plane_index(),
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
        Ok(TransformBlock {
            pixel_component_type: self.pixel_component_type,
            values: block_ndarray,
        })
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

    fn interleave_offset(&self) -> usize {
        match self.pixel_component_type {
            PixelComponentType::Y | PixelComponentType::Cb => 0,
            PixelComponentType::Cr => 1,
        }
    }
    fn interleave_step(&self) -> usize {
        match self.pixel_component_type {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        }
    }

    fn plane_indexes_per_block(&self) -> usize {
        self.block_size * self.block_size * self.interleave_step()
    }
    fn plane_indexes_per_pixel_buffer(&self) -> usize {
        self.pixel_buffer.plane_height(self.pixel_component_type)
            * self.pixel_buffer.plane_row_len(self.pixel_component_type)
    }
}

impl Iterator for TransformBlockIterator {
    type Item = TransformBlock;

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
            .new_transform_block(self.block_size, self.current_block_index)
            .expect("Failed to make new transform block."); // Internal error, should crash rather than returning None.
        self.current_block_index += 1;
        Some(next_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_transform_block_0() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader
            .get_next_pixel_buffer()
            .expect("No pixel buffer.")
            .unwrap();
        let mut iter = TransformBlockIterator::new(pixel_buffer, 8, PixelComponentType::Y);
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
        let iter = TransformBlockIterator::new(pixel_buffer, 8, PixelComponentType::Y);
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
        let iter = TransformBlockIterator::new(pixel_buffer, 4, PixelComponentType::Cb);
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
        let iter = TransformBlockIterator::new(pixel_buffer, 4, PixelComponentType::Cr);
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }

    #[test]
    #[cfg(not(debug_assertions))] // too slow on debug
    fn test_get_transform_block_4() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");

        let mut count = 0;
        for pixel_buffer in reader.pixel_buffer_iter() {
            let iter = TransformBlockIterator::new(pixel_buffer, 8, PixelComponentType::Y);
            count = iter.fold(count, |acc, _| acc + 1);
        }

        assert_eq!(count, 5540400);
    }
}
