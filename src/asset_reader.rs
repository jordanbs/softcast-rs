use ndarray;
use std::{error, path};

use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

use objc2::{rc::Retained, runtime::AnyObject};
use objc2_av_foundation::{
    AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput,
    AVAssetTrack, AVMediaTypeVideo, AVURLAsset,
};

use objc2_core_foundation::{CFRetained, CFString};

use objc2_core_video::{
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    CVImageBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
    CVPixelBufferGetDataSize, CVPixelBufferGetHeightOfPlane, CVPixelBufferIsPlanar,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};

pub struct AssetReader<'a> {
    path: &'a path::Path,
    loaded_reader: Option<LoadedAssetReader>,
}

impl<'a> AssetReader<'a> {
    pub fn new(file_path: &'a str) -> Self {
        AssetReader {
            path: path::Path::new(file_path),
            loaded_reader: None,
        }
    }

    fn av_asset_reader(&mut self) -> Result<Retained<AVAssetReader>, Box<dyn error::Error>> {
        if self.loaded_reader.is_none() {
            self.loaded_reader = Some(LoadedAssetReader::load(self.path)?)
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
            self.loaded_reader = Some(LoadedAssetReader::load(self.path)?)
        }

        Ok(self
            .loaded_reader
            .as_ref()
            .cloned()
            .ok_or("No reader")?
            .av_asset_output)
    }

    #[allow(deprecated)] // blocking i/o is expected here
    pub fn get_next_pixel_buffer(&mut self) -> Result<PixelBuffer, Box<dyn error::Error>> {
        let av_reader = self.av_asset_reader()?;
        let av_output = self.av_asset_output()?;
        loop {
            if let Some(pixel_buffer) = self.get_next_pixel_buffer_helper(&av_reader, &av_output)? {
                return Ok(pixel_buffer);
            }
        }
    }

    #[allow(deprecated)] // blocking i/o is expected here
    fn get_next_pixel_buffer_helper(
        &self,
        av_reader: &AVAssetReader,
        av_output: &AVAssetReaderTrackOutput,
    ) -> Result<Option<PixelBuffer>, Box<dyn error::Error>> {
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

pub struct MacroBlock {
    pixel_component_type: PixelComponentType,
    values: ndarray::Array2<u8>,
}

impl MacroBlock {}

pub struct MacroBlockIterator {
    pixel_buffer: PixelBuffer,
    block_size: usize,
    pixel_component_type: PixelComponentType,
    current_block_index: usize,
}

impl MacroBlockIterator {
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
        }
    }

    fn new_macro_block(
        &self,
        block_size: usize, // only square blocks supported atm
        block_index: usize,
    ) -> Result<MacroBlock, Box<dyn error::Error>> {
        let mut block_ndarray = ndarray::Array2::zeros((block_size, block_size));

        let plane_row_len = self.pixel_buffer.plane_row_len(self.pixel_component_type);
        let plane_height = self.pixel_buffer.plane_height(self.pixel_component_type);
        let plane_len = self.pixel_buffer.plane_data_len();

        assert!(plane_row_len * plane_height <= plane_len);
        assert_eq!(plane_row_len % block_size, 0);
        assert_eq!(plane_height % block_size, 0);

        // CbCr samples are interleaved.
        let interleave_offset = match self.pixel_component_type {
            PixelComponentType::Y | PixelComponentType::Cb => 0,
            PixelComponentType::Cr => 1,
        };
        let interleave_step = match self.pixel_component_type {
            PixelComponentType::Y => 1,
            PixelComponentType::Cb | PixelComponentType::Cr => 2,
        };

        // always include pixels in the plane beyond width
        let column_start = interleave_offset + (block_index * block_size) % plane_row_len;
        let column_end = column_start + block_size * interleave_step;
        let row_start = block_size * ((block_index * block_size) / plane_row_len);
        let row_end = row_start + block_size;

        eprintln!(
            "{}x{} - {}x{}",
            column_start, row_start, column_end, row_end
        );

        if row_end == 1081 {
            let _ = 2 + 2;
        }

        assert!(column_end <= plane_row_len);
        assert!(row_end <= plane_height);

        unsafe {
            let flags = CVPixelBufferLockFlags::ReadOnly;
            CVPixelBufferLockBaseAddress(&self.pixel_buffer.cv_image_buffer, flags);

            let plane_ptr = CVPixelBufferGetBaseAddressOfPlane(
                &self.pixel_buffer.cv_image_buffer,
                self.pixel_component_type.plane_index(),
            ) as *const u8;

            if plane_ptr.is_null() {
                CVPixelBufferUnlockBaseAddress(&self.pixel_buffer.cv_image_buffer, flags);
                return Err("CVPixelBufferGetBaseAddressOfPlane returned NULL.".into());
            }

            let plane_slice = std::slice::from_raw_parts(plane_ptr, plane_len);

            for plane_row in row_start..row_end {
                for plane_column in (column_start..column_end).step_by(interleave_step) {
                    let plane_idx = plane_column + plane_row * plane_row_len;
                    let block_column = plane_column - column_start;
                    let block_row = plane_row - row_start;
                    block_ndarray[(block_column, block_row)] = plane_slice[plane_idx];
                }
            }
            CVPixelBufferUnlockBaseAddress(&self.pixel_buffer.cv_image_buffer, flags);
        }
        Ok(MacroBlock {
            pixel_component_type: self.pixel_component_type,
            values: block_ndarray,
        })
    }
}

impl Iterator for MacroBlockIterator {
    type Item = MacroBlock;

    fn next(&mut self) -> Option<Self::Item> {
        let next_plane_row_start = self.block_size
            * ((self.current_block_index * self.block_size)
                / self.pixel_buffer.plane_row_len(self.pixel_component_type));

        if next_plane_row_start >= self.pixel_buffer.plane_height(self.pixel_component_type) {
            // Completed this pixel buffer.
            return None;
        }

        eprintln!("Getting block {}", self.current_block_index);

        let next_block = self
            .new_macro_block(self.block_size, self.current_block_index)
            .expect("Failed to make new macro block.");
        self.current_block_index += 1;
        Some(next_block)
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
                tracks.firstObject().expect("File has no video tracks.");

            let reader: Retained<AVAssetReader> =
                AVAssetReader::assetReaderWithAsset_error(&asset as &AVAsset)
                    .expect("AVAssetReaderWithAssetWithError failed.");

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

#[cfg(test)]
mod tests {
    use super::*;

    //     #[test]
    //     fn test_get_pixel_buffer_0() {
    //         let mut reader =
    //             AssetReader::new("/Users/jordanbs/Desktop/Screen Recording 2025-11-21 at 19.06.33.mov");
    //         let pixel_buffer = reader.get_next_pixel_buffer().unwrap();
    //         let value = pixel_buffer.get_pixel_value_at_coordinate(PixelComponentType::Y, (0, 0));
    //
    //         eprintln!("pixel_buffer: {:?} value: {}", pixel_buffer, value);
    //         assert_ne!(value, 0);
    //     }

    #[test]
    fn test_get_macro_block_0() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader.get_next_pixel_buffer().expect("No pixel buffer.");
        let mut iter = MacroBlockIterator::new(pixel_buffer, 8, PixelComponentType::Y);
        //         let blocks: Vec<MacroBlock> = iter.collect();
        let block = iter.next().expect("No macro blocks produced.");

        assert_eq!(block.values.len(), 64);
    }

    #[test]
    fn test_get_macro_block_1() {
        let mut reader = AssetReader::new("/Users/jordanbs/Downloads/sample-5s.mp4");
        let pixel_buffer = reader.get_next_pixel_buffer().expect("No pixel buffer.");
        let iter = MacroBlockIterator::new(pixel_buffer, 8, PixelComponentType::Y);
        //         let blocks: Vec<MacroBlock> = iter.collect();
        let count = iter.fold(0, |acc, _| acc + 1);

        assert_eq!(count, 32400);
    }
}
