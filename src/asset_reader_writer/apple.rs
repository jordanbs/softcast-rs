// Copyright 2025-2026 Jordan Schneider
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

#![cfg(target_vendor = "apple")]

use std::{error, path};

use objc2_foundation::{NSArray, NSDictionary, NSMutableDictionary, NSNumber, NSString, NSURL};

use objc2::{rc::Retained, runtime::AnyObject};
use objc2_av_foundation::{
    AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput,
    AVAssetTrack, AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor,
    AVAssetWriterStatus, AVFileTypeMPEG4, AVMediaTypeVideo, AVURLAsset, AVVideoCodecKey,
    AVVideoCodecTypeH264, AVVideoHeightKey, AVVideoWidthKey,
};

use objc2_core_foundation::CFString;
use objc2_core_media::CMTime;
use objc2_core_video::*;

use std::thread::sleep;
use std::time::{Duration, SystemTime};

pub mod asset_reader {
    use super::*;
    use crate::pixel_buffer::*;

    pub struct AssetReader {
        path: path::PathBuf,
        loaded_reader: Option<LoadedAssetReader>,
    }

    impl AssetReader {
        pub fn new(file_path: std::path::PathBuf) -> Self {
            AssetReader {
                path: file_path,
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
        ) -> Result<Option<CVPixelBufferWrapper>, Box<dyn error::Error>> {
            let av_reader = self.av_asset_reader()?;
            let av_output = self.av_asset_output()?;

            unsafe {
                // copyNextSampleBuffer returns the next CMSampleBufferRef or nil at EOF.
                if let Some(sample_buffer) = av_output.copyNextSampleBuffer() {
                    let cv_pixel_buffer = sample_buffer
                        .image_buffer()
                        .ok_or("Failed to get CVPixelBuffer.")?;

                    let pixel_buffer = CVPixelBufferWrapper::new(cv_pixel_buffer);
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
                        &[pixel_format_key],
                        &[pixel_format_value.as_ref()],
                    );

                // Asset / track / reader / output setup.
                let asset: Retained<AVURLAsset> = AVURLAsset::assetWithURL(&url);

                // Get all video tracks.
                let tracks: Retained<NSArray<AVAssetTrack>> =
                    asset.tracksWithMediaType(AVMediaTypeVideo.unwrap());

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
            PixelBufferIterator { asset_reader }
        }
        pub fn macro_block_3d_iterator(
            self,
            gop_len: usize,
        ) -> MacroBlock3DIterator<Self, CVPixelBufferWrapper> {
            MacroBlock3DIterator::new(self, gop_len)
        }
    }

    impl Iterator for PixelBufferIterator<'_> {
        type Item = CVPixelBufferWrapper;
        fn next(&mut self) -> Option<Self::Item> {
            self.asset_reader
                .get_next_pixel_buffer()
                .expect("Failed to get next pixel buffer.")
        }
    }

    pub struct IntoPixelBufferIterator {
        asset_reader: AssetReader,
    }
    impl From<AssetReader> for IntoPixelBufferIterator {
        fn from(asset_reader: AssetReader) -> Self {
            Self { asset_reader }
        }
    }
    impl IntoPixelBufferIterator {
        pub fn into_macro_block_3d_iter(
            self,
            gop_len: usize,
        ) -> MacroBlock3DIterator<Self, CVPixelBufferWrapper> {
            MacroBlock3DIterator::new(self, gop_len)
        }
    }
    impl Iterator for IntoPixelBufferIterator {
        type Item = CVPixelBufferWrapper;
        fn next(&mut self) -> Option<Self::Item> {
            self.asset_reader
                .get_next_pixel_buffer()
                .expect("Failed to get next pixel buffer.")
            // TODO: better error handling
        }
    }
}

pub mod asset_writer {
    use super::*;
    use crate::pixel_buffer::*;

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

    unsafe impl Send for AssetWriter {}

    impl AssetWriter {
        fn new(
            settings: AssetWritterSettings,
            av_asset_writer: Retained<AVAssetWriter>,
            av_asset_writer_input: Retained<AVAssetWriterInput>,
            av_asset_writer_input_pixel_buffer_adaptor: Retained<
                AVAssetWriterInputPixelBufferAdaptor,
            >,
        ) -> Self {
            let timescale = settings.frame_rate as i32;
            AssetWriter {
                //                 settings: settings,
                av_asset_writer,
                av_asset_writer_input,
                av_asset_writer_input_pixel_buffer_adaptor,
                frame_index: 0,
                started_writing: false,
                timescale,
            }
        }

        pub fn load_new(settings: AssetWritterSettings) -> Result<Self, Box<dyn error::Error>> {
            let path_bytes = settings.path.as_path().as_os_str().as_encoded_bytes();
            let ns_path = NSString::from_str(std::str::from_utf8(path_bytes)?);
            let url = NSURL::fileURLWithPath_isDirectory(&ns_path, false);

            unsafe {
                let writer = AVAssetWriter::assetWriterWithURL_fileType_error(
                    &url,
                    AVFileTypeMPEG4.unwrap(),
                )?;
                writer.setMovieFragmentInterval(CMTime::new(1, 1));

                let codec_value = NSString::from_str(&settings.codec.as_string());
                let width_value = NSNumber::new_i32(settings.resolution.0);
                let height_value = NSNumber::new_i32(settings.resolution.1);

                let input_settings_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                    NSMutableDictionary::new();
                input_settings_dict.insert(AVVideoCodecKey.unwrap(), &codec_value);
                input_settings_dict.insert(AVVideoWidthKey.unwrap(), &width_value);
                input_settings_dict.insert(AVVideoHeightKey.unwrap(), &height_value);

                let input = AVAssetWriterInput::assetWriterInputWithMediaType_outputSettings(
                    AVMediaTypeVideo.unwrap(),
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
                const TIMEOUT: Duration = Duration::from_secs(5);
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
                    status => Err(format!(
                        "Failed to become ready to write: {:?} {:?}.",
                        status,
                        self.av_asset_writer.error()
                    )
                    .into()),
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
            pixel_buffer: CVPixelBufferWrapper,
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
                if self.av_asset_writer.finishWriting() {
                    Ok(())
                } else {
                    Err("Failed to finish writing.".into())
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

#[cfg(test)]
mod tests {
    use super::asset_reader::*;
    use super::asset_writer::*;
    use super::*;
    use crate::pixel_buffer::transform_block_3d::*;
    use std::fs;

    #[test]
    fn test_reader_to_writer_0() {
        let mut reader = AssetReader::new("sample-media/sample-5s.mp4".into());
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
        let mut reader = AssetReader::new("sample-media/bipbop-1920x1080-5s.mp4".into()); // 301 frames long

        let num_frames_processed = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator(GOP_SIZE)
            .fold(0, |acc, macro_block| {
                acc + macro_block.y_components.populated_frames_len
                    + macro_block.cb_components.populated_frames_len
                    + macro_block.cr_components.populated_frames_len
            });
        let num_frames_expected = 3 + (300 * 3);
        assert_eq!(num_frames_processed, num_frames_expected);
    }

    #[test]
    fn test_macro_block_3d_move() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
        let mut reader = AssetReader::new(path);

        const MACRO_BLOCK_LEN: usize = 60;

        let macro_block_3d = reader
            .pixel_buffer_iter()
            .macro_block_3d_iterator(MACRO_BLOCK_LEN)
            .next()
            .expect("Failed to get a MacroBlock3D");

        let MacroBlock3D {
            y_components,
            cb_components,
            cr_components,
            ..
        } = macro_block_3d; // demonstrating moving the components

        assert_ne!(y_components.values().len(), 0);
        assert_ne!(cb_components.values().len(), 0);
        assert_ne!(cr_components.values().len(), 0);
    }

    #[test]
    fn test_reader_to_transform_block_3d_to_pb_exact_equality() {
        let path = "sample-media/bipbop-1920x1080-5s.mp4";
        let mut reader_1 = AssetReader::new(path.into());
        let mut reader_2 = AssetReader::new(path.into());

        // reader -> pixel buffer -> macro_block_3d (3x TransformBlock3D) -> PixelBuffer
        let pb1 = reader_1
            .pixel_buffer_iter()
            .macro_block_3d_iterator(20)
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
        let path = "sample-media/bipbop-1920x1080-5s.mp4".into();
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

        let macro_block_3d_iterator = reader.pixel_buffer_iter().macro_block_3d_iterator(90);

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
