// use std::{error, path};
//
// use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
//
// use objc2::{rc::Retained, runtime::AnyObject};
// use objc2_av_foundation::{
//     AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput,
//     AVAssetTrack, AVMediaTypeVideo, AVURLAsset,
// };
//
// use objc2_core_foundation::CFString;
//
// use objc2_core_video::{
//     kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
//     CVPixelBufferGetBaseAddress, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRow,
//     CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight, CVPixelBufferGetWidth,
//     CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
// };
//
// use objc2_core_media::{CMSampleBuffer, CMSampleBufferGetImageBuffer};
//
// fn main() -> Result<(), Box<dyn error::Error>> {
//     let path =
//         path::Path::new("/Users/jordanbs/Desktop/Screen Recording 2025-11-21 at 19.06.33.mov");
//     read_video_pixels(path)?;
//     Ok(())
// }
//
// fn read_video_pixels(path: &path::Path) -> Result<(), Box<dyn error::Error>> {
//     unsafe {
//         // --- 1. Path -> file:// URL ----------------------------------------
//
//         let path_bytes = path.as_os_str().as_encoded_bytes();
//         let path_str = std::str::from_utf8(path_bytes)
//             .map_err(|_| "non-UTF8 paths not handled in this demo")?;
//         let ns_path = NSString::from_str(path_str);
//         let url = NSURL::fileURLWithPath_isDirectory(&ns_path, false);
//
//         let asset: Retained<AVURLAsset> = AVURLAsset::assetWithURL(&url);
//
//         // Get all video tracks.
//         let tracks: Retained<NSArray<AVAssetTrack>> =
//             asset.tracksWithMediaType(&AVMediaTypeVideo.unwrap());
//         if tracks.count() == 0 {
//             return Err("no video tracks in asset".into());
//         }
//         let track: Retained<AVAssetTrack> =
//             tracks.firstObject().expect("File has no video tracks.");
//
//         let pixel_format_key: &NSString =
//             &*(kCVPixelBufferPixelFormatTypeKey as *const CFString as *const NSString);
//         let pixel_format_value = NSNumber::new_u32(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
//
//         let video_settings: Retained<NSDictionary<NSString, AnyObject>> =
//             NSDictionary::from_slices::<NSString>(
//                 &[pixel_format_key.as_ref()],
//                 &[pixel_format_value.as_ref()],
//             );
//
//         // Create the reader (the *_error methods usually return Result<_, NSError>).
//         let reader: Retained<AVAssetReader> =
//             AVAssetReader::assetReaderWithAsset_error(&asset as &AVAsset)
//                 .expect("AVAssetReader::assetReaderWithAsset_error failed");
//
//         // Attach a track output that will give us CVPixelBuffer-backed CMSampleBuffers.
//         let output: Retained<AVAssetReaderTrackOutput> =
//             AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(
//                 &track,
//                 Some(&video_settings),
//             );
//
//         reader.addOutput(&output as &AVAssetReaderOutput);
//
//         if !reader.startReading() {
//             return Err("startReading() failed".into());
//         }
//
//         let mut frame_idx = 0;
//         loop {
//             // copyNextSampleBuffer returns the next CMSampleBufferRef or nil at EOF.
//             let sample: Option<Retained<CMSampleBuffer>> = output.copyNextSampleBuffer();
//             let Some(sample) = sample else {
//                 // Either EOF or an error; check status to distinguish.
//                 match reader.status() {
//                     AVAssetReaderStatus::Completed => {
//                         println!("End of movie.");
//                         break;
//                     }
//                     status => {
//                         eprintln!("Reader stopped with status {:?}", status);
//                         break;
//                     }
//                 }
//             };
//
//             let Some(pixel_buf) = CMSampleBufferGetImageBuffer(&sample) else {
//                 continue;
//             };
//
//             // Lock so we can read the memory.
//             let flags = CVPixelBufferLockFlags::ReadOnly;
//             CVPixelBufferLockBaseAddress(&pixel_buf, flags);
//
//             let width = CVPixelBufferGetWidth(&pixel_buf) as usize;
//             let height = CVPixelBufferGetHeight(&pixel_buf) as usize;
//             let bytes_per_row = CVPixelBufferGetBytesPerRow(&pixel_buf) as usize;
//
//             let base_ptr = CVPixelBufferGetBaseAddress(&pixel_buf) as *const u8;
//             if base_ptr.is_null() {
//                 CVPixelBufferUnlockBaseAddress(&pixel_buf, flags);
//                 continue;
//             }
//
//             let buf_len = bytes_per_row * height;
//             let pixels: &[u8] = std::slice::from_raw_parts(base_ptr, buf_len);
//
//             // 4:2:0 YpCbCr
//             let y_stride = CVPixelBufferGetBytesPerRowOfPlane(&pixel_buf, 0) as usize;
//             let y_base = CVPixelBufferGetBaseAddressOfPlane(&pixel_buf, 0) as *const u8;
//             let y_len = y_stride * height;
//             let y_plane = std::slice::from_raw_parts(y_base, y_len);
//
//             let c_stride = CVPixelBufferGetBytesPerRowOfPlane(&pixel_buf, 1) as usize;
//             let c_base = CVPixelBufferGetBaseAddressOfPlane(&pixel_buf, 1) as *const u8;
//             let c_len = c_stride * (height / 2);
//             let c_plane = std::slice::from_raw_parts(c_base, c_len);
//
//             //             println!("c_len {} c_stride {}", c_len, c_stride);
//
//             let mut y_idx = 0;
//             let mut c_idx = 0;
//
//             for row in 0..height {
//                 for column in 0..width {
//                     let y = y_plane[y_idx];
//                     y_idx += 1;
//                     let mut c_idx_to_use = c_idx;
//
//                     // 4:2:0
//                     if row % 2 == 1 {
//                         c_idx_to_use -= width;
//                     } else if column % 2 == 1 {
//                         c_idx_to_use -= 2;
//                     }
//                     //                     println!("c_idx {} c_idx_to_use {}", c_idx, c_idx_to_use);
//                     let cb = c_plane[c_idx_to_use + 0];
//                     let cr = c_plane[c_idx_to_use + 1];
//
//                     if c_idx == c_idx_to_use {
//                         c_idx += 2;
//                     }
//
//                     println!(
//                         "Frame {} {}×{} {}x{}; YpCbCr = ({},{},{})",
//                         frame_idx, width, height, column, row, y, cb, cr
//                     );
//                 }
//                 // take care of remainder bytes in the row
//                 //                 println!(
//                 //                     "c_idx {} cstride {} adding {}",
//                 //                     c_idx,
//                 //                     c_stride,
//                 //                     c_stride - (c_idx % c_stride)
//                 //                 );
//                 y_idx += y_stride - (y_idx % y_stride);
//                 if c_idx % c_stride != 0 {
//                     c_idx += c_stride - (c_idx % c_stride);
//                 }
//             }
//             frame_idx += 1;
//
//             CVPixelBufferUnlockBaseAddress(&pixel_buf, flags);
//             // Retained<CMSampleBuffer> drops here and releases the sample buffer.
//         }
//
//         Ok(())
//     }
// }
