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

#![cfg(not(target_vendor = "apple"))]

use drm_fourcc;
use libcamera;

use crate::pixel_buffer::*;

pub struct Camera {
    frame_request_rx: Option<std::sync::mpsc::Receiver<NV12PixelBuffer>>,
}

const PIXEL_FORMAT_NV12: libcamera::pixel_format::PixelFormat =
    libcamera::pixel_format::PixelFormat::new(drm_fourcc::DrmFourcc::Nv12 as u32, 0);
const FRAME_RATE: f64 = 30.0;
const FRAME_DURATION_US: i64 = (FRAME_RATE / 1_000_000.0).round() as i64;
const RESOLUTION: (u32, u32) = (1280, 720);

impl Camera {
    pub fn new() -> Self {
        Self {
            frame_request_rx: None,
        }
    }

    pub fn resolution(&self) -> (usize, usize) {
        (RESOLUTION.0 as usize, RESOLUTION.1 as usize)
    }
    pub fn frame_rate(&self) -> f64 {
        FRAME_RATE
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let manager = libcamera::camera_manager::CameraManager::new()?;
        let cameras = manager.cameras();
        let camera = cameras.iter().next().ok_or("No cameras found.")?;

        println!(
            "Using camera: {:?}",
            camera
                .properties()
                .get::<libcamera::properties::Model>()
                .unwrap()
        );

        let mut camera = camera.acquire()?;
        let mut camera_config = camera
            .generate_configuration(&[libcamera::stream::StreamRole::VideoRecording])
            .ok_or("Failed to generate video recording configuration")?;

        let mut stream_config = camera_config.get_mut(0).ok_or("No camera configuration")?;
        stream_config.set_pixel_format(PIXEL_FORMAT_NV12);
        stream_config.set_size(libcamera::geometry::Size::new(RESOLUTION.0, RESOLUTION.1));

        match camera_config.validate() {
            libcamera::camera::CameraConfigurationStatus::Valid => {
                println!("Camera configuration valid!")
            }
            libcamera::camera::CameraConfigurationStatus::Adjusted => {
                println!("Camera configuration was adjusted: {camera_config:#?}")
            }
            libcamera::camera::CameraConfigurationStatus::Invalid => {
                return Err("Error validating camera configuration".into());
            }
        };
        let stream_config = camera_config.get(0).ok_or("No camera configuration.")?;

        // Ensure that pixel format was unchanged
        if PIXEL_FORMAT_NV12 != stream_config.get_pixel_format() {
            return Err("NV12 is not supported by the camera".into());
        };

        let mut controls = libcamera::control::ControlList::new();
        controls.set(libcamera::controls::FrameDurationLimits([
            FRAME_DURATION_US,
            FRAME_DURATION_US,
        ]))?;

        camera.configure(&mut camera_config)?;
        let stream_config = camera_config.get(0).ok_or("No camera configuration.")?;

        let mut framebuffer_allocator =
            libcamera::framebuffer_allocator::FrameBufferAllocator::new(&camera);

        // Allocate frame buffers for the stream
        let stream = stream_config.stream().ok_or("No camera stream.")?;
        let buffers = framebuffer_allocator.alloc(&stream)?;
        println!("Allocated {} buffers", buffers.len());

        // Convert FrameBuffer to MemoryMappedFrameBuffer, which allows reading &[u8]
        let frame_requests: Vec<_> = buffers
            .into_iter()
            .map(|buf| {
                libcamera::framebuffer_map::MemoryMappedFrameBuffer::new(buf).expect("mmap failed")
            })
            .enumerate()
            .map(|(idx, mmap_frame_buffer)| {
                let mut request = camera
                    .create_request(Some(idx as u64))
                    .expect("Failed to make request");
                request
                    .add_buffer(&stream, mmap_frame_buffer)
                    .expect("Failed to add buffer");
                request
            })
            .collect();

        let (tx, rx) = std::sync::mpsc::channel();
        let stream_clone = stream.clone();
        camera.on_request_completed(move |request| {
            println!("Camera request {:?} completed!", request);
            let frame_request = FrameRequest::new(request, stream_clone);
            let pixel_buffer: NV12PixelBuffer = frame_request.into();
            tx.send(pixel_buffer).expect("Failed to send request");
        });

        camera.start(Some(&controls))?;

        // Enqueue all requests to the camera
        for request in frame_requests {
            println!("Request queued for execution: {request:#?}");
            camera.queue_request(request).map_err(|(_, e)| e).unwrap();
        }

        self.frame_request_rx = Some(rx);

        Ok(())
    }
    pub fn pixel_buffer_iter(&mut self) -> impl Iterator<Item = NV12PixelBuffer> {
        let frame_request_rx = self.frame_request_rx.take().expect("No frame_request_rx.");
        frame_request_rx
            .into_iter()
            .inspect(|x| println!("next: {:?}", x))
    }
}

#[derive(Debug)]
pub struct FrameRequest {
    pub request: libcamera::request::Request,
    pub stream: libcamera::stream::Stream,
}
impl FrameRequest {
    pub fn new(request: libcamera::request::Request, stream: libcamera::stream::Stream) -> Self {
        Self { request, stream }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_connected() {
        let mut camera = Camera::new();
        camera.start().expect("Camera ran with no error.");
    }
}
