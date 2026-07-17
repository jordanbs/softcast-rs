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
    camera: ActiveCameraWrapper,
    manager: libcamera::camera_manager::CameraManager,

    frame_request_rx: Option<std::sync::mpsc::Receiver<NV12PixelBuffer>>,
}

const PIXEL_FORMAT_NV12: libcamera::pixel_format::PixelFormat =
    libcamera::pixel_format::PixelFormat::new(drm_fourcc::DrmFourcc::Nv12 as u32, 0);
const FRAME_RATE: f64 = 24.0;
const FRAME_DURATION_US: i64 = (1_000_000.0 / FRAME_RATE).round() as i64;
const RESOLUTION: (u32, u32) = (1280, 720);

impl Camera {
    pub fn new() -> Self {
        let manager = libcamera::camera_manager::CameraManager::new()
            .expect("Failed to create camera manager.");
        let cameras: libcamera::camera_manager::CameraList = manager.cameras();
        let camera = cameras
            .get(0)
            .ok_or("No cameras found.")
            .expect("Failed to get camera.");

        let camera = camera.acquire().expect("Failed to acquire camera.");
        Self {
            camera: camera.into(),
            manager,
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
        println!(
            "Using camera: {:?}",
            self.camera
                .as_ref()
                .properties()
                .get::<libcamera::properties::Model>()
                .unwrap()
        );

        let mut camera_config = self
            .camera
            .as_ref()
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

        self.camera.as_mut().configure(&mut camera_config)?;
        let stream_config = camera_config.get(0).ok_or("No camera configuration.")?;
        let stride_len = stream_config.get_stride();
        let frame_height = stream_config.get_size().height;

        let mut framebuffer_allocator =
            libcamera::framebuffer_allocator::FrameBufferAllocator::new(self.camera.as_ref());

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
                let mut request = self
                    .camera
                    .as_mut()
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
        let camera: ActiveCameraWrapper = self.camera.clone();
        self.camera.as_mut().on_request_completed(move |request| {
            println!("Camera request {:?} completed!", request);
            let frame_request = FrameRequest::new(
                request,
                stream_clone,
                camera.clone(),
                stride_len,
                frame_height,
            );
            let pixel_buffer: NV12PixelBuffer = frame_request.into();
            tx.send(pixel_buffer).expect("Failed to send request");
        });

        self.camera.as_mut().start(Some(&controls))?;

        // Enqueue all requests to the camera
        for request in frame_requests {
            println!("Request queued for execution: {request:#?}");
            self.camera
                .as_ref()
                .queue_request(request)
                .map_err(|(_, e)| e)
                .unwrap();
        }

        self.frame_request_rx = Some(rx);

        Ok(())
    }
    pub fn pixel_buffer_iter(&mut self) -> impl Iterator<Item = NV12PixelBuffer> {
        let frame_request_rx = self.frame_request_rx.take().expect("No frame_request_rx.");
        frame_request_rx.into_iter()
    }
}

#[derive(Clone)]
struct ActiveCameraWrapper {
    camera: std::sync::Arc<std::cell::UnsafeCell<libcamera::camera::ActiveCamera<'static>>>,
}
impl From<libcamera::camera::ActiveCamera<'static>> for ActiveCameraWrapper {
    fn from(camera: libcamera::camera::ActiveCamera<'static>) -> Self {
        Self {
            camera: std::sync::Arc::new(std::cell::UnsafeCell::new(camera)),
        }
    }
}
unsafe impl Send for ActiveCameraWrapper {}
unsafe impl Sync for ActiveCameraWrapper {}
impl AsRef<libcamera::camera::ActiveCamera<'static>> for ActiveCameraWrapper {
    fn as_ref(&self) -> &libcamera::camera::ActiveCamera<'static> {
        unsafe { &*self.camera.get() }
    }
}
impl AsMut<libcamera::camera::ActiveCamera<'static>> for ActiveCameraWrapper {
    fn as_mut(&mut self) -> &mut libcamera::camera::ActiveCamera<'static> {
        // camera is thread safe, trust me rust
        unsafe { &mut *self.camera.get() }
    }
}

unsafe impl Send for Camera {}
unsafe impl Sync for Camera {}

pub struct FrameRequest {
    pub request: Option<libcamera::request::Request>,
    pub stream: libcamera::stream::Stream,
    camera: ActiveCameraWrapper,
    pub stride: u32,
    pub frame_height: u32,
}
impl FrameRequest {
    fn new(
        request: libcamera::request::Request,
        stream: libcamera::stream::Stream,
        camera: ActiveCameraWrapper,
        stride: u32,
        frame_height: u32,
    ) -> Self {
        Self {
            request: Some(request),
            stream,
            camera,
            stride,
            frame_height,
        }
    }
}
impl Drop for FrameRequest {
    fn drop(&mut self) {
        let mut request = self.request.take().expect("No request.");
        request.reuse(libcamera::request::ReuseFlag::REUSE_BUFFERS);
        self.camera
            .as_ref()
            .queue_request(request)
            .expect("Buffer failed to requeue.");
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
