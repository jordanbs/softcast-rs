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

pub const DEFAULT_Y_FRAME_LEN: usize = 0x10000; // ofdm symbols per frame
pub const DEFAULT_CBCR_FRAME_LEN: usize = 0x2000; // ofdm symbols per frame

pub const DEFAULT_Y_WHITEN_LEN: usize = 0x10000; // TODO: whiten crashes when a frame is missed
pub const DEFAULT_CBCR_WHITEN_LEN: usize = 0x2000;

use crate::pixel_buffer::{HasPixelComponentType, PixelComponentType};

#[derive(Clone, Debug)]
pub struct Config {
    pub y: PerPixelTypeConfig,
    pub cbcr: PerPixelTypeConfig,
}
#[derive(Clone, Debug)]
pub struct PerPixelTypeConfig {
    pub frame_length: usize,
    pub whiten_length: usize, // ofdm frames to whiten; must be a power of 2
}
impl Default for Config {
    fn default() -> Self {
        Self {
            y: PerPixelTypeConfig {
                frame_length: DEFAULT_Y_FRAME_LEN,
                whiten_length: DEFAULT_Y_WHITEN_LEN,
            },
            cbcr: PerPixelTypeConfig {
                frame_length: DEFAULT_CBCR_FRAME_LEN,
                whiten_length: DEFAULT_CBCR_WHITEN_LEN,
            },
        }
    }
}

static ONCE: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
impl Config {
    pub fn set(config: Config) {
        ONCE.set(config).expect("config already set");
    }
    pub fn get() -> Config {
        ONCE.get_or_init(|| Config::default()).clone()
    }
    pub fn per_pixel_type<PixelType: HasPixelComponentType>(&self) -> PerPixelTypeConfig {
        match PixelType::TYPE {
            PixelComponentType::Y => &self.y,
            PixelComponentType::Cb | PixelComponentType::Cr => &self.cbcr,
        }
        .clone()
    }
}
