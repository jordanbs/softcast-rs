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

pub const DEFAULT_Y_FRAME_LEN: usize = 0x400; // ofdm symbols per frame
pub const DEFAULT_CBCR_FRAME_LEN: usize = 0x400; // ofdm symbols per frame

pub const DEFAULT_Y_WHITEN_LEN: usize = 0x8000; // TODO: whiten crashes when a frame is missed
pub const DEFAULT_CBCR_WHITEN_LEN: usize = 0x4000;

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

static ONCE: std::sync::Mutex<Option<Config>> = std::sync::Mutex::new(None);
impl Config {
    pub fn set(config: Config) {
        let mut guard = ONCE.lock().expect("failed to grab config");
        assert!(guard.is_none(), "config is already set");
        let _ = guard.insert(config);
    }
    pub fn get() -> Config {
        let guard = ONCE.lock().expect("failed to grab config");
        guard.clone().unwrap_or_default()
    }
    #[cfg(test)]
    pub fn set_for_test(config: Config) -> ConfigGuard {
        let prev_config = Self::get();
        let mut guard = ONCE.lock().expect("failed to grab config");
        let _ = guard.insert(config);
        ConfigGuard { prev_config }
    }
    #[cfg(test)]
    pub fn lock() -> std::sync::MutexGuard<'static, ()> {
        CONFIG_LOCK.lock().expect("failed to grab config lock")
    }
    pub fn per_pixel_type<PixelType: HasPixelComponentType>(&self) -> PerPixelTypeConfig {
        match PixelType::TYPE {
            PixelComponentType::Y => &self.y,
            PixelComponentType::Cb | PixelComponentType::Cr => &self.cbcr,
        }
        .clone()
    }
}

#[cfg(test)]
static CONFIG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(test)]
pub struct ConfigGuard {
    prev_config: Config,
}
#[cfg(test)]
impl Drop for ConfigGuard {
    fn drop(&mut self) {
        let mut guard = ONCE.lock().expect("failed to grab config");
        let _ = guard.insert(self.prev_config.clone());
    }
}
