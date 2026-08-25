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

use crate::framing::FFTW_PLANNER_LOCK;
use crate::sync::Complex32Reader;
use liquid_sys;
use num_complex::Complex32;
use std::assert_matches;

const NUM_SUBCARRIERS: u32 = 64;
const CP_LEN: u32 = 16;
const TAPER_LEN: u32 = 4;

const PAYLOAD_LEN: usize = 0x400;

pub struct DigitalEncoder<R: std::io::Read> {
    reader: R,
    fg: liquid_sys::ofdmflexframegen,
}
impl<R: std::io::Read> From<R> for DigitalEncoder<R> {
    fn from(reader: R) -> Self {
        let mut fgprops = liquid_sys::ofdmflexframegenprops_s {
            check: liquid_sys::crc_scheme_LIQUID_CRC_32,
            fec0: liquid_sys::fec_scheme_LIQUID_FEC_RS_M8,
            fec1: liquid_sys::fec_scheme_LIQUID_FEC_NONE,
            mod_scheme: liquid_sys::modulation_scheme_LIQUID_MODEM_QAM32,
        };
        let _guard = FFTW_PLANNER_LOCK.lock().unwrap();
        let fg = unsafe {
            liquid_sys::ofdmflexframegen_create(
                NUM_SUBCARRIERS,
                CP_LEN,
                TAPER_LEN,
                std::ptr::null_mut(),
                &mut fgprops,
            )
        };

        Self { reader, fg }
    }
}
impl<R: std::io::Read> DigitalEncoder<R> {
    fn iq_symbols_per_flexframe(&self) -> usize {
        let ofdm_symbols = unsafe { liquid_sys::ofdmflexframegen_getframelen(self.fg) } as usize;
        ofdm_symbols * (CP_LEN as usize + NUM_SUBCARRIERS as usize)
    }
}
impl<R: std::io::Read> Iterator for DigitalEncoder<R> {
    type Item = Box<[Complex32]>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let header = [0u8; 8];
            let mut payload = vec![0u8; PAYLOAD_LEN];

            let mut bytes_read = 0;
            while bytes_read < payload.len() {
                let dst = &mut payload[bytes_read..];
                match self.reader.read(dst) {
                    Ok(more_bytes_read) => {
                        if 0 == more_bytes_read {
                            break;
                        }
                        bytes_read += more_bytes_read
                    }
                    Err(_) => break,
                }
            }
            payload.truncate(bytes_read);
            if payload.is_empty() {
                return None;
            }

            let status = liquid_sys::ofdmflexframegen_assemble(
                self.fg,
                header.as_ptr(),
                payload.as_ptr(),
                payload.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let flexframe_len = self.iq_symbols_per_flexframe();
            let mut iq_buf = vec![Complex32::ZERO; flexframe_len];
            let frame_complete = liquid_sys::ofdmflexframegen_write(
                self.fg,
                iq_buf.as_mut_ptr(),
                iq_buf.len() as u32,
            ) != 0;
            assert!(frame_complete);
            Some(iq_buf.into())
        }
    }
}
impl<R: std::io::Read> Complex32Reader for DigitalEncoder<R> {
    fn into_iter(self) -> impl Iterator<Item = Box<[Complex32]>> {
        self
    }
}
impl<R: std::io::Read> Drop for DigitalEncoder<R> {
    fn drop(&mut self) {
        unsafe {
            liquid_sys::ofdmflexframegen_destroy(self.fg);
        }
    }
}

extern "C" fn ofdm_flexframesync_callback(
    _header: *mut ::std::os::raw::c_uchar,
    _header_valid: ::std::os::raw::c_int,
    payload: *mut ::std::os::raw::c_uchar,
    payload_len: ::std::os::raw::c_uint,
    _payload_valid: ::std::os::raw::c_int,
    _stats: liquid_sys::framesyncstats_s,
    _userdata: *mut ::std::os::raw::c_void,
) -> std::os::raw::c_int {
    if 0 == payload_len {
        return 0;
    }
    let context_ptr = _userdata as *mut CallbackContext;
    let context = unsafe { context_ptr.as_mut().expect("NULL context_ptr.") };

    assert_matches!(context.payload, None);

    let payload_slice = unsafe { std::slice::from_raw_parts(payload, payload_len as usize) };
    let payload_vec = payload_slice.to_vec(); // memcpy
    context.payload = Some(payload_vec);

    0
}

struct CallbackContext {
    payload: Option<Vec<u8>>,
}
impl CallbackContext {
    pub fn new() -> Self {
        Self { payload: None }
    }
}

pub struct DigitalDecoder<I: Iterator<Item = Box<[Complex32]>>> {
    iq_iter: std::iter::Flatten<I>,
    fs: liquid_sys::ofdmflexframesync,
    payload: Option<Vec<u8>>,
    payload_bytes_read: usize,
    context: Box<CallbackContext>,
}
impl<I: Iterator<Item = Box<[Complex32]>>> From<I> for DigitalDecoder<I> {
    fn from(iq_iter: I) -> Self {
        let mut context = Box::new(CallbackContext::new());
        let _guard = FFTW_PLANNER_LOCK.lock().unwrap();
        let fs = unsafe {
            let context_ptr: *mut CallbackContext = context.as_mut();

            let fs = liquid_sys::ofdmflexframesync_create(
                NUM_SUBCARRIERS,
                CP_LEN,
                TAPER_LEN,
                std::ptr::null_mut(),
                Some(ofdm_flexframesync_callback),
                context_ptr as *mut core::ffi::c_void,
            );
            let status = liquid_sys::ofdmflexframesync_decode_payload_soft(fs, 1) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
            fs
        };
        Self {
            iq_iter: iq_iter.flatten(),
            fs,
            payload: None,
            payload_bytes_read: 0,
            context,
        }
    }
}
impl<I: Iterator<Item = Box<[Complex32]>>> std::io::Read for DigitalDecoder<I> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        let mut bytes_read = 0;

        loop {
            if let Some(payload) = self.payload.take().or(self.context.payload.take()) {
                let cpy_len = (payload.len() - self.payload_bytes_read).min(buf.len() - bytes_read);
                let dst = &mut buf[bytes_read..(bytes_read + cpy_len)];
                let src = &payload[self.payload_bytes_read..(self.payload_bytes_read + cpy_len)];
                dst.copy_from_slice(src);
                bytes_read += cpy_len;
                self.payload_bytes_read += bytes_read;
                if self.payload_bytes_read < payload.len() {
                    self.payload = Some(payload);
                } else {
                    self.payload_bytes_read = 0;
                }
            }
            if bytes_read == buf.len() {
                break;
            }
            assert!(bytes_read < buf.len());
            if let Some(mut iq) = self.iq_iter.next() {
                unsafe {
                    let status = liquid_sys::ofdmflexframesync_execute(
                        self.fs,
                        &mut iq as *mut Complex32,
                        1,
                    ) as u32;
                    assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                }
            } else {
                break;
            }
        }
        Ok(bytes_read)
    }
}

impl<I: Iterator<Item = Box<[Complex32]>>> Drop for DigitalDecoder<I> {
    fn drop(&mut self) {
        unsafe {
            liquid_sys::ofdmflexframesync_destroy(self.fs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_digital_decode_basic() {
        let input = vec![0xfa; 1];
        let input_reader = std::io::Cursor::new(input.clone());
        let encoder: DigitalEncoder<_> = input_reader.into();
        let mut decoder: DigitalDecoder<_> = encoder.into();
        let mut output = vec![];
        let _bytes_read = decoder.read_to_end(&mut output).expect("failed to decode");
        assert_eq!(input, output);
    }

    #[test]
    fn test_digital_decode_basic_more_bytes() {
        let input = vec![0xfa; 6250];
        let input_reader = std::io::Cursor::new(input.clone());
        let encoder: DigitalEncoder<_> = input_reader.into();
        let mut decoder: DigitalDecoder<_> = encoder.into();
        let mut output = vec![];
        let _bytes_read = decoder.read_to_end(&mut output).expect("failed to decode");
        assert_eq!(input, output);
    }
}
