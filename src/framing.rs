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

use crate::modulation::*;
use liquid_sys;
use num_complex::Complex32;

pub const NUM_SUBCARRIERS: usize = 64;
const CP_LEN: usize = 16;
const TAPER_LEN: usize = 4;
const FRAME_LEN: usize = NUM_SUBCARRIERS + CP_LEN;

static FFTW_PLANNER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug)]
pub struct OFDMSymbol {
    time_domain_symbols: [Complex32; FRAME_LEN],
}

impl Default for OFDMSymbol {
    fn default() -> Self {
        Self {
            time_domain_symbols: [Complex32::default(); FRAME_LEN],
        }
    }
}

pub struct OFDMFrameGenerator<I: Iterator<Item = QuadratureSymbol>> {
    quadrature_symbol_iter: std::iter::Peekable<I>,
    ofdm_framegen: liquid_sys::ofdmframegen,
    state: OFDMFrameGeneratorState,
    subcarrier_allocation: Box<[u8]>,
}

enum OFDMFrameGeneratorState {
    S0a,
    S0b,
    S1,
    Data,
    Complete,
}

impl<I: Iterator<Item = QuadratureSymbol>> From<I> for OFDMFrameGenerator<I> {
    fn from(quadrature_symbol_iter: I) -> Self {
        let mut subcarrier_allocation = Box::new([0u8; NUM_SUBCARRIERS]);
        let status = unsafe {
            liquid_sys::ofdmframe_init_default_sctype(
                NUM_SUBCARRIERS as u32,
                subcarrier_allocation.as_mut_ptr(),
            )
        } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

        let ofdm_framegen = {
            // ofdmframesync calls FFTW_PLANNER, which is not thread safe.
            let _guard = FFTW_PLANNER_LOCK.lock().unwrap(); // drops at end of scope
            unsafe {
                liquid_sys::ofdmframegen_create(
                    NUM_SUBCARRIERS as u32,
                    CP_LEN as u32,
                    TAPER_LEN as u32,
                    std::ptr::null_mut(),
                )
            }
        };
        assert_ne!(std::ptr::null_mut(), ofdm_framegen);
        Self {
            quadrature_symbol_iter: quadrature_symbol_iter.peekable(),
            ofdm_framegen,
            state: OFDMFrameGeneratorState::S0a,
            subcarrier_allocation,
        }
    }
}

impl<I: Iterator<Item = QuadratureSymbol>> Iterator for OFDMFrameGenerator<I> {
    type Item = OFDMSymbol;

    fn next(&mut self) -> Option<Self::Item> {
        let mut symbol = OFDMSymbol::default();

        match self.state {
            OFDMFrameGeneratorState::S0a => unsafe {
                self.state = OFDMFrameGeneratorState::S0b;
                let status = liquid_sys::ofdmframegen_write_S0a(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S0b => unsafe {
                self.state = OFDMFrameGeneratorState::S1;
                let status = liquid_sys::ofdmframegen_write_S0b(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S1 => unsafe {
                self.state = OFDMFrameGeneratorState::Data;
                let status = liquid_sys::ofdmframegen_write_S1(
                    self.ofdm_framegen,
                    symbol.time_domain_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::Data => {
                if self.quadrature_symbol_iter.peek().is_none() {
                    self.state = OFDMFrameGeneratorState::Complete;
                    // write tail
                    let status = unsafe {
                        liquid_sys::ofdmframegen_writetail(
                            self.ofdm_framegen,
                            symbol.time_domain_symbols.as_mut_ptr(),
                        )
                    } as u32;
                    assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                    return Some(symbol);
                }

                let freq_domain: Box<_> = self
                    .subcarrier_allocation
                    .iter()
                    // Insert placeholders for null and pilot subcarriers.
                    .map(|subcarrier_type| match *subcarrier_type as u32 {
                        // Pad frame with zero values; don't drop data.
                        liquid_sys::OFDMFRAME_SCTYPE_DATA => {
                            self.quadrature_symbol_iter.next().unwrap_or_default()
                        }
                        _ => QuadratureSymbol::default(),
                    })
                    .collect();
                let time_domain = &mut symbol.time_domain_symbols;
                let status = unsafe {
                    liquid_sys::ofdmframegen_writesymbol(
                        self.ofdm_framegen,
                        freq_domain.as_ptr() as *mut Complex32,
                        time_domain.as_mut_ptr(),
                    )
                } as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            }
            OFDMFrameGeneratorState::Complete => None,
        }
    }
}

impl<I: Iterator<Item = QuadratureSymbol>> Drop for OFDMFrameGenerator<I> {
    fn drop(&mut self) {
        let status = unsafe { liquid_sys::ofdmframegen_destroy(self.ofdm_framegen) } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
    }
}

pub struct OFDMFrameSynchronizer<I: Iterator<Item = OFDMSymbol>> {
    ofdm_symbol_iter: I,
    ofdm_framesync: liquid_sys::ofdmframesync,
    callback_context: Box<CallbackContext>,
}

#[allow(non_snake_case)]
extern "C" fn ofdm_framesync_callback(
    _y: *mut Complex32,
    _p: *mut u8,
    _M: u32,
    _userdata: *mut core::ffi::c_void,
) -> i32 {
    let subcarrier_samples = unsafe { std::slice::from_raw_parts(_y, _M as usize) };
    let subcarrier_allocation = unsafe { std::slice::from_raw_parts(_p, _M as usize) };

    let context_ptr = _userdata as *mut CallbackContext;
    let context = unsafe { context_ptr.as_mut().expect("NULL context_ptr.") };

    context.ofdm_framesync_callback(subcarrier_samples, subcarrier_allocation);

    liquid_sys::liquid_error_code_LIQUID_OK as i32 // always ok
}

#[derive(Default)]
struct CallbackContext {
    time_domain_symbols: std::collections::VecDeque<QuadratureSymbol>,
}

impl CallbackContext {
    fn ofdm_framesync_callback(
        &mut self,
        subcarrier_samples: &[Complex32],
        subcarrier_allocation: &[u8],
    ) {
        let mut new_samples: std::collections::VecDeque<_> = subcarrier_samples
            .iter()
            .enumerate()
            // ignore null and pilot subcarriers
            .filter(|(idx, _)| {
                liquid_sys::OFDMFRAME_SCTYPE_DATA == subcarrier_allocation[*idx].into()
            })
            .map(|(_, sample)| QuadratureSymbol { value: *sample })
            .collect();
        self.time_domain_symbols.append(&mut new_samples);
    }
}

impl<I: Iterator<Item = OFDMSymbol>> From<I> for OFDMFrameSynchronizer<I> {
    fn from(ofdm_symbol_iter: I) -> Self {
        let mut callback_context_box = Box::new(CallbackContext::default());
        let callback_context_ptr: *mut CallbackContext = callback_context_box.as_mut();
        let callback_context_ptr = callback_context_ptr as *mut core::ffi::c_void;

        let ofdm_framesync = {
            // ofdmframesync calls FFTW_PLANNER, which is not thread safe.
            let _guard = FFTW_PLANNER_LOCK.lock().unwrap(); // drops at end of scope
            unsafe {
                liquid_sys::ofdmframesync_create(
                    NUM_SUBCARRIERS as u32,
                    CP_LEN as u32,
                    TAPER_LEN as u32,
                    std::ptr::null_mut(),
                    Some(ofdm_framesync_callback),
                    callback_context_ptr,
                )
            }
        };
        assert_ne!(std::ptr::null_mut(), ofdm_framesync);

        Self {
            ofdm_symbol_iter,
            ofdm_framesync,
            callback_context: callback_context_box,
        }
    }
}

impl<I: Iterator<Item = OFDMSymbol>> Iterator for OFDMFrameSynchronizer<I> {
    type Item = QuadratureSymbol;

    fn next(&mut self) -> Option<Self::Item> {
        while self.callback_context.time_domain_symbols.is_empty() {
            let ofdm_symbol = self.ofdm_symbol_iter.next()?; // breaks iteration
            let status = unsafe {
                // Pushes samples to self.time_domain_symbols via ofdm_framesync_callback.
                liquid_sys::ofdmframesync_execute(
                    self.ofdm_framesync,
                    ofdm_symbol.time_domain_symbols.as_ptr() as *mut Complex32,
                    FRAME_LEN as u32,
                )
            } as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
        }

        let q_symbol = self
            .callback_context
            .time_domain_symbols
            .pop_front()
            .expect("time_domain_symbols unexepectly empty.");
        Some(q_symbol)
    }
}

impl<I: Iterator<Item = OFDMSymbol>> Drop for OFDMFrameSynchronizer<I> {
    fn drop(&mut self) {
        let status = unsafe { liquid_sys::ofdmframesync_destroy(self.ofdm_framesync) } as u32;
        assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ofdm_basic() {
        let mut quadrature_symbols = vec![
            QuadratureSymbol {
                value: Complex32::default()
            };
            1037
        ];
        for (idx, symbol) in quadrature_symbols.iter_mut().enumerate() {
            symbol.value.re = 0.01 * idx as f32;
            symbol.value.im = 0.01 * -(idx as f32);
        }
        let quadrature_symbols_clone: std::collections::VecDeque<_> =
            quadrature_symbols.clone().into();

        let ofdm_frame_generator: OFDMFrameGenerator<_> = quadrature_symbols.into_iter().into();
        let ofdm_symbols: Vec<OFDMSymbol> = ofdm_frame_generator.collect();
        eprintln!("{:?}", ofdm_symbols);

        let ofdm_frame_synchronizer: OFDMFrameSynchronizer<_> = ofdm_symbols.into_iter().into();

        let new_quadrature_symbols: Vec<_> = ofdm_frame_synchronizer.collect();

        // orig may be shorter than new, because of frame padding.
        assert!(quadrature_symbols_clone.len() <= new_quadrature_symbols.len());

        for (orig, new) in quadrature_symbols_clone
            .iter()
            .zip(new_quadrature_symbols.iter())
        {
            eprintln!("orig:{:?} new:{:?}", orig, new);
            assert!((orig.value.re - new.value.re).abs() < 0.0001);
            assert!((orig.value.im - new.value.im).abs() < 0.0001);
        }
    }

    #[test]
    fn test_ofdm_multiple_frames() {
        let mut quadrature_symbols = vec![
            QuadratureSymbol {
                value: Complex32::default()
            };
            1037
        ];
        for (idx, symbol) in quadrature_symbols.iter_mut().enumerate() {
            symbol.value.re = 0.01 * idx as f32;
            symbol.value.im = 0.01 * -(idx as f32);
        }
        let quadrature_symbols_clone: std::collections::VecDeque<_> =
            quadrature_symbols.clone().into();

        let ofdm_frame_generator: OFDMFrameGenerator<_> = quadrature_symbols.into_iter().into();
        let ofdm_symbols: Vec<OFDMSymbol> = ofdm_frame_generator.collect();
        eprintln!("{:?}", ofdm_symbols);

        let ofdm_frame_synchronizer: OFDMFrameSynchronizer<_> = ofdm_symbols.into_iter().into();

        let new_quadrature_symbols: Vec<_> = ofdm_frame_synchronizer.collect();

        // orig may be shorter than new, because of frame padding.
        assert!(quadrature_symbols_clone.len() <= new_quadrature_symbols.len());

        for (orig, new) in quadrature_symbols_clone
            .iter()
            .zip(new_quadrature_symbols.iter())
        {
            eprintln!("orig:{:?} new:{:?}", orig, new);
            assert!((orig.value.re - new.value.re).abs() < 0.0001);
            assert!((orig.value.im - new.value.im).abs() < 0.0001);
        }
    }
}
