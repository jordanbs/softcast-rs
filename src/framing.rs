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
    quadrature_symbol_iter: I,
    ofdm_framegen: liquid_sys::ofdmframegen,
    state: OFDMFrameGeneratorState,
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
        let ofdm_framegen = unsafe {
            liquid_sys::ofdmframegen_create(
                NUM_SUBCARRIERS as u32,
                CP_LEN as u32,
                TAPER_LEN as u32,
                std::ptr::null_mut(),
            )
        }; // TODO: destroy on drop
        assert_ne!(std::ptr::null_mut(), ofdm_framegen);
        Self {
            quadrature_symbol_iter,
            ofdm_framegen,
            state: OFDMFrameGeneratorState::S0a,
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
                let freq_domain: Box<_> = self
                    .quadrature_symbol_iter
                    .by_ref()
                    .take(NUM_SUBCARRIERS)
                    .map(|quadrature_symbol| quadrature_symbol.value)
                    .collect();
                if freq_domain.is_empty() {
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

pub struct OFDMFrameSynchronizer<I: Iterator<Item = OFDMSymbol>> {
    ofdm_symbol_iter: I,
    ofdm_framesync: liquid_sys::ofdmframesync,
    self_cell_ptr: std::cell::OnceCell<*const Self>,
    time_domain_symbols: std::collections::VecDeque<QuadratureSymbol>,
}

#[allow(non_snake_case)]
extern "C" fn ofdm_framesync_callback<T: OFDMFrameSyncCallback>(
    _y: *mut Complex32,
    _p: *mut u8,
    _M: u32,
    _userdata: *mut core::ffi::c_void,
) -> i32 {
    let subcarrier_samples = unsafe { std::slice::from_raw_parts(_y, _M as usize) };
    let subcarrier_allocation = unsafe { std::slice::from_raw_parts(_p, _M as usize) };

    let cell_ptr = _userdata as *mut std::cell::OnceCell<T>;
    let cell = unsafe { cell_ptr.as_mut().expect("NULL cell_ptr.") };
    let obj = cell.get_mut().expect("Cell not initialized.");

    obj.ofdm_framesync_callback(subcarrier_samples, subcarrier_allocation);

    liquid_sys::liquid_error_code_LIQUID_OK as i32 // always ok
}

trait OFDMFrameSyncCallback {
    fn ofdm_framesync_callback(
        &mut self,
        subcarrier_samples: &[Complex32],
        subcarrier_allocation: &[u8],
    );
}

impl<I: Iterator<Item = OFDMSymbol>> From<I> for OFDMFrameSynchronizer<I> {
    fn from(ofdm_symbol_iter: I) -> Self {
        let mut cell = std::cell::OnceCell::new();
        let cell_cvoid = &mut cell as *mut std::cell::OnceCell<_> as *mut core::ffi::c_void;

        let ofdm_framesync = unsafe {
            liquid_sys::ofdmframesync_create(
                NUM_SUBCARRIERS as u32,
                CP_LEN as u32,
                TAPER_LEN as u32,
                std::ptr::null_mut(),
                Some(ofdm_framesync_callback::<Self>),
                cell_cvoid,
            )
        }; // TODO: destroy on drop
        assert_ne!(std::ptr::null_mut(), ofdm_framesync);

        let new_self = Self {
            ofdm_symbol_iter,
            ofdm_framesync,
            time_domain_symbols: vec![].into(),
            self_cell_ptr: cell, // moved
        };

        // No need for a weak ref, as callback is only synchronously called by .next().
        let cell_ref = &new_self.self_cell_ptr;
        let new_self_ptr = &new_self as *const Self;
        cell_ref.set(new_self_ptr).unwrap();
        return new_self;
    }
}

impl<I: Iterator<Item = OFDMSymbol>> OFDMFrameSyncCallback for OFDMFrameSynchronizer<I> {
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

impl<I: Iterator<Item = OFDMSymbol>> Iterator for OFDMFrameSynchronizer<I> {
    type Item = QuadratureSymbol;

    fn next(&mut self) -> Option<Self::Item> {
        while self.time_domain_symbols.is_empty() {
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
            .time_domain_symbols
            .pop_front()
            .expect("time_domain_symbols unexepectly empty.");
        Some(q_symbol)
    }
}
