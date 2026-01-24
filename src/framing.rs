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
    td_symbols: [Complex32; FRAME_LEN], // in time domain
}

impl Default for OFDMSymbol {
    fn default() -> Self {
        Self {
            td_symbols: [Complex32::default(); FRAME_LEN],
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
                    symbol.td_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S0b => unsafe {
                self.state = OFDMFrameGeneratorState::S1;
                let status = liquid_sys::ofdmframegen_write_S0b(
                    self.ofdm_framegen,
                    symbol.td_symbols.as_mut_ptr(),
                ) as u32;
                assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                Some(symbol)
            },
            OFDMFrameGeneratorState::S1 => unsafe {
                self.state = OFDMFrameGeneratorState::Data;
                let status = liquid_sys::ofdmframegen_write_S1(
                    self.ofdm_framegen,
                    symbol.td_symbols.as_mut_ptr(),
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
                            symbol.td_symbols.as_mut_ptr(),
                        )
                    } as u32;
                    assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);
                    return Some(symbol);
                }
                let time_domain = &mut symbol.td_symbols;
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
