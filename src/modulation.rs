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

use crate::metadata_coding::packetizer::*;
use liquid_sys;
use num_complex::Complex32;

#[derive(Default, Clone, Copy)]
pub struct QuadratureSymbol {
    pub value: Complex32,
}

pub struct MetadataModulator<I: Iterator<Item = EncodedPacket>> {
    modemcf: *mut liquid_sys::modemcf_s,
    inner: I,
    working_packet: Option<EncodedPacket>,
    working_packet_pos: usize,
}
impl<I: Iterator<Item = EncodedPacket>> MetadataModulator<I> {
    const MODULATION_SCHEME: u32 = liquid_sys::modulation_scheme_LIQUID_MODEM_BPSK;

    fn new(encoded_packet_iter: I) -> Self {
        let modemcf = unsafe { liquid_sys::modemcf_create(Self::MODULATION_SCHEME) }; // TODO: destroy on drop
        MetadataModulator {
            modemcf,
            inner: encoded_packet_iter,
            working_packet: None,
            working_packet_pos: 0,
        }
    }
}

impl<I: Iterator<Item = EncodedPacket>> Iterator for MetadataModulator<I> {
    // TODO: size hint

    type Item = [QuadratureSymbol; 8]; // representing one byte
    fn next(&mut self) -> Option<Self::Item> {
        if self.working_packet.is_none() {
            self.working_packet = self.inner.next();
        }
        let working_packet = self.working_packet.as_ref()?;

        let mut quadrature_symbols = [QuadratureSymbol::default(); 8];

        let byte = working_packet.encoded_data[self.working_packet_pos];
        for (bitpos, q_symbol) in quadrature_symbols.iter_mut().enumerate().rev() {
            // MSB-first
            let bitval = 0 != (byte & (1 << bitpos));
            let symbol_ptr: *mut Complex32 = &mut q_symbol.value; // Complex32 is bincompat with C float complex
            let status =
                unsafe { liquid_sys::modemcf_modulate(self.modemcf, bitval.into(), symbol_ptr) };
            assert_eq!(status as u32, liquid_sys::liquid_error_code_LIQUID_OK);
        }
        self.working_packet_pos += 1;
        if working_packet.encoded_data.len() == self.working_packet_pos {
            self.working_packet = None;
            self.working_packet_pos = 0;
        }
        Some(quadrature_symbols)
    }
}

impl<I: Iterator<Item = EncodedPacket>> From<I> for MetadataModulator<I> {
    fn from(encoded_packet_iter: I) -> Self {
        Self::new(encoded_packet_iter)
    }
}
