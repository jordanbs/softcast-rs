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
trait ToByte {
    fn to_byte(self, modem: *mut liquid_sys::modemcf_s) -> u8;
}
trait FromByte {
    fn from_byte(byte: u8, modem: *mut liquid_sys::modemcf_s) -> Self;
}
type BPSKModulatedByte = [QuadratureSymbol; 8];
impl ToByte for &BPSKModulatedByte {
    fn to_byte(self, modem: *mut liquid_sys::modemcf_s) -> u8 {
        let quadature_symbols = self;

        let mut byte = 0u8;
        for (bitpos, q_symbol) in quadature_symbols.iter().enumerate().rev() {
            let mut bitval = 0u32;
            let status =
                unsafe { liquid_sys::modemcf_demodulate(modem, q_symbol.value, &mut bitval) };
            assert_eq!(status as u32, liquid_sys::liquid_error_code_LIQUID_OK);
            let bitval = bitval as u8;
            byte |= bitval << bitpos;
        }
        byte
    }
}
impl FromByte for BPSKModulatedByte {
    fn from_byte(byte: u8, modem: *mut liquid_sys::modemcf_s) -> Self {
        let mut quadrature_symbols = BPSKModulatedByte::default();

        for (bitpos, q_symbol) in quadrature_symbols.iter_mut().enumerate().rev() {
            // MSB-first
            let bitval = 0 != (byte & (1 << bitpos));
            let symbol_ptr: *mut Complex32 = &mut q_symbol.value; // Complex32 is bincompat with C float complex
            let status = unsafe { liquid_sys::modemcf_modulate(modem, bitval.into(), symbol_ptr) };
            assert_eq!(status as u32, liquid_sys::liquid_error_code_LIQUID_OK);
        }

        quadrature_symbols
    }
}

pub mod metadata {
    use super::*;

    const MODULATION_SCHEME: u32 = liquid_sys::modulation_scheme_LIQUID_MODEM_BPSK;

    pub struct MetadataModulator<I: Iterator<Item = EncodedPacket>> {
        modemcf: *mut liquid_sys::modemcf_s,
        inner: I,
        working_packet: Option<EncodedPacket>,
        working_packet_pos: usize,
    }

    impl<I: Iterator<Item = EncodedPacket>> From<I> for MetadataModulator<I> {
        fn from(encoded_packet_iter: I) -> Self {
            let modemcf = unsafe { liquid_sys::modemcf_create(MODULATION_SCHEME) }; // TODO: destroy on drop
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

        type Item = BPSKModulatedByte;
        fn next(&mut self) -> Option<Self::Item> {
            if self.working_packet.is_none() {
                self.working_packet = self.inner.next();
            }
            let working_packet = self.working_packet.as_ref()?;
            let byte = working_packet.encoded_data[self.working_packet_pos];
            let quadrature_symbols = BPSKModulatedByte::from_byte(byte, self.modemcf);

            self.working_packet_pos += 1;
            if working_packet.encoded_data.len() == self.working_packet_pos {
                self.working_packet = None;
                self.working_packet_pos = 0;
            }
            Some(quadrature_symbols)
        }
    }

    pub struct MetadataDemodulator<I: Iterator<Item = QuadratureSymbol>> {
        modemcf: *mut liquid_sys::modemcf_s,
        inner: I,
    }
    impl<I: Iterator<Item = QuadratureSymbol>> From<I> for MetadataDemodulator<I> {
        fn from(quadrature_symbol_iter: I) -> Self {
            let modemcf = unsafe { liquid_sys::modemcf_create(MODULATION_SCHEME) }; // TODO: destroy on drop
            MetadataDemodulator {
                modemcf,
                inner: quadrature_symbol_iter,
            }
        }
    }
    impl<I: Iterator<Item = QuadratureSymbol>> Iterator for MetadataDemodulator<I> {
        // TODO: size hint

        type Item = Result<EncodedPacket, Box<dyn std::error::Error>>;
        fn next(&mut self) -> Option<Self::Item> {
            let mut packet_buf = [0u8; ENCODED_MESSAGE_LENGTH];
            for byte in &mut packet_buf {
                let mut bpsk_byte = BPSKModulatedByte::default();
                let mut bits_consumed = 0;
                for quadrature_symbol in &mut bpsk_byte {
                    if let Some(consumed_symbol) = self.inner.next() {
                        *quadrature_symbol = consumed_symbol;
                        bits_consumed += 1;
                    } else if bits_consumed != 0 {
                        let err =
                            Err("Not enough quadrature symbols to create a full byte.".into());
                        return Some(err);
                    }
                }
                *byte = bpsk_byte.to_byte(self.modemcf);
            }
            let packet: EncodedPacket = packet_buf.into();
            Some(Ok(packet))
        }
    }
}
