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
impl From<Complex32> for QuadratureSymbol {
    fn from(value: Complex32) -> Self {
        Self { value }
    }
}
trait FromByte {
    fn from_byte(byte: u8, modem: *mut liquid_sys::modemcf_s) -> Self;
}
type BPSKModulatedByte = [QuadratureSymbol; 8];
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

        type Item = EncodedPacket;
        fn next(&mut self) -> Option<Self::Item> {
            let mut packet_buf = [0u8; ENCODED_MESSAGE_LENGTH];
            for byte in &mut packet_buf {
                for bitpos in 0..8 {
                    // assume 0 bits if inner runs out, let CRC reject
                    if let Some(q_symbol) = self.inner.next() {
                        let mut bitval = 0u32;
                        let status = unsafe {
                            liquid_sys::modemcf_demodulate(
                                self.modemcf,
                                q_symbol.value,
                                &mut bitval,
                            )
                        };
                        assert_eq!(status as u32, liquid_sys::liquid_error_code_LIQUID_OK);

                        let bitval = bitval as u8;
                        *byte |= bitval << bitpos;
                    } else {
                        return Some(packet_buf.into());
                    }
                }
            }
            Some(packet_buf.into())
        }
    }
}

pub mod slices {
    use super::*;
    use crate::asset_reader_writer::HasPixelComponentType;
    use crate::channel_coding::fwht::ValuesProvider;
    use crate::channel_coding::slice::*;

    pub struct SliceModulator<
        'a,
        const GOP_LENGTH: usize,
        PixelType: HasPixelComponentType,
        I: Iterator<Item = Slice<'a, GOP_LENGTH, PixelType>>,
    > {
        slice_iter: I,
        working_slice: Option<Slice<'a, GOP_LENGTH, PixelType>>,
        working_idx: usize,
    }

    impl<
            'a,
            const GOP_LENGTH: usize,
            PixelType: HasPixelComponentType,
            I: Iterator<Item = Slice<'a, GOP_LENGTH, PixelType>>,
        > From<I> for SliceModulator<'a, GOP_LENGTH, PixelType, I>
    {
        fn from(slice_iter: I) -> Self {
            Self {
                slice_iter,
                working_slice: None,
                working_idx: 0,
            }
        }
    }

    impl<
            'a,
            const GOP_LENGTH: usize,
            PixelType: HasPixelComponentType,
            I: Iterator<Item = Slice<'a, GOP_LENGTH, PixelType>>,
        > SliceModulator<'a, GOP_LENGTH, PixelType, I>
    {
        fn next_real(&mut self) -> Option<f32> {
            if self.working_slice.is_none() {
                self.working_slice = self.slice_iter.next();
            }
            let working_slice = self.working_slice.as_ref()?; // ends iteration

            let values_len = working_slice.values_len();
            if self.working_idx >= values_len {
                return None;
            }
            let real_value = working_slice.value_at(self.working_idx);
            self.working_idx += 1;
            if self.working_idx >= values_len {
                self.working_slice = None;
            }
            Some(real_value)
        }
    }

    impl<
            'a,
            const GOP_LENGTH: usize,
            PixelType: HasPixelComponentType,
            I: Iterator<Item = Slice<'a, GOP_LENGTH, PixelType>>,
        > Iterator for SliceModulator<'a, GOP_LENGTH, PixelType, I>
    {
        type Item = QuadratureSymbol;

        fn next(&mut self) -> Option<Self::Item> {
            // TODO: use size hint for more thorough interleaving.
            let q_val = self.next_real()?;
            let i_val = self.next_real().unwrap_or_default(); // don't drop q_val

            Some(Complex32::new(q_val, i_val).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metadata::*;

    #[test]
    fn test_modem_basic() {
        let mut encoded_packets: Vec<EncodedPacket> =
            vec![[0xbau8; ENCODED_MESSAGE_LENGTH].into(); 333];

        for (idx, packet) in &mut encoded_packets.iter_mut().enumerate() {
            packet.encoded_data[idx] = 0x11u8;
        }
        let cloned_encoded_packets = encoded_packets.clone();

        let modulator = MetadataModulator::from(encoded_packets.into_iter());
        let demodulator = MetadataDemodulator::from(modulator.flatten());

        let mut num_new_packets = 0;
        for (original_packet, new_packet) in cloned_encoded_packets.iter().zip(demodulator) {
            num_new_packets += 1;
            assert_eq!(original_packet.encoded_data, new_packet.encoded_data);
        }
        assert_eq!(cloned_encoded_packets.len(), num_new_packets);
    }
}
