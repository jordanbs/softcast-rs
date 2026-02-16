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

use crate::encoder::OFDMSymbolWriter;
use crate::framing::{OFDM_SYMBOL_LEN, OFDMSymbol};
use crate::sync::*;
use num_complex::Complex32;
use soapysdr;
use std::io::{Read, Write};

pub struct RadioParams {
    pub device_idx: usize,
    pub antenna: String,
    pub channel: usize,
    pub gain: f64,
    pub frequency: f64,
    pub sample_rate: f64,
    pub bandwidth: f64,
}

pub struct TransmitDevice {
    pub sdr: soapysdr::Device,
    pub stream: soapysdr::TxStream<Complex32>,
    dump_file: Option<std::fs::File>,
    activated: bool,
}

impl TransmitDevice {
    pub fn try_new(
        params: RadioParams,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let devices = soapysdr::enumerate("")?;
        let device_args = devices
            .into_iter()
            .skip(params.device_idx)
            .next()
            .ok_or("No device at index.")?;
        let device = soapysdr::Device::new(device_args)?;

        device.set_antenna(soapysdr::Direction::Tx, params.channel, params.antenna)?;
        device.set_gain_mode(soapysdr::Direction::Tx, params.channel, false)?;
        device.set_gain(soapysdr::Direction::Tx, params.channel, params.gain)?;
        device.set_sample_rate(soapysdr::Direction::Tx, params.channel, params.sample_rate)?;
        device.set_frequency(
            soapysdr::Direction::Tx,
            params.channel,
            params.frequency,
            soapysdr::Args::new(),
        )?;
        device.set_bandwidth(soapysdr::Direction::Tx, params.channel, params.bandwidth)?;

        let stream = device.tx_stream(&[params.channel])?;

        Ok(Self {
            sdr: device,
            stream,
            dump_file: dump_file.then(|| create_dump_file(true)),
            activated: false,
        })
    }
}
impl OFDMSymbolWriter for TransmitDevice {
    fn write(&mut self, symbol: OFDMSymbol) -> Result<(), Box<dyn std::error::Error>> {
        if !self.activated {
            self.activated = true;
            self.stream.activate(None)?;
        }

        if let Some(dump_file) = self.dump_file.as_mut() {
            write_ofdm_symbol(dump_file, &symbol)?;
        }

        let buffers = [symbol.time_domain_symbols.as_slice()];
        self.stream
            .write_all(&buffers, None, false, i32::MAX as i64)?; // TODO: consider using burst

        Ok(())
    }
}

pub struct ReceiveDevice {
    stream: soapysdr::RxStream<Complex32>,
    mpsc_writer: MPSCWriter,
    mpsc_reader: Option<MPSCReader>,
    dump_file: Option<std::fs::File>,
}
impl ReceiveDevice {
    pub fn try_new(
        params: RadioParams,
        sdr: &soapysdr::Device,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        sdr.set_antenna(soapysdr::Direction::Rx, params.channel, params.antenna)?;
        sdr.set_gain_mode(soapysdr::Direction::Rx, params.channel, false)?;
        sdr.set_gain(soapysdr::Direction::Rx, params.channel, params.gain)?;
        sdr.set_sample_rate(soapysdr::Direction::Rx, params.channel, params.sample_rate)?;
        sdr.set_frequency(
            soapysdr::Direction::Rx,
            params.channel,
            params.frequency,
            soapysdr::Args::new(),
        )?;
        sdr.set_bandwidth(soapysdr::Direction::Rx, params.channel, params.bandwidth)?;

        let (sender, receiver) = std::sync::mpsc::sync_channel(0x80); // 64 KiB
        let mpsc_reader = MPSCReader { receiver };
        let mpsc_writer = MPSCWriter { sender };

        let stream = sdr.rx_stream(&[params.channel])?;

        Ok(Self {
            stream,
            mpsc_writer,
            mpsc_reader: Some(mpsc_reader),
            dump_file: dump_file.then(|| create_dump_file(false)),
        })
    }
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.activate(None)?;

        loop {
            let mut ofdm_symbol_buf = [Complex32::default(); OFDM_SYMBOL_LEN];
            let mut samples_read = 0;
            while OFDM_SYMBOL_LEN != samples_read {
                let read_buf = &mut ofdm_symbol_buf[samples_read..];
                samples_read += self.stream.read(&mut [read_buf], i32::MAX as i64)?;
            }
            let ofdm_symbol = OFDMSymbol {
                time_domain_symbols: ofdm_symbol_buf,
            };
            if let Some(dump_file) = self.dump_file.as_mut() {
                write_ofdm_symbol(dump_file, &ofdm_symbol)?;
            }
            self.mpsc_writer.write(ofdm_symbol)?;
        }
    }
    pub fn take_mpsc_reader(&mut self) -> MPSCReader {
        self.mpsc_reader.take().expect("MPSCReader already taken.")
    }
    pub fn run_async(mut self) -> std::thread::JoinHandle<Result<(), std::string::String>> {
        std::thread::spawn(move || self.run().map_err(|e| e.to_string()))
    }
}

fn create_dump_file(is_write: bool) -> std::fs::File {
    let mut idx = 0;
    loop {
        let rw = if is_write { "w" } else { "r" };
        let try_path = format!("/tmp/dump{}_{:03}", rw, idx);
        if let Ok(file) = std::fs::File::create_new(try_path) {
            return file;
        }
        idx += 1;
    }
}

fn write_ofdm_symbol(
    file: &mut std::fs::File,
    symbol: &OFDMSymbol,
) -> Result<(), Box<dyn std::error::Error>> {
    for iq in symbol.time_domain_symbols {
        file.write_all(&iq.re.to_be_bytes())?;
        file.write_all(&iq.im.to_be_bytes())?;
    }
    Ok(())
}

pub fn play_dump_file(mut stream: soapysdr::TxStream<Complex32>, path: &std::path::Path) {
    if !stream.active() {
        stream.activate(None).expect("failed to activate");
    }

    let mut file = std::fs::File::open(path).expect("Failed to open dump file.");
    let mut i_buf = [0u8; size_of::<f32>()];
    let mut q_buf = [0u8; size_of::<f32>()];

    loop {
        if file.read_exact(&mut i_buf).is_err() {
            return;
        }
        if file.read_exact(&mut q_buf).is_err() {
            return;
        }

        let iq = Complex32::new(f32::from_be_bytes(i_buf), f32::from_be_bytes(q_buf));
        stream
            .write_all(&[&[iq]], None, false, i32::MAX as i64)
            .expect("Failed to write");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::OFDMSymbolReader;
    use liquid_sys::*;
    use rand::Rng;
    use std::f32::consts::PI;

    #[test]
    fn test_flexframegen() {
        unsafe {
            let mut props = flexframegenprops_s {
                check: 0,
                fec0: 0,
                fec1: 0,
                mod_scheme: 0,
            };
            let status = flexframegenprops_init_default(&mut props) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let flexframegen = flexframegen_create(&mut props);
            assert_ne!(flexframegen, std::ptr::null_mut());

            let payload = [0x9bu8; 60];
            let status = flexframegen_assemble(
                flexframegen,
                std::ptr::null(),
                &payload as *const u8,
                payload.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let frame_len = flexframegen_getframelen(flexframegen) as usize;
            let mut iq_symbols = vec![Complex32::ZERO; frame_len]; // 60 * 8 = 480, bpsk
            while 0
                == flexframegen_write_samples(
                    flexframegen,
                    iq_symbols.as_mut_ptr() as *mut Complex32,
                    iq_symbols.len() as u32,
                )
            {}

            extern "C" fn callback(
                _header: *mut u8,
                _header_valid: i32,
                payload: *mut u8,
                payload_len: u32,
                payload_valid: i32,
                _stats: framesyncstats_s,
                user_data: *mut core::ffi::c_void,
            ) -> i32 {
                unsafe {
                    assert!(payload_valid != 0);

                    let new_payload = std::slice::from_raw_parts(payload, payload_len as usize);
                    let decoded_payload = (user_data as *mut Vec<u8>).as_mut().unwrap();
                    decoded_payload.extend_from_slice(new_payload);

                    0
                }
            }

            let mut decoded_payload: Vec<u8> = vec![];
            let decoded_payload_ptr: *mut Vec<u8> = &mut decoded_payload;

            let flexframesync = flexframesync_create(
                Some(callback),
                decoded_payload_ptr as *mut core::ffi::c_void,
            );
            assert_ne!(flexframesync, std::ptr::null_mut());

            let status = flexframesync_execute(
                flexframesync,
                iq_symbols.as_ptr() as *mut Complex32,
                iq_symbols.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            assert_eq!(payload.to_vec(), decoded_payload);
        }
    }

    #[test]
    fn test_cfo_flexframegen() {
        unsafe {
            let mut props = flexframegenprops_s {
                check: 0,
                fec0: 0,
                fec1: 0,
                mod_scheme: 0,
            };
            let status = flexframegenprops_init_default(&mut props) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let flexframegen = flexframegen_create(&mut props);
            assert_ne!(flexframegen, std::ptr::null_mut());

            let payload = [0x9bu8; 60];
            let status = flexframegen_assemble(
                flexframegen,
                std::ptr::null(),
                &payload as *const u8,
                payload.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let frame_len = flexframegen_getframelen(flexframegen) as usize;
            let mut iq_symbols = vec![Complex32::ZERO; frame_len]; // 60 * 8 = 480, bpsk
            while 0
                == flexframegen_write_samples(
                    flexframegen,
                    iq_symbols.as_mut_ptr() as *mut Complex32,
                    iq_symbols.len() as u32,
                )
            {}

            // cfo
            let dphi: f32 = 0.3; // cfo in radians/sample
            let mut phi: f32 = 0.0;
            for iq in iq_symbols.iter_mut() {
                *iq *= (Complex32::i() * phi).exp();
                phi += dphi;
            }

            // phase offset, shift by 2pi/3 radians  / sample
            let mut phi = 2.0 * PI / 3.0;
            let dphi: f32 = 0.02;
            for iq in iq_symbols.iter_mut() {
                let (r, theta) = iq.to_polar();
                *iq = Complex32::from_polar(r, theta + phi);
                phi += dphi;
            }

            // prefix with random samples
            let mut rng = rand::rng();
            let mut prefix = vec![];
            for _ in 0..40 {
                let i = rng.random_range(-1.0..1.0);
                let q = rng.random_range(-1.0..1.0);
                prefix.push(Complex32::new(i, q));
            }
            prefix.append(&mut iq_symbols);
            let iq_symbols = prefix;

            extern "C" fn callback(
                _header: *mut u8,
                _header_valid: i32,
                payload: *mut u8,
                payload_len: u32,
                payload_valid: i32,
                _stats: framesyncstats_s,
                user_data: *mut core::ffi::c_void,
            ) -> i32 {
                unsafe {
                    assert!(payload_valid != 0);

                    let new_payload = std::slice::from_raw_parts(payload, payload_len as usize);
                    let decoded_payload = (user_data as *mut Vec<u8>).as_mut().unwrap();
                    decoded_payload.extend_from_slice(new_payload);

                    0
                }
            }

            let mut decoded_payload: Vec<u8> = vec![];
            let decoded_payload_ptr: *mut Vec<u8> = &mut decoded_payload;

            let flexframesync = flexframesync_create(
                Some(callback),
                decoded_payload_ptr as *mut core::ffi::c_void,
            );
            assert_ne!(flexframesync, std::ptr::null_mut());

            let status = flexframesync_execute(
                flexframesync,
                iq_symbols.as_ptr() as *mut Complex32,
                iq_symbols.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            assert_eq!(payload.to_vec(), decoded_payload);
        }
    }

    #[test]
    fn test_sdr_loopback_flexframegen() {
        let original_payload = [0x9bu8; 60];
        let iq_symbols = unsafe {
            let mut props = flexframegenprops_s {
                check: 0,
                fec0: 0,
                fec1: 0,
                mod_scheme: 0,
            };
            let status = flexframegenprops_init_default(&mut props) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let flexframegen = flexframegen_create(&mut props);
            assert_ne!(flexframegen, std::ptr::null_mut());

            let status = flexframegen_assemble(
                flexframegen,
                std::ptr::null(),
                &original_payload as *const u8,
                original_payload.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            let frame_len = flexframegen_getframelen(flexframegen) as usize;
            let mut iq_symbols = vec![Complex32::ZERO; frame_len]; // 60 * 8 = 480, bpsk
            while 0
                == flexframegen_write_samples(
                    flexframegen,
                    iq_symbols.as_mut_ptr() as *mut Complex32,
                    iq_symbols.len() as u32,
                )
            {}
            iq_symbols
        };

        let tx_params = RadioParams {
            device_idx: 0,
            antenna: "BAND1".to_string(),
            frequency: 900_000_000.0,
            bandwidth: 6_000_000.0,
            channel: 0,
            gain: 30.0,
            sample_rate: 32_000.0,
        };
        let rx_params = RadioParams {
            device_idx: 0,
            antenna: "LB1".to_string(),
            frequency: 900_000_000.0,
            bandwidth: 6_000_000.0,
            channel: 0,
            gain: 30.0,
            sample_rate: 32_000.0,
        };
        let mut tx_device = TransmitDevice::try_new(tx_params, false).unwrap();
        let mut rx_device = ReceiveDevice::try_new(rx_params, &tx_device.sdr, false).unwrap();

        let iq_iter = rx_device
            .take_mpsc_reader()
            .into_iter()
            .map(|ofdmsymbol| ofdmsymbol.time_domain_symbols)
            .flatten();

        rx_device.run_async();

        let ofdm_symbols: Vec<OFDMSymbol> = iq_symbols
            .chunks(OFDM_SYMBOL_LEN)
            .map(|iq_symbols| {
                let mut ofdm_symbol = OFDMSymbol {
                    time_domain_symbols: [Complex32::ZERO; OFDM_SYMBOL_LEN],
                };
                ofdm_symbol.time_domain_symbols[..iq_symbols.len()].copy_from_slice(iq_symbols);
                ofdm_symbol
            })
            .collect();
        for ofdm_symbol in ofdm_symbols {
            tx_device.write(ofdm_symbol).unwrap();
        }
        drop(tx_device); // for rx_device iter to complete

        unsafe {
            let iq_symbols: Vec<Complex32> = iq_iter.take(32_000 * 2).collect();

            extern "C" fn callback(
                _header: *mut u8,
                _header_valid: i32,
                payload: *mut u8,
                payload_len: u32,
                payload_valid: i32,
                _stats: framesyncstats_s,
                user_data: *mut core::ffi::c_void,
            ) -> i32 {
                unsafe {
                    if 0 != payload_valid {
                        let new_payload = std::slice::from_raw_parts(payload, payload_len as usize);
                        let decoded_payload = (user_data as *mut Vec<u8>).as_mut().unwrap();
                        decoded_payload.extend_from_slice(new_payload);
                    }

                    0
                }
            }

            let mut decoded_payload: Vec<u8> = vec![];
            let decoded_payload_ptr: *mut Vec<u8> = &mut decoded_payload;

            let flexframesync = flexframesync_create(
                Some(callback),
                decoded_payload_ptr as *mut core::ffi::c_void,
            );
            assert_ne!(flexframesync, std::ptr::null_mut());

            let status = flexframesync_execute(
                flexframesync,
                iq_symbols.as_ptr() as *mut Complex32,
                iq_symbols.len() as u32,
            ) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            assert_eq!(original_payload.to_vec(), decoded_payload);
        }
    }
}
