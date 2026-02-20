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

use crate::encoder::Complex32Consumer;
use crate::framing::OFDM_SYMBOL_LEN;
use crate::sync::*;
use limesuite_sys;
use num_complex::Complex32;
use soapysdr;
use std::io::{Read, Write};

#[derive(Default)]
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
            .nth(params.device_idx)
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

    pub fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.activate(None)?;
        self.activated = true;
        Ok(())
    }
}
impl Complex32Consumer for TransmitDevice {
    fn consume(&mut self, buf: Box<[Complex32]>) -> Result<(), Box<dyn std::error::Error>> {
        if !self.activated {
            self.activated = true;
            self.stream.activate(None)?;
        }

        if let Some(dump_file) = self.dump_file.as_mut() {
            write_complex32_symbols(dump_file, &buf)?;
        }

        self.stream
            .write_all(&[&buf], None, false, i32::MAX as i64)?; // TODO: consider using burst

        Ok(())
    }
}

pub struct ReceiveDevice {
    stream: soapysdr::RxStream<Complex32>,
    mpsc_writer: MPSCWriter,
    mpsc_reader: Option<MPSCReader>,
    dump_file: Option<std::fs::File>,
    activated: bool,
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
            activated: false,
        })
    }
    pub fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.activate(None)?;
        self.activated = true;
        Ok(())
    }
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.activated {
            self.stream.activate(None)?;
            self.activated = true;
        }
        loop {
            let mut ofdm_symbol_buf = vec![Complex32::default(); OFDM_SYMBOL_LEN];
            let mut samples_read = 0;
            while OFDM_SYMBOL_LEN != samples_read {
                let read_buf = &mut ofdm_symbol_buf[samples_read..];
                samples_read += self.stream.read(&mut [read_buf], i32::MAX as i64)?;
            }
            if let Some(dump_file) = self.dump_file.as_mut() {
                write_complex32_symbols(dump_file, &ofdm_symbol_buf)?;
            }
            self.mpsc_writer.consume(ofdm_symbol_buf.into())?;
        }
    }
    pub fn take_mpsc_reader(&mut self) -> MPSCReader {
        self.mpsc_reader.take().expect("MPSCReader already taken.")
    }
    pub fn run_async(mut self) -> std::thread::JoinHandle<Result<(), std::string::String>> {
        std::thread::spawn(move || self.run().map_err(|e| e.to_string()))
    }
}

pub struct LimeTransmitDevice {
    pub device: *mut limesuite_sys::lms_device_t,
    stream: Box<limesuite_sys::lms_stream_t>,
    dump_file: Option<std::fs::File>,
}

impl LimeTransmitDevice {
    const SEND_BUF_SIZE_IN_SAMPLES: usize = 0x100_000; // 8MiB of Complex32 samples

    pub fn try_new(
        params: RadioParams,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let num_devices = limesuite_sys::LMS_GetDeviceList(std::ptr::null_mut());
            if 0 == num_devices {
                return Err("No LimeSDR devices found.".into());
            }
            let mut device: *mut limesuite_sys::lms_device_t = std::ptr::null_mut();
            if 0 != limesuite_sys::LMS_Open(&mut device, std::ptr::null_mut(), std::ptr::null_mut())
            {
                return Err("Failed to open LimeSDR device.".into());
            }
            if 0 != limesuite_sys::LMS_Init(device) {
                return Err("Failed to init LimeSDR device.".into());
            }
            if 0 != limesuite_sys::LMS_EnableChannel(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                true,
            ) {
                return Err("Failed to enable LimeSDR channel.".into());
            }
            if 0 != limesuite_sys::LMS_SetSampleRate(device, params.sample_rate, 0) {
                return Err("Failed to set LimeSDR sample rate.".into());
            }
            if 0 != limesuite_sys::LMS_SetLOFrequency(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                params.frequency,
            ) {
                return Err("Failed to set LimeSDR sample rate.".into());
            }

            let antenna_idx = match params.antenna.as_str() {
                "BAND1" => 1,
                "BAND2" => 2,
                _ => return Err("No antenna matching {params.antenna}".into()),
            };
            if 0 != limesuite_sys::LMS_SetAntenna(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                antenna_idx,
            ) {
                return Err("Failed to set LimeSDR antenna.".into());
            }
            if 0 != limesuite_sys::LMS_SetNormalizedGain(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                1.0, // calibrate at full gain
            ) {
                return Err("Failed to set LimeSDR gain.".into());
            }
            if 0 != limesuite_sys::LMS_Calibrate(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                params.bandwidth,
                0,
            ) {
                return Err("Failed to calibrate LimeSDR.".into());
            }
            // lower gain
            if 0 != limesuite_sys::LMS_SetNormalizedGain(
                device,
                limesuite_sys::LMS_CH_TX,
                params.channel,
                params.gain,
            ) {
                return Err("Failed to set LimeSDR gain.".into());
            }
            let mut stream = Box::new(limesuite_sys::lms_stream_t {
                channel: params.channel as u32,
                dataFmt: limesuite_sys::lms_stream_t_LMS_FMT_F32,
                linkFmt: limesuite_sys::lms_stream_t_LMS_LINK_FMT_DEFAULT,
                isTx: true,
                handle: 0,
                fifoSize: Self::SEND_BUF_SIZE_IN_SAMPLES as u32,
                throughputVsLatency: 0.5, // balance latency and throughput to prevent underruns
            });
            if 0 != limesuite_sys::LMS_SetupStream(device, stream.as_mut()) {
                return Err("Failed to set up LimeSDR tx stream.".into());
            }

            Ok(Self {
                device,
                stream,
                dump_file: dump_file.then(|| create_dump_file(true)),
            })
        }
    }

    pub fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            if 0 != limesuite_sys::LMS_StartStream(self.stream.as_mut()) {
                return Err("Failed to start LimeSDR tx stream.".into());
            }
        }
        Ok(())
    }

    pub fn write(&mut self, symbols: &[Complex32]) -> Result<usize, Box<dyn std::error::Error>> {
        let num_symbols_written = unsafe {
            let mut metadata: limesuite_sys::lms_stream_meta_t = std::mem::zeroed();
            metadata.flushPartialPacket = true;
            let num_symbols_sent_or_failure = limesuite_sys::LMS_SendStream(
                self.stream.as_mut(),
                symbols.as_ptr() as *const std::ffi::c_void,
                symbols.len(),
                &metadata,
                u32::MAX, // don't timeout
            );
            if 0 > num_symbols_sent_or_failure {
                return Err("Failed to send symbols".into());
            }
            let num_symbols_sent = num_symbols_sent_or_failure as usize;

            if let Some(dump_file) = self.dump_file.as_mut() {
                write_symbols(dump_file, &symbols)?;
            }
            num_symbols_sent
        };

        Ok(num_symbols_written)
    }
}

impl Drop for LimeTransmitDevice {
    fn drop(&mut self) {
        // unsafe {
        // There are some races I need to figure out how to guard in order to safely close streams
        // let _success = limesuite_sys::LMS_DestroyStream(self.device, self.stream.as_mut());
        // closing the device requires arc and a mutex to perform safely
        // let _success = limesuite_sys::LMS_Close(self.device);
        // }
    }
}

pub struct LimeReceiveDevice {
    //     device: *mut limesuite_sys::lms_device_t,
    stream: Box<limesuite_sys::lms_stream_t>,
    mpsc_sender: std::sync::mpsc::SyncSender<Vec<Complex32>>,
    mpsc_receiver: Option<std::sync::mpsc::Receiver<Vec<Complex32>>>,
    dump_file: Option<std::fs::File>,
}

unsafe impl Send for LimeReceiveDevice {}

impl LimeReceiveDevice {
    const RECEIVE_BUF_SIZE_IN_SAMPLES: usize = 0x100_000; // 8MiB of Complex32 samples
    const READ_BUF_SIZE_IN_SAMPLES: usize = 0x400; // 8KiB of Complex32 samples

    pub fn try_new(
        params: RadioParams,
        device: *mut limesuite_sys::lms_device_t,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            if 0 != limesuite_sys::LMS_EnableChannel(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                true,
            ) {
                return Err("Failed to enable LimeSDR channel.".into());
            }
            if 0 != limesuite_sys::LMS_SetLOFrequency(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                params.frequency,
            ) {
                return Err("Failed to set LimeSDR sample rate.".into());
            }
            let antenna_idx = match params.antenna.as_str() {
                "LNAH" => 1,
                "LNAL" => 2,
                "LNAW" => 3,
                "LB1" => 4, // not verified
                "LB2" => 5, // not verified
                _ => return Err("No antenna matching {params.antenna}".into()),
            };
            if 0 != limesuite_sys::LMS_SetAntenna(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                antenna_idx,
            ) {
                return Err("Failed to set LimeSDR antenna.".into());
            }
            if 0 != limesuite_sys::LMS_SetNormalizedGain(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                params.gain,
            ) {
                return Err("Failed to set LimeSDR gain.".into());
            }
            if 0 != limesuite_sys::LMS_Calibrate(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                params.bandwidth,
                0, // flags
            ) {
                return Err("Failed to calibrate LimeSDR.".into());
            }

            let mut stream = Box::new(limesuite_sys::lms_stream_t {
                channel: params.channel as u32,
                dataFmt: limesuite_sys::lms_stream_t_LMS_FMT_F32,
                linkFmt: limesuite_sys::lms_stream_t_LMS_LINK_FMT_DEFAULT,
                isTx: false,
                handle: 0, // not to be modified manually
                fifoSize: Self::RECEIVE_BUF_SIZE_IN_SAMPLES as u32,
                throughputVsLatency: 1.0, // maximize throughput
            });
            if 0 != limesuite_sys::LMS_SetupStream(device, stream.as_mut()) {
                return Err("Failed to set up LimeSDR tx stream.".into());
            }

            let (mpsc_sender, mpsc_receiver) = std::sync::mpsc::sync_channel(
                Self::RECEIVE_BUF_SIZE_IN_SAMPLES / Self::READ_BUF_SIZE_IN_SAMPLES,
            );

            Ok(Self {
                stream,
                mpsc_sender,
                mpsc_receiver: Some(mpsc_receiver),
                dump_file: dump_file.then(|| create_dump_file(false)),
            })
        }
    }

    pub fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            if 0 != limesuite_sys::LMS_StartStream(self.stream.as_mut()) {
                return Err("Failed to start LimeSDR tx stream.".into());
            }
        }
        Ok(())
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let mut read_buf = vec![Complex32::default(); Self::READ_BUF_SIZE_IN_SAMPLES];
            let samples_read = unsafe {
                let mut metadata: limesuite_sys::lms_stream_meta_t = std::mem::zeroed();
                let num_symbols_read_or_failure = limesuite_sys::LMS_RecvStream(
                    self.stream.as_mut(),
                    read_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    read_buf.len(),
                    &mut metadata,
                    u32::MAX, // no timeout
                );
                if 0 > num_symbols_read_or_failure {
                    return Err("LMS_RecvStream failed.".into());
                }
                num_symbols_read_or_failure as usize
            };
            read_buf.truncate(samples_read);

            if let Some(dump_file) = self.dump_file.as_mut() {
                write_symbols(dump_file, &read_buf)?;
            }
            self.mpsc_sender.send(read_buf)?;
        }
    }
    pub fn take_mpsc_receiver(&mut self) -> std::sync::mpsc::Receiver<Vec<Complex32>> {
        self.mpsc_receiver
            .take()
            .expect("MPSCReader already taken.")
    }
    pub fn run_async(mut self) -> std::thread::JoinHandle<Result<(), std::string::String>> {
        std::thread::spawn(move || self.run().map_err(|e| e.to_string()))
    }
}

impl Drop for LimeReceiveDevice {
    fn drop(&mut self) {
        // unsafe {
        // There are some races I need to figure out how to guard in order to safely close streams
        // let _success = limesuite_sys::LMS_DestroyStream(self.device, self.stream.as_mut());
        // closing the device requires arc and a mutex to perform safely
        // let _success = limesuite_sys::LMS_Close(self.device);
        // }
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

fn write_complex32_symbols(
    file: &mut std::fs::File,
    buf: &[Complex32],
) -> Result<(), Box<dyn std::error::Error>> {
    for iq in buf {
        file.write_all(&iq.re.to_be_bytes())?;
        file.write_all(&iq.im.to_be_bytes())?;
    }
    Ok(())
}

fn write_symbols(
    file: &mut std::fs::File,
    symbols: &[Complex32],
) -> Result<(), Box<dyn std::error::Error>> {
    for iq in symbols {
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
    #[cfg(false)] // needs hardware to run
    fn test_sdr_loopback_flexframegen() {
        use crate::decoder::OFDMSymbolReader;
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

        tx_device.activate().expect("Failed to activate tx");
        rx_device.activate().expect("Failed to activate rx");

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

        std::thread::sleep(std::time::Duration::from_millis(500));

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

    #[test]
    fn test_link_limesuite() {
        unsafe {
            let list = limesuite_sys::LMS_GetDeviceList(std::ptr::null_mut());
            eprintln!("Devices found: {list}");
        }
    }

    #[test]
    #[cfg(false)] // needs hardware to run
    fn test_limesuite_device() {
        let radio_params = RadioParams {
            device_idx: 0,
            channel: 0,
            gain: 1.0,
            antenna: "LNAH".to_string(),
            frequency: 2_400_000_000.0,
            sample_rate: 32_000.0,
            bandwidth: 6_000_000.0,
        };
        let _lime_tx_device = LimeTransmitDevice::try_new(radio_params, false)
            .expect("Failed to create lime tx device.");
    }

    #[test]
    #[cfg(false)] // needs hardware to run
    fn test_limesuite_sdr_loopback_flexframegen_lo() {
        let original_payload = [0xbau8; 0x80];
        let iq_symbols = unsafe {
            let mut props: flexframegenprops_s = std::mem::zeroed();
            let status = flexframegenprops_init_default(&mut props) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            props.check = liquid_sys::crc_scheme_LIQUID_CRC_NONE;

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
            let mut iq_symbols = vec![Complex32::default(); frame_len];
            let frame_complete = flexframegen_write_samples(
                flexframegen,
                iq_symbols.as_mut_ptr() as *mut Complex32,
                iq_symbols.len() as u32,
            );
            assert!(0 != frame_complete);

            // LimeSuite doesn't deal well with values > 1.0
            // flexframegen pushes samples to right about 1.0, sometimes over.
            for iq in iq_symbols.iter_mut() {
                *iq *= 0.5;
            }
            iq_symbols
        };

        let tx_params = RadioParams {
            device_idx: 0,
            antenna: "BAND1".to_string(),
            frequency: 800_000_000.0,
            bandwidth: 2_500_000.0,
            channel: 0,
            gain: 0.7,
            sample_rate: 192_000.0,
        };
        let rx_params = RadioParams {
            device_idx: 0,
            antenna: "LNAL".to_string(),
            frequency: 800_000_000.0,
            bandwidth: 2_500_000.0,
            channel: 1,
            gain: 0.8,
            sample_rate: 192_000.0,
        };
        let mut tx_device = LimeTransmitDevice::try_new(tx_params, false).unwrap();
        let mut rx_device = LimeReceiveDevice::try_new(rx_params, tx_device.device, false).unwrap();

        let iq_iter = rx_device.take_mpsc_receiver().into_iter().flatten();

        // rx should be activated before tx
        rx_device.activate().expect("Failed to activate rx");
        tx_device.activate().expect("Failed to activate tx");

        rx_device.run_async();

        let mut send_buf = &iq_symbols[..];
        while !send_buf.is_empty() {
            let symbols_sent = tx_device.write(&send_buf).unwrap();
            eprintln!("Wrote {symbols_sent} symbols.");
            send_buf = &send_buf[symbols_sent..];
        }

        unsafe {
            let iq_symbols: Vec<Complex32> = iq_iter.take(0x8000).collect();

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

    #[test]
    #[cfg(false)] // needs hardware to run
    fn test_limesuite_sdr_loopback_flexframegen_hi() {
        let original_payload = [0xbau8; 0x80];
        let iq_symbols = unsafe {
            let mut props: flexframegenprops_s = std::mem::zeroed();
            let status = flexframegenprops_init_default(&mut props) as u32;
            assert_eq!(status, liquid_sys::liquid_error_code_LIQUID_OK);

            props.check = liquid_sys::crc_scheme_LIQUID_CRC_NONE;

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
            let mut iq_symbols = vec![Complex32::default(); frame_len];
            let frame_complete = flexframegen_write_samples(
                flexframegen,
                iq_symbols.as_mut_ptr() as *mut Complex32,
                iq_symbols.len() as u32,
            );
            assert!(0 != frame_complete);

            // LimeSuite doesn't deal well with values > 1.0
            // flexframegen pushes samples to right about 1.0, sometimes over.
            for iq in iq_symbols.iter_mut() {
                *iq *= 0.5;
            }
            iq_symbols
        };

        let tx_params = RadioParams {
            device_idx: 0,
            antenna: "BAND2".to_string(),
            frequency: 2_200_000_000.0,
            bandwidth: 2_500_000.0,
            channel: 1,
            gain: 0.7,
            sample_rate: 192_000.0,
        };
        let rx_params = RadioParams {
            device_idx: 0,
            antenna: "LNAH".to_string(),
            frequency: 2_200_000_000.0,
            bandwidth: 2_500_000.0,
            channel: 0,
            gain: 0.8,
            sample_rate: 192_000.0,
        };
        let mut tx_device = LimeTransmitDevice::try_new(tx_params, false).unwrap();
        let mut rx_device = LimeReceiveDevice::try_new(rx_params, tx_device.device, false).unwrap();

        let iq_iter = rx_device.take_mpsc_receiver().into_iter().flatten();

        // rx should be activated before tx
        rx_device.activate().expect("Failed to activate rx");
        tx_device.activate().expect("Failed to activate tx");

        rx_device.run_async();

        let mut send_buf = &iq_symbols[..];
        while !send_buf.is_empty() {
            let symbols_sent = tx_device.write(&send_buf).unwrap();
            eprintln!("Wrote {symbols_sent} symbols.");
            send_buf = &send_buf[symbols_sent..];
        }

        unsafe {
            let iq_symbols: Vec<Complex32> = iq_iter.take(0x8000).collect();

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
