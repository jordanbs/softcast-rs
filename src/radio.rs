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

use crate::rtlsdr_iq;
use crate::sync::*;
use limesuite_sys;
use num_complex::Complex32;
use rtlsdr;
use soapysdr;
use std::io::{Read, Write};

const RECEIVE_BUF_SIZE_IN_SAMPLES: usize = 0x256_000_000;
const READ_BUF_SIZE_IN_SAMPLES: usize = 0x400_000;
const SEND_BUF_SIZE_IN_SAMPLES: usize = 0x320_000;

#[derive(Default, Clone)]
pub struct RadioParams {
    pub device_idx: usize,
    pub antenna: String,
    pub channel: usize,
    pub gain: f64,
    pub frequency: f64,
    pub sample_rate: f64,
    pub bandwidth: f64,
}

pub trait RadioDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait TransmitDevice: RadioDevice + Complex32Consumer {
    fn drain(&mut self);
}

pub trait ReceiveDevice: RadioDevice + Send {
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    fn take_mpsc_reader(&mut self) -> MPSCReader;
}

pub struct SoapyTransmitDevice {
    pub sdr: soapysdr::Device,
    pub stream: soapysdr::TxStream<Complex32>,
    dump_file: Option<std::fs::File>,
    activated: bool,
}

pub fn new_soapy_device(
    params: &RadioParams,
) -> Result<soapysdr::Device, Box<dyn std::error::Error>> {
    let devices = soapysdr::enumerate("")?;
    let device_args = devices
        .into_iter()
        .nth(params.device_idx)
        .ok_or("No device at index.")?;
    Ok(soapysdr::Device::new(device_args)?)
}

impl SoapyTransmitDevice {
    pub fn try_new(
        params: RadioParams,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = new_soapy_device(&params)?;

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
impl Complex32Consumer for SoapyTransmitDevice {
    fn consume(
        &mut self,
        buf: Box<[Complex32]>,
        _flush: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
impl RadioDevice for SoapyTransmitDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.activate(None)?;
        self.activated = true;
        Ok(())
    }
}
impl TransmitDevice for SoapyTransmitDevice {
    fn drain(&mut self) {
        let _ = self.stream.write(&[], None, true, i32::MAX as i64);
    }
}

pub struct SoapyReceiveDevice {
    stream: soapysdr::RxStream<Complex32>,
    mpsc_writer: MPSCWriter,
    mpsc_reader: Option<MPSCReader>,
    dump_file: Option<std::fs::File>,
    activated: bool,
}
impl SoapyReceiveDevice {
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

        let (mpsc_writer, mpsc_reader) =
            MPSCWriter::new_channel(RECEIVE_BUF_SIZE_IN_SAMPLES / READ_BUF_SIZE_IN_SAMPLES);

        let stream = sdr.rx_stream(&[params.channel])?;

        Ok(Self {
            stream,
            mpsc_writer,
            mpsc_reader: Some(mpsc_reader),
            dump_file: dump_file.then(|| create_dump_file(false)),
            activated: false,
        })
    }
}
impl RadioDevice for SoapyReceiveDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.activate(None)?;
        self.activated = true;
        Ok(())
    }
}
pub fn run_async(
    mut rx_device: Box<dyn ReceiveDevice + Send>,
) -> std::thread::JoinHandle<Result<(), std::string::String>> {
    std::thread::spawn(move || rx_device.run().map_err(|e| e.to_string()))
}

impl ReceiveDevice for SoapyReceiveDevice {
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let mut read_buf = vec![Complex32::ZERO; self.stream.mtu()?];
            let samples_read = match self.stream.read(&mut [&mut read_buf], i32::MAX as i64) {
                Ok(samples_read) => samples_read,
                Err(err) => {
                    eprintln!("Read error: {err}");
                    return Err(err.into());
                }
            };
            read_buf.truncate(samples_read);

            if let Some(dump_file) = self.dump_file.as_mut() {
                write_symbols(dump_file, &read_buf)?;
            }
            self.mpsc_writer.consume(read_buf.into(), false)?;
        }
    }
    fn take_mpsc_reader(&mut self) -> MPSCReader {
        self.mpsc_reader.take().expect("MPSCReader already taken.")
    }
}

pub fn new_lime_device(
    device_idx: usize,
) -> Result<*mut limesuite_sys::lms_device_t, &'static str> {
    unsafe {
        let mut device_list: [limesuite_sys::lms_info_str_t; 0x10] = std::mem::zeroed();
        // the lack of a len check in LMS_GetDeviceList is wild.
        let num_devices = limesuite_sys::LMS_GetDeviceList(device_list.as_mut_ptr());
        let success = -1 != num_devices;
        if !success {
            return Err("LMS_GetDeviceList failed.");
        }
        if 0 == num_devices {
            return Err("No LimeSDR devices found.");
        }
        if device_idx as i32 >= num_devices {
            return Err("Device-idx too high.");
        }
        let device_info_str = device_list[device_idx];

        let mut device: *mut limesuite_sys::lms_device_t = std::ptr::null_mut();
        if 0 != limesuite_sys::LMS_Open(&mut device, device_info_str.as_ptr(), std::ptr::null_mut())
        {
            return Err("Failed to open LimeSDR device.");
        }
        if 0 != limesuite_sys::LMS_Init(device) {
            return Err("Failed to init LimeSDR device.");
        }
        Ok(device)
    }
}

pub struct LimeTransmitDevice {
    pub device: *mut limesuite_sys::lms_device_t,
    stream: Box<limesuite_sys::lms_stream_t>,
    dump_file: Option<std::fs::File>,
}

impl LimeTransmitDevice {
    pub fn try_new(
        params: RadioParams,
        skip_cal: bool,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let device = new_lime_device(params.device_idx)?;

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

            if !skip_cal {
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
                linkFmt: limesuite_sys::lms_stream_t_LMS_LINK_FMT_I12,
                isTx: true,
                handle: 0,
                fifoSize: SEND_BUF_SIZE_IN_SAMPLES as u32,
                throughputVsLatency: 1.0, // balance latency and throughput to prevent underruns
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

    fn write(
        &mut self,
        symbols: &[Complex32],
        flush: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let stream_status = unsafe {
            let mut stream_status: limesuite_sys::lms_stream_status_t = std::mem::zeroed();
            let success =
                limesuite_sys::LMS_GetStreamStatus(self.stream.as_mut(), &mut stream_status);
            assert_eq!(0, success);
            stream_status
        };
        if 0 < stream_status.underrun {
            eprintln!(
                "WARNING: Underrun detected. Count:{}",
                stream_status.underrun
            );
        }
        if 0 < stream_status.droppedPackets {
            eprintln!(
                "WARNING: Dropped packets detected. Count:{}",
                stream_status.droppedPackets
            );
        }

        let num_symbols_written = unsafe {
            let mut metadata: limesuite_sys::lms_stream_meta_t = std::mem::zeroed();
            metadata.flushPartialPacket = flush;
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
impl RadioDevice for LimeTransmitDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            if 0 != limesuite_sys::LMS_StartStream(self.stream.as_mut()) {
                return Err("Failed to start LimeSDR tx stream.".into());
            }
        }
        Ok(())
    }
}
impl Complex32Consumer for LimeTransmitDevice {
    fn consume(
        &mut self,
        buf: Box<[Complex32]>,
        flush: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut write_buf = &buf[..];
        while !write_buf.is_empty() {
            let symbols_sent = self.write(&write_buf, flush)?;
            write_buf = &write_buf[symbols_sent..];
        }
        Ok(())
    }
}
impl TransmitDevice for LimeTransmitDevice {
    fn drain(&mut self) {
        loop {
            let drained = unsafe {
                let mut stream_status: limesuite_sys::lms_stream_status_t = std::mem::zeroed();
                let success =
                    limesuite_sys::LMS_GetStreamStatus(self.stream.as_mut(), &mut stream_status);
                assert_eq!(0, success);
                0 == stream_status.fifoFilledCount
            };
            if drained {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
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
    mpsc_writer: MPSCWriter,
    mpsc_reader: Option<MPSCReader>,
    dump_file: Option<std::fs::File>,
}

impl LimeReceiveDevice {
    pub fn try_new(
        params: RadioParams,
        device: *mut limesuite_sys::lms_device_t,
        skip_cal: bool,
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
            if 0 != limesuite_sys::LMS_SetSampleRate(device, params.sample_rate, 0) {
                return Err("Failed to set LimeSDR sample rate.".into());
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
            if !skip_cal {
                if 0 != limesuite_sys::LMS_SetNormalizedGain(
                    device,
                    limesuite_sys::LMS_CH_RX,
                    params.channel,
                    1.0, // calibrate at full gain
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
            }
            if 0 != limesuite_sys::LMS_SetNormalizedGain(
                device,
                limesuite_sys::LMS_CH_RX,
                params.channel,
                params.gain,
            ) {
                return Err("Failed to set LimeSDR gain.".into());
            }

            let mut stream = Box::new(limesuite_sys::lms_stream_t {
                channel: params.channel as u32,
                dataFmt: limesuite_sys::lms_stream_t_LMS_FMT_F32,
                linkFmt: limesuite_sys::lms_stream_t_LMS_LINK_FMT_I12,
                isTx: false,
                handle: 0, // not to be modified manually
                fifoSize: RECEIVE_BUF_SIZE_IN_SAMPLES as u32,
                throughputVsLatency: 1.0, // maximize throughput
            });
            if 0 != limesuite_sys::LMS_SetupStream(device, stream.as_mut()) {
                return Err("Failed to set up LimeSDR tx stream.".into());
            }

            let (mpsc_writer, mpsc_reader) =
                MPSCWriter::new_channel(RECEIVE_BUF_SIZE_IN_SAMPLES / READ_BUF_SIZE_IN_SAMPLES);

            Ok(Self {
                stream,
                mpsc_writer,
                mpsc_reader: Some(mpsc_reader),
                dump_file: dump_file.then(|| create_dump_file(false)),
            })
        }
    }
}
impl RadioDevice for LimeReceiveDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            if 0 != limesuite_sys::LMS_StartStream(self.stream.as_mut()) {
                return Err("Failed to start LimeSDR tx stream.".into());
            }
        }
        Ok(())
    }
}
impl ReceiveDevice for LimeReceiveDevice {
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let stream_status = unsafe {
                let mut stream_status: limesuite_sys::lms_stream_status_t = std::mem::zeroed();
                let success =
                    limesuite_sys::LMS_GetStreamStatus(self.stream.as_mut(), &mut stream_status);
                assert_eq!(0, success);
                stream_status
            };
            if 0 < stream_status.overrun {
                eprintln!("WARNING: Overrun detected. Count:{}", stream_status.overrun);
            }
            if 0 < stream_status.droppedPackets {
                eprintln!(
                    "WARNING: Dropped packets detected. Count:{}",
                    stream_status.droppedPackets
                );
            }

            let mut read_buf = vec![Complex32::default(); READ_BUF_SIZE_IN_SAMPLES];
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
            self.mpsc_writer.sender.send(read_buf.into())?;
        }
    }
    fn take_mpsc_reader(&mut self) -> MPSCReader {
        self.mpsc_reader.take().expect("MPSCReader already taken.")
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

pub struct RtlSdrReceiveDevice {
    device: std::sync::Arc<rtlsdr::Device>,
    mpsc_writer: Option<MPSCWriter>,
    mpsc_reader: Option<MPSCReader>,
    dump_file: Option<std::fs::File>,
}

impl RtlSdrReceiveDevice {
    pub fn try_new(
        params: RadioParams,
        dump_file: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // rtl-sdr only has one channel, ignore

        let (device, err) = rtlsdr::open(params.device_idx as i32);
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to open RTL-SDR: {:?}", err).into());
        }

        let err = device.set_sample_rate(params.sample_rate as i32);
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set sample rate: {:?}", err).into());
        }

        let err = device.set_center_freq(params.frequency as i32);
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set center freq: {:?}", err).into());
        }

        let err = device.set_tuner_bandwidth(params.bandwidth as i32);
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set bandwidth: {:?}", err).into());
        }

        let err = device.set_tuner_gain_mode(true); // manual mode
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set manual gain mode: {:?}", err).into());
        }

        let err = device.set_agc_mode(false); // manual mode
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set manual agc mode: {:?}", err).into());
        }

        let err = device.set_tuner_gain(params.gain.round() as i32);
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to set tuner gain: {:?}", err).into());
        }

        let (mpsc_writer, mpsc_reader) =
            MPSCWriter::new_channel(RECEIVE_BUF_SIZE_IN_SAMPLES / READ_BUF_SIZE_IN_SAMPLES);

        Ok(Self {
            device,
            mpsc_writer: Some(mpsc_writer),
            mpsc_reader: Some(mpsc_reader),
            dump_file: dump_file.then(|| create_dump_file(false)),
        })
    }
}

impl RadioDevice for RtlSdrReceiveDevice {
    fn activate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let err = self.device.reset_buffer();
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("Failed to reset buffer on activate: {:?}", err).into());
        }
        Ok(())
    }
}

struct RtlSdrCallbackContext {
    mpsc_writer: MPSCWriter,
    dump_file: Option<std::fs::File>,
}

unsafe extern "C" fn rtlsdr_read_callback(
    read_buf: *mut core::ffi::c_uchar,
    samples_read: u32,
    ctx: *mut core::ffi::c_void,
) {
    let callback_context_ptr = ctx as *mut RtlSdrCallbackContext;
    let callback_context = unsafe { callback_context_ptr.as_mut().expect("NULL context_ptr") };
    let mpsc_writer = &mut callback_context.mpsc_writer;
    let dump_file = &mut callback_context.dump_file;
    let read_buf_u8 = unsafe { std::slice::from_raw_parts(read_buf, samples_read as usize) };

    // eprintln!("read {samples_read}");

    let (prefix, read_buf_u16, suffix) = unsafe { read_buf_u8.align_to::<u16>() };
    assert_eq!(0, prefix.len());
    assert_eq!(0, suffix.len());

    let read_buf_iq: Box<[Complex32]> = read_buf_u16
        .into_iter()
        .map(|&sample| rtlsdr_iq::IQ[sample])
        .collect();

    if let Some(dump_file) = dump_file.as_mut() {
        let _ = write_symbols(dump_file, &read_buf_iq); // ingore err
    }
    mpsc_writer
        .consume(read_buf_iq, false)
        .expect("Failed to consume sample.");
}

impl ReceiveDevice for RtlSdrReceiveDevice {
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let read_buf_size: i32 = READ_BUF_SIZE_IN_SAMPLES
            .try_into()
            .expect("READ_BUF too big for rtl-sdr.");

        let mut callback_context_box = Box::new(RtlSdrCallbackContext {
            mpsc_writer: self.mpsc_writer.take().expect("No mpsc_writer."),
            dump_file: self.dump_file.take(),
        });
        let callback_context_ptr: *mut RtlSdrCallbackContext = callback_context_box.as_mut();
        let callback_context_ptr = callback_context_ptr as *mut core::ffi::c_void;

        let err = self.device.read_async(
            // blocks until read_cancel is called
            Some(rtlsdr_read_callback),
            callback_context_ptr,
            0,
            read_buf_size,
        );
        if !matches!(err, rtlsdr::Error::NoError) {
            return Err(format!("RTL-SDR read_async failed: {:?}", err).into());
        }
        Ok(())
    }

    fn take_mpsc_reader(&mut self) -> MPSCReader {
        self.mpsc_reader.take().expect("MPSCReader already taken.")
    }
}

impl Drop for RtlSdrReceiveDevice {
    fn drop(&mut self) {
        let _ = self.device.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::FFTW_PLANNER_LOCK;
    use liquid_sys::*;
    use rand::Rng;
    use std::f32::consts::PI;

    #[test]
    fn test_flexframegen() {
        unsafe {
            let _guard = FFTW_PLANNER_LOCK
                .lock()
                .expect("Failed to grab fftw planner lock.");

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
    #[ignore = "doesn't do well in parallel'"]
    fn test_cfo_flexframegen() {
        unsafe {
            let _guard = FFTW_PLANNER_LOCK
                .lock()
                .expect("Failed to grab fftw planner lock.");

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
    fn test_link_limesuite() {
        unsafe {
            let list = limesuite_sys::LMS_GetDeviceList(std::ptr::null_mut());
            eprintln!("Devices found: {list}");
        }
    }

    #[test]
    #[ignore = "needs hardware to run"]
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
        let _lime_tx_device = LimeTransmitDevice::try_new(radio_params, false, false)
            .expect("Failed to create lime tx device.");
    }

    #[test]
    fn test_link_rtlsdr() {
        let count = rtlsdr::get_device_count();
        eprintln!("RTL-SDR devices found: {count}");
    }

    #[test]
    #[ignore = "needs hardware to run"]
    fn test_rtlsdr_open_close() {
        let params = RadioParams {
            device_idx: 0,
            channel: 0,
            gain: 30.0,
            antenna: String::new(),
            frequency: 100_000_000.0,
            sample_rate: 2_048_000.0,
            bandwidth: 0.0,
        };
        let _rx = RtlSdrReceiveDevice::try_new(params, false)
            .expect("Failed to create RTL-SDR rx device.");
    }
}
