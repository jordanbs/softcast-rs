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
