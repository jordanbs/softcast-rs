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

use clap::{Parser, Subcommand};
use parse_int;
use softcast_rs::config::*;
use softcast_rs::encoder::*;
use softcast_rs::radio::*;
use softcast_rs::sync::AbortToken;

const DEFAULT_COMPRESSION_RATIO: f64 = 0.1875;
const DEFAULT_GOP_LEN: usize = 22;
const DEFAULT_Y_CHUNK_DIMENSIONS: &str = "48x40x1";
const DEFAULT_C_CHUNK_DIMENSIONS: &str = "40x30x1";
const DEFAULT_NOISE: f32 = 0.0;
const DEFAULT_TX_GAIN: f64 = 0.08;
const DEFAULT_RX_GAIN: f64 = 0.7;
const DEFAULT_FREQ: f64 = 800_000_000.0;
const DEFAULT_SAMPLE_RATE: f64 = 0x300_000 as f64;
const DEAFULT_BANDWIDTH: f64 = 1.6 * DEFAULT_SAMPLE_RATE;
const DEFAULT_TX_ANTENNA: &str = "BAND1";
const DEFAULT_RX_ANTENNA: &str = "LNAL";
const DEFAULT_TX_CHANNEL: usize = 0;
const DEFAULT_RX_CHANNEL: usize = 1;

#[derive(clap::ValueEnum, Clone, Debug)]
enum Driver {
    Soapy,
    Lime,
    RtlSdr,
}

fn parse_dimensions_3d(s: &str) -> Result<(usize, usize, usize), String> {
    let parts: Box<[&str]> = s.split('x').collect();
    if parts.len() != 3 {
        return Err("Expected WxHxD format".into());
    }

    let x = parts[0].parse().map_err(|_| "Invalid width")?;
    let y = parts[1].parse().map_err(|_| "Invalid height")?;
    let z = parts[2].parse().map_err(|_| "Invalid length")?;

    Ok((x, y, z))
}

fn parse_power_of_two(s: &str) -> Result<usize, String> {
    let u: usize = parse_int::parse(s).map_err(|_| "{s} is not an int")?;

    if u.is_power_of_two() {
        Ok(u)
    } else {
        let error_string = format!("{u} is not a power of 2. Try {}", u.next_power_of_two());
        Err(error_string)
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    use super::*;
    use softcast_rs::decoder::*;
    use softcast_rs::simulator::*;

    #[derive(Parser)]
    struct Args {
        #[command(subcommand)]
        command: Commands,
    }

    #[derive(Subcommand)]
    enum Commands {
        Tx {
            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_exists)]
            infile: std::path::PathBuf,

            #[arg(short, default_value_t = DEFAULT_COMPRESSION_RATIO)]
            compression_ratio: f64,

            #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
            gop_len: usize,

            // defaults set for 1080p
            #[arg(long="y", value_parser = parse_dimensions_3d, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
            y_chunk_dimensions: (usize, usize, usize),

            #[arg(long="cbcr", value_parser = parse_dimensions_3d, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
            c_chunk_dimensions: (usize, usize, usize),

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_Y_WHITEN_LEN)]
            y_whiten_len: usize,

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_CBCR_WHITEN_LEN)]
            cbcr_whiten_len: usize,

            #[arg(long, value_parser = parse_int::parse::<usize>, default_value_t = FRAME_LEN)]
            frame_len: usize,

            #[arg(short, default_value_t = DEFAULT_FREQ)]
            frequency: f64,

            #[arg(short = 'n', default_value_t = DEFAULT_SAMPLE_RATE)]
            sample_rate: f64,

            #[arg(short, default_value_t = DEAFULT_BANDWIDTH)]
            bandwidth: f64,

            #[arg(long, default_value_t = 0)]
            device_idx: usize,

            #[arg(long, default_value = DEFAULT_TX_ANTENNA)]
            antenna: String,

            #[arg(long, default_value_t = DEFAULT_TX_CHANNEL)]
            channel: usize,

            #[arg(long, default_value_t = DEFAULT_TX_GAIN)]
            gain: f64,

            #[arg(long, value_enum, default_value_t = Driver::Lime)]
            driver: Driver,

            #[arg(long, default_value_t = false)]
            skip_cal: bool,
        },
        Rx {
            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_does_not_exist)]
            outfile: std::path::PathBuf,

            #[arg(value_parser = parse_dimensions_2d)]
            asset_resolution: (usize, usize),

            frame_rate: f64,

            #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
            gop_len: usize,

            // defaults set for 1080p
            #[arg(long="y", value_parser = parse_dimensions_3d, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
            y_chunk_dimensions: (usize, usize, usize),

            #[arg(long="cbcr", value_parser = parse_dimensions_3d, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
            c_chunk_dimensions: (usize, usize, usize),

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_Y_WHITEN_LEN)]
            y_whiten_len: usize,

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_CBCR_WHITEN_LEN)]
            cbcr_whiten_len: usize,

            #[arg(long, value_parser = parse_int::parse::<usize>, default_value_t = FRAME_LEN)]
            frame_len: usize,

            #[arg(short, default_value_t = DEFAULT_FREQ)]
            frequency: f64,

            #[arg(short = 'n', default_value_t = DEFAULT_SAMPLE_RATE)]
            sample_rate: f64,

            #[arg(short, default_value_t = DEAFULT_BANDWIDTH)]
            bandwidth: f64,

            #[arg(long, default_value_t = 0)]
            device_idx: usize,

            #[arg(long, default_value = DEFAULT_RX_ANTENNA)]
            antenna: String,

            #[arg(long, default_value_t = DEFAULT_RX_CHANNEL)]
            channel: usize,

            #[arg(long, default_value_t = DEFAULT_RX_GAIN)]
            gain: f64,

            #[arg(long, value_enum, default_value_t = Driver::Lime)]
            driver: Driver,
        },
        Loopback {
            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_exists)]
            infile: std::path::PathBuf,

            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_does_not_exist)]
            outfile: std::path::PathBuf,

            #[arg(short, default_value_t = DEFAULT_COMPRESSION_RATIO)]
            compression_ratio: f64,

            #[arg(long, default_value_t = DEFAULT_NOISE)]
            noise: f32,

            #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
            gop_len: usize,

            // defaults set for 1080p
            #[arg(long="y", value_parser = parse_dimensions_3d, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
            y_chunk_dimensions: (usize, usize, usize),

            #[arg(long="cbcr", value_parser = parse_dimensions_3d, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
            c_chunk_dimensions: (usize, usize, usize),

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_Y_WHITEN_LEN)]
            y_whiten_len: usize,

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_CBCR_WHITEN_LEN)]
            cbcr_whiten_len: usize,

            #[arg(long, value_parser = parse_int::parse::<usize>, default_value_t = FRAME_LEN)]
            frame_len: usize,

            #[arg(short, default_value_t = DEFAULT_FREQ)]
            frequency: f64,

            #[arg(short = 'n', default_value_t = DEFAULT_SAMPLE_RATE)]
            sample_rate: f64,

            #[arg(short, default_value_t = DEAFULT_BANDWIDTH)]
            bandwidth: f64,

            #[arg(long, default_value_t = 0)]
            device_idx: usize,

            #[arg(long, default_value = DEFAULT_TX_ANTENNA)]
            tx_antenna: String,

            #[arg(long, default_value = DEFAULT_RX_ANTENNA)]
            rx_antenna: String,

            #[arg(long, default_value_t = DEFAULT_TX_CHANNEL)]
            tx_channel: usize,

            #[arg(long, default_value_t = DEFAULT_RX_CHANNEL)]
            rx_channel: usize,

            #[arg(long, default_value_t = DEFAULT_TX_GAIN)]
            tx_gain: f64,

            #[arg(long, default_value_t = DEFAULT_RX_GAIN)]
            rx_gain: f64,

            #[arg(long, value_enum, default_value_t = Driver::Lime)]
            driver: Driver,

            #[arg(long, default_value_t = false)]
            skip_cal: bool,
        },
        Simulate {
            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_exists)]
            infile: std::path::PathBuf,

            #[arg(value_hint = clap::ValueHint::FilePath)]
            #[arg(value_parser = validate_file_does_not_exist)]
            outfile: std::path::PathBuf,

            #[arg(short, default_value_t = DEFAULT_COMPRESSION_RATIO)]
            compression_ratio: f64,

            #[arg(long, default_value_t = DEFAULT_NOISE)]
            noise: f32,

            #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
            gop_len: usize,

            // defaults set for 1080p
            #[arg(long="y", value_parser = parse_dimensions_3d, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
            y_chunk_dimensions: (usize, usize, usize),

            #[arg(long="cbcr", value_parser = parse_dimensions_3d, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
            c_chunk_dimensions: (usize, usize, usize),

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_Y_WHITEN_LEN)]
            y_whiten_len: usize,

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_CBCR_WHITEN_LEN)]
            cbcr_whiten_len: usize,

            #[arg(long, value_parser = parse_int::parse::<usize>, default_value_t = FRAME_LEN)]
            frame_len: usize,
        },
    }

    fn parse_dimensions_2d(s: &str) -> Result<(usize, usize), String> {
        let parts: Box<[&str]> = s.split('x').collect();
        if parts.len() != 2 {
            return Err("Expected WxHxD format".into());
        }

        let x = parts[0].parse().map_err(|_| "Invalid width")?;
        let y = parts[1].parse().map_err(|_| "Invalid height")?;

        Ok((x, y))
    }

    fn validate_file_exists(path: &str) -> Result<std::path::PathBuf, String> {
        let path: std::path::PathBuf = path.into();
        if !path.try_exists().map_err(|e| e.to_string())? {
            return Err(format!("File does not exist: {}", path.display()));
        }
        if !path.is_file() {
            return Err(format!("Not a regular file: {}", path.display()));
        }
        Ok(path)
    }

    fn validate_file_does_not_exist(path: &str) -> Result<std::path::PathBuf, String> {
        let path: std::path::PathBuf = path.into();
        if path.try_exists().map_err(|e| e.to_string())? {
            return Err(format!("File already exists: {}", path.display()));
        }
        Ok(path)
    }

    fn loopback(
        infile: std::path::PathBuf,
        outfile: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        noise: f32,
        y_chunk_dimensions: (usize, usize, usize),
        c_chunk_dimensions: (usize, usize, usize),
        y_whiten_len: usize,
        cbcr_whiten_len: usize,
        frame_len: usize,
        frequency: f64,
        sample_rate: f64,
        bandwidth: f64,
        device_idx: usize,
        tx_antenna: &str,
        rx_antenna: &str,
        tx_channel: usize,
        rx_channel: usize,
        tx_gain: f64,
        rx_gain: f64,
        driver: Driver,
        skip_cal: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config {
            frame_length: frame_len,
            y: PerPixelTypeConfig {
                whiten_length: y_whiten_len,
            },
            cbcr: PerPixelTypeConfig {
                whiten_length: cbcr_whiten_len,
            },
        };
        Config::set(config);

        let mut tx_params = RadioParams::default();
        tx_params.frequency = frequency;
        tx_params.sample_rate = sample_rate;
        tx_params.bandwidth = bandwidth;
        tx_params.device_idx = device_idx;
        tx_params.channel = tx_channel;
        tx_params.antenna = tx_antenna.to_string();
        tx_params.gain = tx_gain;

        let mut rx_params = tx_params.clone();
        rx_params.antenna = rx_antenna.to_string();
        rx_params.channel = rx_channel;
        rx_params.gain = rx_gain;

        let (mut tx_radio, mut rx_radio): (Box<dyn TransmitDevice>, Box<dyn ReceiveDevice>) =
            match driver {
                Driver::Lime => {
                    let tx_radio = LimeTransmitDevice::try_new(tx_params, skip_cal, false)?;
                    let rx_radio =
                        LimeReceiveDevice::try_new(rx_params, tx_radio.device, skip_cal, false)?;
                    (Box::new(tx_radio), Box::new(rx_radio))
                }
                Driver::Soapy => {
                    let tx_radio = SoapyTransmitDevice::try_new(tx_params, false)?;
                    let rx_radio = SoapyReceiveDevice::try_new(rx_params, &tx_radio.sdr, false)?;
                    (Box::new(tx_radio), Box::new(rx_radio))
                }
                _ => return Err("Driver does not support transmit.".into()),
            };

        let abort_token = AbortToken::new();
        let abort_token_clone = abort_token.clone();

        let mut encoder = FileReaderEncoder::with_file(
            infile,
            gop_len,
            compression_ratio,
            noise,
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )?;
        let asset_resolution = encoder.asset_resolution();
        let frame_rate = encoder.frame_rate();
        let mut decoder = FileWriterDecoder::try_new(
            outfile,
            asset_resolution,
            frame_rate,
            gop_len,
            encoder.y_chunk_dimensions,
            encoder.cb_chunk_dimensions,
            encoder.cr_chunk_dimensions,
        )?;

        rx_radio.activate()?;
        tx_radio.activate()?;

        let iq_reader = rx_radio.take_mpsc_reader();
        let rx_radio_join = run_async(rx_radio);
        let decoder_join = std::thread::spawn(move || {
            let result = decoder
                .run(iq_reader, abort_token_clone)
                .map_err(|e| e.to_string());
            eprintln!("decoder result: {:?}", result);
            result
        });

        encoder.run(tx_radio.as_mut(), abort_token)?;

        //     play_dump_file(tx_radio.stream, &std::path::PathBuf::from("/tmp/dumpw_063"));

        let _ = rx_radio_join.join().map_err(|_| "decoder thread panic'd")?; // TODO: preserve inner error
        let _ = decoder_join.join().map_err(|_| "decoder thread panic'd")?; // TODO: preserve inner error
        tx_radio.drain();

        Ok(())
    }

    fn simulate(
        infile: std::path::PathBuf,
        outfile: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        noise: f32,
        y_chunk_dimensions: (usize, usize, usize),
        c_chunk_dimensions: (usize, usize, usize),
        y_whiten_len: usize,
        cbcr_whiten_len: usize,
        frame_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config {
            frame_length: frame_len,
            y: PerPixelTypeConfig {
                whiten_length: y_whiten_len,
            },
            cbcr: PerPixelTypeConfig {
                whiten_length: cbcr_whiten_len,
            },
        };
        Config::set(config);

        let encoder = FileReaderEncoder::with_file(
            infile,
            gop_len,
            compression_ratio,
            noise,
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )?;
        let asset_resolution = encoder.asset_resolution();
        let frame_rate = encoder.frame_rate();
        let decoder = FileWriterDecoder::try_new(
            outfile,
            asset_resolution,
            frame_rate,
            gop_len,
            encoder.y_chunk_dimensions,
            encoder.cb_chunk_dimensions,
            encoder.cr_chunk_dimensions,
        )?;
        run_simulation(encoder, decoder)?;

        Ok(())
    }

    fn transmit(
        infile: std::path::PathBuf,
        gop_len: usize,
        compression_ratio: f64,
        y_chunk_dimensions: (usize, usize, usize),
        c_chunk_dimensions: (usize, usize, usize),
        y_whiten_len: usize,
        cbcr_whiten_len: usize,
        frame_len: usize,
        frequency: f64,
        sample_rate: f64,
        bandwidth: f64,
        device_idx: usize,
        antenna: &str,
        channel: usize,
        gain: f64,
        driver: Driver,
        skip_cal: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config {
            frame_length: frame_len,
            y: PerPixelTypeConfig {
                whiten_length: y_whiten_len,
            },
            cbcr: PerPixelTypeConfig {
                whiten_length: cbcr_whiten_len,
            },
        };
        Config::set(config);

        let mut tx_params = RadioParams::default();
        tx_params.frequency = frequency;
        tx_params.sample_rate = sample_rate;
        tx_params.bandwidth = bandwidth;
        tx_params.device_idx = device_idx;
        tx_params.channel = channel;
        tx_params.antenna = antenna.to_string();
        tx_params.gain = gain;

        let mut tx_radio: Box<dyn TransmitDevice> = match driver {
            Driver::Lime => {
                let tx_radio = LimeTransmitDevice::try_new(tx_params, skip_cal, false)?;
                Box::new(tx_radio)
            }
            Driver::Soapy => {
                let tx_radio = SoapyTransmitDevice::try_new(tx_params, false)?;
                Box::new(tx_radio)
            }
            _ => return Err("Driver does not support transmit.".into()),
        };
        let mut encoder = FileReaderEncoder::with_file(
            infile,
            gop_len,
            compression_ratio,
            0.0,
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )?;

        tx_radio.activate()?;

        encoder.run(tx_radio.as_mut(), AbortToken::new())?;

        Ok(())
    }

    fn receive(
        outfile: std::path::PathBuf,
        asset_resolution: (usize, usize),
        frame_rate: f64,
        gop_len: usize,
        y_chunk_dimensions: (usize, usize, usize),
        c_chunk_dimensions: (usize, usize, usize),
        y_whiten_len: usize,
        cbcr_whiten_len: usize,
        frame_len: usize,
        frequency: f64,
        sample_rate: f64,
        bandwidth: f64,
        device_idx: usize,
        antenna: &str,
        channel: usize,
        gain: f64,
        driver: Driver,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config {
            frame_length: frame_len,
            y: PerPixelTypeConfig {
                whiten_length: y_whiten_len,
            },
            cbcr: PerPixelTypeConfig {
                whiten_length: cbcr_whiten_len,
            },
        };
        Config::set(config);

        let mut rx_params = RadioParams::default();
        rx_params.frequency = frequency;
        rx_params.sample_rate = sample_rate;
        rx_params.bandwidth = bandwidth;
        rx_params.channel = channel;
        rx_params.antenna = antenna.to_string();
        rx_params.gain = gain;
        rx_params.device_idx = device_idx;

        let mut rx_radio: Box<dyn ReceiveDevice> = match driver {
            Driver::Lime => {
                let rx_radio =
                    LimeReceiveDevice::try_new(rx_params, new_lime_device()?, false, false)?;
                Box::new(rx_radio)
            }
            Driver::Soapy => {
                let device = &new_soapy_device(&rx_params)?;
                let rx_radio = SoapyReceiveDevice::try_new(rx_params, device, true)?;
                Box::new(rx_radio)
            }
            Driver::RtlSdr => {
                let device = RtlSdrReceiveDevice::try_new(rx_params, false)?;
                Box::new(device)
            }
        };

        // invert dimensions to match encode
        let y_chunk_dimensions = (
            y_chunk_dimensions.2,
            y_chunk_dimensions.1,
            y_chunk_dimensions.0,
        );
        let c_chunk_dimensions = (
            c_chunk_dimensions.2,
            c_chunk_dimensions.1,
            c_chunk_dimensions.0,
        );

        let mut decoder = FileWriterDecoder::try_new(
            outfile,
            asset_resolution,
            frame_rate,
            gop_len,
            y_chunk_dimensions,
            c_chunk_dimensions,
            c_chunk_dimensions,
        )?;

        rx_radio.activate()?;

        let iq_reader = rx_radio.take_mpsc_reader();
        let decoder_join = std::thread::spawn(move || {
            let result = decoder
                .run(iq_reader, AbortToken::new())
                .map_err(|e| e.to_string());
            eprintln!("decoder result: {:?}", result);
            result
        });

        rx_radio.run()?;

        let _ = decoder_join.join().map_err(|_| "decoder thread panic'd")?; // TODO: preserve inner error

        Ok(())
    }

    pub fn main() -> Result<(), String> {
        let args = Args::parse();

        match args.command {
            Commands::Tx {
                infile,
                compression_ratio,
                gop_len,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                antenna,
                channel,
                gain,
                driver,
                skip_cal,
            } => transmit(
                infile,
                gop_len,
                compression_ratio,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                &antenna,
                channel,
                gain,
                driver,
                skip_cal,
            ),
            Commands::Rx {
                outfile,
                asset_resolution,
                frame_rate,
                gop_len,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                antenna,
                channel,
                gain,
                driver,
            } => receive(
                outfile,
                asset_resolution,
                frame_rate,
                gop_len,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                &antenna,
                channel,
                gain,
                driver,
            ),
            Commands::Loopback {
                infile,
                outfile,
                compression_ratio,
                noise,
                gop_len,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                tx_antenna,
                rx_antenna,
                tx_channel,
                rx_channel,
                tx_gain,
                rx_gain,
                driver,
                skip_cal,
            } => loopback(
                infile,
                outfile,
                gop_len,
                compression_ratio,
                noise,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                &tx_antenna,
                &rx_antenna,
                tx_channel,
                rx_channel,
                tx_gain,
                rx_gain,
                driver,
                skip_cal,
            ),
            Commands::Simulate {
                infile,
                outfile,
                gop_len,
                compression_ratio,
                noise,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
            } => simulate(
                infile,
                outfile,
                gop_len,
                compression_ratio,
                noise,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
            ),
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(not(target_vendor = "apple"))]
mod linux {
    use super::*;

    #[derive(Parser)]
    struct Args {
        #[command(subcommand)]
        command: Commands,
    }
    #[derive(Subcommand)]
    enum Commands {
        TxCamera {
            #[arg(short, default_value_t = DEFAULT_COMPRESSION_RATIO)]
            compression_ratio: f64,

            #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
            gop_len: usize,

            // defaults set for 1080p
            #[arg(long="y", value_parser = parse_dimensions_3d, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
            y_chunk_dimensions: (usize, usize, usize),

            #[arg(long="cbcr", value_parser = parse_dimensions_3d, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
            c_chunk_dimensions: (usize, usize, usize),

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_Y_WHITEN_LEN)]
            y_whiten_len: usize,

            #[arg(long, value_parser = parse_power_of_two, default_value_t = DEFAULT_CBCR_WHITEN_LEN)]
            cbcr_whiten_len: usize,

            #[arg(long, value_parser = parse_int::parse::<usize>, default_value_t = FRAME_LEN)]
            frame_len: usize,

            #[arg(short, default_value_t = DEFAULT_FREQ)]
            frequency: f64,

            #[arg(short = 'n', default_value_t = DEFAULT_SAMPLE_RATE)]
            sample_rate: f64,

            #[arg(short, default_value_t = DEAFULT_BANDWIDTH)]
            bandwidth: f64,

            #[arg(long, default_value_t = 0)]
            device_idx: usize,

            #[arg(long, default_value = DEFAULT_TX_ANTENNA)]
            antenna: String,

            #[arg(long, default_value_t = DEFAULT_TX_CHANNEL)]
            channel: usize,

            #[arg(long, default_value_t = DEFAULT_TX_GAIN)]
            gain: f64,

            #[arg(long, value_enum, default_value_t = Driver::Lime)]
            driver: Driver,

            #[arg(long, default_value_t = false)]
            skip_cal: bool,
        },
    }

    fn transmit(
        gop_len: usize,
        compression_ratio: f64,
        y_chunk_dimensions: (usize, usize, usize),
        c_chunk_dimensions: (usize, usize, usize),
        y_whiten_len: usize,
        cbcr_whiten_len: usize,
        frame_len: usize,
        frequency: f64,
        sample_rate: f64,
        bandwidth: f64,
        device_idx: usize,
        antenna: &str,
        channel: usize,
        gain: f64,
        driver: Driver,
        skip_cal: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }

    pub fn main() -> Result<(), String> {
        let args = Args::parse();

        println!("Hello, Linux.");

        match args.command {
            Commands::TxCamera {
                compression_ratio,
                gop_len,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                antenna,
                channel,
                gain,
                driver,
                skip_cal,
            } => transmit(
                gop_len,
                compression_ratio,
                y_chunk_dimensions,
                c_chunk_dimensions,
                y_whiten_len,
                cbcr_whiten_len,
                frame_len,
                frequency,
                sample_rate,
                bandwidth,
                device_idx,
                &antenna,
                channel,
                gain,
                driver,
                skip_cal,
            ),
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn main() -> Result<(), String> {
    #[cfg(target_vendor = "apple")]
    {
        apple::main()
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        linux::main()
    }
}
