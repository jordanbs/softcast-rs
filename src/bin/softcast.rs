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
use softcast_rs::decoder::*;
use softcast_rs::encoder::*;
use softcast_rs::radio::*;
use softcast_rs::simulator::*;

const DEFAULT_COMPRESSION_RATIO: f64 = 0.1875;
const DEFAULT_GOP_LEN: usize = 22;
const DEFAULT_Y_CHUNK_DIMENSIONS: &str = "48x40x1";
const DEFAULT_C_CHUNK_DIMENSIONS: &str = "40x30x1";
const DEFAULT_NOISE_POWER: f32 = 0.0;
const DEFAULT_TX_GAIN: f64 = 0.08;
const DEFAULT_RX_GAIN: f64 = 0.7;
const DEFAULT_FREQ: f64 = 800_000_000.0;
const DEFAULT_SAMPLE_RATE: f64 = 0x400_000 as f64;
const DEAFULT_BANDWIDTH: f64 = 1.6 * DEFAULT_SAMPLE_RATE;
const DEFAULT_TX_ANTENNA: &str = "BAND1";
const DEFAULT_RX_ANTENNA: &str = "LNAL";
const DEFAULT_TX_CHANNEL: usize = 0;
const DEFAULT_RX_CHANNEL: usize = 1;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Loopback {
        #[arg(value_hint = clap::ValueHint::FilePath)]
        #[arg(value_parser = validate_file_exists)]
        infile: std::path::PathBuf,

        #[arg(value_hint = clap::ValueHint::FilePath)]
        #[arg(value_parser = validate_file_does_not_exist)]
        outfile: std::path::PathBuf,

        #[arg(short, default_value_t = DEFAULT_COMPRESSION_RATIO)]
        compression_ratio: f64,

        #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
        gop_len: usize,

        // defaults set for 1080p
        #[arg(long="y", value_parser = parse_dimensions, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
        y_chunk_dimensions: (usize, usize, usize),

        #[arg(long="cbcr", value_parser = parse_dimensions, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
        c_chunk_dimensions: (usize, usize, usize),

        #[arg(short, default_value_t = DEFAULT_FREQ)]
        frequency: f64,

        #[arg(short = 'n', default_value_t = DEFAULT_SAMPLE_RATE)]
        sample_rate: f64,

        #[arg(short, default_value_t = DEAFULT_BANDWIDTH)]
        bandwidth: f64,

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

        #[arg(short, default_value_t = DEFAULT_NOISE_POWER)]
        noise_power: f32,

        #[arg(short, default_value_t = DEFAULT_GOP_LEN)]
        gop_len: usize,

        // defaults set for 1080p
        #[arg(long="y", value_parser = parse_dimensions, default_value = DEFAULT_Y_CHUNK_DIMENSIONS)]
        y_chunk_dimensions: (usize, usize, usize),

        #[arg(long="cbcr", value_parser = parse_dimensions, default_value = DEFAULT_C_CHUNK_DIMENSIONS)]
        c_chunk_dimensions: (usize, usize, usize),
    },
}

fn parse_dimensions(s: &str) -> Result<(usize, usize, usize), String> {
    let parts: Box<[&str]> = s.split('x').collect();
    if parts.len() != 3 {
        return Err("Expected WxHxD format".into());
    }

    let x = parts[0].parse().map_err(|_| "Invalid width")?;
    let y = parts[1].parse().map_err(|_| "Invalid height")?;
    let z = parts[2].parse().map_err(|_| "Invalid length")?;

    Ok((x, y, z))
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
    y_chunk_dimensions: (usize, usize, usize),
    c_chunk_dimensions: (usize, usize, usize),
    frequency: f64,
    sample_rate: f64,
    bandwidth: f64,
    tx_antenna: &str,
    rx_antenna: &str,
    tx_channel: usize,
    rx_channel: usize,
    tx_gain: f64,
    rx_gain: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tx_params = RadioParams::default();
    tx_params.frequency = frequency;
    tx_params.sample_rate = sample_rate;
    tx_params.bandwidth = bandwidth;
    tx_params.channel = tx_channel;
    tx_params.antenna = tx_antenna.to_string();
    tx_params.gain = tx_gain;

    let mut rx_params = tx_params.clone();
    rx_params.antenna = rx_antenna.to_string();
    rx_params.channel = rx_channel;
    rx_params.gain = rx_gain;

    let mut tx_radio = LimeTransmitDevice::try_new(tx_params, false)?;
    let mut rx_radio = LimeReceiveDevice::try_new(rx_params, tx_radio.device, true)?;
    let mut encoder = FileReaderEncoder::try_new(
        infile,
        gop_len,
        compression_ratio,
        0.0,
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
        y_chunk_dimensions,
        c_chunk_dimensions,
        c_chunk_dimensions,
    )?;

    rx_radio.activate()?;
    tx_radio.activate()?;

    let iq_reader = rx_radio.take_mpsc_reader();
    let rx_radio_join = rx_radio.run_async();
    let decoder_join = std::thread::spawn(move || {
        let result = decoder.run(iq_reader).map_err(|e| e.to_string());
        eprintln!("decoder result: {:?}", result);
        result
    });

    encoder.run(tx_radio)?;

    //     play_dump_file(tx_radio.stream, &std::path::PathBuf::from("/tmp/dumpw_063"));

    let _ = rx_radio_join.join().map_err(|_| "decoder thread panic'd")?; // TODO: preserve inner error
    let _ = decoder_join.join().map_err(|_| "decoder thread panic'd")?; // TODO: preserve inner error

    Ok(())
}

fn simulate(
    infile: std::path::PathBuf,
    outfile: std::path::PathBuf,
    gop_len: usize,
    compression_ratio: f64,
    noise_power: f32,
    y_chunk_dimensions: (usize, usize, usize),
    c_chunk_dimensions: (usize, usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let encoder = FileReaderEncoder::try_new(
        infile,
        gop_len,
        compression_ratio,
        noise_power,
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
        y_chunk_dimensions,
        c_chunk_dimensions,
        c_chunk_dimensions,
    )?;
    run_simulation(encoder, decoder)?;

    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    match args.command {
        Commands::Loopback {
            infile,
            outfile,
            compression_ratio,
            gop_len,
            y_chunk_dimensions,
            c_chunk_dimensions,
            frequency,
            sample_rate,
            bandwidth,
            tx_antenna,
            rx_antenna,
            tx_channel,
            rx_channel,
            tx_gain,
            rx_gain,
        } => loopback(
            infile,
            outfile,
            gop_len,
            compression_ratio,
            y_chunk_dimensions,
            c_chunk_dimensions,
            frequency,
            sample_rate,
            bandwidth,
            &tx_antenna,
            &rx_antenna,
            tx_channel,
            rx_channel,
            tx_gain,
            rx_gain,
        ),
        Commands::Simulate {
            infile,
            outfile,
            gop_len,
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            c_chunk_dimensions,
        } => simulate(
            infile,
            outfile,
            gop_len,
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            c_chunk_dimensions,
        ),
    }
    .map_err(|e| e.to_string())?;
    Ok(())
}
