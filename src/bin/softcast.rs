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
use softcast_rs::simulator::*;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Simulate {
        #[arg(value_hint = clap::ValueHint::FilePath)]
        #[arg(value_parser = validate_file_exists)]
        infile: std::path::PathBuf,

        #[arg(value_hint = clap::ValueHint::FilePath)]
        #[arg(value_parser = validate_file_does_not_exist)]
        outfile: std::path::PathBuf,

        #[arg(short, default_value_t = 0.1875)]
        compression_ratio: f64,

        #[arg(short, default_value_t = 0.0)]
        noise_power: f32,

        #[arg(short, default_value_t = 22)]
        gop_len: usize,

        // defaults set for 1080p
        #[arg(long="y", value_parser = parse_dimensions, default_value = "48x40x1")]
        y_chunk_dimensions: (usize, usize, usize),

        #[arg(long="cbcr", value_parser = parse_dimensions, default_value = "40x30x1")]
        c_chunk_dimensions: (usize, usize, usize),
    },
}

fn parse_dimensions(s: &str) -> Result<(usize, usize, usize), String> {
    let parts: Box<[&str]> = s.split('x').collect();
    if parts.len() != 3 {
        return Err(format!("Expected WxHxD format"));
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
        Commands::Simulate {
            infile,
            outfile,
            gop_len,
            compression_ratio,
            noise_power,
            y_chunk_dimensions,
            c_chunk_dimensions,
        } => {
            simulate(
                infile,
                outfile,
                gop_len,
                compression_ratio,
                noise_power,
                y_chunk_dimensions,
                c_chunk_dimensions,
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
