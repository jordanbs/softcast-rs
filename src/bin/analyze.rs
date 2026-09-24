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

use softcast_rs::sync::Complex32IterReadWrapper;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_hint = clap::ValueHint::FilePath)]
    infile: std::path::PathBuf,
}

fn papr(dump_file: std::fs::File) -> Result<f64, std::io::Error> {
    let buf_reader = std::io::BufReader::new(dump_file);
    let c32_reader = Complex32IterReadWrapper::from(buf_reader);

    let mut peak_power = 0f64;
    let mut average_power = 0f64;
    for (idx, iq) in c32_reader.flatten().enumerate() {
        let iq_power = iq.norm_sqr() as f64;
        average_power += (iq_power - average_power) / (1 + idx) as f64;
        peak_power = peak_power.max(iq_power);
    }

    Ok(peak_power / average_power)
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let dump_file = std::fs::File::open(args.infile).map_err(|_e| stringify!(e))?;
    let papr = papr(dump_file).map_err(|_e| stringify!(e))?;
    println!("PAPR: {papr}");
    Ok(())
}
