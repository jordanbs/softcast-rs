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

fn main() {
    let limesuite = pkg_config::Config::new()
        .probe("LimeSuite")
        .expect("pkg-config could not find LimeSuite.");
    let limesuite_include = limesuite
        .include_paths
        .first()
        .expect("pkg-config LimeSuite returned no include paths");

    let limesuite_h = limesuite_include.join("lime").join("LimeSuite.h");

    let bindings = bindgen::Builder::default()
        .header(limesuite_h.display().to_string())
        .generate()
        .expect("Unable to generate LimeSuite bindings.");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
