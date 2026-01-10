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

use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Rebuild triggers (avoid rebuilds)
    println!("cargo:rerun-if-changed=build.rs");
    // libfec always triggers rebuilds.
    // Please cargo clean if a rebuild is truly needed.
    //     println!(
    //         "cargo:rerun-if-changed={}",
    //         manifest_dir.join("../../vendor/libfec").display()
    //     );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("../../vendor/liquid-dsp").display()
    );

    // 1) Build + stage libfec
    let libfec_src = manifest_dir.join("../../vendor/libfec");
    if !libfec_src.exists() {
        panic!("missing submodule at {}", libfec_src.display());
    }

    run(
        Command::new("./configure").current_dir(&libfec_src),
        "building libfec",
    );
    run(
        Command::new("make").current_dir(&libfec_src),
        "building libfec",
    );

    let libfec_a = find_file(&libfec_src, "libfec.a").unwrap_or_else(|| {
        panic!(
            "built libfec, but could not find libfec.a under {}",
            libfec_src.display()
        )
    });

    let fec_h = find_file(&libfec_src, "fec.h").unwrap_or_else(|| {
        panic!(
            "built libfec, but could not find fec.h under {}",
            libfec_src.display()
        )
    });

    let stage = out_dir.join("stage");
    let stage_lib = stage.join("lib");
    let stage_inc = stage.join("include");
    fs::create_dir_all(&stage_lib).unwrap();
    fs::create_dir_all(&stage_inc).unwrap();

    let staged_libfec_a = stage_lib.join("libfec.a");
    fs::copy(&libfec_a, &staged_libfec_a).unwrap_or_else(|e| {
        panic!(
            "copy {} -> {}: {e}",
            libfec_a.display(),
            staged_libfec_a.display()
        )
    });

    let staged_fec_h = stage_inc.join("fec.h");
    fs::copy(&fec_h, &staged_fec_h).unwrap_or_else(|e| {
        panic!(
            "copy {} -> {}: {e}",
            fec_h.display(),
            staged_fec_h.display()
        )
    });

    // Probe system FFTW (STATIC) via pkg-config, and use it for CMake include paths
    let fftw = pkg_config::Config::new()
        .statik(true)
        .cargo_metadata(true) // still emit link flags for the Rust link step
        .probe("fftw3f")
        .expect("pkg-config could not find static fftw3f");

    // Help CMake compile liquid (header search + library search)
    let fftw_inc = fftw
        .include_paths
        .get(0)
        .expect("pkg-config fftw3f returned no include_paths");

    // 3) Build + install liquid-dsp
    let liquid_src = manifest_dir.join("../../vendor/liquid-dsp");
    if !liquid_src.exists() {
        panic!("missing submodule at {}", liquid_src.display());
    }

    // Build liquid-dsp as a static library in its own CMake build/install under OUT_DIR.
    let liquid_install = cmake::Config::new(&liquid_src)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_AUTOTESTS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("BUILD_SANDBOX", "OFF")
        .define("BUILD_DOC", "OFF")
        // Make libfec visible to liquid's CMake checks
        .define("CMAKE_PREFIX_PATH", stage.to_string_lossy().to_string())
        .define("ENABLE_LIBFEC", "1")
        .define(
            "CMAKE_INCLUDE_PATH",
            format!("{};{}", stage_inc.display(), fftw_inc.display()),
        )
        .define(
            "CMAKE_LIBRARY_PATH",
            stage_lib.to_string_lossy().to_string(),
        )
        .define("FEC_LIBRARY", staged_libfec_a.to_string_lossy().to_string())
        .cflag(format!(
            "-I{} -I{}",
            stage_inc.display(),
            fftw_inc.display()
        ))
        .build();

    // cmake crate installs into `liquid_install` (returned path).
    // Usually libs end up in <install>/lib
    let liquid_libdir = liquid_install.join("lib");
    let liquid_incdir = liquid_install.join("include");

    if !liquid_libdir.exists() {
        panic!(
            "liquid install libdir not found: {} (install prefix {})",
            liquid_libdir.display(),
            liquid_install.display()
        );
    }

    // 4) Tell Rust how to link
    println!("cargo:rustc-link-search=native={}", stage_lib.display());
    println!("cargo:rustc-link-search=native={}", liquid_libdir.display());

    // Static link to libliquid + libfec
    println!("cargo:rustc-link-lib=static=liquid");
    println!("cargo:rustc-link-lib=static=fec");

    // liquid uses libm; harmless on macOS, required on many Unixes
    println!("cargo:rustc-link-lib=m");

    // Export includes for bindgen/users (optional)
    println!("cargo:include={}", liquid_incdir.display());
    println!("cargo:include={}", stage_inc.display());

    // 5) Generate Rust bindings for liquid.h
    let header = {
        // liquid installs either include/liquid/liquid.h or include/liquid.h depending on layout
        let a = liquid_incdir.join("liquid").join("liquid.h");
        let b = liquid_incdir.join("liquid.h");
        if a.exists() {
            a
        } else if b.exists() {
            b
        } else {
            // last resort: search
            find_file(&liquid_incdir, "liquid.h").unwrap_or_else(|| {
                panic!("could not find liquid.h under {}", liquid_incdir.display())
            })
        }
    };

    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        // Make sure bindgen/clang can find headers it needs
        .clang_arg(format!("-I{}", liquid_incdir.display()))
        .clang_arg(format!("-I{}", stage_inc.display())) // fec.h, if referenced
        .clang_arg(format!("-I{}", fftw_inc.display())) // fftw3.h
        .blocklist_type("liquid_float_complex")
        .blocklist_type("liquid_double_complex")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed generating bindings for liquid.h");

    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("could not write bindings.rs");
}

fn run(cmd: &mut Command, what: &str) {
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("{what}: failed to spawn: {e}"));
    if !output.status.success() {
        panic!(
            "{what} failed.\ncmd: {:?}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            cmd,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn find_file(root: &Path, filename: &str) -> Option<PathBuf> {
    // Simple recursive walk without extra deps
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = fs::read_dir(&dir).ok()?;
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.file_name() == Some(OsStr::new(filename)) {
                return Some(p);
            }
        }
    }
    None
}
