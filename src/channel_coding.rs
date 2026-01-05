// Copyright 2025-2026 Jordan Schneider
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

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    #[test]
    fn ofdm_flexframe() {
        use liquid_dsp_sys;

        unsafe {
            let fg = liquid_dsp_sys::ofdmflexframegen_create(
                64,
                16,
                4,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert!(!fg.is_null());

            let header = [0u8; 8];
            let payload = [0u8; 20];
            let mut buf = [Complex32::ZERO; 120];

            let status = liquid_dsp_sys::ofdmflexframegen_assemble(
                fg,
                header.as_ptr(),
                payload.as_ptr(),
                payload.len().try_into().unwrap(),
            );
            assert_eq!(status, 0);

            let status = liquid_dsp_sys::ofdmflexframegen_print(fg);
            assert_eq!(status, 0);

            let mut frame_complete = 0;
            while 0 == frame_complete {
                frame_complete = liquid_dsp_sys::ofdmflexframegen_write(
                    fg,
                    buf.as_mut_ptr(),
                    buf.len().try_into().unwrap(),
                );

                eprintln!(
                    "ofdmflexframegen wrote to buffer{}",
                    match frame_complete {
                        0 => "",
                        _ => " (frame complete)",
                    },
                );
            }
        }
    }
}
