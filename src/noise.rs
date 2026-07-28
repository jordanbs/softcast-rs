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

use num_complex::Complex32;
use rand::SeedableRng;
use rand_xoshiro;

pub struct AdditiveWhiteGaussianNoise<I: Iterator<Item = Box<[Complex32]>>> {
    inner: I,
    rng: rand_xoshiro::Xoshiro128PlusPlus,
    noise_σ: f32,
    bypass: bool,
}
impl<I: Iterator<Item = Box<[Complex32]>>> AdditiveWhiteGaussianNoise<I> {
    pub fn new(inner: I, noise_power: f32, seed: u64) -> Self {
        Self {
            inner,
            rng: rand_xoshiro::Xoshiro128PlusPlus::seed_from_u64(seed), // deterministic
            noise_σ: (noise_power / 2.0).sqrt(),
            bypass: 0.0 == noise_power,
        }
    }
}
impl<I: Iterator<Item = Box<[Complex32]>>> Iterator for AdditiveWhiteGaussianNoise<I> {
    type Item = Box<[Complex32]>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut iqs = self.inner.next()?;
        if !self.bypass {
            for iq in iqs.iter_mut() {
                *iq = iq.add_random_noise(self.noise_σ, &mut self.rng);
            }
        }
        Some(iqs)
    }
}
pub trait AddRandomNoise {
    fn add_random_noise<R: rand::Rng>(&self, noise_σ: f32, rng: &mut R) -> Self;
}
impl AddRandomNoise for Complex32 {
    fn add_random_noise<R: rand::Rng>(&self, noise_σ: f32, rng: &mut R) -> Self {
        let u1: f32 = rng.random_range(f32::EPSILON..1.0);
        let u2: f32 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt() * noise_σ;
        let theta = 2.0 * std::f32::consts::PI * u2;
        self + Self::new(r * theta.cos(), r * theta.sin())
    }
}
