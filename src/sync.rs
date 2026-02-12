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

use crate::decoder::OFDMSymbolReader;
use crate::encoder::OFDMSymbolWriter;
use crate::framing::OFDMSymbol;

pub struct MPSCWriter {
    pub sender: std::sync::mpsc::SyncSender<OFDMSymbol>,
}
impl OFDMSymbolWriter for MPSCWriter {
    fn write(&mut self, symbol: OFDMSymbol) -> Result<(), Box<dyn std::error::Error>> {
        self.sender.send(symbol).map_err(|e| e.into())
    }
}
impl MPSCWriter {
    pub fn new_channel() -> (Self, MPSCReader) {
        let (sender, receiver) = std::sync::mpsc::sync_channel(0x80); // 64 KiB
        let writer = Self { sender };
        let reader = MPSCReader { receiver };
        (writer, reader)
    }
}

pub struct MPSCReader {
    pub receiver: std::sync::mpsc::Receiver<OFDMSymbol>,
}

impl OFDMSymbolReader for MPSCReader {
    fn into_iter(self) -> impl Iterator<Item = OFDMSymbol> {
        self.receiver.into_iter()
    }
}
