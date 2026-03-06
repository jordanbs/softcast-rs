# softcast-rs

softcast-rs is an implementation of [Szymon Jakubczak\'s 2011 thesis](https://dspace.mit.edu/handle/1721.1/66006 "SoftCast : exposing a waveform interface to the wireless channel for scalable video broadcast") of a hybrid analog digital video transmission mechanism, SoftCast. SoftCast achieves a linear relationship between video quality and signal-to-noise-ratio by employing joint source channel coding of video and the wireless channel. SoftCast is more robust in challenging radio environments than digital video transmitted via traditional means, and requires less bandwidth at comparable quality than analog video systems (NTSC, PAL) by employing techniques from video codecs. 

This project is in its infancy. In its current state, video broadcast and recovery over a software defined radio has been achieved. This project aims to continue to refine and optimize the software and protocol for real time transmission and reception, suitable for real world deployments.

Given the early state of the project, the employed protocol should be considered unstable and subject to change. The current implementation omits self describing information to minimize metadata.

softcast-rs currently only runs on macOS.

## Requirements
- AVFoundation
- fftw
- limesuite

## Usage
```
softcast loopback path/to/infile.mp4 path/to/outfile.mp4
```

Without a software defined radio, a digital simulation can be performed.
```
softcast simulate --noise 0.01 path/to/infile/mp4 path/to/outfile.mp4
```

## Contact
If you would like to collaborate on, deploy, or commercially license softcast-rs, please email me at rockers.grate.4y at icloud.com.

## License
This project is licensed under the [GNU GPLv3](LICENSE).