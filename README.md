# Deconvolution on Simulated JWST Data
### A physics-based forward modeling and restoration pipeline for diffraction-limited astronomical data.


## Overview
This project implements a computational optics engine to mathematically reverse the inherent convolution that happens during the capturing process of an image. Instead of simple image 
sharpening, we built a Forward Model to simulate the quantum physics of photon arrival and the wave optics of hexagonal apertures. We then validated various deconvolution algorithms against 
this ground truth, proving that Bayesian methods (Richardson-Lucy) outperform linear filters (Wiener) in high-contrast astronomical regimes.


