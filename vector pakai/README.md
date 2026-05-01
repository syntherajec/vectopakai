# VectorFlow — Adobe Stock Processor

100% browser-based · No API key · Unlimited · Free forever

## Features
- Upload massal JPG / JPEG / JFIF / PNG
- Proses otomatis satu per satu (antrian)
- Tab-safe: ganti tab, proses tetap jalan (Web Worker)
- Upscale 4x (Real-ESRGAN ONNX / Lanczos fallback)
- Remove Background (RMBG-1.4 ONNX / flood-fill fallback)
- Edge Cleaning (alpha matting + erode)
- Vectorize → SVG (multi-color tracer)
- SVG → EPS (metadata lengkap: RGB, artboard 0,0, title)
- Download ZIP berisi semua file .EPS
- Preview before/after per file

## Deploy ke GitHub Pages
1. Fork / upload repo ini ke GitHub
2. Settings → Pages → Source: main branch / root
3. Akses via https://username.github.io/repo-name

## Tech Stack (semua gratis)
| Komponen | Library |
|----------|---------|
| Remove BG | RMBG-1.4 via ONNX Runtime Web |
| Upscale | Real-ESRGAN via ONNX Runtime Web |
| Vectorize | Custom multi-color tracer |
| EPS builder | Pure JS |
| ZIP | JSZip 3.10 |
| Background thread | Web Worker |

## Catatan Adobe Stock
- Centang "Created using generative AI tools" saat upload
- Output EPS sudah include metadata: RGB, BoundingBox, Title
- Ukuran artboard minimum 1000x1000px (sudah otomatis)
