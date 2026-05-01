// ============================================================
// Web Worker — runs pipeline in background
// Tab can be switched, process continues
// ============================================================

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

var sessionRmbg = null;
var sessionEsrgan = null;
var modelsLoaded = { rmbg: false, esrgan: false };

// Receive messages from main thread
self.onmessage = async function(e) {
  var msg = e.data;
  switch(msg.type) {
    case 'LOAD_MODELS':
      await loadModels();
      break;
    case 'PROCESS_FILE':
      await processFile(msg.file, msg.idx, msg.options);
      break;
  }
};

function send(type, data) {
  self.postMessage(Object.assign({ type: type }, data));
}

function log(msg, level) {
  send('LOG', { msg: msg, level: level || 'info' });
}

// ============================================================
// LOAD MODELS
// ============================================================
async function loadModels() {
  send('MODEL_STATUS', { model: 'rmbg', state: 'loading', text: 'downloading...' });
  send('MODEL_STATUS', { model: 'esrgan', state: 'loading', text: 'downloading...' });

  // Configure ONNX paths
  if (typeof ort !== 'undefined') {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    ort.env.wasm.numThreads = 2; // conservative for i5-6300U
  }

  // Load RMBG-1.4
  try {
    log('Loading RMBG-1.4 from HuggingFace (~40MB, cached after first run)...', 'info');
    sessionRmbg = await ort.InferenceSession.create(
      'https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    );
    modelsLoaded.rmbg = true;
    send('MODEL_STATUS', { model: 'rmbg', state: 'ready', text: 'ready' });
    log('RMBG-1.4 loaded OK', 'success');
  } catch(e) {
    send('MODEL_STATUS', { model: 'rmbg', state: 'error', text: 'fallback' });
    log('RMBG-1.4 failed (' + e.message + '), using flood-fill fallback', 'warn');
  }

  // Load Real-ESRGAN lightweight
  try {
    log('Loading Real-ESRGAN (~5MB)...', 'info');
    sessionEsrgan = await ort.InferenceSession.create(
      'https://huggingface.co/rocca/realesrgan-onnx/resolve/main/realesrgan-x4plus-anime.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    );
    modelsLoaded.esrgan = true;
    send('MODEL_STATUS', { model: 'esrgan', state: 'ready', text: 'ready' });
    log('Real-ESRGAN loaded OK', 'success');
  } catch(e) {
    send('MODEL_STATUS', { model: 'esrgan', state: 'error', text: 'lanczos fallback' });
    log('Real-ESRGAN failed (' + e.message + '), using Lanczos fallback', 'warn');
  }

  send('MODELS_READY', { rmbg: modelsLoaded.rmbg, esrgan: modelsLoaded.esrgan });
}

// ============================================================
// PROCESS FILE — full pipeline in worker
// ============================================================
async function processFile(fileData, idx, options) {
  try {
    var baseName = fileData.name.replace(/\.[^.]+$/, '');
    send('STEP', { idx: idx, step: 0, stepText: 'Step 1/5: Reading image...', progress: 3 });

    // Decode image from ArrayBuffer
    var blob = new Blob([fileData.buffer], { type: fileData.type });
    var bitmap = await createImageBitmap(blob);
    var origW = bitmap.width, origH = bitmap.height;
    log(baseName + ': original ' + origW + 'x' + origH, 'info');

    // Draw to OffscreenCanvas
    var origCanvas = new OffscreenCanvas(origW, origH);
    origCanvas.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();

    // STEP 1: Upscale 4x
    send('STEP', { idx: idx, step: 0, stepText: 'Step 1/5: Upscaling 4x...', progress: 10 });
    var upCanvas = await upscale4x(origCanvas);
    log(baseName + ': upscaled → ' + upCanvas.width + 'x' + upCanvas.height, 'success');

    // STEP 2: Remove Background
    send('STEP', { idx: idx, step: 1, stepText: 'Step 2/5: Removing background...', progress: 30 });
    var bgCanvas = await removeBackground(upCanvas, baseName);
    log(baseName + ': background removed', 'success');

    // STEP 3: Edge Cleaning
    send('STEP', { idx: idx, step: 2, stepText: 'Step 3/5: Cleaning edges...', progress: 55 });
    var cleanCanvas = await edgeCleaning(bgCanvas);
    log(baseName + ': edges cleaned', 'success');

    // STEP 4: Vectorize
    send('STEP', { idx: idx, step: 3, stepText: 'Step 4/5: Vectorizing...', progress: 70 });
    var svgStr = await vectorize(cleanCanvas);
    log(baseName + ': vectorized', 'success');

    // STEP 5: SVG → EPS
    send('STEP', { idx: idx, step: 4, stepText: 'Step 5/5: Building EPS...', progress: 85 });
    var epsStr = svgToEps(svgStr, baseName, cleanCanvas.width, cleanCanvas.height);
    log(baseName + ': EPS built', 'success');

    // Get preview PNG
    send('STEP', { idx: idx, step: 5, stepText: 'Packaging...', progress: 95 });
    var previewBlob = await cleanCanvas.convertToBlob({ type: 'image/png' });
    var previewBuf = await previewBlob.arrayBuffer();

    send('FILE_DONE', {
      idx: idx,
      name: baseName,
      eps: epsStr,
      previewBuffer: previewBuf,
      fileName: fileData.name
    });

  } catch(e) {
    send('FILE_ERROR', { idx: idx, error: e.message });
  }
}

// ============================================================
// UPSCALE 4x
// ============================================================
async function upscale4x(canvas) {
  if (sessionEsrgan) {
    try {
      // Only use ONNX if image is small enough for i5-6300U RAM
      if (canvas.width <= 256 && canvas.height <= 256) {
        return await upscaleOnnx(canvas);
      }
    } catch(e) {
      log('ESRGAN ONNX error, falling back to Lanczos: ' + e.message, 'warn');
    }
  }
  return upscaleLanczos(canvas, 4);
}

async function upscaleOnnx(canvas) {
  var w = canvas.width, h = canvas.height;
  var ctx = canvas.getContext('2d');
  var imgd = ctx.getImageData(0, 0, w, h);
  var data = imgd.data;

  var float32 = new Float32Array(3 * h * w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var i = (y * w + x) * 4;
      float32[0 * h * w + y * w + x] = data[i] / 255;
      float32[1 * h * w + y * w + x] = data[i + 1] / 255;
      float32[2 * h * w + y * w + x] = data[i + 2] / 255;
    }
  }

  var tensor = new ort.Tensor('float32', float32, [1, 3, h, w]);
  var feeds = {};
  feeds[sessionEsrgan.inputNames[0]] = tensor;
  var results = await sessionEsrgan.run(feeds);
  var out = results[sessionEsrgan.outputNames[0]];
  var outH = out.dims[2], outW = out.dims[3];
  var outData = out.data;

  var outCanvas = new OffscreenCanvas(outW, outH);
  var octx = outCanvas.getContext('2d');
  var outImgd = octx.createImageData(outW, outH);
  for (var oy = 0; oy < outH; oy++) {
    for (var ox = 0; ox < outW; ox++) {
      var oi = (oy * outW + ox) * 4;
      outImgd.data[oi]     = clamp(outData[0 * outH * outW + oy * outW + ox] * 255);
      outImgd.data[oi + 1] = clamp(outData[1 * outH * outW + oy * outW + ox] * 255);
      outImgd.data[oi + 2] = clamp(outData[2 * outH * outW + oy * outW + ox] * 255);
      outImgd.data[oi + 3] = 255;
    }
  }
  octx.putImageData(outImgd, 0, 0);
  return outCanvas;
}

function upscaleLanczos(canvas, scale) {
  // Multi-pass 2x each time for better quality
  var tmp = canvas;
  var passes = Math.log2(scale);
  for (var p = 0; p < passes; p++) {
    var next = new OffscreenCanvas(tmp.width * 2, tmp.height * 2);
    var ctx = next.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, next.width, next.height);
    tmp = next;
  }
  return tmp;
}

// ============================================================
// REMOVE BACKGROUND
// ============================================================
async function removeBackground(canvas, baseName) {
  if (sessionRmbg) {
    try {
      return await removeBgOnnx(canvas);
    } catch(e) {
      log(baseName + ': RMBG ONNX error, using flood-fill: ' + e.message, 'warn');
    }
  }
  return removeBgFallback(canvas);
}

async function removeBgOnnx(canvas) {
  var targetSize = 512; // smaller for i5-6300U RAM
  var resized = resizeOffscreen(canvas, targetSize, targetSize);
  var ctx = resized.getContext('2d');
  var imgd = ctx.getImageData(0, 0, targetSize, targetSize);
  var data = imgd.data;

  var mean = [0.485, 0.456, 0.406];
  var std  = [0.229, 0.224, 0.225];
  var sz = targetSize * targetSize;
  var float32 = new Float32Array(3 * sz);

  for (var y = 0; y < targetSize; y++) {
    for (var x = 0; x < targetSize; x++) {
      var i = (y * targetSize + x) * 4;
      var pi = y * targetSize + x;
      float32[0 * sz + pi] = ((data[i]     / 255) - mean[0]) / std[0];
      float32[1 * sz + pi] = ((data[i + 1] / 255) - mean[1]) / std[1];
      float32[2 * sz + pi] = ((data[i + 2] / 255) - mean[2]) / std[2];
    }
  }

  var tensor = new ort.Tensor('float32', float32, [1, 3, targetSize, targetSize]);
  var feeds = {};
  feeds[sessionRmbg.inputNames[0]] = tensor;
  var out = await sessionRmbg.run(feeds);
  var maskData = out[sessionRmbg.outputNames[0]].data;
  var maskDims = out[sessionRmbg.outputNames[0]].dims;
  var maskH = maskDims[2], maskW = maskDims[3];

  // Apply mask to original canvas
  var origW = canvas.width, origH = canvas.height;
  var origCtx = canvas.getContext('2d');
  var origImgd = origCtx.getImageData(0, 0, origW, origH);

  var result = new OffscreenCanvas(origW, origH);
  var rctx = result.getContext('2d');
  var rImgd = rctx.createImageData(origW, origH);

  for (var ry = 0; ry < origH; ry++) {
    for (var rx = 0; rx < origW; rx++) {
      var si = (ry * origW + rx) * 4;
      var my = Math.min(maskH - 1, Math.floor((ry / origH) * maskH));
      var mx = Math.min(maskW - 1, Math.floor((rx / origW) * maskW));
      var alpha = clamp(maskData[my * maskW + mx] * 255);
      rImgd.data[si]     = origImgd.data[si];
      rImgd.data[si + 1] = origImgd.data[si + 1];
      rImgd.data[si + 2] = origImgd.data[si + 2];
      rImgd.data[si + 3] = alpha;
    }
  }
  rctx.putImageData(rImgd, 0, 0);
  return result;
}

function removeBgFallback(canvas) {
  var w = canvas.width, h = canvas.height;
  var ctx = canvas.getContext('2d');
  var imgd = ctx.getImageData(0, 0, w, h);
  var data = imgd.data;

  var result = new OffscreenCanvas(w, h);
  var rctx = result.getContext('2d');
  var rImgd = rctx.createImageData(w, h);
  for (var k = 0; k < data.length; k++) rImgd.data[k] = data[k];

  // Sample background from corners and edges
  var samplePoints = [
    [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
    [Math.floor(w/2), 0], [0, Math.floor(h/2)],
    [w-1, Math.floor(h/2)], [Math.floor(w/2), h-1],
    [Math.floor(w/4), 0], [Math.floor(3*w/4), 0],
    [0, Math.floor(h/4)], [0, Math.floor(3*h/4)]
  ];

  var bgColors = samplePoints.map(function(p) {
    var i = (p[1] * w + p[0]) * 4;
    return { r: data[i], g: data[i+1], b: data[i+2] };
  });

  var tolerance = 50;

  function isBg(r, g, b) {
    return bgColors.some(function(c) {
      return Math.abs(r - c.r) + Math.abs(g - c.g) + Math.abs(b - c.b) < tolerance * 3;
    });
  }

  // BFS flood fill from edges
  var visited = new Uint8Array(w * h);
  var queue = [];
  var qi = 0;

  // Seed all edge pixels that match BG
  for (var ex = 0; ex < w; ex++) {
    var ti = ex; var tpi = ex * 4;
    if (!visited[ti] && isBg(data[tpi], data[tpi+1], data[tpi+2])) { visited[ti]=1; queue.push(ti); }
    var bi = (h-1)*w+ex; var bpi = bi*4;
    if (!visited[bi] && isBg(data[bpi], data[bpi+1], data[bpi+2])) { visited[bi]=1; queue.push(bi); }
  }
  for (var ey = 0; ey < h; ey++) {
    var li = ey*w; var lpi = li*4;
    if (!visited[li] && isBg(data[lpi], data[lpi+1], data[lpi+2])) { visited[li]=1; queue.push(li); }
    var ri = ey*w+(w-1); var rpi = ri*4;
    if (!visited[ri] && isBg(data[rpi], data[rpi+1], data[rpi+2])) { visited[ri]=1; queue.push(ri); }
  }

  while (qi < queue.length) {
    var ci = queue[qi++];
    rImgd.data[ci*4+3] = 0; // transparent
    var cx = ci % w, cy = Math.floor(ci / w);
    var neighbors = [
      cy > 0     ? ci - w : -1,
      cy < h - 1 ? ci + w : -1,
      cx > 0     ? ci - 1 : -1,
      cx < w - 1 ? ci + 1 : -1
    ];
    for (var n = 0; n < 4; n++) {
      var ni = neighbors[n];
      if (ni < 0 || visited[ni]) continue;
      var npi = ni * 4;
      if (isBg(data[npi], data[npi+1], data[npi+2])) {
        visited[ni] = 1;
        queue.push(ni);
      }
    }
  }

  rctx.putImageData(rImgd, 0, 0);
  return result;
}

// ============================================================
// EDGE CLEANING
// ============================================================
function edgeCleaning(canvas) {
  var w = canvas.width, h = canvas.height;
  var ctx = canvas.getContext('2d');
  var imgd = ctx.getImageData(0, 0, w, h);
  var data = imgd.data;
  var result = new Uint8ClampedArray(data.length);
  for (var k = 0; k < data.length; k++) result[k] = data[k];

  for (var i = 0; i < data.length; i += 4) {
    // Hard alpha threshold
    if (data[i+3] < 25) { result[i+3] = 0; }
    else if (data[i+3] > 230) { result[i+3] = 255; }
  }

  // Erode semi-transparent fringe pixels
  for (var y = 1; y < h - 1; y++) {
    for (var x = 1; x < w - 1; x++) {
      var idx = (y * w + x) * 4;
      if (result[idx+3] === 0) continue;
      var neighbors = [
        ((y-1)*w+x)*4, ((y+1)*w+x)*4,
        (y*w+x-1)*4,   (y*w+x+1)*4
      ];
      var hasTransNeighbor = neighbors.some(function(ni) { return result[ni+3] < 128; });
      if (hasTransNeighbor && result[idx+3] < 210) {
        result[idx+3] = 0;
      }
    }
  }

  var out = new OffscreenCanvas(w, h);
  out.getContext('2d').putImageData(new ImageData(result, w, h), 0, 0);
  return out;
}

// ============================================================
// VECTORIZE (ImageTracer inline in worker)
// ============================================================
function vectorize(canvas) {
  var maxDim = 800;
  var c = canvas;
  if (c.width > maxDim || c.height > maxDim) {
    var ratio = Math.min(maxDim / c.width, maxDim / c.height);
    c = resizeOffscreen(canvas, Math.round(c.width * ratio), Math.round(c.height * ratio));
  }

  var w = c.width, h = c.height;
  var ctx = c.getContext('2d');
  var imgd = ctx.getImageData(0, 0, w, h);
  var data = imgd.data;

  var nc = 20; // number of colors
  var colorMap = {};
  var colors = [];

  for (var i = 0; i < data.length; i += 4) {
    if (data[i+3] < 20) continue;
    var r = Math.round(data[i] / 16) * 16;
    var g = Math.round(data[i+1] / 16) * 16;
    var b = Math.round(data[i+2] / 16) * 16;
    var key = r + ',' + g + ',' + b;
    if (!colorMap[key]) {
      colorMap[key] = { r: data[i], g: data[i+1], b: data[i+2], a: data[i+3], count: 0 };
      colors.push(key);
    }
    colorMap[key].count++;
  }

  colors.sort(function(a, b) { return colorMap[b].count - colorMap[a].count; });
  colors = colors.slice(0, nc);

  var svgPaths = [];
  var tolerance = 40;
  var minPatchSize = 6;

  colors.forEach(function(key) {
    var c2 = colorMap[key];
    var mask = new Uint8Array(w * h);

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var idx = (y * w + x) * 4;
        if (data[idx+3] < 20) continue;
        var dr = Math.abs(data[idx]   - c2.r);
        var dg = Math.abs(data[idx+1] - c2.g);
        var db = Math.abs(data[idx+2] - c2.b);
        if (dr + dg + db < tolerance * 3) mask[y * w + x] = 1;
      }
    }

    var visited = new Uint8Array(w * h);
    var rects = [];

    for (var ry = 0; ry < h; ry++) {
      for (var rx = 0; rx < w; rx++) {
        var ridx = ry * w + rx;
        if (!mask[ridx] || visited[ridx]) continue;
        var minX = rx, minY = ry, maxX = rx, maxY = ry;
        var bfsQ = [ridx];
        visited[ridx] = 1;
        var bqi = 0;
        while (bqi < bfsQ.length) {
          var ci = bfsQ[bqi++];
          var cx = ci % w, cy2 = Math.floor(ci / w);
          if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
          if (cy2 < minY) minY = cy2; if (cy2 > maxY) maxY = cy2;
          var nbrs = [ci-1, ci+1, ci-w, ci+w];
          var vld  = [cx>0, cx<w-1, cy2>0, cy2<h-1];
          for (var n = 0; n < 4; n++) {
            if (!vld[n]) continue;
            var ni = nbrs[n];
            if (!visited[ni] && mask[ni]) { visited[ni]=1; bfsQ.push(ni); }
          }
        }
        var rw = maxX - minX + 1, rh2 = maxY - minY + 1;
        if (rw * rh2 >= minPatchSize) rects.push({ x: minX, y: minY, w: rw, h: rh2 });
      }
    }

    if (rects.length === 0) return;
    var hex = '#' + ('0' + c2.r.toString(16)).slice(-2)
                  + ('0' + c2.g.toString(16)).slice(-2)
                  + ('0' + c2.b.toString(16)).slice(-2);
    var opa = (c2.a / 255).toFixed(2);
    var d = rects.map(function(r) {
      return 'M' + r.x + ' ' + r.y + 'H' + (r.x+r.w) + 'V' + (r.y+r.h) + 'H' + r.x + 'Z';
    }).join(' ');
    svgPaths.push('<path fill="'+hex+'" fill-opacity="'+opa+'" d="'+d+'"/>');
  });

  return '<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'">'
    + svgPaths.join('') + '</svg>';
}

// ============================================================
// SVG → EPS
// ============================================================
function svgToEps(svgStr, title, width, height) {
  var parser = new DOMParser();
  var doc = parser.parseFromString(svgStr, 'image/svg+xml');
  var svgEl = doc.documentElement;
  var vb = (svgEl.getAttribute('viewBox') || '0 0 '+width+' '+height).split(' ').map(Number);
  var vbW = vb[2] || width;
  var vbH = vb[3] || height;

  var targetW = Math.max(width, 1000);
  var targetH = Math.max(height, 1000);
  var scaleX = targetW / vbW;
  var scaleY = targetH / vbH;

  var paths = doc.querySelectorAll('path');
  var dateStr = new Date().toISOString().slice(0, 10);

  var eps = '%!PS-Adobe-3.0 EPSF-3.0\n'
    + '%%BoundingBox: 0 0 ' + Math.round(targetW) + ' ' + Math.round(targetH) + '\n'
    + '%%HiResBoundingBox: 0.0 0.0 ' + targetW.toFixed(4) + ' ' + targetH.toFixed(4) + '\n'
    + '%%Title: (' + title.replace(/[()]/g, '') + ')\n'
    + '%%Creator: (VectorFlow — Adobe Stock Processor)\n'
    + '%%CreationDate: (' + dateStr + ')\n'
    + '%%DocumentData: Clean7Bit\n'
    + '%%Origin: 0 0\n'
    + '%%ColorUsage: Color\n'
    + '%%DocumentProcessColors: Red Green Blue\n'
    + '%%LanguageLevel: 3\n'
    + '%%Pages: 1\n'
    + '%%EndComments\n'
    + '%%BeginSetup\n%%EndSetup\n'
    + '%%Page: 1 1\n'
    + 'gsave\n'
    + '1 -1 scale\n'
    + '0 -' + targetH + ' translate\n';

  paths.forEach(function(path) {
    var fill = path.getAttribute('fill') || '#000000';
    var opacity = parseFloat(path.getAttribute('fill-opacity') || '1');
    var d = path.getAttribute('d') || '';
    if (!d || fill === 'none') return;

    var rgb = hexToRgb(fill);
    if (!rgb) return;

    eps += rgb.r.toFixed(4) + ' ' + rgb.g.toFixed(4) + ' ' + rgb.b.toFixed(4) + ' setrgbcolor\n';

    var ps = svgPathToPs(d, scaleX, scaleY, vbH);
    if (ps) {
      eps += 'newpath\n' + ps + '\nfill\n';
    }
  });

  eps += 'grestore\n%%Trailer\n%%EOF\n';
  return eps;
}

function hexToRgb(hex) {
  hex = hex.replace('#', '');
  if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
  if (hex.length !== 6) return null;
  return {
    r: parseInt(hex.slice(0,2), 16) / 255,
    g: parseInt(hex.slice(2,4), 16) / 255,
    b: parseInt(hex.slice(4,6), 16) / 255
  };
}

function svgPathToPs(d, scaleX, scaleY, vbH) {
  var tokens = d.match(/[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?/gi);
  if (!tokens) return '';
  var ps = '', cmd = '', i = 0, cx = 0, cy = 0;

  function num() { return parseFloat(tokens[i++]); }
  function tx(x) { return (x * scaleX).toFixed(3); }
  function ty(y) { return ((vbH - y) * scaleY).toFixed(3); }

  while (i < tokens.length) {
    var t = tokens[i];
    if (/[MmLlHhVvCcSsQqTtAaZz]/.test(t)) { cmd = t; i++; } else {
      switch(cmd) {
        case 'M': cx=num(); cy=num(); ps+='  '+tx(cx)+' '+ty(cy)+' moveto\n'; cmd='L'; break;
        case 'm': cx+=num(); cy+=num(); ps+='  '+tx(cx)+' '+ty(cy)+' moveto\n'; cmd='l'; break;
        case 'L': cx=num(); cy=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'l': cx+=num(); cy+=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'H': cx=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'h': cx+=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'V': cy=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'v': cy+=num(); ps+='  '+tx(cx)+' '+ty(cy)+' lineto\n'; break;
        case 'C': var x1=num(),y1=num(),x2=num(),y2=num(); cx=num(); cy=num();
          ps+='  '+tx(x1)+' '+ty(y1)+' '+tx(x2)+' '+ty(y2)+' '+tx(cx)+' '+ty(cy)+' curveto\n'; break;
        case 'c': var rx1=cx+num(),ry1=cy+num(),rx2=cx+num(),ry2=cy+num(); cx+=num(); cy+=num();
          ps+='  '+tx(rx1)+' '+ty(ry1)+' '+tx(rx2)+' '+ty(ry2)+' '+tx(cx)+' '+ty(cy)+' curveto\n'; break;
        case 'S': var sx2=num(),sy2=num(); cx=num(); cy=num();
          ps+='  '+tx(sx2)+' '+ty(sy2)+' '+tx(cx)+' '+ty(cy)+' '+tx(cx)+' '+ty(cy)+' curveto\n'; break;
        case 'Z': case 'z': ps+='  closepath\n'; break;
        default: i++; break;
      }
    }
  }
  return ps;
}

// ============================================================
// HELPERS
// ============================================================
function clamp(v) { return Math.min(255, Math.max(0, Math.round(v))); }

function resizeOffscreen(canvas, w, h) {
  var out = new OffscreenCanvas(w, h);
  var ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, w, h);
  return out;
}
