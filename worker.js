// VectorFlow Web Worker — standalone file
// Must be served from same origin as index.html (GitHub Pages OK)

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

var sessionRmbg = null, sessionEsrgan = null;
var modelsLoaded = { rmbg: false, esrgan: false };

self.onmessage = async function(e) {
  var m = e.data;
  if (m.type === 'LOAD_MODELS')  await loadModels();
  if (m.type === 'PROCESS_FILE') await processFile(m.file, m.idx);
};

function send(type, data) { self.postMessage(Object.assign({ type: type }, data)); }
function log(msg, level)  { send('LOG', { msg: msg, level: level || 'info' }); }

// ── LOAD MODELS ──────────────────────────────────────────────
async function loadModels() {
  if (typeof ort !== 'undefined') {
    ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    ort.env.wasm.numThreads = 2;
  }

  // RMBG-1.4
  send('MODEL_STATUS', { model: 'rmbg', state: 'loading', text: 'downloading...' });
  try {
    sessionRmbg = await ort.InferenceSession.create(
      'https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    );
    modelsLoaded.rmbg = true;
    send('MODEL_STATUS', { model: 'rmbg', state: 'ready', text: 'ready v' });
    log('RMBG-1.4 loaded OK', 'success');
  } catch(e) {
    send('MODEL_STATUS', { model: 'rmbg', state: 'error', text: 'flood-fill' });
    log('RMBG failed, flood-fill fallback: ' + e.message, 'warn');
  }

  // Real-ESRGAN
  send('MODEL_STATUS', { model: 'esrgan', state: 'loading', text: 'downloading...' });
  try {
    sessionEsrgan = await ort.InferenceSession.create(
      'https://huggingface.co/rocca/realesrgan-onnx/resolve/main/realesrgan-x4plus-anime.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    );
    modelsLoaded.esrgan = true;
    send('MODEL_STATUS', { model: 'esrgan', state: 'ready', text: 'ready v' });
    log('Real-ESRGAN loaded OK', 'success');
  } catch(e) {
    send('MODEL_STATUS', { model: 'esrgan', state: 'error', text: 'Lanczos' });
    log('ESRGAN failed, Lanczos fallback: ' + e.message, 'warn');
  }

  send('MODELS_READY', { rmbg: modelsLoaded.rmbg, esrgan: modelsLoaded.esrgan });
}

// ── PROCESS FILE ─────────────────────────────────────────────
async function processFile(fileData, idx) {
  try {
    var baseName = fileData.name.replace(/\.[^.]+$/, '');
    send('STEP', { idx: idx, step: 0, stepText: 'Reading image...', progress: 3 });

    var blob   = new Blob([fileData.buffer], { type: fileData.type || 'image/jpeg' });
    var bitmap = await createImageBitmap(blob);
    var origW  = bitmap.width, origH = bitmap.height;
    log(baseName + ': ' + origW + 'x' + origH, 'info');

    var oc = new OffscreenCanvas(origW, origH);
    oc.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();

    // Step 1: Upscale 4x
    send('STEP', { idx: idx, step: 0, stepText: 'Step 1/5: Upscaling 4x...', progress: 10 });
    var up = await upscale4x(oc);
    log(baseName + ': upscaled to ' + up.width + 'x' + up.height, 'success');

    // Step 2: Remove BG
    send('STEP', { idx: idx, step: 1, stepText: 'Step 2/5: Removing background...', progress: 30 });
    var bg = await removeBg(up, baseName);
    log(baseName + ': background removed', 'success');

    // Step 3: Edge clean
    send('STEP', { idx: idx, step: 2, stepText: 'Step 3/5: Cleaning edges...', progress: 55 });
    var cl = edgeClean(bg);
    log(baseName + ': edges cleaned', 'success');

    // Step 4: Vectorize
    send('STEP', { idx: idx, step: 3, stepText: 'Step 4/5: Vectorizing...', progress: 70 });
    var svg = vectorize(cl);
    log(baseName + ': vectorized', 'success');

    // Step 5: SVG -> EPS
    send('STEP', { idx: idx, step: 4, stepText: 'Step 5/5: Building EPS...', progress: 85 });
    var eps = svgToEps(svg, baseName, cl.width, cl.height);
    log(baseName + ': EPS built', 'success');

    // Package
    send('STEP', { idx: idx, step: 5, stepText: 'Packaging...', progress: 95 });
    var prevBlob = await cl.convertToBlob({ type: 'image/png' });
    var prevBuf  = await prevBlob.arrayBuffer();

    send('FILE_DONE', {
      idx: idx, name: baseName, eps: eps,
      previewBuffer: prevBuf, fileName: fileData.name
    });

  } catch(e) {
    send('FILE_ERROR', { idx: idx, error: e.message });
  }
}

// ── UPSCALE ──────────────────────────────────────────────────
async function upscale4x(canvas) {
  if (sessionEsrgan && canvas.width <= 256 && canvas.height <= 256) {
    try { return await upscaleOnnx(canvas); }
    catch(e) { log('ESRGAN error, Lanczos: ' + e.message, 'warn'); }
  }
  return upscaleLanczos(canvas, 4);
}

async function upscaleOnnx(canvas) {
  var w = canvas.width, h = canvas.height;
  var data = canvas.getContext('2d').getImageData(0, 0, w, h).data;
  var f32  = new Float32Array(3 * h * w);
  for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
    var i = (y * w + x) * 4;
    f32[0*h*w + y*w+x] = data[i]   / 255;
    f32[1*h*w + y*w+x] = data[i+1] / 255;
    f32[2*h*w + y*w+x] = data[i+2] / 255;
  }
  var t = new ort.Tensor('float32', f32, [1, 3, h, w]);
  var feeds = {}; feeds[sessionEsrgan.inputNames[0]] = t;
  var res = await sessionEsrgan.run(feeds);
  var out = res[sessionEsrgan.outputNames[0]];
  var oH  = out.dims[2], oW = out.dims[3], od = out.data;
  var oc  = new OffscreenCanvas(oW, oH);
  var id  = oc.getContext('2d').createImageData(oW, oH);
  for (var oy = 0; oy < oH; oy++) for (var ox = 0; ox < oW; ox++) {
    var oi = (oy * oW + ox) * 4;
    id.data[oi]   = clamp(od[0*oH*oW + oy*oW+ox] * 255);
    id.data[oi+1] = clamp(od[1*oH*oW + oy*oW+ox] * 255);
    id.data[oi+2] = clamp(od[2*oH*oW + oy*oW+ox] * 255);
    id.data[oi+3] = 255;
  }
  oc.getContext('2d').putImageData(id, 0, 0);
  return oc;
}

function upscaleLanczos(canvas, scale) {
  var tmp = canvas, passes = Math.log2(scale);
  for (var p = 0; p < passes; p++) {
    var n = new OffscreenCanvas(tmp.width * 2, tmp.height * 2);
    var c = n.getContext('2d');
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    c.drawImage(tmp, 0, 0, n.width, n.height); tmp = n;
  }
  return tmp;
}

// ── REMOVE BACKGROUND ────────────────────────────────────────
async function removeBg(canvas, name) {
  if (sessionRmbg) {
    try { return await removeBgOnnx(canvas); }
    catch(e) { log(name + ': RMBG error, fallback: ' + e.message, 'warn'); }
  }
  return removeBgFlood(canvas);
}

async function removeBgOnnx(canvas) {
  var SZ   = 512;
  var rs   = resizeCanvas(canvas, SZ, SZ);
  var data = rs.getContext('2d').getImageData(0, 0, SZ, SZ).data;
  var mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  var sz   = SZ * SZ, f32 = new Float32Array(3 * sz);
  for (var y = 0; y < SZ; y++) for (var x = 0; x < SZ; x++) {
    var i = (y * SZ + x) * 4, pi = y * SZ + x;
    f32[0*sz + pi] = ((data[i]   / 255) - mean[0]) / std[0];
    f32[1*sz + pi] = ((data[i+1] / 255) - mean[1]) / std[1];
    f32[2*sz + pi] = ((data[i+2] / 255) - mean[2]) / std[2];
  }
  var t = new ort.Tensor('float32', f32, [1, 3, SZ, SZ]);
  var feeds = {}; feeds[sessionRmbg.inputNames[0]] = t;
  var res  = await sessionRmbg.run(feeds);
  var mOut = res[sessionRmbg.outputNames[0]];
  var mData = mOut.data, mH = mOut.dims[2], mW = mOut.dims[3];

  var oW = canvas.width, oH = canvas.height;
  var origData = canvas.getContext('2d').getImageData(0, 0, oW, oH).data;
  var out = new OffscreenCanvas(oW, oH);
  var id  = out.getContext('2d').createImageData(oW, oH);
  for (var ry = 0; ry < oH; ry++) for (var rx = 0; rx < oW; rx++) {
    var si = (ry * oW + rx) * 4;
    var my = Math.min(mH-1, Math.floor(ry / oH * mH));
    var mx = Math.min(mW-1, Math.floor(rx / oW * mW));
    id.data[si]   = origData[si];
    id.data[si+1] = origData[si+1];
    id.data[si+2] = origData[si+2];
    id.data[si+3] = clamp(mData[my * mW + mx] * 255);
  }
  out.getContext('2d').putImageData(id, 0, 0);
  return out;
}

function removeBgFlood(canvas) {
  var w = canvas.width, h = canvas.height;
  var data = canvas.getContext('2d').getImageData(0, 0, w, h).data;
  var out  = new OffscreenCanvas(w, h);
  var id   = out.getContext('2d').createImageData(w, h);
  for (var k = 0; k < data.length; k++) id.data[k] = data[k];

  var pts = [
    [0,0],[w-1,0],[0,h-1],[w-1,h-1],
    [Math.floor(w/2),0],[0,Math.floor(h/2)],
    [w-1,Math.floor(h/2)],[Math.floor(w/2),h-1],
    [Math.floor(w/4),0],[Math.floor(3*w/4),0]
  ];
  var bgC = pts.map(function(p) {
    var i = (p[1]*w + p[0]) * 4;
    return { r: data[i], g: data[i+1], b: data[i+2] };
  });
  var tol = 50;
  function isBg(r, g, b) {
    return bgC.some(function(c) {
      return Math.abs(r-c.r) + Math.abs(g-c.g) + Math.abs(b-c.b) < tol * 3;
    });
  }

  var visited = new Uint8Array(w * h), queue = [], qi = 0;
  for (var ex = 0; ex < w; ex++) {
    var t1 = ex, d1 = t1*4;
    if (!visited[t1] && isBg(data[d1],data[d1+1],data[d1+2])) { visited[t1]=1; queue.push(t1); }
    var b1 = (h-1)*w+ex, db = b1*4;
    if (!visited[b1] && isBg(data[db],data[db+1],data[db+2])) { visited[b1]=1; queue.push(b1); }
  }
  for (var ey = 0; ey < h; ey++) {
    var l = ey*w, dl = l*4;
    if (!visited[l] && isBg(data[dl],data[dl+1],data[dl+2])) { visited[l]=1; queue.push(l); }
    var r = ey*w+(w-1), dr = r*4;
    if (!visited[r] && isBg(data[dr],data[dr+1],data[dr+2])) { visited[r]=1; queue.push(r); }
  }
  while (qi < queue.length) {
    var ci = queue[qi++];
    id.data[ci*4+3] = 0;
    var cx = ci % w, cy = Math.floor(ci / w);
    var nb = [cy>0?ci-w:-1, cy<h-1?ci+w:-1, cx>0?ci-1:-1, cx<w-1?ci+1:-1];
    for (var n = 0; n < 4; n++) {
      var ni = nb[n]; if (ni < 0 || visited[ni]) continue;
      var np = ni * 4;
      if (isBg(data[np],data[np+1],data[np+2])) { visited[ni]=1; queue.push(ni); }
    }
  }
  out.getContext('2d').putImageData(id, 0, 0);
  return out;
}

// ── EDGE CLEANING ─────────────────────────────────────────────
function edgeClean(canvas) {
  var w = canvas.width, h = canvas.height;
  var data = canvas.getContext('2d').getImageData(0, 0, w, h).data;
  var res  = new Uint8ClampedArray(data.length);
  for (var k = 0; k < data.length; k++) res[k] = data[k];
  for (var i = 0; i < data.length; i += 4) {
    if (data[i+3] < 25)  res[i+3] = 0;
    else if (data[i+3] > 230) res[i+3] = 255;
  }
  for (var y = 1; y < h-1; y++) for (var x = 1; x < w-1; x++) {
    var idx = (y*w+x)*4;
    if (res[idx+3] === 0) continue;
    var nb = [((y-1)*w+x)*4,((y+1)*w+x)*4,(y*w+x-1)*4,(y*w+x+1)*4];
    if (nb.some(function(ni){ return res[ni+3] < 128; }) && res[idx+3] < 210) res[idx+3] = 0;
  }
  var out = new OffscreenCanvas(w, h);
  out.getContext('2d').putImageData(new ImageData(res, w, h), 0, 0);
  return out;
}

// ── VECTORIZE ────────────────────────────────────────────────
function vectorize(canvas) {
  var maxDim = 800, c = canvas;
  if (c.width > maxDim || c.height > maxDim) {
    var ratio = Math.min(maxDim/c.width, maxDim/c.height);
    c = resizeCanvas(canvas, Math.round(c.width*ratio), Math.round(c.height*ratio));
  }
  var w = c.width, h = c.height;
  var data = c.getContext('2d').getImageData(0, 0, w, h).data;
  var nc = 24, cmap = {}, colors = [], tol = 40;

  for (var i = 0; i < data.length; i += 4) {
    if (data[i+3] < 20) continue;
    var r = Math.round(data[i]/16)*16;
    var g = Math.round(data[i+1]/16)*16;
    var b = Math.round(data[i+2]/16)*16;
    var key = r+','+g+','+b;
    if (!cmap[key]) { cmap[key]={r:data[i],g:data[i+1],b:data[i+2],a:data[i+3],count:0}; colors.push(key); }
    cmap[key].count++;
  }
  colors.sort(function(a,b){ return cmap[b].count - cmap[a].count; });
  colors = colors.slice(0, nc);

  var paths = [];
  colors.forEach(function(key) {
    var c2 = cmap[key], mask = new Uint8Array(w*h);
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      var idx = (y*w+x)*4; if (data[idx+3] < 20) continue;
      if (Math.abs(data[idx]-c2.r)+Math.abs(data[idx+1]-c2.g)+Math.abs(data[idx+2]-c2.b) < tol*3)
        mask[y*w+x] = 1;
    }
    var vis = new Uint8Array(w*h), rects = [];
    for (var ry = 0; ry < h; ry++) for (var rx = 0; rx < w; rx++) {
      var ri = ry*w+rx; if (!mask[ri]||vis[ri]) continue;
      var mnX=rx,mnY=ry,mxX=rx,mxY=ry,q=[ri]; vis[ri]=1; var qi2=0;
      while (qi2 < q.length) {
        var ci=q[qi2++], cx=ci%w, cy=Math.floor(ci/w);
        if(cx<mnX)mnX=cx; if(cx>mxX)mxX=cx; if(cy<mnY)mnY=cy; if(cy>mxY)mxY=cy;
        var nb2=[ci-1,ci+1,ci-w,ci+w], vl=[cx>0,cx<w-1,cy>0,cy<h-1];
        for (var n=0;n<4;n++){ if(!vl[n])continue; var ni=nb2[n]; if(!vis[ni]&&mask[ni]){vis[ni]=1;q.push(ni);} }
      }
      var rw=mxX-mnX+1, rh2=mxY-mnY+1;
      if (rw*rh2 >= 6) rects.push({x:mnX,y:mnY,w:rw,h:rh2});
    }
    if (!rects.length) return;
    var hex = '#'+('0'+c2.r.toString(16)).slice(-2)+('0'+c2.g.toString(16)).slice(-2)+('0'+c2.b.toString(16)).slice(-2);
    var opa = (c2.a/255).toFixed(2);
    var d   = rects.map(function(r){ return 'M'+r.x+' '+r.y+'H'+(r.x+r.w)+'V'+(r.y+r.h)+'H'+r.x+'Z'; }).join(' ');
    paths.push('<path fill="'+hex+'" fill-opacity="'+opa+'" d="'+d+'"/>');
  });
  return '<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'">'+paths.join('')+'</svg>';
}

// ── SVG → EPS ────────────────────────────────────────────────
function svgToEps(svgStr, title, width, height) {
  var vbM  = svgStr.match(/viewBox=["']([^"']+)["']/);
  var vbP  = vbM ? vbM[1].trim().split(/\s+|,/).map(Number) : [0,0,width,height];
  var vbW  = vbP[2]||width, vbH = vbP[3]||height;
  var tW   = Math.max(width,1000), tH = Math.max(height,1000);
  var sX   = tW/vbW, sY = tH/vbH;
  var date = new Date().toISOString().slice(0,10);

  var eps = '%!PS-Adobe-3.0 EPSF-3.0\n'
    +'%%BoundingBox: 0 0 '+Math.round(tW)+' '+Math.round(tH)+'\n'
    +'%%HiResBoundingBox: 0.0 0.0 '+tW.toFixed(4)+' '+tH.toFixed(4)+'\n'
    +'%%Title: ('+title.replace(/[()\\]/g,'')+') \n'
    +'%%Creator: (VectorFlow Adobe Stock Processor)\n'
    +'%%CreationDate: ('+date+')\n'
    +'%%DocumentData: Clean7Bit\n'
    +'%%Origin: 0 0\n'
    +'%%ColorUsage: Color\n'
    +'%%DocumentProcessColors: Red Green Blue\n'
    +'%%LanguageLevel: 3\n'
    +'%%Pages: 1\n'
    +'%%EndComments\n'
    +'%%BeginSetup\n%%EndSetup\n'
    +'%%Page: 1 1\n'
    +'gsave\n'
    +'1 -1 scale\n'
    +'0 -'+tH+' translate\n';

  var pathRe = /<path([^>]+?)(?:\/>|>)/g, pm;
  while ((pm = pathRe.exec(svgStr)) !== null) {
    var attrs = pm[1];
    var fillM = attrs.match(/fill=["']([^"']+)["']/);
    var dM    = attrs.match(/\sd=["']([^"']+)["']/) || attrs.match(/d=["']([^"']+)["']/);
    var fill  = fillM ? fillM[1] : '#000000';
    if (fill==='none'||!dM) continue;
    var rgb = hexRgb(fill); if (!rgb) continue;
    eps += rgb.r.toFixed(4)+' '+rgb.g.toFixed(4)+' '+rgb.b.toFixed(4)+' setrgbcolor\n';
    var psd = pathToPs(dM[1], sX, sY, vbH);
    if (psd) eps += 'newpath\n'+psd+'fill\n';
  }
  eps += 'grestore\n%%Trailer\n%%EOF\n';
  return eps;
}

function hexRgb(h) {
  h = h.replace('#','');
  if (h.length===3) h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  if (h.length!==6) return null;
  return { r:parseInt(h.slice(0,2),16)/255, g:parseInt(h.slice(2,4),16)/255, b:parseInt(h.slice(4,6),16)/255 };
}

function pathToPs(d, sX, sY, vbH) {
  var toks = d.match(/[MmLlHhVvCcSsZz]|[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?/gi);
  if (!toks) return '';
  var ps='', cmd='', i=0, cx=0, cy=0;
  function num(){ return parseFloat(toks[i++]); }
  function tx(x){ return (x*sX).toFixed(3); }
  function ty(y){ return ((vbH-y)*sY).toFixed(3); }
  while (i < toks.length) {
    var t = toks[i];
    if (/[MmLlHhVvCcSsZz]/.test(t)){ cmd=t; i++; continue; }
    switch(cmd){
      case 'M': cx=num();cy=num();ps+=' '+tx(cx)+' '+ty(cy)+' moveto\n';cmd='L';break;
      case 'm': cx+=num();cy+=num();ps+=' '+tx(cx)+' '+ty(cy)+' moveto\n';cmd='l';break;
      case 'L': cx=num();cy=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'l': cx+=num();cy+=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'H': cx=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'h': cx+=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'V': cy=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'v': cy+=num();ps+=' '+tx(cx)+' '+ty(cy)+' lineto\n';break;
      case 'C': var x1=num(),y1=num(),x2=num(),y2=num();cx=num();cy=num();
        ps+=' '+tx(x1)+' '+ty(y1)+' '+tx(x2)+' '+ty(y2)+' '+tx(cx)+' '+ty(cy)+' curveto\n';break;
      case 'c': var rx1=cx+num(),ry1=cy+num(),rx2=cx+num(),ry2=cy+num();cx+=num();cy+=num();
        ps+=' '+tx(rx1)+' '+ty(ry1)+' '+tx(rx2)+' '+ty(ry2)+' '+tx(cx)+' '+ty(cy)+' curveto\n';break;
      case 'Z':case 'z': ps+=' closepath\n';break;
      default: i++;break;
    }
  }
  return ps;
}

function resizeCanvas(canvas, w, h) {
  var out = new OffscreenCanvas(w, h);
  var c   = out.getContext('2d');
  c.imageSmoothingEnabled=true; c.imageSmoothingQuality='high';
  c.drawImage(canvas,0,0,w,h); return out;
}
function clamp(v){ return Math.min(255,Math.max(0,Math.round(v))); }
