const $ = (id) => document.getElementById(id);

const fileInput = $('fileInput');
const dropzone = $('dropzone');
const fileName = $('fileName');
const preview = $('preview');
const overlay = $('overlay');
const placeholder = $('placeholder');
const submitBtn = $('submitBtn');
const loading = $('loading');
const errorBox = $('errorBox');
const summary = $('summary');
const layoutThreshold = $('layoutThreshold');
const ocrThreshold = $('ocrThreshold');

let selectedFile = null;
let objectUrl = null;
let lastResult = null;

const regionColors = {
  image: '#22c55e',
  chart: '#f59e0b',
  table: '#8b5cf6',
  header_image: '#06b6d4',
  footer_image: '#06b6d4',
};

function setError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || '';
}

function chooseFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  preview.hidden = false;
  placeholder.hidden = true;
  overlay.hidden = true;
  submitBtn.disabled = false;
  lastResult = null;
  summary.textContent = '已选择图片，等待识别';
  clearResults();
}

fileInput.addEventListener('change', (event) => chooseFile(event.target.files[0]));

['dragenter', 'dragover'].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));

layoutThreshold.addEventListener('input', () => {
  $('layoutValue').textContent = Number(layoutThreshold.value).toFixed(2);
});
ocrThreshold.addEventListener('input', () => {
  $('ocrValue').textContent = Number(ocrThreshold.value).toFixed(2);
});

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach((item) => item.classList.remove('active'));
    tab.classList.add('active');
    $(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

preview.addEventListener('load', () => {
  if (lastResult) drawOverlay(lastResult);
});

async function checkHealth() {
  const badge = $('modelStatus');
  try {
    const response = await fetch('/health');
    const data = await response.json();
    if (data.ready) {
      badge.textContent = '模型已就绪';
      badge.className = 'status status-ready';
    } else {
      badge.textContent = `模型不可用${data.detail ? '：' + data.detail : ''}`;
      badge.className = 'status status-error';
    }
  } catch (error) {
    badge.textContent = `模型状态检查失败：${error.message}`;
    badge.className = 'status status-error';
  }
}

function clearResults() {
  $('textRows').innerHTML = '';
  $('regionRows').innerHTML = '';
  $('markdownOutput').textContent = '';
  $('jsonOutput').textContent = '';
}

async function runAnalysis() {
  if (!selectedFile) return;
  setError('');
  loading.hidden = false;
  submitBtn.disabled = true;
  summary.textContent = '识别中…';

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('region_types', $('regionTypes').value);
  form.append('layout_threshold', layoutThreshold.value);
  form.append('ocr_threshold', ocrThreshold.value);
  form.append('include_text', String($('includeText').checked));
  form.append('text_in_region', $('textInRegion').value);
  form.append('region_association', $('association').value);
  form.append('return_markdown', String($('returnMarkdown').checked));

  try {
    const response = await fetch('/v1/analyze', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '识别失败');
    lastResult = data;
    renderResult(data);
    drawOverlay(data);
    const orientation = data.orientation || {};
    const orientationText = orientation.corrected
      ? `方向 ${orientation.detected_angle}° 已矫正`
      : orientation.low_confidence
        ? `方向 ${orientation.detected_angle || 0}° 置信度低，未矫正`
        : `方向 ${orientation.detected_angle || 0}°`;
    summary.textContent = `${orientationText} · 区域 ${data.regions.length} 个 · 文字 ${data.texts.length} 条 · ${data.timings_ms.layout_ms || 0} + ${data.timings_ms.ocr_ms || 0} ms`;
  } catch (error) {
    setError(error.message);
    summary.textContent = '识别失败';
  } finally {
    loading.hidden = true;
    submitBtn.disabled = false;
  }
}

submitBtn.addEventListener('click', runAnalysis);

function renderResult(result) {
  const textRows = $('textRows');
  textRows.innerHTML = '';
  result.texts.forEach((item) => {
    const row = document.createElement('tr');
    appendCell(row, item.id);
    appendCell(row, item.text);
    appendCell(row, item.score.toFixed(3));
    appendCell(row, item.bbox.map((v) => Math.round(v)).join(', '));
    appendCell(row, item.region ? `${item.region.id} ${item.region.type} (${item.region.relation})` : '-');
    textRows.appendChild(row);
  });
  if (!result.texts.length) appendEmptyRow(textRows, 5, '未识别到文字');

  const regionRows = $('regionRows');
  regionRows.innerHTML = '';
  result.regions.forEach((item) => {
    const row = document.createElement('tr');
    appendCell(row, item.id);
    appendCell(row, item.type);
    appendCell(row, item.score.toFixed(3));
    appendCell(row, item.bbox.map((v) => Math.round(v)).join(', '));
    regionRows.appendChild(row);
  });
  if (!result.regions.length) appendEmptyRow(regionRows, 4, '未检测到图片或图形区域');

  $('markdownOutput').textContent = result.layout_markdown || '（空）';
  $('jsonOutput').textContent = JSON.stringify(result, null, 2);
}

function appendCell(row, value) {
  const cell = document.createElement('td');
  cell.textContent = String(value);
  row.appendChild(cell);
}

function appendEmptyRow(tbody, columns, text) {
  const row = document.createElement('tr');
  const cell = document.createElement('td');
  cell.colSpan = columns;
  cell.textContent = text;
  cell.style.color = '#68738a';
  row.appendChild(cell);
  tbody.appendChild(row);
}

function drawOverlay(result) {
  if (!result || preview.hidden) return;
  const rect = preview.getBoundingClientRect();
  const wrapRect = $('imageWrap').getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.style.left = `${preview.offsetLeft}px`;
  overlay.style.top = `${preview.offsetTop}px`;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  overlay.hidden = false;

  const ctx = overlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const sx = rect.width / result.image.width;
  const sy = rect.height / result.image.height;

  result.regions.forEach((region) => {
    const [x1, y1, x2, y2] = region.bbox;
    const color = regionColors[region.type] || '#38bdf8';
    ctx.strokeStyle = color;
    ctx.fillStyle = `${color}22`;
    ctx.lineWidth = 2;
    ctx.fillRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillStyle = color;
    ctx.font = '12px sans-serif';
    ctx.fillText(`${region.type} ${region.id}`, x1 * sx + 4, Math.max(12, y1 * sy - 4));
  });

  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 1;
  result.texts.forEach((text) => {
    const [x1, y1, x2, y2] = text.bbox;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  });

  void wrapRect;
}

window.addEventListener('resize', () => {
  if (lastResult) drawOverlay(lastResult);
});

checkHealth();
