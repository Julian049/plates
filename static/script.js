const fileInput     = document.getElementById('image');
const fileLabelText = document.getElementById('file-label-text');
const fileHint      = document.getElementById('file-hint');
const form          = document.getElementById('upload-form');
const submitBtn     = document.getElementById('submit-btn');
const fileLabel     = document.querySelector('.file-label');

/* ── File selection feedback ── */
if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
      fileLabelText.textContent = 'Selecciona o arrastra una imagen aquí';
      return;
    }

    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    fileLabelText.textContent = `${file.name}  ·  ${sizeMB} MB`;

    if (file.size > 5 * 1024 * 1024) {
      fileHint.textContent = '⚠ El archivo supera el límite de 5 MB.';
      fileHint.style.color = '#ff5b5b';
    } else {
      fileHint.textContent = '✓ Imagen lista para analizar.';
      fileHint.style.color = '#3bffb0';
    }
  });
}

/* ── Drag & drop ── */
if (fileLabel) {
  ['dragenter', 'dragover'].forEach(evt =>
    fileLabel.addEventListener(evt, (e) => {
      e.preventDefault();
      fileLabel.style.borderColor = 'rgba(59,255,176,0.65)';
      fileLabel.style.background  = 'rgba(59,255,176,0.06)';
    })
  );

  ['dragleave', 'drop'].forEach(evt =>
    fileLabel.addEventListener(evt, () => {
      fileLabel.style.borderColor = '';
      fileLabel.style.background  = '';
    })
  );

  fileLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    const dt = e.dataTransfer;
    if (dt && dt.files.length) {
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event('change'));
    }
  });
}

/* ── Submit loading state ── */
if (form && submitBtn) {
  form.addEventListener('submit', () => {
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
  });
}

/* ── Scroll-reveal for image cards ── */
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll('.image-card').forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(28px)';
  card.style.transition = `opacity 0.55s ${i * 0.12}s ease, transform 0.55s ${i * 0.12}s ease`;
  observer.observe(card);
});