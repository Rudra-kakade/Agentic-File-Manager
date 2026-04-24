// src/extractor.rs
//
// Text extraction from files.
// Uses file extension to select the right strategy:
//   - PDF:   shell out to `pdftotext` (poppler-utils)
//   - Image: shell out to `tesseract` (OCR, optional)
//   - Text-like (txt, md, rs, py, …): read directly as UTF-8
//
// All extraction is async and runs on Tokio's blocking thread pool
// because it is I/O- (and sometimes CPU-) bound.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::debug;

/// Maximum file size we're willing to read into memory (16 MB).
const MAX_FILE_SIZE: u64 = 16 * 1024 * 1024;

pub async fn extract_text(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || extract_sync(&path))
        .await
        .context("Extraction task panicked")?
}

fn extract_sync(path: &Path) -> Result<String> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat {}", path.display()))?;

    if meta.len() > MAX_FILE_SIZE {
        anyhow::bail!("File too large ({} bytes, max {})", meta.len(), MAX_FILE_SIZE);
    }

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf(path),
        "png" | "jpg" | "jpeg" | "tiff" | "bmp" | "webp" => extract_ocr(path),
        _ => extract_plain(path),
    }
}

fn extract_plain(path: &Path) -> Result<String> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    debug!(path = ?path, chars = content.len(), "Extracted plain text");
    Ok(content)
}

fn extract_pdf(path: &Path) -> Result<String> {
    let output = std::process::Command::new("pdftotext")
        .arg(path.as_os_str())
        .arg("-")  // write to stdout
        .output()
        .context("Failed to run pdftotext (is poppler-utils installed?)")?;

    if !output.status.success() {
        anyhow::bail!("pdftotext exited with {}", output.status);
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    debug!(path = ?path, chars = text.len(), "Extracted PDF text");
    Ok(text)
}

fn extract_ocr(path: &Path) -> Result<String> {
    let output = std::process::Command::new("tesseract")
        .arg(path.as_os_str())
        .arg("stdout")
        .output()
        .context("Failed to run tesseract (is tesseract-ocr installed?)")?;

    if !output.status.success() {
        anyhow::bail!("tesseract exited with {}", output.status);
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    debug!(path = ?path, chars = text.len(), "Extracted OCR text");
    Ok(text)
}
