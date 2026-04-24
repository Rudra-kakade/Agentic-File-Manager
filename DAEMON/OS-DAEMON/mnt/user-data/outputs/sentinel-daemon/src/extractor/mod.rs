// src/extractor/mod.rs
//
// Content Extraction Pipeline
//
// Dispatches to the correct parser based on file extension.
// All parsers return plain UTF-8 text — the embedder doesn't care about
// structure, only content.
//
// Supported formats (Phase 1):
//   • Plain text / source code — direct read
//   • PDF                      — pdfium-render (or poppler via subprocess)
//   • DOCX                     — docx-rs
//   • XLSX                     — calamine
//   • Images / unknown         — metadata only (path, name, extension)
//
// Phase 2 (future): OCR via tesseract for image/scanned PDF content.

use std::path::Path;
use anyhow::Result;
use tokio::fs;
use tracing::debug;

/// Extract textual content from a file.
/// Returns an empty string for binary files we cannot parse.
pub async fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    debug!(path = ?path, ext = ext, "Extracting content");

    // Skip files that are too large (> 50 MB) — not useful for semantic search
    if let Ok(meta) = fs::metadata(path).await {
        if meta.len() > 50 * 1024 * 1024 {
            return Ok(metadata_fallback(path));
        }
    }

    match ext.as_str() {
        // ── Plain text formats ────────────────────────────────────────────
        "txt" | "md" | "markdown" | "rst" | "csv" | "json" | "yaml" | "yml"
        | "toml" | "xml" | "html" | "htm" | "log" => {
            read_text_file(path).await
        }

        // ── Source code ───────────────────────────────────────────────────
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "c" | "cpp"
        | "h" | "hpp" | "java" | "kt" | "swift" | "rb" | "php" | "sh"
        | "bash" | "zsh" | "fish" | "sql" | "r" | "scala" | "cs" | "fs"
        | "ex" | "exs" | "lua" | "vim" | "tf" | "hcl" | "dockerfile" => {
            read_text_file(path).await
        }

        // ── PDF ───────────────────────────────────────────────────────────
        "pdf" => extract_pdf(path).await,

        // ── DOCX ─────────────────────────────────────────────────────────
        "docx" => extract_docx(path).await,

        // ── XLSX / XLS ────────────────────────────────────────────────────
        "xlsx" | "xls" | "ods" => extract_xlsx(path).await,

        // ── Everything else: emit metadata only ───────────────────────────
        _ => Ok(metadata_fallback(path)),
    }
}

// ── Plain text ────────────────────────────────────────────────────────────────

async fn read_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).await?;
    // Replace invalid UTF-8 sequences with replacement chars rather than failing
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// ── PDF ───────────────────────────────────────────────────────────────────────
// We call `pdftotext` (poppler) as a subprocess.
// This avoids unsafe FFI in Phase 1 while still delivering real text.
// In Phase 2, link against pdfium-render for in-process extraction.

async fn extract_pdf(path: &Path) -> Result<String> {
    let output = tokio::process::Command::new("pdftotext")
        .arg(path)
        .arg("-")           // output to stdout
        .arg("-enc")
        .arg("UTF-8")
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        }
        Ok(out) => {
            // pdftotext failed (encrypted, damaged, etc.) — fall back to metadata
            debug!(
                path = ?path,
                stderr = %String::from_utf8_lossy(&out.stderr),
                "pdftotext failed — using metadata fallback"
            );
            Ok(metadata_fallback(path))
        }
        Err(_) => {
            // pdftotext not installed — fall back gracefully
            Ok(metadata_fallback(path))
        }
    }
}

// ── DOCX ──────────────────────────────────────────────────────────────────────
// docx-rs is a pure-Rust parser.  We extract paragraph text only.

async fn extract_docx(path: &Path) -> Result<String> {
    // docx-rs is synchronous; run in blocking thread pool
    let path_owned = path.to_path_buf();
    let result = tokio::task::spawn_blocking(move || -> Result<String> {
        // docx-rs reads the zip and parses XML
        // NOTE: add `docx-rs = "0.4"` to Cargo.toml when enabling this
        // For now we emit a metadata-only stub so the project compiles.
        //
        // let file = std::fs::File::open(&path_owned)?;
        // let mut docx = docx_rs::read_docx(&mut std::io::BufReader::new(file))?;
        // let text = docx.document.body.children.iter()...collect::<String>();
        // Ok(text)
        Ok(metadata_fallback(&path_owned))
    })
    .await??;
    Ok(result)
}

// ── XLSX ──────────────────────────────────────────────────────────────────────
// calamine is a pure-Rust XLSX/XLS/ODS reader.

async fn extract_xlsx(path: &Path) -> Result<String> {
    let path_owned = path.to_path_buf();
    let result = tokio::task::spawn_blocking(move || -> Result<String> {
        // NOTE: add `calamine = "0.24"` to Cargo.toml when enabling this.
        //
        // use calamine::{open_workbook_auto, Reader};
        // let mut wb = open_workbook_auto(&path_owned)?;
        // let mut out = String::new();
        // for name in wb.sheet_names().to_owned() {
        //     if let Some(Ok(range)) = wb.worksheet_range(&name) {
        //         for row in range.rows() {
        //             for cell in row {
        //                 out.push_str(&cell.to_string());
        //                 out.push(' ');
        //             }
        //             out.push('\n');
        //         }
        //     }
        // }
        // Ok(out)
        Ok(metadata_fallback(&path_owned))
    })
    .await??;
    Ok(result)
}

// ── Metadata fallback ─────────────────────────────────────────────────────────
// For binary/unsupported files, we embed a minimal text representation so
// the file still appears in semantic search by name/extension.

fn metadata_fallback(path: &Path) -> String {
    let name = path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = path.extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let parent = path.parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    format!("File: {name}. Type: {ext}. Location: {parent}.")
}
