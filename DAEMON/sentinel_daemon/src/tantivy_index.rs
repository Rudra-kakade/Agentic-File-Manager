// src/tantivy_index.rs
//
// Lexical index (BM25) — Tier 3 fallback retrieval engine.

use anyhow::Result;
use std::path::Path;
use tantivy::{
    doc,
    schema::{Schema, TEXT, STORED, STRING},
    Index, IndexWriter, TantivyDocument,
    query::QueryParser,
    collector::TopDocs,
};

pub struct LexicalIndex {
    index:  Index,
    writer: IndexWriter,
    // Schema fields
    pub f_path:    tantivy::schema::Field,
    pub f_name:    tantivy::schema::Field,
    pub f_content: tantivy::schema::Field,
}

impl LexicalIndex {
    pub fn open_or_create(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let mut schema_builder = Schema::builder();
        let f_path    = schema_builder.add_text_field("path",    STRING | STORED);
        let f_name    = schema_builder.add_text_field("name",    TEXT   | STORED);
        let f_content = schema_builder.add_text_field("content", TEXT);
        let schema    = schema_builder.build();

        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema)?
        };

        let writer = index.writer(50_000_000)?; // 50 MB heap
        Ok(Self { index, writer, f_path, f_name, f_content })
    }

    /// Upsert a document. Tantivy doesn't support true upsert — we
    /// delete-by-path then re-add. Safe because we hold the writer lock.
    pub fn upsert(&mut self, path: &str, name: &str, content: &str) -> Result<()> {
        use tantivy::Term;
        let term = Term::from_field_text(self.f_path, path);
        self.writer.delete_term(term);
        self.writer.add_document(doc!(
            self.f_path    => path,
            self.f_name    => name,
            self.f_content => content,
        ))?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        Ok(())
    }

    /// BM25 search — Tier 3 fallback retrieval.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<String>> {
        let reader  = self.index.reader()?;
        let searcher = reader.searcher();
        let qp = QueryParser::for_index(
            &self.index,
            vec![self.f_name, self.f_content],
        );
        let query   = qp.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
        let mut results = Vec::new();
        for (_score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = doc.get_first(self.f_path) {
                if let Some(s) = v.as_str() {
                    results.push(s.to_string());
                }
            }
        }
        Ok(results)
    }
}
