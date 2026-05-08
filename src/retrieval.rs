//! Knowledge retrieval records, provenance, and fixed-capacity stores.
//!
//! Retrieval results include provenance and confidence metadata as required by
//! the cognition and cognitive-device specifications.

use crate::{CognitionError, CognitionResult, DataClass};

/// Maximum namespace length.
pub const MAX_NAMESPACE_LEN: usize = 96;

/// Maximum record key length.
pub const MAX_RECORD_KEY_LEN: usize = 128;

/// Maximum record text length for this skeleton.
pub const MAX_RECORD_TEXT_LEN: usize = 2048;

/// Maximum provenance source length.
pub const MAX_PROVENANCE_SOURCE_LEN: usize = 160;

/// Maximum retrieval hits returned by one query.
pub const MAX_RETRIEVAL_HITS: usize = 16;

/// Knowledge record type.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeRecordKind {
    /// Typed fact suitable for deterministic lookup.
    Fact = 1,
    /// Semantic text chunk suitable for vector-like retrieval.
    SemanticChunk = 2,
    /// System instruction or policy note.
    Instruction = 3,
    /// Corpus-derived example.
    CorpusExample = 4,
}

impl KnowledgeRecordKind {
    /// Stable record kind label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::SemanticChunk => "semantic_chunk",
            Self::Instruction => "instruction",
            Self::CorpusExample => "corpus_example",
        }
    }
}

/// Source provenance for a knowledge record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Provenance<'a> {
    /// Human-readable source identifier.
    pub source: &'a str,
    /// License or policy label.
    pub license: &'a str,
    /// Stable content digest when available.
    pub digest: [u8; 32],
    /// Ingest sequence or snapshot generation.
    pub generation: u64,
}

impl<'a> Provenance<'a> {
    /// Creates provenance metadata.
    pub const fn new(source: &'a str, license: &'a str, digest: [u8; 32]) -> Self {
        Self {
            source,
            license,
            digest,
            generation: 0,
        }
    }

    /// Sets ingest generation.
    pub const fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    /// Validates provenance metadata.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.source.is_empty() || self.license.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.source.len() > MAX_PROVENANCE_SOURCE_LEN
            || self.license.len() > MAX_PROVENANCE_SOURCE_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        Ok(())
    }
}

/// A knowledge record visible to retrieval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeRecord<'a> {
    /// Stable record identifier.
    pub id: u64,
    /// Namespace or memory scope.
    pub namespace: &'a str,
    /// Stable lookup key.
    pub key: &'a str,
    /// Record text or redacted summary.
    pub text: &'a str,
    /// Record kind.
    pub kind: KnowledgeRecordKind,
    /// Source provenance.
    pub provenance: Provenance<'a>,
    /// Confidence in basis points, 0..=10000.
    pub confidence_bps: u16,
    /// Embedding/vector dimensions when relevant.
    pub vector_dimensions: u16,
    /// Data sensitivity class.
    pub data_class: DataClass,
}

impl<'a> KnowledgeRecord<'a> {
    /// Creates a knowledge record.
    pub const fn new(
        id: u64,
        namespace: &'a str,
        key: &'a str,
        text: &'a str,
        kind: KnowledgeRecordKind,
        provenance: Provenance<'a>,
    ) -> Self {
        Self {
            id,
            namespace,
            key,
            text,
            kind,
            provenance,
            confidence_bps: 10_000,
            vector_dimensions: 0,
            data_class: DataClass::Operational,
        }
    }

    /// Sets confidence in basis points.
    pub const fn with_confidence(mut self, confidence_bps: u16) -> Self {
        self.confidence_bps = confidence_bps;
        self
    }

    /// Sets vector dimensions.
    pub const fn with_vector_dimensions(mut self, vector_dimensions: u16) -> Self {
        self.vector_dimensions = vector_dimensions;
        self
    }

    /// Sets data classification.
    pub const fn with_data_class(mut self, data_class: DataClass) -> Self {
        self.data_class = data_class;
        self
    }

    /// Validates required fields, limits, confidence, and provenance.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.id == 0 || self.namespace.is_empty() || self.key.is_empty() || self.text.is_empty()
        {
            return Err(CognitionError::MissingField);
        }
        if self.namespace.len() > MAX_NAMESPACE_LEN
            || self.key.len() > MAX_RECORD_KEY_LEN
            || self.text.len() > MAX_RECORD_TEXT_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        if self.confidence_bps > 10_000 {
            return Err(CognitionError::InvalidRetrieval);
        }
        self.provenance.validate()
    }
}

/// Retrieval query over knowledge records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetrievalQuery<'a> {
    /// Namespace or memory scope.
    pub namespace: &'a str,
    /// Query text or lookup key.
    pub query: &'a str,
    /// Maximum number of returned hits.
    pub top_k: usize,
    /// Minimum confidence in basis points.
    pub min_confidence_bps: u16,
    /// Highest data class allowed in returned records.
    pub max_data_class: DataClass,
    /// Include semantic chunks.
    pub include_semantic: bool,
    /// Include typed facts.
    pub include_facts: bool,
}

impl<'a> RetrievalQuery<'a> {
    /// Creates a retrieval query with safe defaults.
    pub const fn new(namespace: &'a str, query: &'a str) -> Self {
        Self {
            namespace,
            query,
            top_k: 4,
            min_confidence_bps: 0,
            max_data_class: DataClass::Operational,
            include_semantic: true,
            include_facts: true,
        }
    }

    /// Sets the maximum hit count.
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets minimum confidence.
    pub const fn with_min_confidence(mut self, min_confidence_bps: u16) -> Self {
        self.min_confidence_bps = min_confidence_bps;
        self
    }

    /// Sets maximum returned data class.
    pub const fn with_max_data_class(mut self, max_data_class: DataClass) -> Self {
        self.max_data_class = max_data_class;
        self
    }

    /// Validates query limits and fields.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.namespace.is_empty() || self.query.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.namespace.len() > MAX_NAMESPACE_LEN || self.query.len() > MAX_RECORD_TEXT_LEN {
            return Err(CognitionError::FieldTooLong);
        }
        if self.top_k == 0 || self.top_k > MAX_RETRIEVAL_HITS || self.min_confidence_bps > 10_000 {
            return Err(CognitionError::InvalidRetrieval);
        }
        Ok(())
    }
}

/// One retrieval hit with provenance and confidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetrievalHit<'a> {
    /// Retrieved record.
    pub record: KnowledgeRecord<'a>,
    /// Query score in basis points.
    pub score_bps: u16,
    /// One-based rank.
    pub rank: u16,
}

impl<'a> RetrievalHit<'a> {
    /// Creates a retrieval hit.
    pub const fn new(record: KnowledgeRecord<'a>, score_bps: u16, rank: u16) -> Self {
        Self {
            record,
            score_bps,
            rank,
        }
    }

    /// Validates score, rank, and record metadata.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.score_bps > 10_000 || self.rank == 0 {
            return Err(CognitionError::InvalidRetrieval);
        }
        self.record.validate()
    }
}

/// Fixed-capacity retrieval result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalResult<'a, const N: usize> {
    hits: [Option<RetrievalHit<'a>>; N],
    len: usize,
    truncated: bool,
}

impl<'a, const N: usize> RetrievalResult<'a, N> {
    /// Creates an empty retrieval result.
    pub const fn new() -> Self {
        Self {
            hits: [None; N],
            len: 0,
            truncated: false,
        }
    }

    /// Returns number of hits.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no hits were returned.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` when result capacity truncated matches.
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Adds a hit.
    pub fn push(&mut self, hit: RetrievalHit<'a>) -> CognitionResult<()> {
        hit.validate()?;
        if self.len == N {
            self.truncated = true;
            return Err(CognitionError::CapacityExceeded);
        }
        self.hits[self.len] = Some(hit);
        self.len += 1;
        Ok(())
    }

    /// Marks the result truncated.
    pub fn mark_truncated(&mut self) {
        self.truncated = true;
    }

    /// Returns a hit by zero-based index.
    pub fn hit(&self, index: usize) -> Option<&RetrievalHit<'a>> {
        if index < self.len {
            self.hits[index].as_ref()
        } else {
            None
        }
    }

    /// Iterates over hits.
    pub fn iter(&self) -> impl Iterator<Item = &RetrievalHit<'a>> {
        self.hits[..self.len].iter().filter_map(Option::as_ref)
    }
}

impl<'a, const N: usize> Default for RetrievalResult<'a, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-capacity host-mode knowledge store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeStore<'a, const N: usize> {
    records: [Option<KnowledgeRecord<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> KnowledgeStore<'a, N> {
    /// Creates an empty store.
    pub const fn new() -> Self {
        Self {
            records: [None; N],
            len: 0,
        }
    }

    /// Returns number of stored records.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the store is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts a record.
    pub fn put(&mut self, record: KnowledgeRecord<'a>) -> CognitionResult<()> {
        record.validate()?;
        if self.len == N {
            return Err(CognitionError::CapacityExceeded);
        }
        if self.get(record.id).is_some() {
            return Err(CognitionError::DuplicateRecord);
        }
        self.records[self.len] = Some(record);
        self.len += 1;
        Ok(())
    }

    /// Gets a record by id.
    pub fn get(&self, id: u64) -> Option<&KnowledgeRecord<'a>> {
        self.iter().find(|record| record.id == id)
    }

    /// Executes a deterministic host-mode retrieval query.
    pub fn query<const OUT: usize>(
        &self,
        query: RetrievalQuery<'_>,
    ) -> CognitionResult<RetrievalResult<'a, OUT>> {
        query.validate()?;
        let mut result = RetrievalResult::new();
        let mut rank = 1_u16;
        for record in self.iter() {
            if !record_matches(*record, query) {
                continue;
            }
            if result.len() == query.top_k {
                result.mark_truncated();
                break;
            }
            let hit = RetrievalHit::new(*record, record.confidence_bps, rank);
            if result.push(hit).is_err() {
                break;
            }
            rank = rank.saturating_add(1);
        }
        Ok(result)
    }

    /// Iterates over stored records.
    pub fn iter(&self) -> impl Iterator<Item = &KnowledgeRecord<'a>> {
        self.records[..self.len].iter().filter_map(Option::as_ref)
    }
}

impl<'a, const N: usize> Default for KnowledgeStore<'a, N> {
    fn default() -> Self {
        Self::new()
    }
}

fn record_matches(record: KnowledgeRecord<'_>, query: RetrievalQuery<'_>) -> bool {
    if record.namespace != query.namespace {
        return false;
    }
    if record.confidence_bps < query.min_confidence_bps {
        return false;
    }
    if data_class_rank(record.data_class) > data_class_rank(query.max_data_class) {
        return false;
    }
    match record.kind {
        KnowledgeRecordKind::Fact => query.include_facts,
        KnowledgeRecordKind::SemanticChunk
        | KnowledgeRecordKind::Instruction
        | KnowledgeRecordKind::CorpusExample => query.include_semantic,
    }
}

const fn data_class_rank(data_class: DataClass) -> u8 {
    match data_class {
        DataClass::Public => 0,
        DataClass::Operational => 1,
        DataClass::Sensitive => 2,
        DataClass::Secret => 3,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalDescriptor<'a> {
    /// Component name.
    pub name: &'a str,
    /// Component version marker.
    pub version: u32,
}

impl<'a> RetrievalDescriptor<'a> {
    /// Creates a retrieval component descriptor.
    pub const fn new(name: &'a str, version: u32) -> Self {
        Self { name, version }
    }
}
