# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-03-12

### Added

#### Entity Linking — Three-Stage Cascade Pipeline
- **Direct match** (Stage 1): exact name+country lookup against all ROR names, aliases, acronyms, and labels. ~35% coverage at ~98% precision with zero latency.
- **Dense retrieval** (Stage 2): FAISS HNSW index with `SIRIS-Lab/affilgood-dense-retriever` encoder (1024-dim XLM-RoBERTa). Multi-variant queries (ORG+CITY+COUNTRY, ORG+COUNTRY, ORG only) merged by max score. R@1=0.905.
- **Cross-encoder reranking** (Stage 2b): optional reranker (e.g. `cometadata/jina-reranker-v2-multilingual-affiliations-large`) with quality gate threshold (default 0.038 from experiments). Uses entity name as query to avoid multi-entity affiliation confusion.
- **LLM judge** (Stage 3): optional instruction-following LLM for low-confidence cases. First-token logit scoring (FIRST-style) — one forward pass, no generation. Respects "N" (none-of-the-above) probability: rejects when N > best candidate.
- **Reranker quality gate**: hard rejection when reranker's best score < threshold, preventing garbage candidates from reaching the output.

#### Inactive Record Resolution
- DirectMatcher indexes both active and inactive ROR records.
- Successor chain resolution: inactive records (e.g. INRA) automatically resolve to their active successor (e.g. INRAE).
- Works in both direct match and dense retrieval paths.

#### Unicode-Safe Normalization (DirectMatcher)
- Casefold + NFKD decomposition for cross-locale matching ("TÜRKİYE" = "Türkiye" = "turkiye").
- Dot removal: "C.N.R.S." matches "CNRS", "I.N.R.A." matches "INRA".
- Hyphen → space: "INSA-Lyon" matches "INSA Lyon".
- Case-insensitive: "LAmCoS" matches "LAMCOS".

#### Translation for Non-Latin Scripts
- New `AffiliationTranslator` component using instruction-following LLMs (e.g. Qwen2.5-0.5B-Instruct).
- Auto-detects and translates Chinese, Japanese, Korean, Arabic, Russian, Persian, Greek, Thai, Hindi, Ukrainian, and more.
- Configured via `translate_config` parameter.

#### Organization Type Classification
- New `OrganizationTypeClassifier` component with two-level taxonomy.
- Level 1: Company, Healthcare, Higher Education, Research Organization, Public Administration, etc.
- Level 2: HEI.institution, PRC.sme, REC.institute, REC.lab, etc.
- Context from DuckDuckGo search (bing backend with brave fallback) + RoBERTa classifiers (`SIRIS-Lab/acty2de-roberta_lvl1_ctx`, `lvl2_ctx`).
- SQLite cache for DDG queries and classification results.
- Level 2 validation: rejects if lvl2 prefix doesn't match lvl1.
- Output formatting: "HEI.institution" → {"lvl1": "Higher Education", "lvl2": "Institution"}.
- Region-calibrated search (country → DDG region code) for better results.

#### ROR → Geocode Feedback Loop
- When NER misses CITY/COUNTRY but entity linking found a match, uses ROR city+country to geocode via Nominatim.
- Source tagged as `"ror-osm"` to distinguish from NER-based geocoding.

#### Language Detection
- Combined langdetect method for affiliation language identification.
- Configured via `enable_language_detect` and `language_config`.

### Changed
- **Output schema**: `"query"` field renamed to `"raw"` in both institutions and subunits.
- **Output schema**: added `"type"` field to institutions with lvl1/lvl2 classification.
- **Output schema**: location sources now distinguish `"osm"` vs `"ror-osm"` vs `"ner"`.
- **Output schema**: subunits now include `"raw"` field (original NER text).
- **Pipeline architecture**: fully modular seven-stage pipeline (Span → Language → Translation → NER → Entity Linking → Geocoding → Org Type).
- **FAISS index**: uses `METRIC_INNER_PRODUCT` (was defaulting to L2 distance).
- **Reranker query**: uses entity name only, not full affiliation string. Prevents multi-entity affiliations from confusing the reranker.
- **Score handling**: raw reranker scores passed through directly. No fusion, no min-max normalization.
- **`RegistryRecord`**: added `successor_id` field for inactive → active resolution.
- **`RegistryManager`**: both v1 and v2 ROR normalizers now parse successor relationships.
- **Dense index record map**: uses all records (active + inactive) for metadata lookups.
- **`requires-python`**: bumped to >=3.10.

### Removed
- Score fusion (`_fuse_scores`, `score_fusion_alpha` parameter) — replaced by raw reranker scores + quality gate.
- Whoosh-based entity linking (replaced by FAISS dense retrieval).
- `rank-bm25` dependency.
- `whoosh` dependency.

### Fixed
- FAISS HNSW index was built with L2 distance metric instead of inner product, causing inverted score ordering (best matches ranked last).
- Turkish İ/ı normalization in DirectMatcher (casefold + NFKD handles "TÜRKİYE" correctly).
- Reranker confused by multi-entity affiliations (e.g. "Università di Roma Tre AND INFN sezione di Roma Tre" — reranker was matching the wrong entity).
- LLM judge was ignoring "N" (none) probability and always picking a candidate even when no match existed.
- LLM judge fired on garbage candidates after reranker quality gate rejected — now hard rejection prevents this.
- Score inflation: min-max normalization made worst-match-in-set show as 1.0 confidence.
- Device conflict between pipeline and entity linker components.
- `duckduckgo_search` package renamed to `ddgs`.

## [1.0.0] - 2024-08-15

### Added
- Initial release with basic pipeline functionality
- Support for span identification, NER, and entity linking
- Integration with Research Organization Registry (ROR)
- Basic documentation and usage examples

[2.0.0]: https://github.com/sirisacademic/affilgood/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/sirisacademic/affilgood/releases/tag/v1.0.0