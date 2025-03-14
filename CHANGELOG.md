# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Improved Whoosh-based entity linking with support for multiple languages
- LLM-based candidate reranking for more accurate entity matching
- Support for multiple entity linkers with result merging
- Enhanced country and location normalization
- Extensive API documentation and usage examples

### Changed
- Modularized pipeline architecture for better extensibility
- Improved multilingual support with updated models
- More efficient processing of large datasets with parallelization
- Better caching mechanisms for geocoding and entity linking

### Fixed
- Issues with non-Latin script handling in entity recognition
- Memory consumption issues when processing large batches
- Various edge cases in location normalization

## [1.0.0] - 2024-08-15

### Added
- Initial release with basic pipeline functionality
- Support for span identification, NER, and entity linking
- Integration with Research Organization Registry (ROR)
- Basic documentation and usage examples

[Unreleased]: https://github.com/yourusername/affilgood/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/affilgood/releases/tag/v1.0.0
