# AffilGood Library 🔍

AffilGood provides tools and annotated datasets to improve the accuracy of attributing scientific works to research organizations, especially in multilingual and complex contexts. The framework addresses key challenges in institution name disambiguation through a modular pipeline approach.

![AffilGood Pipeline](figure1.png)

## Publication

This is the official repository for the paper ["AffilGood: Building reliable institution name disambiguation tools to improve scientific literature analysis"](https://aclanthology.org/2024.sdp-1.13/), published in the Scholarly Document Processing (SDP) 2024 Workshop at ACL 2024 Conference. Slides used in the presentation are available [here](https://docs.google.com/presentation/d/1wX7zInjoUrjO1hRL3U8tpSzxU6KOX0FknTaEqSf6ML0/edit#slide=id.g2effd47279e_0_22).

## 🌟 Key Features

- **Modular Pipeline Architecture**: Separate components for span identification, named entity recognition, entity linking, and metadata normalization
- **Multilingual Support**: Models trained on data in multiple languages
- **Multiple Entity Linking Options**: Support for various linking strategies including Whoosh-based search and S2AFF integration
- **Location Normalization**: Integration with OpenStreetMap for standardizing geographic data
- **Country Code Normalization**: Mapping of country names to standard codes

## 📚 Documentation

For more detailed information about using and extending AffilGood, check out our documentation:

- [AffilGood Modules](docs/modules.md) - Detailed reference for classes and methods in all modules
- [Usage Examples](docs/usage-examples.md) - Code examples for different use cases
- [Technical Overview](docs/technical-overview.md) - In-depth explanation of system architecture and design
- [Entity Linking Extension Guide](docs/entity-linking-extension.md) - Guide for extending the entity linking module
- [Contribution Guide](docs/contribution-guide.md) - Guidelines for contributing to the project

## 🛠️ Installation

```bash
pip install affilgood
```

Or for development:

```bash
git clone https://github.com/sirisacademic/affilgood.git
cd affilgood
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from affilgood import AffilGood

# Initialize with default settings (Whoosh linker)
affil_good = AffilGood()

# Or customize components
affil_good = AffilGood(
    span_separator='',  # Use model-based span identification
    span_model_path='nicolauduran45/affilgood-span-v2',  # Custom span model
    ner_model_path='nicolauduran45/affilgood-ner-multilingual-v2',  # Custom NER model
    entity_linkers='Whoosh',  # Use Whoosh for entity linking ('S2AFF' also available)
    return_scores=True,  # Return confidence scores with predictions
    metadata_normalization=True,  # Enable location normalization
    verbose=False,  # Detailed logging
    device=None  # Auto-detect device (CPU or CUDA)
)

# Process affiliation strings
affiliations = [
    "Granges Terragrisa SL, Paratge de La Gleva, Camí de Burrissola s/n, E-08508 Les Masies de Voltregà (Barcelona), Catalonia, Spain",
    "Treuman Katz Center for Pediatric Bioethics, Seattle Children's Research Institute, Seattle, WA, USA"
]

# Full pipeline processing (span identification, NER, normalization, entity linking)
results = affil_good.process(affiliations)

# Or use individual components
spans = affil_good.get_span(affiliations)
entities = affil_good.get_ner(spans)
normalized = affil_good.get_normalization(entities)
linked = affil_good.get_entity_linking(normalized)

print(linked)
```

## 📦 Project Structure

The repository is structured as follows:

```
affilgood/
├── __init__.py                   # Package initialization
├── affilgood.py                  # Main AffilGood class implementation
├── span_identification/          # Span identification module
│   ├── span_identifier.py        # Model-based span identification
│   ├── simple_span_identifier.py # Character-based span splitter
│   └── noop_span_identifier.py   # Pass-through identifier for pre-segmented data
├── ner/                          # Named Entity Recognition module
│   └── ner.py                    # NER implementation
├── entity_linking/               # Entity linking module
│   ├── entity_linker.py          # Main entity linking orchestrator
│   ├── base_linker.py            # Base class for entity linkers
│   ├── whoosh_linker.py          # Whoosh-based entity linker
│   ├── s2aff_linker.py           # S2AFF-based entity linker
│   ├── base_reranker.py          # Base class for rerankers
│   ├── llm_reranker.py           # LLM-based reranker for candidate selection
│   └── constants.py              # Constants for entity linking
├── metadata_normalization/       # Metadata normalization module
│   └── normalizer.py             # Location and country normalization
└── utils/                        # Utility functions
    ├── data_manager.py           # Data loading and caching
    ├── text_utils.py             # Text processing utilities
    └── translation_mappings.py   # Institution name translation mappings
```

## 🤗 Pre-trained Models

AffilGood uses several pre-trained models available on Hugging Face:

- 🤗 [nicolauduran45/affilgood-ner-multilingual-v2](https://huggingface.co/nicolauduran45/affilgood-ner-multilingual-v2) - Multilingual NER model
- 🤗 [nicolauduran45/affilgood-span-v2](https://huggingface.co/nicolauduran45/affilgood-span-v2) - Span identification model
- 🤗 [SIRIS-Lab/affilgood-NER](https://huggingface.co/SIRIS-Lab/affilgood-NER) - English NER model
- 🤗 [SIRIS-Lab/affilgood-NER-multilingual](https://huggingface.co/SIRIS-Lab/affilgood-NER-multilingual) - Multilingual NER model
- 🤗 [SIRIS-Lab/affilgood-SPAN](https://huggingface.co/SIRIS-Lab/affilgood-span) - English span model
- 🤗 [SIRIS-Lab/affilgood-span-multilingual](https://huggingface.co/SIRIS-Lab/affilgood-span-multilingual) - Multilingual span model
- 🤗 [SIRIS-Lab/affilgood-affilRoBERTa](https://huggingface.co/SIRIS-Lab/affilgood-affilroberta) - RoBERTa adapted for affiliation data
- 🤗 [SIRIS-Lab/affilgood-affilXLM](https://huggingface.co/SIRIS-Lab/affilgood-affilxlm) - XLM-RoBERTa adapted for affiliation data

## 📊 Performance (from paper)

Note: These results can be outdated as the pipeline is in development and new features are being included.

AffilGood achieves state-of-the-art performance on institution name disambiguation tasks compared to existing systems:

| **Model**                     | **MA** | **FA** | **NRMO** | **S2AFF*** | **CORDIS** | **ETERe** | **ETERm** |
|-------------------------------|--------|--------|----------|------------|------------|-----------|-----------|
| ElasticSearch                 | .545   | .407   | .470     | .515       | .751       | .855      | .847      |
| OpenAlex                      | .394   | .118   | .769     | **.871**🔥  | .648       | .859      | .852      |
| S2AFF                         | .546   | .367   | .617     | .785       | .649       | .668      | .720      |
| AffRo                         | .452   | .408   | .558     | .726       | .641       | .709      | .617      |
| AffilGoodNERm + S2AFF<sub>Linker</sub> | .596 | .685 | .762 | .841       | .827       | .887      | .863      |
| AffilGoodNER + S2AFF<sub>Linker</sub>  | .579 | .685 | .758 | .850       | .839       | .895      | .855      |
| AffilGoodNERm + Elastic       | .690   | .587   | .747     | .640       | .849       | .887      | .894      |
| AffilGoodNER + Elastic        | .649   | .610   | .755     | .648       | .855       | .893      | .881      |
| AffilGoodNERm + Elastic+qLLM  | **.710**🔥 | .721 | **.774**🔥 | .790 | .881       | **.936**🔥 | **.916**🔥 |
| AffilGoodNER + Elastic+qLLM   | .653   | **.747**🔥 | .767 | .799       | **.891**🔥 | **.936**🔥 | .909      |

## 📝 Citation

If you use AffilGood in your research, please cite our paper:

```bibtex
@inproceedings{duran-silva-etal-2024-affilgood,
    title = "{A}ffil{G}ood: Building reliable institution name disambiguation tools to improve scientific literature analysis",
    author = "Duran-Silva, Nicolau  and
      Accuosto, Pablo  and
      Przyby{\l}a, Piotr  and
      Saggion, Horacio",
    editor = "Ghosal, Tirthankar  and
      Singh, Amanpreet  and
      Waard, Anita  and
      Mayr, Philipp  and
      Naik, Aakanksha  and
      Weller, Orion  and
      Lee, Yoonjoo  and
      Shen, Shannon  and
      Qin, Yanxia",
    booktitle = "Proceedings of the Fourth Workshop on Scholarly Document Processing (SDP 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sdp-1.13",
    pages = "135--144",
}
```

## 🙋‍♀️ Contributing

We welcome contributions to the AffilGood project! Instead of a single main branch, we use two branches:

- `develop`: Development and default branch for new features and bug fixes.
- `main`: Production branch used to deploy the server components to the production environment.

Please follow our [Contribution Guidelines](/docs/contribute.md) to participate in this project.

## 📫 Contact

For further information, please contact <nicolau.duransilva@sirisacademic.com>.

## ⚖️ License

This work is distributed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
