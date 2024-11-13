# AffilGood Library 🕺🏿

AffilGood provides annotated datasets and tools to improve the accuracy of attributing scientific works to research organizations, especially in multilingual and complex contexts. The accurate attribution of scientific works to research organizations is hindered by the lack of openly available manually annotated data--in particular when multilingual and complex affiliation strings are considered. The AffilGood framework introduced in this paper addresses this gap. We identify three sub-tasks relevant for institution name disambiguation and make available annotated datasets and tools aimed at each of them, including i) a dataset annotated with affiliation spans in noisy automatically-extracted strings; ii) a dataset annotated with named entities for the identification of organizations and their locations; iii) seven datasets annotated with ROR identifiers for the evaluation of entity-linking systems. In addition, we describe, evaluate and make available newly developed tools that use these datasets to provide solutions for each of the identified sub-tasks. Our results confirm the value of the developed resources and methods in addressing key challenges in institution name disambiguation.

This is the official repository for the paper ["AffilGood: Building reliable institution name disambiguation tools to improve scientific literature analysis"](https://aclanthology.org/2024.sdp-1.13/), published in the Scholarly Document Processing (SDP) 2024 Workshop at ACL 2024 Conference. Slides used in the presentation are available [here](https://docs.google.com/presentation/d/1wX7zInjoUrjO1hRL3U8tpSzxU6KOX0FknTaEqSf6ML0/edit#slide=id.g2effd47279e_0_22).

![Figure 1](figure1.png)

## Project Structure

The repository is structured as follows:

```
affilgood/
├── __init__.py               # Initializes the library and imports the AffilGood class.
├── span_identification/
│   ├── span_identifier.py    # SpanIdentifier class for span identification.
├── named_entity_recognition/
│   ├── ner.py                # NER class for named entity recognition.
├── entity_linking/
│   ├── entity_linker.py      # EntityLinker class for entity linking.
├── metadata_normalization/
│   ├── normalizer.py         # Normalizer class for metadata normalization.
└── affilgood.py              # AffilGood main class with process, get_span, and get_ner methods.
```

### Modules


## How It Works

```python
from affilgood import AffilGood

# Instantiate the AffilGood class with required configurations
affilgood = AffilGood(
    span_model_path="path/to/span/model",
    ner_model_path="path/to/ner/model",
    linker_model="path/to/linker/model",
    normalization_rules={"rule1": "value1"},  # Example normalization rules
    device="cuda"
)

# Run the full pipeline
result = affilgood.process("Sample input string or list of affiliation texts")

# Run individual steps if needed
spans = affilgood.get_span("Sample input string")
entities = affilgood.get_ner(spans)
linked_entities = affilgood.get_entity_linking(entities)
normalized_data = affilgood.get_normalization(linked_entities)
```

## 🤗 Models are available at HuggingFace

- 🤗 [SIRIS-Lab/affilgood-NER](https://huggingface.co/SIRIS-Lab/affilgood-NER)
- 🤗 [SIRIS-Lab/affilgood-NER-multilingual](https://huggingface.co/SIRIS-Lab/affilgood-NER-multilingual)
- 🤗 [SIRIS-Lab/affilgood-SPAN](https://huggingface.co/SIRIS-Lab/affilgood-span)
- 🤗 [SIRIS-Lab/affilgood-span-multilingual](https://huggingface.co/SIRIS-Lab/affilgood-span-multilingual)
- 🤗 [SIRIS-Lab/affilgood-affilRoBERTa](https://huggingface.co/SIRIS-Lab/affilgood-affilroberta)
- 🤗 [SIRIS-Lab/affilgood-affilXLM](https://huggingface.co/SIRIS-Lab/affilgood-affilxlm)

## 📝 Results
### Pipeline
Pipeline (NER+EL) results, evaluated by example-based F1-score. AffilGood-NER-multilingual correspond to the best-performing fine-tuned NER model with adapted XLM-RoBERTa, and AffilGood-NER, to the best with adapted English RoBERTa. Entities in pre-segmented datasets have concatenated with coma-separator.  

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

Disclaimer: we cannot guarantee that any of the baseline systems, such as OpenAlex or S2AFF, used samples from the original version of the S2AFF dataset for training, since it is open.

## 📣 Citation
```
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

## Dependencies

Ensure you have all necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## 🙋‍♀️ Contribute to this project

Instead of a single main branch, we use two branches to record the history of the project:

- `develop`: development and default branch for new features and bug fixes.
- `main`: production branch is used to deploy the server components to the `production` environment.

Please, follow the 📗 [Contribution guidelines](/docs/contribute.md) in order to participate to this project.

### 🛟 A note on language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).


## 📝 Changelog

All notable changes to this project will be documented in the 📝 [CHANGELOG](CHANGELOG.md) file, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 📫 Contact

For further information, please contact <nicolau.duransilva@sirisacademic.com>.

## ⚖️ License

This work is distributed under a [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
