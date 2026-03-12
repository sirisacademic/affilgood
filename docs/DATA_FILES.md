# Data Files: Where to Put Them

## Project structure

```
affilgood/
├── __init__.py
├── api.py
├── pipeline.py
├── output.py
├── components/
│   ├── __init__.py
│   ├── span.py
│   ├── ner.py
│   ├── geocoder.py              ← resolves data dir via Path(__file__).parent / "data"
│   └── data/
│       ├── __init__.py           ← required for setuptools to find it
│       ├── country_data.tsv      ← existing file (move here)
│       ├── openrefine-countries-normalized.csv  ← existing file (move here)
│       └── nuts/
│           ├── NUTS_RG_01M_2021_4326.shp
│           ├── NUTS_RG_01M_2021_4326.dbf
│           ├── NUTS_RG_01M_2021_4326.shx
│           └── NUTS_RG_01M_2021_4326.prj
├── pyproject.toml
└── ...
```

## How it works

### Path resolution (geocoder.py)

```python
def _get_data_dir() -> Path:
    return Path(__file__).parent / "data"
```

`__file__` = `.../affilgood/components/geocoder.py`
→ `parent` = `.../affilgood/components/`
→ `/ "data"` = `.../affilgood/components/data/`

This works in both:
- **dev mode** (`pip install -e .`): points to your source tree
- **installed mode** (`pip install affilgood`): points to site-packages

### pyproject.toml (package data)

```toml
[tool.setuptools.package-data]
"affilgood.components.data" = [
    "*.tsv",
    "*.csv",
    "nuts/*.shp",
    "nuts/*.dbf",
    "nuts/*.shx",
    "nuts/*.prj",
    "nuts/*.cpg",
]
```

This tells setuptools to include those files when building the wheel.

### The data/__init__.py

Create an empty file — just needs to exist so setuptools treats
`affilgood.components.data` as a package:

```python
# affilgood/components/data/__init__.py
```

## Migration steps

1. Create `affilgood/components/data/` directory
2. Create empty `affilgood/components/data/__init__.py`
3. Move `country_data.tsv` → `affilgood/components/data/country_data.tsv`
4. Move `openrefine-countries-normalized.csv` → `affilgood/components/data/`
5. Create `affilgood/components/data/nuts/` directory
6. Download NUTS shapefile from Eurostat and place all 4 files there

## NUTS shapefile download

Source: https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/nuts

Direct link (2021, 1:1M, EPSG:4326):
https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_01M_2021_4326.shp.zip

```bash
cd affilgood/components/data/nuts/
wget https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_01M_2021_4326.shp.zip
unzip NUTS_RG_01M_2021_4326.shp.zip
rm NUTS_RG_01M_2021_4326.shp.zip
```

## Overriding at runtime

Users can point to custom data locations:

```python
ag = AffilGood(
    normalization_config={
        "data_dir": "/custom/path/to/data",
        "nuts_shapefile": "/custom/path/to/nuts.shp",
        "cache_dir": "/tmp/affilgood_cache",
    }
)
```

## .gitignore

The NUTS shapefile is ~15MB. You may want to:
- Include it in the package (simpler for users)
- Or gitignore it and document the download step

```gitignore
# Option: exclude from git, include in sdist via MANIFEST.in
affilgood/components/data/nuts/*.shp
affilgood/components/data/nuts/*.dbf
affilgood/components/data/nuts/*.shx
affilgood/components/data/nuts/*.prj
```

If excluded from git, add a `MANIFEST.in` for source distributions:
```
recursive-include affilgood/components/data *.tsv *.csv
recursive-include affilgood/components/data/nuts *.shp *.dbf *.shx *.prj
```