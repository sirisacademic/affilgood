"""
Output normalization for AffilGood.

Converts a raw pipeline item (one affiliation span) into
the stable, user-facing output schema.

Design principles:
- Pure (no side effects)
- Defensive (never crashes)
- Schema-driven (no assumptions about which components ran)
- OSM data preferred over NER for location fields when available
- Clean output: None-only fields are stripped

Location hierarchy (from OSM address fields):
- region   ← STATE     (Catalonia, Nouvelle-Aquitaine, Washington)
- province ← PROVINCE or STATE_DISTRICT (Girona, Barcelona)
- county   ← COUNTY    (Upper Empordà, Pyrénées-Atlantiques, Barcelonès)
- city     ← CITY      (Barcelona, Pau)
"""

from typing import Dict, Any, Optional, List, Tuple


def _format_org_type(org_type: Optional[Dict]) -> Optional[Dict]:
    """
    Format org type for output.

    "HEI.institution" → "Institution"
    "PRC.sme"         → "Sme"
    "EDU.primary"     → "Primary"
    "Individual"      → "Individual"
    "Unknown"         → "Unknown"
    """
    if not org_type:
        return None

    result = {}
    lvl1 = org_type.get("lvl1")
    lvl2 = org_type.get("lvl2")

    if lvl1:
        result["lvl1"] = lvl1

    if lvl2:
        # Strip prefix: "HEI.institution" → "institution", "Individual" → "Individual"
        if "." in lvl2:
            lvl2_clean = lvl2.split(".", 1)[1]
        else:
            lvl2_clean = lvl2
        # Title case, replace underscores with spaces
        result["lvl2"] = lvl2_clean.replace("_", " ").title()
    else:
        result["lvl2"] = None

    return result if result else None


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw pipeline item (one affiliation span) into
    normalized public output.
    """
    score_lookup = _build_score_lookup(raw)

    institutions = _extract_institutions(raw, score_lookup)
    subunits = _extract_subunits(raw, score_lookup)

    return {
        "input": raw.get("raw_text", ""),
        "institutions": institutions,
        "subunits": subunits,
        "location": _extract_location(raw, score_lookup),
        "language": _extract_language(raw),
        "confidence": _extract_confidence(institutions),
    }


# ------------------------------------------------------------------
# Score lookup
# ------------------------------------------------------------------

def _build_score_lookup(raw: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    ner_raw = raw.get("ner_raw", [])
    if not isinstance(ner_raw, list):
        return lookup
    for span_raw in ner_raw:
        if not isinstance(span_raw, list):
            continue
        for entity in span_raw:
            if not isinstance(entity, dict):
                continue
            group = entity.get("entity_group")
            word = entity.get("word")
            score = entity.get("score")
            if group and word and score is not None:
                try:
                    lookup[(group, word)] = float(score)
                except (TypeError, ValueError):
                    pass
    return lookup


def _get_score(
    score_lookup: Dict[Tuple[str, str], float],
    entity_group: str,
    name: str,
) -> Optional[float]:
    score = score_lookup.get((entity_group, name))
    if score is not None:
        return round(score, 6)
    return None


# ------------------------------------------------------------------
# Extractors
# ------------------------------------------------------------------

def _extract_institutions(
    raw: Dict[str, Any],
    score_lookup: Dict[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    institutions: List[Dict[str, Any]] = []

    # 1. Entity linking (preferred when match found)
    entity_linking = raw.get("entity_linking", {})
    if isinstance(entity_linking, dict):
        linked = entity_linking.get("institutions", [])
        if isinstance(linked, list) and linked:
            for inst in linked:
                if not isinstance(inst, dict):
                    continue

                # Handle rich id format: id={ror_id, ror_name, ...} or legacy id=str
                id_field = inst.get("id")
                if isinstance(id_field, dict):
                    # New format: id is a dict with ror_id, ror_name, etc.
                    name = id_field.get("ror_name") or inst.get("query", "")
                    record_id = id_field
                else:
                    # Legacy format or no match: id is str or None
                    name = inst.get("name") or inst.get("query", "")
                    record_id = id_field

                if not name and not inst.get("query"):
                    continue

                confidence = inst.get("score") or inst.get("confidence")
                org_type = inst.get("type")  # {"lvl1": ..., "lvl2": ...} or None
                entry = {
                    "name": name or inst.get("query", ""),
                    "raw": inst.get("query"),
                    "id": record_id,
                    "confidence": confidence,
                    "source": inst.get("source", "entity_linking"),
                }
                if org_type:
                    entry["type"] = _format_org_type(org_type)
                institutions.append(entry)
            if institutions:
                return institutions

    # 2. Fallback: NER ORG entities
    ner = raw.get("ner", [])
    if not isinstance(ner, list):
        return institutions
    for span_ner in ner:
        if not isinstance(span_ner, dict):
            continue
        orgs = span_ner.get("ORG", [])
        if not isinstance(orgs, list):
            continue
        for org in orgs:
            if not org:
                continue
            institutions.append({
                "name": org,
                "id": None,
                "confidence": _get_score(score_lookup, "ORG", org),
                "source": "ner",
            })
    return institutions


def _extract_subunits(
    raw: Dict[str, Any],
    score_lookup: Dict[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    subunits: List[Dict[str, Any]] = []

    # 1. Entity linking subunits (when available)
    entity_linking = raw.get("entity_linking", {})
    if isinstance(entity_linking, dict):
        linked_subs = entity_linking.get("subunits", [])
        if isinstance(linked_subs, list) and linked_subs:
            for sub in linked_subs:
                if not isinstance(sub, dict):
                    continue

                id_field = sub.get("id")
                if isinstance(id_field, dict):
                    name = id_field.get("ror_name") or sub.get("query", "")
                    record_id = id_field
                else:
                    name = sub.get("name") or sub.get("query", "")
                    record_id = id_field

                if not name and not sub.get("query"):
                    continue

                confidence = sub.get("score") or sub.get("confidence")
                entry = {
                    "name": name or sub.get("query", ""),
                    "raw": sub.get("query"),
                    "id": record_id,
                    "confidence": confidence,
                    "source": sub.get("source", "entity_linking"),
                }
                org_type = None # sub.get("type")
                if org_type:
                    entry["type"] = _format_org_type(org_type)
                subunits.append(entry)
            if subunits:
                return subunits

    # 2. Fallback: NER SUB entities
    ner = raw.get("ner", [])
    if not isinstance(ner, list):
        return subunits
    for span_ner in ner:
        if not isinstance(span_ner, dict):
            continue
        subs = span_ner.get("SUB", [])
        if not isinstance(subs, list):
            continue
        for sub in subs:
            if not sub:
                continue
            subunits.append({
                "name": sub,
                "confidence": _get_score(score_lookup, "SUB", sub),
                "source": "ner",
            })
    return subunits


def _extract_location(
    raw: Dict[str, Any],
    score_lookup: Dict[Tuple[str, str], float],
) -> Optional[Dict[str, Any]]:
    """
    Extract location information.

    Priority: OSM > NER for each field.

    OSM address field mapping:
        STATE          → region   (Catalonia, Nouvelle-Aquitaine)
        PROVINCE / STATE_DISTRICT → province (Girona, Barcelona)
        COUNTY         → county   (Upper Empordà, Pyrénées-Atlantiques)
        CITY           → city     (Barcelona, Pau)
        COUNTRY        → country  (Spain — already name_short)
        COUNTRY_CODE   → country_code (ESP)
        CONTINENT      → continent (Europe)
        UN_REGION      → un_region (Southern Europe)
    """
    # Working dict — all possible fields
    location: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    scores: Dict[str, Optional[float]] = {}

    # -------------------------------------------------
    # 1. OSM (preferred — geocoded, normalized data)
    # -------------------------------------------------
    osm = raw.get("osm", [])
    if isinstance(osm, list) and osm:
        entry = osm[0]
        if isinstance(entry, dict):
            # Source label: "ror-osm" when geocoded from ROR data, "osm" from NER
            src = entry.get("_source_type", "osm")

            # City
            osm_city = entry.get("CITY")
            if osm_city:
                location["city"] = osm_city
                sources["city"] = src

            # County
            osm_county = entry.get("COUNTY")
            if osm_county:
                location["county"] = osm_county
                sources["county"] = src

            # Province: PROVINCE first, then STATE_DISTRICT as fallback
            osm_province = entry.get("PROVINCE") or entry.get("STATE_DISTRICT")
            if osm_province:
                location["province"] = osm_province
                sources["province"] = src

            # Region: STATE (the main administrative division)
            osm_region = entry.get("STATE")
            if osm_region:
                location["region"] = osm_region
                sources["region"] = src

            # Country (already normalized to name_short by geocoder)
            osm_country = entry.get("COUNTRY")
            if osm_country:
                location["country"] = osm_country
                sources["country"] = src

            # Country code
            osm_code = entry.get("COUNTRY_CODE")
            if osm_code:
                location["country_code"] = osm_code
                sources["country_code"] = src

            # Continent
            osm_continent = entry.get("CONTINENT")
            if osm_continent:
                location["continent"] = osm_continent
                sources["continent"] = src

            # UN region
            osm_un_region = entry.get("UN_REGION")
            if osm_un_region:
                location["un_region"] = osm_un_region
                sources["un_region"] = src

            # Coordinates
            coords = entry.get("COORDS")
            lat, lon = _parse_coords(coords)
            if lat is not None:
                location["lat"] = lat
                location["lon"] = lon

            # NUTS
            nuts_data = entry.get("NUTS")
            if isinstance(nuts_data, dict) and nuts_data:
                location["nuts"] = nuts_data

    # -------------------------------------------------
    # 2. NER fallback / enrichment (fills gaps only)
    # -------------------------------------------------
    # NER entity labels → output field mapping
    ner_field_map = {
        "city":        ("CITY", None),
        "region":      ("REGION", None),
        "country":     ("COUNTRY", None),
        "postal_code": ("POSTALCODE", None),
        "address":     ("ADDRESS", None),
    }

    ner = raw.get("ner", [])
    if isinstance(ner, list):
        for span_ner in ner:
            if not isinstance(span_ner, dict):
                continue

            for field, (label, alt_label) in ner_field_map.items():
                # Only fill if OSM didn't provide this field
                if field in location:
                    continue

                values = span_ner.get(label)
                used_label = label

                if (not isinstance(values, list) or not values) and alt_label:
                    values = span_ner.get(alt_label)
                    used_label = alt_label

                if isinstance(values, list) and values:
                    location[field] = values[0]
                    sources[field] = "ner"
                    scores[field] = _get_score(score_lookup, used_label, values[0])

    # -------------------------------------------------
    # 3. Finalize
    # -------------------------------------------------
    if not location:
        return None

    # Attach source/confidence only for fields that have values
    if sources:
        location["source"] = sources
    if scores:
        location["confidence"] = scores

    return location


def _parse_coords(coords: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse coordinates from various formats OSM/cache might return."""
    if coords is None:
        return None, None

    if isinstance(coords, (tuple, list)) and len(coords) == 2:
        try:
            return float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            return None, None

    if isinstance(coords, str):
        try:
            cleaned = coords.strip("()").replace("'", "")
            lat_s, lon_s = cleaned.split(",")
            return float(lat_s), float(lon_s)
        except Exception:
            return None, None

    return None, None


def _extract_language(raw: Dict[str, Any]) -> Optional[str]:
    lang_info = raw.get("language_info", {})
    if not isinstance(lang_info, dict):
        return None
    return lang_info.get("language") or lang_info.get("detected_language")


def _extract_confidence(institutions: List[Dict[str, Any]]) -> Optional[float]:
    scores = [
        inst.get("confidence")
        for inst in institutions
        if isinstance(inst, dict) and inst.get("confidence") is not None
    ]
    if not scores:
        return None
    try:
        return float(max(scores))
    except Exception:
        return None