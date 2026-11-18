"""Experimental graph-to-existing-analytics adapter; no production mutations."""
from functools import lru_cache
import hashlib
from pathlib import Path

from .matcher import ConceptGraphMatcher

STATUS_NAMES = {
    "practical": "Есть описание практики", "participation": "Участие в общей работе",
    "declared": "Навык только заявлен", "learning": "Обучение или планы",
    "negated": "Отрицание опыта", "other_actor": "Действие другого человека",
    "uncertain_actor": "Исполнитель не установлен", "related": "Только ассоциация",
    "no_evidence": "Свидетельств не найдено", "unsupported": "Требование не распознано",
}
POLICY = {
    "version": "graph-app-v2-presence",
    "practical": 1.,
    "participation": .35,
    "declared": .20,
    "other": 0,
}

ENGINE_REVISION = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
    for name in ("scoring.py", "knowledge.json", "engine.py", "matcher.py", "attribution.py"))).hexdigest()


@lru_cache(maxsize=1)
def matcher():
    return ConceptGraphMatcher()


def contribution(row, required_level=None):
    """Equal requirement weights. Association never contributes to fit."""
    status = row["status"]
    return POLICY.get(status, 0.)


def assess_resume(text, profile, graph_matcher=None):
    engine = graph_matcher or matcher()
    requirements = []
    for category, skills in profile["categories"].items():
        for skill, spec in skills.items():
            requirements.append(dict(skill=skill, description=spec.get("description", ""), category=category))
    assessment = engine.assess(requirements, text)
    categories = {name: dict(score=0., weight=1., percentage=0., matches=[], partials=[], missing=[])
                  for name in profile["categories"]}
    details, extracted = [], {}
    for requirement, row in zip(requirements, assessment["requirements"]):
        value = contribution(row)
        row = {k: v for k, v in row.items() if k != "level"}
        row = dict(row, category=requirement["category"],
                   contribution=round(value, 6))
        details.append(row)
        data = categories[row["category"]]
        data["score"] += value
        description = STATUS_NAMES[row["status"]]
        base = dict(skill=row["skill"], description=description, confidence=value, graph_status=row["status"])
        if row["status"] == "practical":
            data["matches"].append(base)
        elif row["status"] in ("practical", "participation", "declared"):
            data["partials"].append(base)
        else:
            data["missing"].append(base)
        if row["status"] in ("practical", "participation", "declared"):
            extracted.setdefault(row["skill"], base)
    for category, data in categories.items():
        count = len(profile["categories"][category])
        data["percentage"] = round(100 * data["score"] / count, 1) if count else 0.
    total = sum(row["contribution"] for row in details)
    return dict(total_percentage=round(100 * total / len(details), 1) if details else 0.,
                categories_analysis=categories, extracted_skills=list(extracted.values()),
                confirmed_skills_count=sum(r["status"] == "practical" for r in details),
                graph_details=details, graph_policy=POLICY.copy(),
                graph_support=dict(supported=sum(r["status"] != "unsupported" for r in details),
                                   total=len(details)),
                graph_version=engine.graph.get("schema_version"))
