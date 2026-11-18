"""Local canvas network; only actual evidence paths and semantic links."""
import hashlib
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from .scoring import matcher, STATUS_NAMES


def graph_data(rows, selected=None):
    engine = matcher()
    nodes, edges, groups, overview = {}, {}, [], {}
    group = None

    def node(cid, label=None, kind=None, active=False, detail=""):
        source = engine.nodes.get(cid, {})
        if cid not in nodes:
            nodes[cid] = dict(id=cid, label=label or source.get("label", cid),
                              kind=kind or source.get("kind", "concept"), active=active, detail=detail,
                              group=group, groups=[])
        if group not in nodes[cid]["groups"]:
            nodes[cid]["groups"].append(group)
        nodes[cid]["active"] |= active
        if kind == "requirement":
            nodes[cid].update(kind=kind, label=label, detail=detail, group=group)

    def edge(a, b, practical=False, relation=""):
        if a != b:
            key = (a, b)
            if key not in edges:
                edges[key] = dict(source=a, target=b, practical=practical, relation=relation)
            edges[key]["practical"] |= practical

    selected_id = None
    for index, row in enumerate(rows):
        group = f"requirement:{index}"
        groups.append(dict(id=group, label=row["skill"], category=row["category"], status=row["status"]))
        sid = row["binding"].get("skill_id") or f"unmapped:{index}"
        node(sid, label=row["skill"], kind="requirement", active=row["status"] == "practical",
             detail=STATUS_NAMES[row["status"]] + ("\n" + row["evidence"][0]["quote"] if row["evidence"] else ""))
        nodes[sid]["status"] = row["status"]
        if index == selected:
            selected_id = sid
        for item in row["evidence"][:2]:
            practical = item["status"] == "practical"
            for fact in item["facts"][:4]:
                qid = "quote:" + hashlib.sha256(f"{fact['start']}:{fact['end']}:{fact['quote']}".encode()).hexdigest()[:16]
                node(qid, label=fact["quote"], kind="quote", active=practical, detail=item["quote"])
                overview[(sid, qid)] = dict(source=sid, target=qid, practical=practical,
                                           relation="Свидетельство из резюме")
                path = fact["path"]
                for cid in path:
                    node(cid, active=practical)
                edge(qid, path[0], practical, "Фрагмент резюме")
                for a, b in zip(path, path[1:]):
                    edge(a, b, practical, "Понятие")
                if item["route"]:
                    node(item["route"], active=practical)
                    edge(path[-1], item["route"], practical, "Часть составного основания")
                    edge(item["route"], sid, practical, "Основание")
        association = row.get("association")
        if association:
            for cid in association["path"]:
                node(cid)
            for a, b, relation in zip(association["path"], association["path"][1:], association["relations"]):
                edge(a, b, relation=relation)
    # Overview links between skills must be supported by an actual semantic
    # path. They visualize related concepts and never change candidate scores.
    requirement_ids = [cid for cid, n in nodes.items() if n["kind"] == "requirement" and cid in engine.nodes]
    relationships = {}
    for source in requirement_ids:
        paths = engine.semantic_paths(source)
        targets = sorted(((target, paths[target]) for target in requirement_ids
                          if target != source and target in paths and paths[target]["score"] >= .16),
                         key=lambda item: -item[1]["score"])[:2]
        for target, path in targets:
            pair = tuple(sorted((source, target)))
            if pair not in relationships or path["score"] > relationships[pair]["score"]:
                relationships[pair] = dict(source=source, target=target, practical=False,
                                           relation="Смысловая связь понятий", score=path["score"],
                                           path=path["path"])
    return dict(nodes=list(nodes.values()), edges=list(edges.values()), groups=groups,
                overview_edges=list(overview.values()), relationships=list(relationships.values()), selected=selected_id)


def render_network(rows, selected=None):
    data = json.dumps(graph_data(rows, selected), ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("graph.html").read_text(encoding="utf-8")
    components.html(template.replace("__GRAPH_DATA__", data), height=665, scrolling=False)
