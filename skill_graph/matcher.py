"""V2 extraction plus proof transport over explicitly reviewed equivalences.

No transport over thematic links, used_for, or generic graph proximity. Actor
and local-context checks are inherited unchanged, including their limitations.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
from .engine import GraphMatcher


class ConceptGraphMatcher(GraphMatcher):
    def __init__(self,path=None):
        super().__init__(path or HERE/"knowledge.json")
        for edge in self.graph["edges"]:
            if edge["relation"]=="equivalent" and edge.get("proof_transfer"):
                if edge["weight"]<.95:raise ValueError("Weak association cannot transport proof")
                a,b=edge["source"],edge["target"]
                if b not in self.parents[a]:self.parents[a].append(b)
                # Ground this wording as the canonical concept. Do not infer
                # the narrower wording back from any broad canonical fact.

    def _extract_details(self,text):
        evidence,signals=super()._extract_details(text)
        # Once an AND route has actually been satisfied for the candidate,
        # that established competence can seed THEMATIC paths. It still cannot
        # create practical proof of a different competence.
        for item in evidence:
            if item["status"]=="practical" and item["actor"] in ("candidate","candidate_assumed"):
                signals.append(dict(fact=dict(concept=item["skill_id"],start=item["start"],end=item["end"],
                                              quote=item["quote"],path=item["path"],derived_from_route=item["route"]),
                                    actor=item["actor"],actor_reason=item["actor_reason"]))
        return evidence,signals
