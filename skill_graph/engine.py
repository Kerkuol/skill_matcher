"""Pure-stdlib typed graph inference with attributed text evidence.

Evidence spans refer to the exact input text. Normalization preserves positions.
The graph supplies knowledge; this module only implements shared operations.
No semantic probability, candidate percentage or inferred grade is fabricated.
"""
from collections import defaultdict, deque
from functools import lru_cache
import json
from pathlib import Path
import re
import heapq
from .attribution import attributed_fragments, RESULT_USER, CLAIM, ACTION


def normalized(text):
    return text.lower().replace("ё", "е")


@lru_cache(maxsize=2048)
def alias_pattern(alias):
    parts=[]
    for token in re.findall(r"[^\s]+",normalized(alias)):
        escaped=re.escape(token).replace(r"\*",r"[\w-]*")
        parts.append(escaped)
    return re.compile(r"(?<!\w)"+r"\s+".join(parts)+r"(?!\w)",re.UNICODE)


NEGATION=re.compile(r"\bне\s+(?:\w+\s+){0,2}(?:работал\w*|занимал\w*|разрабатывал\w*|создавал\w*|писал\w*|проводил\w*|знаю|руководил\w*|вел|обучал\w*|настраивал\w*|использовал\w*|запускал\w*|оценивал\w*|согласовывал\w*|выделял\w*|ставил\w*|составлял\w*)\b|\b(?:no practical experience|never used|have not worked|did not implement)\b",re.I)
LEARNING=re.compile(r"\b(?:планирую|пока планирую|хочу стать|изучаю|прохожу\s+(?:вводный\s+)?курс|начал курс|слушаю курс|осваиваю на курсе|currently learning|learning to|studying|plan to)\b",re.I)
PARTICIPATION=re.compile(r"\b(?:участвовал\w*|помогал\w*|ассистировал\w*|contributed to|helped|assisted)\b",re.I)
LEVELS={
    3:re.compile(r"\b(?:эксперт\w*|продвинут\w*|expert|advanced)\b",re.I),
    2:re.compile(r"\b(?:средн\w* уровень|уверенно владею|intermediate)\b",re.I),
    1:re.compile(r"\b(?:базов\w*|начальн\w*|beginner|basic)\b",re.I),
}


def fragments(text):
    """Preserve PDF bullets; join wrapped lines, keep at most two sentences.

    There is no coreference/dependency parser here. Nearby sentences can still
    be associated incorrectly; the benchmark includes such counterexamples.
    """
    boundaries=[0]
    for match in re.finditer(r"\n\s*\n|\n(?=\s*[•●▪])",text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    for start,end in zip(boundaries,boundaries[1:]):
        block=text[start:end]
        sentences=list(re.finditer(r"[^.!?]+(?:[.!?]+|$)",block))
        if not sentences and block.strip():
            sentences=[re.match(r"[\s\S]+",block)]
        for index in range(0,len(sentences),2):
            batch=sentences[index:index+2]
            a=start+batch[0].start(); b=start+batch[-1].end()
            # Bound long layout artifacts; use overlapping raw character windows.
            for offset in range(a,b,600):
                right=min(b,offset+800)
                if text[offset:right].strip():
                    yield offset,right,text[offset:right]
                if right==b:
                    break


class GraphMatcher:
    def __init__(self,path=None):
        path=Path(path) if path else Path(__file__).with_name("knowledge.json")
        self.graph=json.loads(path.read_text(encoding="utf-8"))
        self.nodes={node["id"]:node for node in self.graph["nodes"]}
        self.parents=defaultdict(list)
        self.rules=defaultdict(list)
        self.rule_targets={}
        self.semantic=defaultdict(list)
        self.policy=self.graph.get("association_policy",dict(hop_decay=.85,max_hops=4,min_score=.08))
        if not 0 < self.policy["hop_decay"] < 1:
            raise ValueError("Every semantic step must decrease relatedness")
        if not isinstance(self.policy["max_hops"],int) or self.policy["max_hops"]<1:
            raise ValueError("Association traversal must have a positive hop limit")
        self.skills={key:node for key,node in self.nodes.items() if node["kind"]=="skill"}
        for edge in self.graph["edges"]:
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes:
                raise ValueError("Dangling graph edge")
            if edge["relation"] in ("is_a","related_to","used_for","equivalent"):
                weight=float(edge.get("weight",.85))
                if not 0 < weight <= 1:raise ValueError("Semantic edge weight must be in (0,1]")
                self.semantic[edge["source"]].append((edge["target"],weight,edge["relation"],edge.get("provenance","unknown")))
                if edge["relation"] in ("related_to","equivalent"):
                    self.semantic[edge["target"]].append((edge["source"],weight,edge["relation"],edge.get("provenance","unknown")))
            if edge["relation"]=="is_a":self.parents[edge["source"]].append(edge["target"])
            elif edge["relation"]=="requires_all":self.rules[edge["target"]].append(edge["source"])
            elif edge["relation"]=="supports_practice":self.rule_targets[edge["source"]]=edge["target"]
        self.patterns=[(node["id"],alias,alias_pattern(alias)) for node in self.nodes.values()
                       if node["kind"]!="inference" for alias in node["aliases"]]
        self.skill_patterns=[(sid,alias,alias_pattern(alias)) for sid,node in self.skills.items()
                             for alias in [node["label"]]+node["aliases"]]

    def bind(self,name,description=""):
        """Resolve a requirement into the shared graph without a hand-coded ID.

        Exact titles win. Otherwise distinct matched skills cause abstention.
        Description-only mapping is visibly weaker, never invented knowledge.
        """
        title=normalized(name).strip()
        exact={sid for sid,node in self.skills.items() if title in {normalized(node["label"]),*(normalized(a) for a in node["aliases"])}}
        if len(exact)==1:return dict(skill_id=next(iter(exact)),method="exact_title")
        hits={sid for sid,alias,pattern in self.skill_patterns if pattern.search(title)}
        if len(hits)==1:return dict(skill_id=next(iter(hits)),method="title_alias")
        if len(hits)>1:return dict(skill_id=None,method="ambiguous",candidates=sorted(hits))
        hits={sid for sid,alias,pattern in self.skill_patterns if pattern.search(normalized(description))}
        if len(hits)==1:return dict(skill_id=next(iter(hits)),method="description_alias")
        return dict(skill_id=None,method="ambiguous" if hits else "unmapped",candidates=sorted(hits))

    def matrix_graph(self,requirements,matrix_id="research"):
        nodes=[dict(id="matrix:"+matrix_id,kind="matrix",label=matrix_id)]
        edges=[]
        bindings=[]
        for index,requirement in enumerate(requirements):
            rid=f"requirement:{matrix_id}:{index}"
            binding=self.bind(requirement["skill"],requirement.get("description",""))
            nodes.append(dict(id=rid,kind="requirement",label=requirement["skill"],
                              required_level=requirement.get("level",2),binding=binding))
            edges.append(dict(source="matrix:"+matrix_id,relation="requires",target=rid))
            if binding["skill_id"]:
                edges.append(dict(source=rid,relation="refers_to",target=binding["skill_id"]))
            bindings.append(binding)
        return dict(nodes=nodes,edges=edges,bindings=bindings)

    @staticmethod
    def valid_action(quote,match):
        surface=match.group().strip().lower()
        prefix=quote[:match.start()].lower()
        # Receipt/use of another person's product is not its creation. A later
        # coordinated personal action can still be independently supported.
        received=list(RESULT_USER.finditer(prefix))
        if received:
            tail=prefix[received[-1].end():]
            if not re.search(r"\b(?:и|and)\b",tail):return False
        words=re.findall(r"[a-zа-яё]+",surface)
        finite=any(re.fullmatch(r"[а-яё]{2,}(?:л|ла|ли|ю|ем|ит|ут|ют)(?:ся|сь)?",w) or
                   w in {"пишу","built","wrote","led","set","read"} or
                   w.endswith("ed") for w in words)
        if finite or CLAIM.search(surface):return True
        # Noun phrases such as 'queries' or 'data-quality analysis' need an
        # actual candidate predicate/experience statement in the same frame.
        return bool(ACTION.search(quote) or CLAIM.search(quote))

    @lru_cache(maxsize=4096)
    def semantic_paths(self,source):
        """Bounded strongest-product paths. Never sum duplicate weak paths.

        State includes hop count, preventing a longer high-weight path from
        suppressing a shorter path which still has expansion budget.
        Only semantic edges participate: AND prerequisites are NOT shortcuts.
        """
        decay=self.policy["hop_decay"];limit=self.policy["max_hops"]
        best={(source,0):1.0};results={source:dict(score=1.0,path=[source],relations=[])}
        queue=[(-1.0,0,source,[source],[])]
        while queue:
            negative,hops,current,path,relations=heapq.heappop(queue)
            score=-negative
            if score+1e-12<best.get((current,hops),0) or hops>=limit:continue
            for target,weight,relation,provenance in self.semantic[current]:
                if target in path:continue
                value=score*weight*decay
                if value<self.policy["min_score"] or value<=best.get((target,hops+1),0)+1e-12:continue
                best[(target,hops+1)]=value
                new_path=path+[target]
                new_relations=relations+[dict(relation=relation,weight=weight,decay=decay,provenance=provenance)]
                if value>results.get(target,dict(score=-1))["score"]:
                    results[target]=dict(score=value,path=new_path,relations=new_relations)
                heapq.heappush(queue,(-value,hops+1,target,new_path,new_relations))
        return results

    def _extract_details(self,text):
        evidence=[];signals=[]
        for frame in attributed_fragments(text):
            start,end,quote=frame["start"],frame["end"],frame["quote"]
            norm=normalized(quote);facts={}
            for cid,alias,pattern in self.patterns:
                for match in pattern.finditer(norm):
                    if self.nodes[cid]["kind"]=="action" and not self.valid_action(norm,match):continue
                    if cid not in facts or start+match.start()<facts[cid]["start"]:
                        facts[cid]=dict(concept=cid,start=start+match.start(),end=start+match.end(),
                                        quote=text[start+match.start():start+match.end()],matched_alias=alias,path=[cid])
                    break
            direct=dict(facts)
            queue=deque(list(facts))
            while queue:
                child=queue.popleft()
                for parent in self.parents[child]:
                    if parent not in facts:
                        facts[parent]=facts[child] | dict(concept=parent,path=facts[child]["path"]+[parent])
                        queue.append(parent)
            actor=frame["actor"]
            status="other_actor" if actor=="other" else "uncertain_actor" if actor=="uncertain" else "participation" if actor=="collective" else frame["stance"]
            if actor in ("candidate","candidate_assumed") and frame["stance"]=="practical":
                signals.extend(dict(fact=f,actor=actor,actor_reason=frame["actor_reason"]) for f in direct.values())
            for route,inputs in self.rules.items():
                if all(cid in facts for cid in inputs):
                    sid=self.rule_targets[route]
                    evidence.append(dict(skill_id=sid,status=status,route=route,
                        start=start,end=end,quote=quote,facts=[facts[cid] for cid in inputs],
                        path=[route,sid],level=None,actor=actor,actor_reason=frame["actor_reason"]))
            for sid in self.skills:
                if sid in facts and not any(e["skill_id"]==sid and e["start"]==start for e in evidence):
                    evidence.append(dict(skill_id=sid,status=status if status!="practical" else "declared",
                        route=None,start=start,end=end,quote=quote,facts=[facts[sid]],path=[sid],level=None,
                        actor=actor,actor_reason=frame["actor_reason"]))
        return evidence,signals

    def extract(self,text):
        return self._extract_details(text)[0]

    def assess(self,requirements,text,matrix_id="research"):
        compiled=self.matrix_graph(requirements,matrix_id)
        evidence,signals=self._extract_details(text)
        output=[]
        priority={"practical":0,"participation":1,"declared":2,"learning":3,"negated":4,"uncertain_actor":5,"other_actor":6}
        for requirement,binding in zip(requirements,compiled["bindings"]):
            sid=binding["skill_id"]
            if not sid:
                output.append(dict(skill=requirement["skill"],status="unsupported",binding=binding,evidence=[],level=None,association=None,evidence_score=0.0))
                continue
            found=sorted([e for e in evidence if e["skill_id"]==sid],key=lambda e:priority[e["status"]])
            # Do not propagate a level attached to another skill in the block.
            # Only explicit per-skill clauses are eligible, and conflicting
            # explicit levels remain unresolved. General work implies no grade.
            strongest=None
            for signal in signals:
                fact=signal["fact"]
                path=self.semantic_paths(fact["concept"]).get(sid)
                if path and (strongest is None or path["score"]>strongest["score"]):
                    strongest=path | dict(source=fact,actor=signal["actor"],actor_reason=signal["actor_reason"],
                                         percentage=round(path["score"]*100,1),meaning="graph_relatedness_not_probability_or_grade")
            levels=set()
            for match in found:
                if match["actor"] not in ("candidate","candidate_assumed"):continue
                for clause in re.split(r"[.;\n]",match["quote"]):
                    linked={target for target,_,pattern in self.skill_patterns if pattern.search(normalized(clause))}
                    if linked=={sid}:
                        levels.update(level for level,pattern in LEVELS.items() if pattern.search(clause))
            status=found[0]["status"] if found else "related" if strongest else "no_evidence"
            # Association is never sufficient to promote a skill to practical.
            evidence_score=(.90 if found[0]["actor"]=="candidate_assumed" else 1.0) if status=="practical" else 0.0
            output.append(dict(skill=requirement["skill"],status=status,
                               binding=binding,evidence=found,
                               level=next(iter(levels)) if len(levels)==1 else None,
                               association=strongest,evidence_score=evidence_score))
        return dict(matrix_id=matrix_id,requirements=output,graph=compiled)
