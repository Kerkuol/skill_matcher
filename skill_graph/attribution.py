"""Conservative bilingual clause attribution; stdlib, research heuristics.

This is not a full dependency parser. Every decision exposes its reason.
Relative clauses can change actor temporarily without changing the outer actor.
Unknown and collective actors cannot certify independent candidate practice.
"""
import re

OTHER = re.compile(r"\b(?:коллега|коллеги|руководитель|аналитик|подрядчик|поставщик|дизайнер|разработчик|другая команда|другой сотрудник|команда|сотрудники|colleague|coworker|manager|contractor|supplier|designer|another team|the team)\b",re.I)
SELF = re.compile(r"\b(?:я|лично|мною|мной|самостоятельно|сам|сама|i|myself)\b",re.I)
COLLECTIVE = re.compile(r"\b(?:мы|наша команда|моя команда|we|our team|my team)\b",re.I)
RELATIVE = re.compile(r"^\s*(?:котор\w*|где|which|who|that)\b",re.I)
ACTION = re.compile(r"\b(?:разработал\w*|разрабатывал\w*|настроил\w*|создал\w*|создавал\w*|написал\w*|писал\w*|пишу|построил\w*|распределял\w*|проводил\w*|согласовал\w*|выполнял\w*|обучал\w*|обучил\w*|внедрил\w*|implemented|configured|developed|created|built|wrote|assigned|conducted|trained|led|performed|applied|conducted)\b",re.I)
NEGATION = re.compile(r"\bне\s+(?:\w+\s+){0,2}(?:работал\w*|занимал\w*|разрабатывал\w*|создавал\w*|писал\w*|знаю|проводил\w*|руководил\w*|вел|обучал\w*|настраивал\w*|использ\w*|запускал\w*|оценивал\w*|согласовывал\w*|выделял\w*|ставил\w*|составлял\w*)\b|\b(?:no practical experience|never used|have not worked|did not implement)\b",re.I)
LEARNING = re.compile(r"\b(?:планирую|хочу стать|изучаю|прохожу\s+(?:вводный\s+)?курс|начал курс|слушаю курс|осваиваю на курсе|currently learning|learning to|studying|plan to)\b",re.I)
PARTICIPATION = re.compile(r"\b(?:участвовал\w*|помогал\w*|ассистировал\w*|contributed to|helped|assisted)\b",re.I)
RESULT_USER = re.compile(r"\b(?:получал\w*|получил\w*|пользовал\w*|открывал\w*|читал\w*|пересылал\w*|received|reviewed|read|consumed)\b",re.I)
CLAIM = re.compile(r"\b(?:есть опыт|имею опыт|опыт\s+(?:в\s+)?написания|опыт работы|experience\s+(?:in|with|building)|track record)\b",re.I)


def actor_of(text, inherited="candidate_assumed", relative=False):
    """Explicit subjects override inherited attribution; objects don't.

    Nominative role words are intentionally narrow. Unusual names/syntax can
    still defeat this heuristic and are represented in independent examples.
    """
    own=SELF.search(text)
    other=OTHER.search(text)
    collective=COLLECTIVE.search(text)
    if own and (not other or own.start()<other.start()):
        return "candidate","explicit_self"
    if collective and (not other or collective.start()<=other.start()):
        return "collective","collective_subject"
    if other:
        return "other","third_party_subject"
    if own:
        return "candidate","explicit_self"
    if re.search(r"\b(?:разработали|настроили|создали|внедрили|собрали|реализовали|распределяли)\b",text,re.I):
        return "collective","plural_action_without_individual_role"
    # A postposed Cyrillic proper name directly after a predicate is a likely
    # explicit executor. Treat it as unresolved rather than guessing a person
    # versus a named product (there is no NER model in this prototype).
    for predicate in ACTION.finditer(text):
        if re.match(r"\s+[А-ЯЁ][а-яё]{2,}\b",text[predicate.end():]):
            return "uncertain","postposed_proper_name"
    named=re.search(r"\b([А-ЯЁ][а-яё]{2,}|[A-Z][a-z]{2,})\s+",text)
    if named:
        action=ACTION.match(text,named.end())
        if action and named.group(1).lower() not in {"успешно","затем","позже","самостоятельно","successfully","later"}:
            return "other","named_subject"
    if relative:
        return "uncertain","relative_actor_unresolved"
    return inherited,"resume_subject_assumed" if inherited=="candidate_assumed" else "inherited_subject"


def attributed_fragments(text):
    # Sentence punctuation excludes decimals and common technical identifiers.
    sentence_ends=[0]
    for m in re.finditer(r"(?<!\d)[.!?](?=\s|$)|[.!?](?!\d)(?=\s|$)|\n\s*\n|\n(?=\s*[•●▪])",text):
        sentence_ends.append(m.end())
    sentence_ends.append(len(text))
    previous_actor="candidate_assumed"
    for left,right in zip(sentence_ends,sentence_ends[1:]):
        sentence=text[left:right]
        if not sentence.strip():continue
        # Semicolons separate subjects; commas delimit potential relative clauses.
        chunks=[];start=0
        for m in re.finditer(r"[,;]|\s+(?:а|но|but|whereas)\s+|\s+(?=(?:я|лично|i)\b)",sentence,re.I):
            if m.start()>start:chunks.append((left+start,left+m.start()))
            start=m.end()
        if start<len(sentence):chunks.append((left+start,right))
        new_item=bool(re.match(r"\s*[•●▪]",sentence) or re.search(r"\n\s*\n$",text[max(0,left-40):left]))
        inherited="uncertain" if previous_actor in ("other","collective","uncertain") and not new_item else "candidate_assumed"
        pending=[]
        def emit(items):
            if not items:return None
            a,b=items[0]["start"],items[-1]["end"]
            # Never merge across another actor/stance. Gaps between identical
            # candidate clauses contain only separators, never another clause.
            return dict(start=a,end=b,quote=text[a:b],actor=items[0]["actor"],
                        actor_reason=items[0]["actor_reason"],stance=items[0]["stance"],
                        actor_assumed=items[0]["actor"]=="candidate_assumed")
        for a,b in chunks:
            quote=text[a:b]
            if not quote.strip():continue
            relative=bool(RELATIVE.search(quote))
            actor,reason=actor_of(quote,inherited,relative)
            if not relative and reason in ("explicit_self","third_party_subject","collective_subject","named_subject"):
                inherited=actor
            stance="negated" if NEGATION.search(quote) else "learning" if LEARNING.search(quote) else "participation" if PARTICIPATION.search(quote) else "practical"
            item=dict(start=a,end=b,actor=actor,actor_reason=reason,stance=stance)
            if pending and (pending[-1]["actor"]!=actor or pending[-1]["stance"]!=stance or ";" in text[pending[-1]["end"]:a] or b-pending[0]["start"]>800):
                result=emit(pending)
                if result:yield result
                pending=[]
            pending.append(item)
        result=emit(pending)
        if result:yield result
        previous_actor=inherited
