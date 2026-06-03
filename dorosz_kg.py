"""Dorosz Causal Knowledge Graph loader + drug-mention injection.

The KG built from the Dorosz PDF (`dorosz_ckg.json`) contains noisy "drug" nodes
mixed with section-title fragments. We apply 4 KG-research techniques to make
it actually useful:

  (1) Morphological filter on drug labels (pharmacologic suffixes)
        → keeps ~260 real drug names, drops ~2400 PDF-fragment titles.
  (2) Hybrid registry: curated table (high quality, 126 names with rich facts)
      ∪ filtered KG (broad coverage, 5 relation types).
  (3) Multi-relation aggregation per drug:
        has_route, indication / treats, has_dosage, contraindication.
  (4) Single longest-match scan with word-boundary regex (no partial hits).

Used ONLY on the big multilingual dataset; the small BrainHealthAI dataset
trains without KG injection per project decision.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path


# ── (1) Morphological filter ──────────────────────────────────────────────
# These suffixes cover the vast majority of real-drug names in pharmacopeia
# (INN naming conventions, French + English).
_DRUG_SUFFIX_RE = re.compile(
    r"(?:"
    r"ol|olol|ine|ide|ate|ase|ide|ane|"           # general (-ol, -olol, -ine…)
    r"mab|nib|tinib|"                              # biologics / kinase inhibitors
    r"pril|sartan|"                                # ACEi / ARBs
    r"statine?|"                                   # statins
    r"mycine?|cycline?|cilline?|"                  # antibiotics
    r"azole|conazole|"                             # antifungals / PPIs
    r"azepam|"                                     # benzodiazepines
    r"caine|"                                      # local anesthetics
    r"profen|fenac|"                               # NSAIDs
    r"prazole|tidine|"                             # GI (PPIs / H2 blockers)
    r"gliflozine?|gliptine?|glutide|"              # diabetes
    r"setron|asetron|"                             # antiemetics
    r"sone|asone|olone|"                           # corticosteroids
    r"thiazide|spirone|"                           # diuretics
    r"barbital|"                                   # barbiturates
    r"vir|navir|fovir|"                            # antivirals
    r"floxacine?|"                                 # fluoroquinolones
    r"parine?|"                                    # heparins
    r"xaban|rudin|"                                # DOACs
    r"orphine|adone"                               # opioids
    r")$",
    flags=re.IGNORECASE)

_DRUG_REJECT_RE = re.compile(
    r"\b(?:et|de|le|la|les|du|des|en|aux|dans|pour|sur|par|à|au|"
    r"and|the|of|in|to|for|on|by|with|from)\b",
    flags=re.IGNORECASE)

_STOP_LABELS = {
    "guide pratique des", "médicaments", "dorosz", "maloine",
    "pour tous pays", "vii", "xii", "et anesthésiques locaux",
    "anti-inflammatoires non stéroïdiens",
}


def _looks_like_drug_name(label: str) -> bool:
    """Heuristic: does this label look like a real INN drug name?"""
    low = label.strip().lower()
    if len(low) < 4 or len(low) > 35:           return False
    if low in _STOP_LABELS:                     return False
    if _DRUG_REJECT_RE.search(low):             return False
    # Allow 1-2 token names (some real drugs have hyphens or compound names like
    # "amoxicilline-clavulanate"), but reject 3+ token chunks (likely titles).
    if low.count(" ") >= 2:                     return False
    if not _DRUG_SUFFIX_RE.search(low):         return False
    if not re.search(r"[a-zà-ÿ]{4,}", low):     return False
    return True


# ── Curated common-drug fact table (high quality, 126 entries) ───────────
_COMMON_DRUGS: dict[str, str] = {
    # Analgesics / antipyretics
    "paracétamol": "voies: po, iv ; indications: douleur, fièvre",
    "paracetamol": "routes: po, iv ; indications: pain, fever",
    "acetaminophen": "routes: po, iv ; indications: pain, fever",
    "ibuprofène": "voie: po ; indications: douleur, inflammation, fièvre ; AINS",
    "ibuprofen": "route: po ; indications: pain, inflammation, fever ; NSAID",
    "aspirine": "voie: po ; indications: douleur, antiagrégant plaquettaire (faible dose)",
    "aspirin": "route: po ; indications: pain, antiplatelet (low dose)",
    "tramadol": "voies: po, iv ; douleur modérée à sévère ; opiacé faible",
    "morphine": "voies: po, iv, sc ; douleur sévère ; opiacé fort",
    "codéine": "voie: po ; douleur légère à modérée ; opiacé faible",
    "codeine": "route: po ; mild-to-moderate pain ; weak opioid",
    "fentanyl": "voies: iv, transdermique ; douleur sévère ; opiacé fort",
    "kétoprofène": "voies: po, iv ; AINS ; douleur, inflammation",
    "naproxen": "route: po ; NSAID ; pain, inflammation",
    "diclofenac": "voies: po, im ; AINS ; douleur, inflammation",
    # Antibiotics
    "amoxicilline": "voies: po, iv ; infections bactériennes ; ATC J01CA04",
    "amoxicillin": "routes: po, iv ; bacterial infections ; ATC J01CA04",
    "augmentin": "voies: po, iv ; β-lactamine + inhibiteur ; large spectre",
    "ciprofloxacine": "voies: po, iv ; infections urinaires/GI ; fluoroquinolone",
    "ciprofloxacin": "routes: po, iv ; UTI/GI infections ; fluoroquinolone",
    "azithromycine": "voie: po ; infections respiratoires ; macrolide",
    "azithromycin": "route: po ; respiratory infections ; macrolide",
    "doxycycline": "voie: po ; infections atypiques, acné, Lyme ; tétracycline",
    "ceftriaxone": "voies: iv, im ; céphalosporine 3G ; large spectre",
    "vancomycine": "voie: iv ; SARM, infections graves",
    "vancomycin": "route: iv ; MRSA, severe infections",
    "métronidazole": "voies: po, iv ; anaérobies, parasites",
    "metronidazole": "routes: po, iv ; anaerobic infections, parasites",
    "clarithromycine": "voie: po ; infections respiratoires ; macrolide",
    "rifampicine": "voie: po ; tuberculose ; antituberculeux",
    "isoniazide": "voie: po ; tuberculose",
    "cotrimoxazole": "voies: po, iv ; UTI, pneumocystose",
    # Cardiology
    "atorvastatine": "voie: po ; hypercholestérolémie ; statine",
    "atorvastatin": "route: po ; hypercholesterolemia ; statin",
    "rosuvastatine": "voie: po ; hypercholestérolémie ; statine",
    "simvastatine": "voie: po ; hypercholestérolémie ; statine",
    "métoprolol": "voies: po, iv ; HTA, post-IDM, IC ; β-bloquant",
    "metoprolol": "routes: po, iv ; HTN, post-MI, HF ; β-blocker",
    "aténolol": "voie: po ; HTA, angor ; β-bloquant",
    "bisoprolol": "voie: po ; insuffisance cardiaque, HTA ; β-bloquant",
    "carvédilol": "voie: po ; insuffisance cardiaque ; α/β-bloquant",
    "amlodipine": "voie: po ; HTA, angor ; inhibiteur calcique",
    "nifédipine": "voie: po ; HTA ; inhibiteur calcique",
    "diltiazem": "voies: po, iv ; HTA, angor, FA ; inhibiteur calcique",
    "verapamil": "voies: po, iv ; HTA, FA ; inhibiteur calcique",
    "lisinopril": "voie: po ; HTA, IC ; IEC",
    "ramipril": "voie: po ; HTA, IC ; IEC",
    "perindopril": "voie: po ; HTA ; IEC",
    "enalapril": "voie: po ; HTA, IC ; IEC",
    "losartan": "voie: po ; HTA ; ARA-II",
    "valsartan": "voie: po ; HTA, IC ; ARA-II",
    "warfarine": "voie: po ; anticoagulation ; AVK",
    "warfarin": "route: po ; anticoagulation ; vitamin-K antagonist",
    "rivaroxaban": "voie: po ; AOD ; TVP/EP, FA",
    "apixaban": "voie: po ; AOD ; TVP/EP, FA",
    "héparine": "voies: iv, sc ; anticoagulation ; HBPM ou HNF",
    "enoxaparine": "voie: sc ; HBPM ; prophylaxie/traitement TVP",
    "clopidogrel": "voie: po ; antiagrégant plaquettaire ; post-IDM, AVC",
    "furosémide": "voies: po, iv ; diurétique de l'anse ; OAP, IC",
    "furosemide": "routes: po, iv ; loop diuretic ; HF, edema",
    "spironolactone": "voie: po ; diurétique épargneur K+ ; IC",
    "digoxine": "voies: po, iv ; FA, IC ; index thérapeutique étroit",
    "amiodarone": "voies: po, iv ; antiarythmique ; FA, TV",
    # Endocrinology / Diabetes
    "metformine": "voie: po ; diabète type 2 (1ère ligne) ; ATC A10BA02",
    "metformin": "route: po ; type 2 diabetes (first-line) ; ATC A10BA02",
    "insuline": "voies: sc, iv ; diabète ; basale, rapide, lente",
    "insulin": "routes: sc, iv ; diabetes ; basal, rapid, long-acting",
    "sitagliptine": "voie: po ; iDPP-4 ; diabète type 2",
    "empagliflozine": "voie: po ; iSGLT2 ; diabète type 2, IC",
    "liraglutide": "voie: sc ; agoniste GLP-1 ; diabète type 2, obésité",
    "lévothyroxine": "voie: po ; hypothyroïdie ; T4 synthétique",
    "levothyroxine": "route: po ; hypothyroidism ; synthetic T4",
    # Respiratory
    "salbutamol": "voie: inhalée ; β2-mimétique courte action ; asthme, BPCO",
    "albuterol": "route: inhaled ; short-acting β2-agonist ; asthma, COPD",
    "budésonide": "voie: inhalée ; corticoïde ; asthme, BPCO",
    "fluticasone": "voie: inhalée ; corticoïde ; asthme",
    "montelukast": "voie: po ; antileucotriène ; asthme",
    "salmétérol": "voie: inhalée ; β2-mimétique longue action ; asthme",
    "tiotropium": "voie: inhalée ; anticholinergique ; BPCO",
    # GI
    "oméprazole": "voies: po, iv ; IPP ; ulcère, RGO",
    "omeprazole": "routes: po, iv ; PPI ; ulcer, GERD",
    "pantoprazole": "voies: po, iv ; IPP ; ulcère, RGO",
    "ésoméprazole": "voies: po, iv ; IPP ; RGO",
    "lansoprazole": "voie: po ; IPP ; RGO, ulcère",
    "ranitidine": "voies: po, iv ; anti-H2 ; ulcère, RGO",
    "lopéramide": "voie: po ; antidiarrhéique",
    "loperamide": "route: po ; antidiarrheal",
    "métoclopramide": "voies: po, iv ; antiémétique",
    "ondansétron": "voies: po, iv ; antiémétique 5HT3",
    # Mental health
    "sertraline": "voie: po ; ISRS ; dépression, anxiété, TOC",
    "fluoxétine": "voie: po ; ISRS ; dépression, TOC, boulimie",
    "fluoxetine": "route: po ; SSRI ; depression, OCD, bulimia",
    "paroxétine": "voie: po ; ISRS ; dépression, anxiété",
    "escitalopram": "voie: po ; ISRS ; dépression, anxiété",
    "venlafaxine": "voie: po ; ISRSN ; dépression, anxiété",
    "duloxétine": "voie: po ; ISRSN ; dépression, douleur neuropathique",
    "alprazolam": "voie: po ; benzodiazépine courte ; anxiété",
    "diazepam": "voies: po, iv ; benzodiazépine longue ; anxiété, sevrage, convulsions",
    "lorazépam": "voies: po, iv, im ; benzodiazépine ; anxiété, sevrage",
    "olanzapine": "voies: po, im ; antipsychotique atypique",
    "rispéridone": "voie: po ; antipsychotique atypique",
    "haloperidol": "voies: po, im, iv ; antipsychotique typique",
    "lithium": "voie: po ; thymorégulateur ; trouble bipolaire ; ITE étroit",
    "valproate": "voies: po, iv ; antiépileptique, thymorégulateur",
    "lamotrigine": "voie: po ; antiépileptique, thymorégulateur",
    "lévétiracétam": "voies: po, iv ; antiépileptique large spectre",
    # Steroids / Immunosuppressants
    "prednisone": "voie: po ; corticoïde ; nombreuses indications inflammatoires",
    "prednisolone": "voies: po, iv ; corticoïde ; idem prednisone",
    "hydrocortisone": "voies: po, iv, im ; corticoïde ; insuffisance surrénalienne, choc",
    "dexaméthasone": "voies: po, iv, im ; corticoïde puissant ; oedème cérébral, COVID",
    "méthylprednisolone": "voies: po, iv ; corticoïde ; poussées inflammatoires",
    "méthotrexate": "voies: po, sc ; immunosuppresseur ; PR, psoriasis, oncologie",
    "azathioprine": "voie: po ; immunosuppresseur ; auto-immunes, transplantation",
    # Allergy
    "loratadine": "voie: po ; antihistaminique 2G ; rhinite allergique",
    "cetirizine": "voie: po ; antihistaminique 2G ; rhinite allergique, urticaire",
    "desloratadine": "voie: po ; antihistaminique 2G",
    "diphenhydramine": "voies: po, iv, im ; antihistaminique 1G",
    # Misc / Emergency
    "naloxone": "voies: iv, im, sc, intra-nasale ; antagoniste opiacé ; surdose",
    "adrenaline": "voies: iv, im ; choc anaphylactique, ACR",
    "épinéphrine": "voies: iv, im ; choc anaphylactique, ACR",
    "atropine": "voies: iv, im, sc ; bradycardie, intoxication organophosphorée",
    "midazolam": "voies: iv, im, intranasale ; benzodiazépine courte ; sédation",
    "ketamine": "voies: iv, im ; anesthésique dissociatif, douleur",
    "propofol": "voie: iv ; anesthésique IV ; induction, sédation USI",
    "succinylcholine": "voie: iv ; curare dépolarisant ; intubation rapide",
    "rocuronium": "voie: iv ; curare non-dépolarisant ; intubation",
}


# Map of relation labels (raw) → human-friendly French label used in fact line
_REL_LABEL = {
    "has_route":       "voies",
    "indication":      "indications",
    "treats":          "traite",
    "has_dosage":      "posologie",
    "contraindication": "contre-indications",
}


class DoroszKG:
    """Hybrid drug knowledge: curated (precise) ∪ filtered Dorosz KG (broad)."""

    def __init__(self, kg_path: Path | str) -> None:
        path = Path(kg_path)
        self.empty = False
        self.kg_facts: dict[str, str] = {}     # name → pre-rendered fact line

        n_kg_total = 0; n_kg_kept = 0; n_edges = 0
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            nodes = {n["id"]: n for n in raw.get("nodes", [])}
            edges = raw.get("edges", [])

            # Build neighbor index per (source_id, relation)
            out_neigh: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
            for e in edges:
                src = e.get("source"); tgt = e.get("target")
                rel = e.get("relation") or e.get("key")
                if not (src and tgt and rel):  continue
                lab = (nodes.get(tgt, {}).get("label") or tgt).strip()
                out_neigh[src][rel].append(lab)
                n_edges += 1

            # ── (1) Filter drug nodes by morphology ─────────────────
            for nid, node in nodes.items():
                if node.get("type") != "drug":
                    continue
                n_kg_total += 1
                label = (node.get("label") or "").strip()
                if not _looks_like_drug_name(label):
                    continue
                low = label.lower()
                # ── (3) Multi-relation aggregation (drop duplicates) ─
                neigh = out_neigh.get(nid, {})
                parts: list[str] = []
                for rel_key, fr_lab in _REL_LABEL.items():
                    vals = sorted({v for v in neigh.get(rel_key, []) if v})[:3]
                    if vals:
                        parts.append(f"{fr_lab}: {', '.join(vals)}")
                if not parts:           # KG node with no useful edges → skip
                    continue
                self.kg_facts[low] = " ; ".join(parts)
                n_kg_kept += 1

        # ── (2) Hybrid: curated wins, KG fills gaps ───────────────────
        self.curated = {k.lower(): v for k, v in _COMMON_DRUGS.items()}
        self._all_facts: dict[str, str] = {**self.kg_facts, **self.curated}
        # (curated overrides KG when both present)

        if not self._all_facts:
            self.empty = True
            self._scan_re = None
            return

        # ── (4) Single longest-match scan regex ────────────────────────
        names = sorted(self._all_facts.keys(), key=len, reverse=True)
        escaped = [re.escape(n) for n in names]
        self._scan_re = re.compile(
            r"(?<![A-Za-zÀ-ÿ])(" + "|".join(escaped) + r")(?![A-Za-zÀ-ÿ])",
            flags=re.IGNORECASE)

        print(f"[DoroszKG] {n_kg_kept}/{n_kg_total} KG drugs kept after morphological "
              f"filter ; {n_edges:,} edges ; {len(self.curated)} curated drugs ; "
              f"{len(self._all_facts)} total drug terms.")

    # ── Public API ────────────────────────────────────────────────────────
    def lookup_facts(self, text: str, max_drugs: int = 3) -> list[str]:
        """Return up to max_drugs short bullet lines for drugs found in text."""
        if self.empty or not text or self._scan_re is None:
            return []
        seen: set[str] = set()
        facts: list[str] = []
        for m in self._scan_re.finditer(text):
            name = m.group(1).lower()
            if name in seen:
                continue
            seen.add(name)
            fact_str = self._all_facts.get(name)
            if not fact_str:
                continue
            facts.append(f"- {name} — {fact_str}")
            if len(facts) >= max_drugs:
                break
        return facts

    @property
    def name_to_id(self) -> dict[str, str]:
        """Compatibility shim for the smoke test."""
        return {k: ("curated" if k in self.curated else "kg") for k in self._all_facts}


# Singleton accessor
_KG: DoroszKG | None = None

def get_kg(kg_path: Path | str | None = None) -> DoroszKG:
    global _KG
    if _KG is None:
        _KG = DoroszKG(kg_path or Path(__file__).parent / "dorosz_ckg.json")
    return _KG
