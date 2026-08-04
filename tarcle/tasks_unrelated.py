"""Twelve unrelated ICL tasks: the negative control for the circulant test.

CLAUDE.md rule 2 and BRIEF §5 require that n unrelated tasks show **no** circulant
structure. Per prereg §5 a positive circulant result here voids the entire run,
which makes this the control that licenses every stage-2 diagnostic: without it a
circulant Gram matrix on months is unreadable, because nothing rules out the
diagnostic finding circulant structure in anything.

Design constraints, each load-bearing:

- **Twelve tasks**, so the Gram matrix is the same size as the months one and the
  diagnostics run at identical n. A different n would change every threshold.
- **Arbitrary order.** The tasks are listed alphabetically by name, which carries
  no structure. Stage 2 should also re-run the circulant test under random
  permutations of the ordering and report the distribution — under the null there
  is nothing for a permutation to destroy.
- **24 pairs per task.** `docs/pilot_findings.md` §9 shows FVs collapse to a
  degenerate default when demonstrations draw from too few distinct operands, with
  the threshold between 6 and 9. At 24 pairs a 16-shot held-out prompt draws
  ~12-13 distinct operands, comfortably clear.
- **Single-token targets where possible**, so forced-choice scoring is one forward
  pass, as it is for months. Multi-token targets fall back to teacher forcing.
- **No cyclic or ordinal structure in any task**, and no two tasks related to each
  other — the point is that the parameter index is meaningless.
"""
from __future__ import annotations

UNRELATED_TASKS: dict[str, list[tuple[str, str]]] = {
    "antonym": [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
        ("light", "dark"), ("hard", "soft"), ("wet", "dry"), ("open", "closed"),
        ("rich", "poor"), ("young", "old"), ("full", "empty"), ("clean", "dirty"),
        ("loud", "quiet"), ("strong", "weak"), ("sharp", "blunt"), ("deep", "shallow"),
        ("thick", "thin"), ("early", "late"), ("tight", "loose"), ("smooth", "rough"),
        ("brave", "cowardly"), ("cheap", "expensive"), ("simple", "complex"),
        ("wide", "narrow"),
    ],
    "capital": [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Greece", "Athens"),
        ("Russia", "Moscow"), ("China", "Beijing"), ("Cuba", "Havana"),
        ("Peru", "Lima"), ("Chile", "Santiago"), ("Norway", "Oslo"),
        ("Sweden", "Stockholm"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
        ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Turkey", "Ankara"),
        ("Kenya", "Nairobi"), ("Morocco", "Rabat"), ("Iceland", "Reykjavik"),
        ("Finland", "Helsinki"), ("Hungary", "Budapest"), ("Denmark", "Copenhagen"),
    ],
    # Adjectives chosen disjoint from "antonym"'s operands: both tasks take
    # adjectives, and a shared operand vocabulary would couple their FVs through
    # the operand distribution — the coupling the months controls exist to exclude.
    "comparative": [
        ("tall", "taller"), ("warm", "warmer"), ("cool", "cooler"),
        ("bright", "brighter"), ("calm", "calmer"), ("bold", "bolder"),
        ("kind", "kinder"), ("smart", "smarter"), ("safe", "safer"),
        ("mild", "milder"), ("crisp", "crisper"), ("harsh", "harsher"),
        ("fierce", "fiercer"), ("humble", "humbler"), ("gentle", "gentler"),
        # "brief"/"briefer" replaced: it shared a first token with "bolder",
        # which would make two candidates indistinguishable under first-token
        # forced-choice scoring.
        ("plain", "plainer"), ("quick", "quicker"), ("keen", "keener"),
        ("dense", "denser"), ("rude", "ruder"), ("fine", "finer"),
        ("tough", "tougher"), ("neat", "neater"), ("shy", "shyer"),
    ],
    # Countries disjoint from "capital"'s operands, for the same reason.
    "currency": [
        ("India", "rupee"), ("Mexico", "peso"), ("Israel", "shekel"),
        ("Vietnam", "dong"), ("Thailand", "baht"), ("Korea", "won"),
        ("Ghana", "cedi"), ("Nigeria", "naira"), ("Ethiopia", "birr"),
        # "rial" and "kip" replaced: they shared first tokens with "rupiah" and
        # "kuna" respectively, making those candidates indistinguishable under
        # first-token forced-choice scoring.
        ("Brazil", "real"), ("Haiti", "gourde"), ("Ukraine", "hryvnia"),
        ("Czechia", "koruna"), ("Bulgaria", "lev"), ("Croatia", "kuna"),
        ("Malaysia", "ringgit"), ("Indonesia", "rupiah"), ("Bangladesh", "taka"),
        ("Romania", "leu"), ("Serbia", "dinar"), ("Kazakhstan", "tenge"),
        ("Mongolia", "tugrik"), ("Afghanistan", "afghani"), ("Argentina", "peso"),
    ],
    "english_french": [
        ("dog", "chien"), ("cat", "chat"), ("house", "maison"),
        ("water", "eau"), ("bread", "pain"), ("book", "livre"),
        ("tree", "arbre"), ("milk", "lait"), ("sun", "soleil"),
        ("moon", "lune"), ("fire", "feu"), ("night", "nuit"),
        ("king", "roi"), ("queen", "reine"), ("horse", "cheval"),
        ("flower", "fleur"), ("street", "rue"), ("cheese", "fromage"),
        ("window", "fenetre"), ("garden", "jardin"), ("winter", "hiver"),
        ("summer", "ete"), ("school", "ecole"), ("friend", "ami"),
    ],
    "first_letter": [
        ("apple", "a"), ("banana", "b"), ("cherry", "c"), ("dolphin", "d"),
        ("engine", "e"), ("forest", "f"), ("guitar", "g"), ("harbor", "h"),
        ("island", "i"), ("jacket", "j"), ("kitten", "k"), ("lantern", "l"),
        ("mirror", "m"), ("needle", "n"), ("orange", "o"), ("pencil", "p"),
        ("quartz", "q"), ("rabbit", "r"), ("silver", "s"), ("tunnel", "t"),
        ("umbrella", "u"), ("violin", "v"), ("window", "w"), ("yellow", "y"),
    ],
    "occupation": [
        ("Einstein", "physicist"), ("Mozart", "composer"), ("Picasso", "painter"),
        ("Shakespeare", "playwright"), ("Darwin", "biologist"), ("Chopin", "pianist"),
        ("Freud", "psychologist"), ("Napoleon", "general"), ("Socrates", "philosopher"),
        ("Gutenberg", "printer"), ("Magellan", "explorer"), ("Pasteur", "chemist"),
        ("Beethoven", "composer"), ("Rembrandt", "painter"), ("Euclid", "mathematician"),
        ("Hippocrates", "physician"), ("Homer", "poet"), ("Herodotus", "historian"),
        ("Galileo", "astronomer"), ("Nightingale", "nurse"), ("Caesar", "emperor"),
        ("Confucius", "philosopher"), ("Michelangelo", "sculptor"), ("Cicero", "orator"),
    ],
    "past_tense": [
        ("go", "went"), ("eat", "ate"), ("run", "ran"), ("see", "saw"),
        ("take", "took"), ("give", "gave"), ("write", "wrote"), ("sing", "sang"),
        ("drink", "drank"), ("swim", "swam"), ("break", "broke"), ("speak", "spoke"),
        ("drive", "drove"), ("choose", "chose"), ("fall", "fell"), ("know", "knew"),
        ("grow", "grew"), ("throw", "threw"), ("wear", "wore"), ("steal", "stole"),
        ("begin", "began"), ("forget", "forgot"), ("freeze", "froze"), ("ride", "rode"),
    ],
    "plural": [
        ("mouse", "mice"), ("goose", "geese"), ("child", "children"),
        ("tooth", "teeth"), ("foot", "feet"), ("person", "people"),
        ("man", "men"), ("woman", "women"), ("ox", "oxen"),
        ("cactus", "cacti"), ("fungus", "fungi"), ("nucleus", "nuclei"),
        ("thesis", "theses"), ("crisis", "crises"), ("analysis", "analyses"),
        ("datum", "data"), ("medium", "media"), ("index", "indices"),
        ("matrix", "matrices"), ("vertex", "vertices"), ("appendix", "appendices"),
        ("knife", "knives"), ("leaf", "leaves"), ("wolf", "wolves"),
    ],
    "product_company": [
        ("iPhone", "Apple"), ("Windows", "Microsoft"), ("Photoshop", "Adobe"),
        ("Prius", "Toyota"), ("Mustang", "Ford"), ("PlayStation", "Sony"),
        ("Kindle", "Amazon"), ("Android", "Google"), ("Instagram", "Meta"),
        ("Nutella", "Ferrero"), ("Kleenex", "Kimberly"), ("Gillette", "Procter"),
        ("Pampers", "Procter"), ("Nescafe", "Nestle"), ("Sprite", "Coca"),
        ("Doritos", "Frito"), ("Xbox", "Microsoft"), ("Beetle", "Volkswagen"),
        ("Civic", "Honda"), ("Galaxy", "Samsung"), ("Roomba", "iRobot"),
        ("Lego", "Lego"), ("Rolex", "Rolex"), ("Airbus", "Airbus"),
    ],
    "singular_verb": [
        ("walk", "walks"), ("talk", "talks"), ("jump", "jumps"), ("swim", "swims"),
        ("read", "reads"), ("write", "writes"), ("sing", "sings"), ("dance", "dances"),
        ("cook", "cooks"), ("clean", "cleans"), ("paint", "paints"), ("build", "builds"),
        ("climb", "climbs"), ("drive", "drives"), ("laugh", "laughs"), ("think", "thinks"),
        ("listen", "listens"), ("travel", "travels"), ("watch", "watches"),
        ("teach", "teaches"), ("catch", "catches"), ("wash", "washes"),
        ("push", "pushes"), ("mix", "mixes"),
    ],
    # Verbs disjoint from both "singular_verb" and "past_tense", which also take
    # verbs as operands.
    "verb_noun": [
        ("farm", "farmer"), ("bake", "baker"), ("play", "player"),
        ("work", "worker"), ("lead", "leader"), ("own", "owner"),
        ("buy", "buyer"), ("sell", "seller"), ("employ", "employer"),
        ("train", "trainer"), ("design", "designer"), ("report", "reporter"),
        ("print", "printer"), ("manage", "manager"), ("sail", "sailor"),
        ("act", "actor"), ("direct", "director"), ("invent", "inventor"),
        ("visit", "visitor"), ("edit", "editor"), ("collect", "collector"),
        ("inspect", "inspector"), ("garden", "gardener"), ("govern", "governor"),
    ],
}

# Fixed, arbitrary order: alphabetical by task name. Carries no structure, which
# is the point — the index is a label, not a parameter.
TASK_NAMES: list[str] = sorted(UNRELATED_TASKS)


def task_choices(task: str) -> list[str]:
    """Forced-choice candidate set: the task's target vocabulary, deduplicated
    and order-stable. A few tasks map two operands to the same target (two
    composers, two Microsoft products), so this is shorter than the pair list."""
    seen, out = set(), []
    for _, target in UNRELATED_TASKS[task]:
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


def validate() -> None:
    """Invariants the negative control depends on."""
    assert len(UNRELATED_TASKS) == 12, len(UNRELATED_TASKS)
    for name, pairs in UNRELATED_TASKS.items():
        assert len(pairs) == 24, (name, len(pairs))
        operands = [a for a, _ in pairs]
        assert len(set(operands)) == 24, f"{name}: duplicate operands"
        assert len(task_choices(name)) >= 20, f"{name}: too few distinct targets"
