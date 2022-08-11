# +
from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import sys
from typing import List, Tuple
import unicodedata
import warnings

import pandas as pd
import requests
from tqdm.auto import tqdm
import transformers
# -

IS_INTERACTIVE = False
try:
    get_ipython()
except NameError: pass
else: IS_INTERACTIVE = True
IS_INTERACTIVE

p = ArgumentParser()
p.add_argument("raw_text_dir", type=Path)
p.add_argument("aligned_corpora", type=Path, nargs="+")
p.add_argument("-m", "--model", default="GroNLP/gpt2-small-dutch",
               help="Huggingface model ref. NB this processing is GPT2 specific at the moment (see code comments).")

if IS_INTERACTIVE:
    args = Namespace(raw_text_dir=Path("../../data/gillis2021/raw_text/"),
                     aligned_corpora=list(Path(".").glob("DKZ_*.csv")),
                     model="GroNLP/gpt2-small-dutch")
else:
    args = p.parse_args()

raw_text = {}
for text_file in args.raw_text_dir.glob("*.txt"):
    raw_text[text_file.stem] = text_file.read_text()

aligned_corpora = {p.stem: p for p in args.aligned_corpora}

# assert set(aligned_corpora.keys()) == set(raw_text.keys())

# #####

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

# +
# # From http://www.sprookjes.org/index.php?option=com_content&view=article&id=175&Itemid=82
# # Also available here but just as many mismatches https://storiesguide.com/nl/node/2180
# raw_text["DKZ_1"] = \
# """De kleine zeemeermin.

# Ver in zee is het water zo blauw als de blaadjes van de mooiste korenbloem en zo helder als het zuiverste glas, maar het is heel diep, dieper als een ankerketting ooit kan komen. Je zou een heleboel kerktorens boven op elkaar moeten zetten om van de bodem van de zee tot aan de oppervlakte te komen. En daar in de diepte wonen de zeemensen. Maar je moet niet denken dat de bodem van kaal, wit zand is. Nee hoor, er groeien de prachtigste bomen en planten en hun stengels en bladeren zijn zo soepel dat ze bij de minste beweging van het water heen en weer gaan, net of ze levend zijn. Grote en kleine vissen, allemaal glippen ze tussen de takken door, net als bij ons de vogels in de lucht. Op het allerdiepste plekje ligt het paleis van de zeekoning. De muren zijn van koraal en de lange, spits toelopende ramen het zuiverste barnsteen, maar het dak is van schelpen die met de bewegingen van het water open en dicht gaan. Het ziet er beeldig uit, want in iedere schelp zit een schitterende parel en als je er maar eentje van in de kroon van de koningin zou doen, zou dat al heel bijzonder zijn.

# De zeekoning was al jaren weduwnaar, maar zijn oude moeder deed de huishouding voor hem. Ze was een wijze vrouw, maar ze was trots op haar adellijke afkomst. Daarom had ze twaalf oesters op haar staart; de anderen van hoge komaf mochten er maar zes. Verder verdiende ze alle lof, vooral omdat ze zoveel van de kleine zeeprinsesjes, haar kleindochters, hield. Het waren zes mooie kinderen, maar de jongste was de mooiste van allemaal. Haar huid was zo blank en zo teer als een rozenblaadje, haar ogen zo blauw als het diepste meer, maar ze had geen voeten, net zomin als de anderen: haar lichaam eindigde in een vissenstaart. Ze speelden de hele dag in het paleis, in de grote zalen, waar levende bloemen uit de muren groeiden. De grote ramen van barnsteen gingen open en dan zwommen de vissen bij ze binnen, zoals bij ons de zwaluwen binnenvliegen als wij de ramen openzetten, maar de vissen zwommen recht op de prinsesjes af, aten uit hun hand en lieten zich aaien. Buiten het paleis was een grote tuin met vuurrode en donkerblauwe bomen. De vruchten glansden als goud en de bloemen als brandend vuur, doordat ze de hele tijd hun stengels en bladeren bewogen. De grond was van het fijnste zand, maar blauw, als de vlam van zwavel. Er lag een wonderlijk blauw schijnsel over alles heen. Je zou eerder denken dat je hoog boven in de lucht was en alleen lucht boven en onder je zag, dan dat je op de bodem van de zee was. Bij echt stil weer kon je de zon zien, die leek op een purperen bloem met een kelk waaruit al het licht stroomde. Ieder prinsesje had haar eigen plekje in de tuin, waar ze mocht graven en planten wat ze maar wilde. Eentje gaf haar bloementuintje de vorm van een walvis, een ander vond het leuk als haar tuintje op een zeemeermin leek, maar de jongste maakte haar tuintje helemaal rond, net als de zon, en ze had alleen maar roodglanzende bloemen. Het was een wonderlijk kind, stil en in zichzelf gekeerd, en terwijl de andere zusjes hun tuintjes versierden met de gekste dingen die ze uit gestrande schepen hadden gehaald, wilde zij, behalve de rozerode bloemen die op de verre zon leken, alleen een mooi marmeren beeld hebben, in de vorm van een mooie jongen, dat uit glanzend witte Steen was gehakt en bij een schipbreuk op de bodem van de zee terecht was gekomen.

# Bij het beeld plantte ze een rozerode treurwilg die prachtig groeide en die zijn takken over het beeldje heen liet hangen, tot op de blauwe zandbodem, waar de violette schaduw net als de takken heen en weer bewoog; het leek wel of de top van de boom en de wortels een spelletje deden, waarbij het de bedoeling was dat ze haar kusten. Ze vond niets leuker dan verhalen over de mensenwereld boven water. Haar oude oma moest alles vertellen wat ze wist over schepen en steden, mensen en dieren. Het leek haar vooral zo bijzonder dat de bloemen op aarde een geur hadden. Op de bodem van de zee was dat niet zo en dat de bossen groen waren, en dat de vissen die je daar tussen de takken zag, zo hard en zo mooi konden zingen dat het een lust was. Dat waren de vogeltjes, maar oma noemde ze vissen, want anders begrepen ze haar niet omdat ze nog nooit een vogel hadden gezien. 'Als jullie vijftien worden,' zei oma, 'dan mogen jullie uit de we opduiken en in de maneschijn op de rotsen zitten kijken naar de grote schepen die voorbijvaren, en dan zal je bossen en steden zien!' Het jaar daarop werd het eerste zusje vijftien, maar de anderen tja, de een was een jaar jonger dan de ander, de jongste moest dus nog vijfjaar wachten voordat ze weg mocht van de bodem van de zee om te zien hoe het er Het bij ons uitziet. Maar de een beloofde de ander om te vertellen wat ze de eerste dag had gezien, want hun oma had ze niet genoeg verteld; er was zoveel dat ze wilden weten.

# Geen van de zusjes was zo vol verlangen. Als de jongste, juist degene die nog het langst moest wachten en die zo stil en zo in zichzelf gekeerd was. Vaak stond ze 's nachts aan het open raam door het donkerblauwe water te kijken hoe de vissen hun vinnen en hun staart bewogen. De maan en de sterren kon ze zien. Ze schenen maar heel bleekjes, maar door het water leken ze veel groter dan ze in onze ogen zijn. Als er dan een soort zwarte wolk voor gleed, wist ze dat dit een walvis was die boven haar hoofd zwom of een schip vol mensen. Ze dachten er vast niet aan dat in de diepte een zeemeerminnetje haar blanke handen naar de kiel uitstrekte. Toen werd de oudste prinses vijftien en mocht ze naar boven. Toen ze terugkwam, had ze honderd uit te vertellen, maar het heerlijkste, zei ze, was in de maneschijn op een zandbank in de kalme zee te liggen en dicht bij de kust de grote stad te zien waar de lichtjes blonken, alsof het honderd sterren waren, en naar de muziek te luisteren en naar het lawaai en de drukte van rijtuigen en mensen, de vele kerktorens met hun spitsen te zien en te horen hoe de klokken luidden. En juist omdat ze er niet heen kon, verlangde ze daar het meest naar. O, wat zat het jongste zusje met gespitste oortjes te luisteren. Als ze daarna 's nachts bij het open raam door het diepblauwe water stond te kijken, dacht ze aan de grote stad met al het lawaai en de drukte en dan leek het of ze de kerkklokken kon horen luiden.

# 'Ach, was ik maar alvast vijftien,' zei ze, 'ik weet zeker dat ik veel van de wereld en de mensen die daar wonen, zal houden.' Eindelijk werd ze vijftien. 'Zo, nu benijd ook groot,' zei haar oma, de oude koningin-weduwe. 'Kom, dan maak ik je mooi, net als je zusjes.' Ze zette haar een krans van witte lelies op, maar ieder bloemblad was de helft van een parel, en de oude vrouw liet acht grote oesters zich vastklemmen aan de staart van de prinses om te laten zien dat ze van hoge komaf was. 'Het doet zo pijn,' zei de kleine zeemeermin. 'Tja, wie mooi wil zijn, moet pijn lijden!' zei de oude vrouw. O, wat had ze graag al dat moois van zich afgeschud en de zware krans afgedaan. De rode bloemen in haar tuintje pasten veel heter bij haar, maar nu durfde ze het niet meer te veranderen. 'Daag,' zei ze en ze steeg op, zo licht en zo doorzichtig als een luchtbel.

# De zon was net ondergegaan toen ze haar hoofd boven water stak, maar de wolken glansden nog als rozen van goud en midden in de bleekrode lucht stond de avondster helder te stralen. De lucht was zacht en fris en de zee was kalm. Er lag een groot schip met de masten. Er was maar een zeil gehesen, want er stond geen zuchtje wind. In het want en op de raas zaten de matrozen. Er klonk muziek en gezang en toen het donker werd, gingen er honderd gekleurde lantarens aan. Het leek wel of de vlaggen van albe landen daar wapperden. De kleine zeemeermin zwom naar het raam van de kajuit en iedere keer als ze door de golven werd opgetild, kon ze door de glasheldere ramen naar binnen kijken, waar heel veel mooi aangeklede mensen stonden. Maar het mooist was toch tjonge erin met zijn grote, zwarte ogen. Hij was niet veel ouder dan zestien, hij was jarig en daar was al die pracht en praam voor. De matrozen dansten op het dek en toen de jonge prins naar buiten kwam, stegen er meer dan honderd vuurpijlen op. Het was zo licht als op klaarlichte dag, zo dat de kleine zeemeermin van schrik onder water dook, maar ze stak haar hoofd gauw weer boven water en toen leek het alsof alle sterren van de hemel op haar neerkwamen. Zo vuurwerk had ze nog nooit gezien. Grote zonnen draaiden rond, schitterende vissen van vuur spartelden in de blauwe lucht en de heldere, kalme zee weerkaatste alles. Op het schip zelf was het zo licht dat je ieder touwtje kon zien, dus de mensen al helemaal. Wat was tjonge prins mooi. Hij gaf de zeelui een hand, hij was vriendelijk en hij lachte, terwijl de muziek weerklonk in de nacht.

# Het werd laat, maar de kleine zeemeermin kon haar ogen niet van het schip en de mooie prins afhouden. De gekleurde lantarens gingen uit, er stegen geen vuur pijlen meer op, er weerklonken ook geen kanonschoten meer, maar diep in de zee rommelde en stommelde het. De kleine zeemeermin deinde intussen op de golven op en neer, zodat ze in de kajuit kon kijken. Maar het schip kreeg meer vaart, het ene zeil na het andere werd gehesen, de golven gingen hoger, er kwamen grote wolken opzetten, het bliksemde in de verte. O, wat een vreselijk weer zou het worden! Daarom haalden de matrozen de zeilen in. Op en neer ging het grote schip, in vliegende vaart over de woeste zee. Het water rees als grote, zwarte bergen die zich op de mast wilden storten, maar het schip dook als een zwaan tussen de hoge golven en liet zich weer door het op torenende water optillen. De kleine zeemeermin vond dat het lekker snel ging, maar de zeelui dachten daar anders over. Het schip kraakte en kreunde, de dikke planken kromden zich onder de harde windstoten, de zee sloeg over het schip, de mast knakte Als een net doormidden en het schip maakte slingerend slagzij, terwijl het water het ruim binnendrong. Toen zag de kleine zeemeermin dat ze in gevaar waren. Zelf moest ze ook uitkijken voor de balken en wrakstukken van het schip die op het water dreven. Het ene ogenblik was het zo pikdonker dat ze geen niets zag, maar als het dan bliksemde, werd het weer zo licht dat ze iedereen op het schip kon het kennen. Iedereen probeerde zich zo goed en zo kwaad Als het ging te redden. Zij keek vooral uit naar de jonge prins en toen het schip in tweeen werd gespleten, zag ze hem in de diepe zee zinken. Het eerste moment was ze daar blij om, want nu kwam hij naar haar toe. Maar toen dacht ze er weer aan dat mensen niet in het water kunnen leven en dat hij niet naar haar vaders paleis zou kunnen komen, behalve als hij dood was. Maar doodgaan mocht hij niet. Daarom zwom ze tussen planken en balken, die in het water dreven, door zonder eraan te denken dat die haar konden verpletteren.

# Ze dook diep onder en kwam op de golven weer omhoog, en eindelijk was ze dan bij de prins, die in de woeste zee bijna niet meer kon zwemmen. Zijn armen en benen begonnen moe te worden, zijn mooie ogen sloten zich; hij zou gestorven zijn als de kleine zeemeermin er niet was geweest. Ze hield zijn hoofd boven water en liet zich met hem door de golven dragen waarheen die maar wilden. De volgende ochtend was het noodweer voorbij; van het schip was en geen spaan meer heel. De zon steeg rood glanzend uit het water op, het leek net of er daardoor weer leven op de wangen van de prins kwam, maar zijn ogen waren nog steeds gesloten. De zeemeermin kuste zijn mooie, hoge voorhoofd en streek de natte haren uit zijn gezicht. Ze vond dat hij op het marmeren beeld in haar tuintje leek, ze kuste hem nog eens en ze wenste dat hij in leven zou blijven. Toen zag ze het vasteland voor zich: hoge, blauwe bergen waar op de toppen de witte sneeuw blonk, alsof daar zwanen lagen. Bij de kust waren er prachtige, groene bossen en daarvoor lag een kerk of een klooster, dat wist ze niet precies, maar een gebouw was het in elk geval. Er groeiden citroen- en sinaasappelbomen in de tuin en voor de poort stonden hoge palmen. De zee had hier een kleine inham. Hij was kalm, maar heel diep, tot aan de rots waar fijn, wit zand op was gespoeld. Daar zwom ze met de mooie prins naar toe. Ze legde hem in het zand, maar zorgde en vooral voor dat zijn hoofd hoog kwam te liggen, in de warmte van de zon."""

# There's a huge section in the middle of DKZ_1 (from "Het jaar daarna" until "Ach, was ik maar alvast")
# which is excluded from the FA. What's up with that?
# Anyway, excluded from the raw text above.

# Patches
raw_text_replacements = {
    "DKZ_1": [
        # Transcription differences
        ("O,", "Oo,"),
        ("tjonge", "de jonge"),

        # # Typos?
        ("op torenende", "optorende"),

        # TODO this should be "des" I think, it's an abbreviation of the
        # genitive 's in the raw text. why is it "ss" in the corpus?
        ("'s nachts", "ss nachts"),

        ("dieper als", "dieper dan"),
        ("ramen het zuiverste", "ramen van het zuiverste"),
        ("dan dat je", "en dan zag je dat je"),
        ("eigen plekje", "eigen plek"),
        ("glanzend witte", "glanzend wit"),
        ("dat ze haar kusten", "dat ze elkaar kusten"),
        ("leek haar vooral", "leek vooral"),
        ("jullie uit de we opduiken", "jullie uit de zee opduiken"),
        ("en dan zal je", "en dan zul je"),
        ("zien hoe het er", "zien hoe het er hier"),
        ("was zo vol verlangen", "was zo verlangend"),

        ("wist ze dat dit een", "wist ze dat dat een"),
        ("ze er niet heen kon", "ze er niet naartoe kon"),
        ("nu benijd ook", "nu ben jij ook"),
        ("Het doet zo pijn", "Het doet zo een pijn"),
        ("pasten veel heter bij", "pasten veel beter bij"),
        ("schip met de masten", "schip met drie masten"),
        ("gekleurde lantarens", "gekleurde lantaarns"),
        ("van albe landen", "van alle landen"),
        ("was toch de jonge erin", "was toch de jonge prins"),
        ("pracht en praam voor", "pracht en praal voor"),
        ("Zo vuurwerk", "Zo een vuurwerk"),
        ("stegen geen vuur pijlen meer", "stegen geen vuurpijlen meer"),
        ("Als een net doormidden", "Als een riet doormidden"),
        ("dat ze geen niets zag, maar", "dat ze geen hand voor de ogen zag, maar"),
        ("schip kon het kennen", "schip kon herkennen"),
        ("naar haar vaders paleis", "naar het paleis van haar vader"),
        ("het schip was en geen", "het schip was er geen"),
        ("op het marmeren", "op het mooie marmeren"),
        ("fijn, wit zand", "fijn, zacht zand"),
        ("zorgde en", "zorgde er"),
    ],
    
    "DKZ_2": [
        
    ],
}
# -

# Preprocess raw text lightly.
for story in raw_text:
    raw_text[story] = re.sub(r"\s+", " ", raw_text[story])

# -----

# +
def process_token(token):
    return token.replace("Ġ", "").lower()

punct_re = re.compile(r"[^A-zÀ-ž]")
only_punct_re = re.compile(r"^[^A-zÀ-ž]+$")

# FA annotations
skip_re = re.compile(r"\(SKIP(\d)\)")
recap_re = re.compile(r"\(RECAP(\d+)\)")

# NB this is GPT-2 specific!
subword_re = re.compile(r"^[^Ġ]")

def align_corpora(fa_words, tokens_flat):
    tok_cursor = 0
    tok_el = process_token(tokens_flat[tok_cursor])
    
    # Tracks alignment between indices in FA corpus (original, prior to filtering)
    # and indices in tokens_flat. Third element indicates various metadata about
    # alignment (see `flag_types`).
    alignment: List[Tuple[int, int, int]] = []
    flag_types = {
        "recap": 0,  # the word was repeated one or more times in the FA
    }

    def advance(tok_cursor, first_delta=1, skip_subwords=False):
        """
        If `skip_subwords` is `True`, will advance until we reach a BPE
        sentinel.
        """
        next_token_raw, next_token = None, None
        while next_token is None or only_punct_re.match(next_token) or (skip_subwords and subword_re.match(next_token_raw)):
            tok_cursor += first_delta if next_token is None else 1
            
            next_token_raw = tokens_flat[tok_cursor]
            next_token = process_token(next_token_raw)

        # print("///", tok_cursor, next_token)

        return tok_cursor, next_token

    def commit(fa_row, tok_cursor, flags=None, do_advance=True):
        # print(f"{fa_words.loc[fa_idx].text} -- {tokens_flat[tok_cursor]}")
        alignment.append((fa_row.original_idx, tok_cursor, flags))

        if do_advance:
            try:
                return advance(tok_cursor)
            except IndexError:
                raise StopIteration
        else:
            return tok_cursor

    stop = False

    try:
        # NB here `idx` is not the original idx in the FA corpus but the index in the
        # filtered dataframe. These are guaranteed to be contiguous, which is useful
        # for checking neighbor contents.
        # But what should be inserted into `alignment` is the original index in the
        # FA corpus, AKA `row.original_idx`.
        for idx, row in tqdm(fa_words.iterrows(), total=len(fa_words)):
            if stop:
                break

            fa_el = row.text
            if skip_re.search(fa_el):
                # FA corpus indicates that we are missing transcriptions for the preceding `n` words.
                # First try: blindly advance the same number of tokens.
                skip_n = int(skip_re.search(fa_el).group(1))
                tok_cursor, tok_el = advance(tok_cursor, skip_n, skip_subwords=True)

                # Now proceed.
                fa_el = skip_re.sub("", fa_el)
            if recap_re.search(fa_el):
                # This was handled in the previous iteration. Drop.
                fa_el = recap_re.sub("", fa_el)

            if fa_el == tok_el:
                tok_cursor, tok_el = commit(row, tok_cursor)
            elif fa_el.startswith(tok_el):
                while fa_el.startswith(tok_el):
                    fa_el = fa_el[len(tok_el):]
                    tok_cursor, tok_el = commit(row, tok_cursor)

                if fa_el:
                    # There is residual FA el not covered by tokens. Stop.
                    print(fa_words.iloc[idx - 5:idx + 5])
                    print(tokens_flat[tok_cursor - 5:tok_cursor + 5])
                    raise ValueError(str((fa_el, tok_el)))
                    break
            elif recap_re.search(fa_words.loc[idx + 1].text):
                # Next row indicates that the current FA row was a repetition
                # of the previous.

                # Find all tokens associated with previous FA and duplicate them.
                prev_commits = []
                for prev_fa_idx, tok_idx, _ in alignment[::-1]:
                    if prev_fa_idx != idx - 1:
                        break

                    prev_commits.append(tok_idx)

                for tok_idx in prev_commits:
                    commit(row, tok_idx, flags=flag_types["recap"],
                           do_advance=False)

                # NB don't advance token cursor.
            else:
                print(fa_words.iloc[idx - 5:idx + 5])
                print(tokens_flat[tok_cursor - 5:tok_cursor])
                print(tokens_flat[tok_cursor:tok_cursor + 5])
                print(alignment[-5:])
                raise ValueError(str((fa_el, tok_el)))
    except StopIteration:
        pass
    
    return alignment


# -

def patch_story(fa_words, name):
    """
    Perform manual fixes on transcription data in order to facilitate
    matching with raw text stimulus.
    """
    if name == "DKZ_1":
        assert fa_words.loc[995].text == "ons(SKIP1)"
        fa_words.loc[995, "text"] = "ons(SKIP2)"

        # Fix mistake in recap semantics
        assert fa_words.loc[1928].text == "er(RECAP1)"
        assert fa_words.loc[1929].text == "stegen"
        fa_words.loc[1928, "text"] = "er"
        fa_words.loc[1929, "text"] = "stegen(RECAP1)"
    elif name == "DKZ_2":
        # Fix mistake in recap semantics
        # I may have learned from a mistaken row in the beginning
        # In any case, keeping this consistent :)
        fa_words.loc[347, "text"] = "het"
        fa_words.loc[350, "text"] = "kon(SKIP1)(RECAP1)"
    else:
        raise ValueError(f"unknown story name {name}")
        
    # Drop rows that are not useful to us.
    fa_words = fa_words[~fa_words.text.isin(("GBG-LOOP", "STUT"))]

    # Reset indexing after dropping 
    fa_words = fa_words.reset_index().rename(columns={"index": "original_idx"})
    
    return fa_words


def process_story(name):
    # Do manual replacements to match content of two story transcriptions.
    text = raw_text[name]
    for src, tgt in raw_text_replacements[name]:
        assert text.count(src) > 0, src
        text = text.replace(src, tgt)
    
    # Tokenize raw text.
    encoded = tokenizer(text)
    tokens_flat = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    # Retrieve and patch FA corpus.
    fa_df = pd.read_csv(aligned_corpora[name])
    fa_words = fa_df[fa_df.tier == "words"]
    fa_words = patch_story(fa_words, name)
    
    alignment = pd.DataFrame(align_corpora(fa_words, tokens_flat),
                             columns=["textgrid_idx", "tok_idx", "flags"])
    
    # Merge with existing words df.
    fa_words = pd.merge(
        fa_words,
        alignment.rename(columns={"textgrid_idx": "original_idx"}).drop(columns=["flags"]),
        on="original_idx")
    
    # Asof merge to store FA word + token index data in phoneme data.
    fa_phonemes = fa_df[fa_df.tier == "phonemes"]
    fa_phonemes = pd.merge_asof(fa_phonemes, fa_words[["start", "original_idx", "tok_idx"]],
                                on="start", direction="backward")
    
    # Annotate with story name.
    for df in [fa_words, fa_phonemes]:
        df["story"] = name
        df.set_index("story", append=True, inplace=True)
        df.index = df.index.reorder_levels((1, 0))
    
    return tokens_flat, fa_words, fa_phonemes


# +
raw_text_replacements["DKZ_2"] = [
    ("'s Avonds", "ss Avonds"),
    ("'s morgens", "ss morgens"),
    
    ("gebouw en er kwamen", "gebouw en kwamen"),
    ("Ze bedekte heur haar", "Ze bedekte haar haar"),
    ("wie en naar", "wie er naar"),
    ("de arme prins roe kwam", "de arme prins toe kwam"),  # typo
    ("haalde ze er andere", "haalde ze de andere"),
    ("andere mensen bij", "andere mensen erbij"),
    ("vroegen haar war", "vroegen haar wat"),
    ("in de ruin rijpten", "in de tuin rijpten"),
    ("zag ze niet een", "zag ze niet en"),
    ("bladeren naakten", "bladeren raakten"),
    
]
process_story("DKZ_2")
# -

all_tokens, all_aligned_words, all_aligned_phonemes = [], [], []
stories = sorted(aligned_corpora)
for story in stories:
    try:
        tokens_flat, aligned_words, aligned_phonemes = process_story(story)
    except KeyError:
        warnings.warn(f"Story {story} not yet prepared.")
    else:
        all_tokens.append(tokens_flat)
        all_aligned_words.append(aligned_words)
        all_aligned_phonemes.append(aligned_phonemes)

tok_dir = Path("tokenized")
tok_dir.mkdir(exist_ok=True)
for story, tokens in zip(stories, all_tokens):
    with (tok_dir / f"{story}.txt").open("w") as f:
        f.write(" ".join(tokens))

pd.concat(all_aligned_words).to_csv("aligned_words.csv")

pd.concat(all_aligned_phonemes).to_csv("aligned_phonemes.csv")
