import hashlib
import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil

from . import base
from . import downloader

logger = logutil.getLogger()

# ==================================================================================================

BASE_URL = "https://lingualibre.org/datasets/{}.zip"
LANGUAGES = {
    "afr": "Q150-afr-Afrikaans",
    "amh": "Q154-amh-Amharic",
    "ara": "Q219-ara-Arabic",
    "arq": "Q6714-arq-AlgerianArabic",
    "ary": "Q264201-ary-MoroccanArabic",
    "atj": "Q52295-atj-Atikamekw",
    "bam": "Q318-bam-Bambara",
    "bas": "Q405-bas-Basaalanguage",
    "bbj": "Q52067-bbj-Ghomala%27%20language",
    "bci": "Q19858-bci-Baoul%C3%A9",
    "bcl": "Q115107-bcl-CentralBikol",
    "bdu": "Q52073-bdu-Oroko",
    "ben": "Q307-ben-Bengali",
    "bum": "Q52068-bum-Bululanguage",
    "bzm": "Q52074-bzm-Londo",
    "cat": "Q203-cat-Catalan",
    "ces": "Q392-ces-Czech",
    "cmn": "Q113-cmn-MandarinChinese",
    "cym": "Q141-cym-Welsh",
    "deu": "Q24-deu-German",
    "dua": "Q52071-dua-Duala",
    "dyu": "Q159-dyu-Dioulalanguage",
    "eng": "Q22-eng-English",
    "epo": "Q25-epo-Esperanto",
    "eus": "Q299-eus-Basque",
    "fin": "Q33-fin-Finnish",
    "fon": "Q242-fon-Fon",
    "fra": "Q21-fra-French",
    "gaa": "Q321-gaa-Ga",
    "gcf": "Q83641-gcf-GuadeloupeanCreoleFrench",
    "gre": "Q205-gre-Greek",
    "hat": "Q165-hat-HaitianCreole",
    "hav": "Q51299-hav-Havu",
    "heb": "Q397-heb-Hebrew",
    "hin": "Q123-hin-Hindi",
    "hye": "Q131-hye-Armenian",
    "ita": "Q385-ita-Italian",
    "jpn": "Q389-jpn-Japanese",
    "kab": "Q273-kab-Kabyle",
    "kan": "Q80-kan-Kannada",
    "ken": "Q204940-ken-Nyanglanguage",
    "ltz": "Q46-ltz-Luxembourgish",
    "mal": "Q437-mal-Malayalam",
    "mar": "Q34-mar-Marathi",
    "mis-can": "Q221062-mis-Cantonese",
    "mis-teo": "Q4465-mis-Teochewdialect",
    "mis-sur": "Q74905-mis-Sursilvan",
    "mis-gas": "Q930-mis-Gascondialect",
    "mis-lan": "Q931-mis-Languedociendialect",
    "mos": "Q170137-mos-Mossi",
    "myv": "Q231-myv-Erzya",
    "nld": "Q35-nld-Dutch",
    "nor": "Q45-nor-Norwegian",
    "nso": "Q258-nso-NorthernSotho",
    "oci": "Q311-oci-Occitan",
    "ori": "Q336-ori-Odia",
    "pan": "Q446-pan-Punjabi",
    "pol": "Q298-pol-Polish",
    "por": "Q126-por-Portuguese",
    "que": "Q388-que-Quechua",
    "rus": "Q129-rus-Russian",
    "sat": "Q339-sat-Santali",
    "shy": "Q4901-shy-Shawiyalanguage",
    "spa": "Q386-spa-Spanish",
    "srr": "Q101-srr-Serer",
    "swe": "Q44-swe-Swedish",
    "tam": "Q127-tam-Tamil",
    "tay": "Q51302-tay-Atayal",
    "tel": "Q39-tel-Telugu",
    "tgl": "Q169-tgl-Tagalog",
    "vie": "Q208-vie-Vietnamese",
    "zho": "Q130-zho-Chinese",
}


# ==================================================================================================


class LinguaLibreDownloader(downloader.ArchiveDownloader):
    def __init__(self, lang="deu"):
        if lang in LANGUAGES:
            link = BASE_URL.format(LANGUAGES[lang])
            super(LinguaLibreDownloader, self).__init__(
                link, move_files_up=True, num_threads=1
            )
        else:
            msg = "There is no lingualibre URL present for language {}!"
            raise ValueError(msg.format(lang))

    @classmethod
    def type(cls):
        return "lingualibre"


# ==================================================================================================


class LinguaLibreReader(base.CorpusReader):
    """ Reader for collections of lingualibre audio data.
    The reader expects extracted  files in the given folder """

    @classmethod
    def type(cls):
        return "lingualibre"

    def _check_for_missing_files(self, path):
        return []

    # ==============================================================================================

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        all_files = []
        for dirpath, dirs, files in os.walk(path):
            for file in files:
                all_files.append([dirpath, file])

        for dirpath, file in logger.progress(all_files):
            file_path = os.path.join(dirpath, file)
            issuer_idx = os.path.basename(dirpath)
            transcription = os.path.splitext(os.path.basename(file_path))[0]

            # Create files ...
            file_idx = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
            corpus.new_file(file_path, file_idx)

            # Issuers, use folder name
            issuer = issuers.Speaker(issuer_idx)
            corpus.import_issuers(issuer)

            # Utterances with labels ...
            utterance = corpus.new_utterance(file_idx, file_idx, issuer_idx)
            ll = annotations.LabelList.create_single(
                transcription, idx=audiomate.corpus.LL_WORD_TRANSCRIPT
            )
            utterance.set_label_list(ll)

        return corpus
