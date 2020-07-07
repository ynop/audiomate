import hashlib
import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil
from audiomate.utils import download

from . import base

logger = logutil.getLogger()

# ==================================================================================================

urls = {
    "afr": "https://lingualibre.org/datasets/Q150-afr-Afrikaans.zip",
    "amh": "https://lingualibre.org/datasets/Q154-amh-Amharic.zip",
    "ara": "https://lingualibre.org/datasets/Q219-ara-Arabic.zip",
    "arq": "https://lingualibre.org/datasets/Q6714-arq-AlgerianArabic.zip",
    "ary": "https://lingualibre.org/datasets/Q264201-ary-MoroccanArabic.zip",
    "atj": "https://lingualibre.org/datasets/Q52295-atj-Atikamekw.zip",
    "bam": "https://lingualibre.org/datasets/Q318-bam-Bambara.zip",
    "bas": "https://lingualibre.org/datasets/Q405-bas-Basaalanguage.zip",
    "bbj": "https://lingualibre.org/datasets/Q52067-bbj-Ghomala%27%20language.zip",
    "bci": "https://lingualibre.org/datasets/Q19858-bci-Baoul%C3%A9.zip",
    "bcl": "https://lingualibre.org/datasets/Q115107-bcl-CentralBikol.zip",
    "bdu": "https://lingualibre.org/datasets/Q52073-bdu-Oroko.zip",
    "ben": "https://lingualibre.org/datasets/Q307-ben-Bengali.zip",
    "bum": "https://lingualibre.org/datasets/Q52068-bum-Bululanguage.zip",
    "bzm": "https://lingualibre.org/datasets/Q52074-bzm-Londo.zip",
    "cat": "https://lingualibre.org/datasets/Q203-cat-Catalan.zip",
    "ces": "https://lingualibre.org/datasets/Q392-ces-Czech.zip",
    "cmn": "https://lingualibre.org/datasets/Q113-cmn-MandarinChinese.zip",
    "cym": "https://lingualibre.org/datasets/Q141-cym-Welsh.zip",
    "deu": "https://lingualibre.org/datasets/Q24-deu-German.zip",
    "dua": "https://lingualibre.org/datasets/Q52071-dua-Duala.zip",
    "dyu": "https://lingualibre.org/datasets/Q159-dyu-Dioulalanguage.zip",
    "eng": "https://lingualibre.org/datasets/Q22-eng-English.zip",
    "epo": "https://lingualibre.org/datasets/Q25-epo-Esperanto.zip",
    "eus": "https://lingualibre.org/datasets/Q299-eus-Basque.zip",
    "fin": "https://lingualibre.org/datasets/Q33-fin-Finnish.zip",
    "fon": "https://lingualibre.org/datasets/Q242-fon-Fon.zip",
    "fra": "https://lingualibre.org/datasets/Q21-fra-French.zip",
    "gaa": "https://lingualibre.org/datasets/Q321-gaa-Ga.zip",
    "gcf": "https://lingualibre.org/datasets/Q83641-gcf-GuadeloupeanCreoleFrench.zip",
    "gre": "https://lingualibre.org/datasets/Q205-gre-Greek.zip",
    "hat": "https://lingualibre.org/datasets/Q165-hat-HaitianCreole.zip",
    "hav": "https://lingualibre.org/datasets/Q51299-hav-Havu.zip",
    "heb": "https://lingualibre.org/datasets/Q397-heb-Hebrew.zip",
    "hin": "https://lingualibre.org/datasets/Q123-hin-Hindi.zip",
    "hye": "https://lingualibre.org/datasets/Q131-hye-Armenian.zip",
    "ita": "https://lingualibre.org/datasets/Q385-ita-Italian.zip",
    "jpn": "https://lingualibre.org/datasets/Q389-jpn-Japanese.zip",
    "kab": "https://lingualibre.org/datasets/Q273-kab-Kabyle.zip",
    "kan": "https://lingualibre.org/datasets/Q80-kan-Kannada.zip",
    "ken": "https://lingualibre.org/datasets/Q204940-ken-Nyanglanguage.zip",
    "ltz": "https://lingualibre.org/datasets/Q46-ltz-Luxembourgish.zip",
    "mal": "https://lingualibre.org/datasets/Q437-mal-Malayalam.zip",
    "mar": "https://lingualibre.org/datasets/Q34-mar-Marathi.zip",
    "mis-can": "https://lingualibre.org/datasets/Q221062-mis-Cantonese.zip",
    "mis-teo": "https://lingualibre.org/datasets/Q4465-mis-Teochewdialect.zip",
    "mis-sur": "https://lingualibre.org/datasets/Q74905-mis-Sursilvan.zip",
    "mis-gas": "https://lingualibre.org/datasets/Q930-mis-Gascondialect.zip",
    "mis-lan": "https://lingualibre.org/datasets/Q931-mis-Languedociendialect.zip",
    "mos": "https://lingualibre.org/datasets/Q170137-mos-Mossi.zip",
    "myv": "https://lingualibre.org/datasets/Q231-myv-Erzya.zip",
    "nld": "https://lingualibre.org/datasets/Q35-nld-Dutch.zip",
    "nor": "https://lingualibre.org/datasets/Q45-nor-Norwegian.zip",
    "nso": "https://lingualibre.org/datasets/Q258-nso-NorthernSotho.zip",
    "oci": "https://lingualibre.org/datasets/Q311-oci-Occitan.zip",
    "ori": "https://lingualibre.org/datasets/Q336-ori-Odia.zip",
    "pan": "https://lingualibre.org/datasets/Q446-pan-Punjabi.zip",
    "pol": "https://lingualibre.org/datasets/Q298-pol-Polish.zip",
    "por": "https://lingualibre.org/datasets/Q126-por-Portuguese.zip",
    "que": "https://lingualibre.org/datasets/Q388-que-Quechua.zip",
    "rus": "https://lingualibre.org/datasets/Q129-rus-Russian.zip",
    "sat": "https://lingualibre.org/datasets/Q339-sat-Santali.zip",
    "shy": "https://lingualibre.org/datasets/Q4901-shy-Shawiyalanguage.zip",
    "spa": "https://lingualibre.org/datasets/Q386-spa-Spanish.zip",
    "srr": "https://lingualibre.org/datasets/Q101-srr-Serer.zip",
    "swe": "https://lingualibre.org/datasets/Q44-swe-Swedish.zip",
    "tam": "https://lingualibre.org/datasets/Q127-tam-Tamil.zip",
    "tay": "https://lingualibre.org/datasets/Q51302-tay-Atayal.zip",
    "tel": "https://lingualibre.org/datasets/Q39-tel-Telugu.zip",
    "tgl": "https://lingualibre.org/datasets/Q169-tgl-Tagalog.zip",
    "vie": "https://lingualibre.org/datasets/Q208-vie-Vietnamese.zip",
    "zho": "https://lingualibre.org/datasets/Q130-zho-Chinese.zip",
}


# ==================================================================================================


class LinguaLibreDownloader(base.CorpusDownloader):
    def __init__(self, lang="deu"):
        if lang in urls.keys():
            self.url = urls[lang]
        else:
            msg = "There is no lingualibre URL present for language {}!"
            raise ValueError(msg.format(lang))

    @classmethod
    def type(cls):
        return "lingualibre"

    # ==============================================================================================

    def _download(self, target_path):
        """ Download the data to target_path """

        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, "tmp_lingualibre.zip")

        download.download_file(self.url, tmp_file, num_threads=1)
        download.extract_zip(tmp_file, target_path)
        os.remove(tmp_file)


# ==================================================================================================


class LinguaLibreReader(base.CorpusReader):
    """ Reader for collections of lingualibre audio data.
    The reader expects extracted .zip files in the given folder """

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
