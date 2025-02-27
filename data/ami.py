import json
import os
import wave
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset

from data.audio import load_audio
from params import sample_rate

script_dir = os.path.dirname(os.path.abspath(__file__))

speaker_change_token = "<|startoflm|>"


def find_files(root, extension):
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.abspath(os.path.join(root, file))


def get_data_dict():
    if not os.path.exists(os.path.join(script_dir, "..", "ami", "amicorpus")):
        raise FileNotFoundError(
            "AMI corpus not found at "
            + os.path.abspath(os.path.join(script_dir, "..", "ami", "amicorpus"))
            + ". Please download the corpus using data/download_ami_corpus.sh."
        )

    if not os.path.exists(
        os.path.join(script_dir, "..", "ami", "ami_public_manual_1.6.2", "words")
    ):
        raise FileNotFoundError(
            "AMI words not found at "
            + os.path.abspath(
                os.path.join(
                    script_dir, "..", "ami", "ami_public_manual_1.6.2", "words"
                )
            )
            + ". Please download and extract these files from here: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        )
    wav_files = find_files(os.path.join(script_dir, "..", "ami", "amicorpus"), ".wav")
    word_files = find_files(
        os.path.join(script_dir, "..", "ami", "ami_public_manual_1.6.2", "words"),
        ".xml",
    )

    data_dict = {}

    for wav_file in wav_files:
        meeting_id = wav_file.split("/")[-1].split(".")[0]
        if meeting_id not in data_dict:
            data_dict[meeting_id] = {"wav_file": wav_file, "word_files": []}
        data_dict[meeting_id]["wav_file"] = wav_file

    for word_file in word_files:
        meeting_id = word_file.split("/")[-1].split(".")[0]
        if meeting_id not in data_dict:
            continue

        data_dict[meeting_id]["word_files"].append(word_file)

    return {k: v for k, v in data_dict.items() if len(v["word_files"]) > 0}


def create_word_from_xml_child(child):
    id = child.attrib["{http://nite.sourceforge.net/}id"]
    return {
        "speaker": id.split(".")[1],
        "start_time": float(child.attrib["starttime"]),
        "end_time": float(child.attrib["endtime"]),
        "text": child.text,
        "is_punctuation": child.attrib.get("punc", "false") == "true",
        "is_truncated": child.attrib.get("trunc", "false") == "true",
    }


def extract_words(word_files):
    words = []
    for word_file in word_files:
        with open(word_file, "r") as f:
            root = ET.parse(f).getroot()

            for child in root:
                if (
                    child.tag == "w"
                    and "starttime" in child.attrib
                    and "endtime" in child.attrib
                ):
                    words.append(create_word_from_xml_child(child))

    return words


disfluencies = ["um", "uh", "mm-hmm", "mm"]


def create_text_from_words(words):
    text = ""
    sorted_words = sorted(words, key=lambda x: x["start_time"])

    latest_speaker = None
    last_word = None
    for word in sorted_words:
        if (
            last_word
            and last_word["speaker"] != word["speaker"]
            and word["is_punctuation"]
        ):
            continue

        if word["text"].lower() in disfluencies:
            continue

        if word["is_truncated"]:
            continue

        if latest_speaker != word["speaker"] and latest_speaker is not None:
            text += "</" + latest_speaker + ">"

        if latest_speaker != word["speaker"] or latest_speaker is None:
            text += "<" + word["speaker"] + ">"

        text += (" " if not word["is_punctuation"] else "") + word["text"]

        latest_speaker = word["speaker"]
        last_word = word

    if latest_speaker is not None:
        text += "</" + latest_speaker + ">"

    return text


def create_text_from_words_with_speaker_change(words):
    text = ""
    sorted_words = sorted(words, key=lambda x: x["start_time"])

    latest_speaker = None
    last_word = None
    for word in sorted_words:
        if (
            last_word
            and last_word["speaker"] != word["speaker"]
            and word["is_punctuation"]
        ):
            continue

        if word["text"].lower() in disfluencies:
            continue

        if word["is_truncated"]:
            continue

        if latest_speaker != word["speaker"] and latest_speaker is not None:
            text += speaker_change_token

        text += (" " if not word["is_punctuation"] else "") + word["text"]

        latest_speaker = word["speaker"]
        last_word = word

    return text


def chunk_transcripts(wav_file_path, word_files):
    chunks = []

    words = extract_words(word_files)

    with wave.open(wav_file_path, "rb") as wav_file:
        length = wav_file.getnframes() / wav_file.getframerate()

    for i in range(0, int(length), 30):
        start = i
        end = min(i + 30, length)
        chunk_length = end - start
        if chunk_length < 30:
            continue

        chunk_words = [
            word
            for word in words
            if word["start_time"] >= start and word["end_time"] <= end
        ]

        text = create_text_from_words_with_speaker_change(chunk_words)

        chunks.append(
            {"wav_file": wav_file_path, "start": start, "end": end, "text": text}
        )

    return chunks


dump_path = os.path.join(script_dir, "ami_dataset.json")
if not os.path.exists(dump_path):
    data_dict = get_data_dict()
    dataset = []

    for meeting_data in data_dict.values():
        wav_file = meeting_data["wav_file"]
        word_files = meeting_data["word_files"]
        chunks = chunk_transcripts(wav_file, word_files)
        dataset.extend(chunks)

    with open(dump_path, "w") as f:
        json.dump(dataset, f)

else:
    with open(dump_path, "r") as f:
        dataset = json.load(f)


class AMIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        start = item["start"]
        end = item["end"]
        text = item["text"]

        wav_file = item["wav_file"]
        audio = load_audio(wav_file, sample_rate, start, end)
        return audio, text
