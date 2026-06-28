import os
import shutil
import string
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from sacremoses import MosesTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from global_variables import DATA_DIR, EXT_DATA_DIR

n_processors = cpu_count()


def _dataset_cache_ready(path) -> bool:
    """Return True if path contains a complete datasets save_to_disk cache."""
    cache_path = Path(path)
    return cache_path.is_dir() and (cache_path / "state.json").is_file()


def _needs_dataset_build(path) -> bool:
    """Return True if the dataset at path must be (re)built."""
    if _dataset_cache_ready(path):
        return False
    cache_path = Path(path)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    return True


def _get_translation_sentences(dataset, language: str) -> list:
    """Extract source or target sentences from a WMT-style translation column."""
    return [translation[language] for translation in dataset["translation"]]


class TokenizerWrapper:
    """
    Wrapper class for tokenization using MosesTokenizer for English and French.
    """

    def __init__(self, tokenizer_en, tokenizer_fr):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

    def preprocess_text(self, text):
        """
        Preprocess the input text, e.g., convert to lowercase.
        """
        return text.lower()

    def tokenize_function(self, examples):
        """
        Tokenize English and French translations in the examples.
        """
        preprocessed_en = self.preprocess_text(examples["translation"]["en"])
        preprocessed_fr = self.preprocess_text(examples["translation"]["fr"])

        tokenized_en = self.tokenizer_en.tokenize(preprocessed_en)
        tokenized_fr = self.tokenizer_fr.tokenize(preprocessed_fr)

        # length_en = len(tokenized_en)
        # length_fr = len(tokenized_fr)

        return {
            "tokenized_en": tokenized_en,
            "tokenized_fr": tokenized_fr,
            # "Tx": length_en,
            # "Ty": length_fr,
        }


class toWordCount:
    """
    Transform class to convert tokenized sentences to corresponding word IDs.
    """

    def __init__(self, cont):
        self.cont = cont

    def __call__(self, tokenized):
        count_en = self.cont()
        count_fr = self.cont()
        count_en.update(tokenized["tokenized_en"])
        count_fr.update(tokenized["tokenized_fr"])

        return {
            "count_en_words": list(count_en.keys()),
            "count_en_freq": list(count_en.values()),
            "count_fr_words": list(count_fr.keys()),
            "count_fr_freq": list(count_fr.values()),
        }


class toIdTransform:
    """
    Transform class to convert tokenized sentences to corresponding word IDs.
    """

    def __init__(self, most_frequent_words_en, most_frequent_words_fr, tor):
        self.most_frequent_words_en = most_frequent_words_en
        self.most_frequent_words_fr = most_frequent_words_fr
        self.word_to_id_en = {word: i for i, word in enumerate(most_frequent_words_en)}
        self.word_to_id_fr = {word: i for i, word in enumerate(most_frequent_words_fr)}
        self.tor = tor

    def __call__(self, tokenized):
        """
        Convert tokenized sentences to word IDs.
        """
        return {
            "ids_en": [
                self.word_to_id_en.get(token, len(self.most_frequent_words_en) - 1)
                for token in tokenized["tokenized_en"]
            ],
            "ids_fr": [
                self.word_to_id_fr.get(token, len(self.most_frequent_words_fr) - 1)
                for token in tokenized["tokenized_fr"]
            ],
        }


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.languages = ["english", "french"]
        self.types = ["idx", "sentences"]

    def __len__(self):
        return len(
            self.data["english"]["idx"]
        )  # Assuming all language and data types have the same length

    def __getitem__(self, index):
        sample = {}
        for language in self.languages:
            sample[language] = {}
            for type in self.types:
                sample[language][type] = self.data[language][type][index]
        return sample


class to_tensor:
    """
    Transform class to convert word IDs to PyTorch tensors.
    """

    def __init__(self, tor):
        self.tor = tor

    def __call__(self, ids):
        """
        Convert word IDs to PyTorch tensors.
        """
        return {
            "ids_en": self.tor.tensor(ids["ids_en"]),
            "ids_fr": self.tor.tensor(ids["ids_fr"]),
        }


def extract_word_frequency(data):
    word_freq = {"en": Counter(), "fr": Counter()}

    for x in tqdm(data, desc="Extracting word frequency", unit="sample"):
        for lang in ["en", "fr"]:
            word_freq_dict = dict(
                zip(
                    x["count_{}_words".format(lang)],
                    x["count_{}_freq".format(lang)],
                )
            )
            # Exclude simple punctuation marks
            word_freq_dict = {
                word: freq
                for word, freq in word_freq_dict.items()
                if word not in string.punctuation
            }
            word_freq[lang].update(word_freq_dict)
    df_en = pd.DataFrame(word_freq["en"].items(), columns=["word", "freq"])
    df_fr = pd.DataFrame(word_freq["fr"].items(), columns=["word", "freq"])

    df_en.sort_values(by=["freq"], ascending=False, inplace=True)
    df_fr.sort_values(by=["freq"], ascending=False, inplace=True)

    return df_en, df_fr


# Helper function to pad sequences to a specified length
def pad_to_length(x, length, k):
    sos_value = k - 2
    pad_value = k - 1

    if len(x) < length - 1:
        return [sos_value] + x + [pad_value] * (length - len(x) - 1)
    else:
        return [sos_value] + x[: length - 1]


# Function to pad a slice of tokenized data (used by multiprocessing workers)
def _slice_data(data, start, end):
    if hasattr(data, "select"):
        return data.select(range(start, end))
    return data[start:end]


def _pad_slice(args):
    records, Tx, kx, Ty, ky = args
    n = len(records)
    idx_en = torch.zeros((n, Tx), dtype=torch.int16)
    idx_fr = torch.zeros((n, Ty), dtype=torch.int16)
    for i, x in enumerate(records):
        ids_en = x["ids_en"]
        ids_fr = x["ids_fr"]
        idx_en[i] = torch.tensor(pad_to_length(ids_en, Tx, kx), dtype=torch.int16)
        idx_fr[i] = torch.tensor(pad_to_length(ids_fr, Ty, ky), dtype=torch.int16)
    return idx_en, idx_fr


def pad_sequences(data, Tx, Ty, kx, ky, multiprocess=True):
    """Pad tokenized sequences to fixed length, optionally using multiple processes."""
    n = len(data)
    idx_en = torch.zeros((n, Tx), dtype=torch.int16)
    idx_fr = torch.zeros((n, Ty), dtype=torch.int16)

    if multiprocess and n_processors > 1 and n > n_processors:
        slices = []
        for i in range(n_processors):
            start = i * n // n_processors
            end = (i + 1) * n // n_processors
            slices.append((_slice_data(data, start, end), Tx, kx, Ty, ky))
        with ProcessPoolExecutor(max_workers=n_processors) as executor:
            results = list(
                tqdm(
                    executor.map(_pad_slice, slices),
                    total=len(slices),
                    desc="Padding sequences",
                    unit="chunk",
                )
            )
        offset = 0
        for en, fr in results:
            length = en.shape[0]
            idx_en[offset : offset + length] = en
            idx_fr[offset : offset + length] = fr
            offset += length
    else:
        for i, x in tqdm(
            enumerate(data),
            total=n,
            desc="Padding sequences",
            unit="sample",
        ):
            idx_en[i] = torch.tensor(pad_to_length(x["ids_en"], Tx, kx), dtype=torch.int16)
            idx_fr[i] = torch.tensor(pad_to_length(x["ids_fr"], Ty, ky), dtype=torch.int16)

    return idx_en, idx_fr


# Backward-compatible alias
pad_multiprocess = pad_sequences


# import autotokenizer
def load_data(
    train_len,
    val_len,
    kx=30000,
    ky=30000,
    Tx=30,
    Ty=30,
    batch_size=32,
    tokenizer="Moses",
    vocab_source="train",
    mp=True,
    only_vocab=False,
):
    """
    Load and preprocess data for training and validation.
    """

    tqdm.write("Loading and preprocessing data...")
    mt_en = (
        MosesTokenizer(lang="en")
        if tokenizer == "Moses"
        else AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    )
    mt_fr = (
        MosesTokenizer(lang="fr")
        if tokenizer == "Moses"
        else AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    )

    if not only_vocab:
        # Load WMT14 dataset (HF Hub id changed from "wmt14" to "wmt/wmt14")
        wmt14 = load_dataset("wmt/wmt14", "fr-en")

        # Accessing example data
        train_data = wmt14["train"]
        val_data = wmt14["validation"]

        # shuffle data
        train_data = train_data.shuffle(seed=42)
        val_data = val_data.shuffle(seed=42)

        # Select a subset of data if specified
        if train_len is not None:
            train_data = train_data.select(range(train_len))
        else:
            train_len = len(train_data)
        if val_len is not None:
            val_data = val_data.select(range(val_len))
        else:
            val_len = len(val_data)

        tokenizer_wrapper = TokenizerWrapper(mt_en, mt_fr)

        # Tokenize and save train data if not already done
        tokenized_train_path = DATA_DIR / f"processed_data/tokenized_train_data_{train_len}"
        if _needs_dataset_build(tokenized_train_path):
            tqdm.write("Tokenizing train data...")
            tokenized_train_data = train_data.map(
                tokenizer_wrapper.tokenize_function,
                batched=False,
                num_proc=n_processors,
                remove_columns=["translation"],
            )
            tokenized_train_data.save_to_disk(tokenized_train_path)

        # Tokenize and save validation data if not already done
        tokenized_val_path = DATA_DIR / f"processed_data/tokenized_val_data_{val_len}"
        if _needs_dataset_build(tokenized_val_path):
            tqdm.write("Tokenizing validation data...")
            tokenized_val_data = val_data.map(
                tokenizer_wrapper.tokenize_function,
                batched=False,
                num_proc=n_processors,
                remove_columns=["translation"],
            )
            tokenized_val_data.save_to_disk(tokenized_val_path)

        tokenized_train_data = load_from_disk(tokenized_train_path)
        tokenized_val_data = load_from_disk(tokenized_val_path)

        word_count_path = DATA_DIR / f"processed_data/word_count_{train_len}"
        if _needs_dataset_build(word_count_path):
            tqdm.write("Counting word frequency...")
            word_count_train = tokenized_train_data.map(
                toWordCount(Counter), batched=False, num_proc=n_processors
            )
            word_count_val = tokenized_val_data.map(
                toWordCount(Counter), batched=False, num_proc=n_processors
            )
            word_count = concatenate_datasets([word_count_train, word_count_val])
            word_count.save_to_disk(word_count_path)
        word_count = load_from_disk(word_count_path)

    tqdm.write("Building vocabulary...")
    if not os.path.exists(DATA_DIR / "dictionaries/"):
        os.mkdir(DATA_DIR / "dictionaries/")
    if vocab_source == "train":
        if not os.path.exists(
            DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len)
        ):
            if only_vocab:
                raise ValueError(
                    "No vocabulary file found, please rerun with only_vocab = False to extract"
                )
            df_en, df_fr = extract_word_frequency(word_count)
            df_en.to_csv(
                DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len),
                index=False,
            )
            df_fr.to_csv(
                DATA_DIR / "dictionaries/unigram_freq_fr_{}.csv".format(train_len),
                index=False,
            )
        df_en = pd.read_csv(
            DATA_DIR / "dictionaries/unigram_freq_en_{}.csv".format(train_len)
        )
        df_fr = pd.read_csv(
            DATA_DIR / "dictionaries/unigram_freq_fr_{}.csv".format(train_len)
        )
        bow_english, bow_french = df_en, df_fr
    else:
        bow_english = pd.read_csv(EXT_DATA_DIR / "dictionaries/unigram_freq_en_ext.csv")
        bow_french = pd.read_csv(EXT_DATA_DIR / "dictionaries/unigram_freq_fr_ext.csv")
    tqdm.write("Vocabulary ready.")
    bow_english = bow_english[:kx]
    bow_french = bow_french[:ky]

    # Get most frequent English and French words
    most_frequent_english_words = bow_english["word"].apply(lambda x: str(x)).tolist()
    most_frequent_french_words = bow_french["word"].apply(lambda x: str(x)).tolist()
    tokenized_most_frequent_english_words= mt_en.tokenize(" ".join(most_frequent_english_words))[: kx - 2]
    tokenized_most_frequent_french_words =  mt_fr.tokenize(" ".join(most_frequent_french_words))[: ky - 2]
    # we'll use <sos> and <pad> and <unk> as special tokens
    if not only_vocab:
        to_id_transform = toIdTransform(
            tokenized_most_frequent_english_words,
            tokenized_most_frequent_french_words,
            torch.tensor,
        )

        id_train_path = DATA_DIR / f"processed_data/id_train_data_{train_len}_{kx}_{ky}_{vocab_source}"
        if _needs_dataset_build(id_train_path):
            tqdm.write("Converting train data to word IDs...")
            tokenized_train_data = tokenized_train_data.map(
                to_id_transform, batched=False, num_proc=n_processors
            )
            tokenized_train_data.save_to_disk(id_train_path)

        id_val_path = DATA_DIR / f"processed_data/id_val_data_{val_len}_{kx}_{ky}_{vocab_source}"
        if _needs_dataset_build(id_val_path):
            tqdm.write("Converting validation data to word IDs...")
            tokenized_val_data = tokenized_val_data.map(
                to_id_transform, batched=False, num_proc=n_processors
            )
            tokenized_val_data.save_to_disk(id_val_path)

        tokenized_train_data = load_from_disk(id_train_path)
        tokenized_val_data = load_from_disk(id_val_path)

        # Pad sequences to fixed length
        idx_train_tensor_en, idx_train_tensor_fr = pad_sequences(
            tokenized_train_data, Tx, Ty, kx, ky, mp
        )
        idx_val_tensor_en, idx_val_tensor_fr = pad_sequences(
            tokenized_val_data, Tx, Ty, kx, ky, mp
        )

        # Extract English and French sentences for train and validation data
        train_english_sentences = _get_translation_sentences(train_data, "en")
        train_french_sentences = _get_translation_sentences(train_data, "fr")
        val_english_sentences = _get_translation_sentences(val_data, "en")
        val_french_sentences = _get_translation_sentences(val_data, "fr")
    else:
        idx_train_tensor_en = torch.zeros((1, Tx), dtype=torch.int16)
        idx_train_tensor_fr = torch.zeros((1, Ty), dtype=torch.int16)
        idx_val_tensor_en = torch.zeros((1, Tx), dtype=torch.int16)
        idx_val_tensor_fr = torch.zeros((1, Ty), dtype=torch.int16)
        train_english_sentences = []
        train_french_sentences = []
        val_english_sentences = []
        val_french_sentences = []
    # Organize data into a dictionary
    data = dict(
        train=dict(
            english=dict(idx=idx_train_tensor_en, sentences=train_english_sentences),
            french=dict(idx=idx_train_tensor_fr, sentences=train_french_sentences),
        ),
        val=dict(
            english=dict(idx=idx_val_tensor_en, sentences=val_english_sentences),
            french=dict(idx=idx_val_tensor_fr, sentences=val_french_sentences),
        ),
        bow=dict(
            english=np.array(tokenized_most_frequent_english_words),
            french=np.array(tokenized_most_frequent_french_words),
        ),
    )
    train_dataset = TranslationDataset(data["train"])
    val_dataset = TranslationDataset(data["val"])

    num_workers = min(4, n_processors) if n_processors > 1 else 0
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )

    tqdm.write(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val samples.")
    for label, dataset in (("train", train_dataset), ("val", val_dataset)):
        sample = dataset[0]
        tqdm.write(
            f"  [{label}] en: {sample['english']['sentences'][:120]}..."
            if len(sample["english"]["sentences"]) > 120
            else f"  [{label}] en: {sample['english']['sentences']}"
        )
        tqdm.write(
            f"  [{label}] fr: {sample['french']['sentences'][:120]}..."
            if len(sample["french"]["sentences"]) > 120
            else f"  [{label}] fr: {sample['french']['sentences']}"
        )

    tqdm.write("Data loading and preprocessing complete.")

    return (
        (data["train"], train_dataloader),
        (data["val"], val_dataloader),
        (data["bow"]["english"], data["bow"]["french"]),
    )
