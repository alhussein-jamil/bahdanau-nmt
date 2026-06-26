"""End-to-end smoke test on synthetic data (no HF download)."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_preprocessing import TranslationDataset
from models.translation_models import AlignAndTranslate


def _build_tiny_model(device: str, tx: int, ty: int, kx: int):
    english_vocab = np.array([f"en{i}" for i in range(kx - 2)])
    french_vocab = np.array([f"fr{i}" for i in range(kx - 2)])
    output_vocab_size = len(french_vocab) + 2

    config_rnn_decoder = dict(
        input_size=16 * 2 + 16,
        hidden_size=16,
        num_layers=1,
        device=device,
        type="LSTM",
        bidirectional=False,
    )
    decoder_cfg = dict(
        alignment=dict(input_size=16 * 3, device=device, dropout=0.0),
        rnn=config_rnn_decoder,
        output_nn=dict(
            embedding_size=16,
            max_out_units=8,
            hidden_size=16,
            vocab_size=output_vocab_size,
            device=device,
            dropout=0.0,
        ),
        embedding=dict(
            embedding_size=16,
            vocab_size=output_vocab_size,
            device=device,
        ),
        Ty=ty,
        traditional=False,
    )
    encoder_cfg = dict(
        rnn_hidden_size=16,
        rnn_num_layers=1,
        rnn_device=device,
        vocab_size=len(english_vocab) + 2,
        rnn_type="LSTM",
        embedding_size=16,
        reverse_source=True,
    )
    training_cfg = dict(
        device=device,
        output_vocab_size=output_vocab_size,
        english_vocab=english_vocab,
        french_vocab=french_vocab,
        epochs=0.05,
        load_last_model=False,
        beam_search_eval=False,
        display_every_epochs=1,
        grad_accum_steps=1,
        print_every=10_000,
        save_every=10_000,
        Tx=tx,
        Ty=ty,
    )
    return AlignAndTranslate(
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        training=training_cfg,
    ).to(device)


def _make_dataloader(n_samples: int, tx: int, ty: int, kx: int, ky: int, batch_size: int):
    pad_en, pad_fr = kx - 1, ky - 1
    sos_en, sos_fr = kx - 2, ky - 2
    x = torch.randint(0, kx - 3, (n_samples, tx - 1))
    y = torch.randint(0, ky - 3, (n_samples, ty - 1))
    x = torch.cat([torch.full((n_samples, 1), sos_en), x], dim=1).to(torch.int16)
    y = torch.cat([torch.full((n_samples, 1), sos_fr), y], dim=1).to(torch.int16)
    x[:, -1] = pad_en
    y[:, -1] = pad_fr

    dataset = TranslationDataset(
        {
            "english": {
                "idx": x,
                "sentences": [f"english sentence {i}" for i in range(n_samples)],
            },
            "french": {
                "idx": y,
                "sentences": [f"french sentence {i}" for i in range(n_samples)],
            },
        }
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestPipelineSmoke(unittest.TestCase):
    def test_train_and_validate_one_epoch(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tx, ty, kx, ky = 8, 8, 20, 20
        train_loader = _make_dataloader(8, tx, ty, kx, ky, batch_size=4)
        val_loader = _make_dataloader(4, tx, ty, kx, ky, batch_size=4)

        with patch(
            "models.translation_models.DATA_DIR",
            Path(self._tmpdir.name) / "data",
        ):
            model = _build_tiny_model(device, tx, ty, kx)
            model.train(train_loader=train_loader, val_loader=val_loader)

        self.assertGreater(len(model.train_losses), 0)
        self.assertGreater(len(model.val_losses), 1)
        self.assertFalse(torch.isnan(torch.tensor(model.train_losses[-1])))

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
