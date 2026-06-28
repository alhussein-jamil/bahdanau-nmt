import datetime
import math
import os
from typing import Any, Dict, List

import torch
from sacremoses import MosesDetokenizer, MosesTokenizer
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from tqdm import tqdm

from data_preprocessing import TokenizerWrapper, pad_sequences, toIdTransform
from global_variables import DATA_DIR, DEVICE, maybe_autocast
from metrics import bleu_seq
from metrics.losses import Loss
from models.decoder import Decoder
from models.encoder import Encoder
from utils.plotting import plot_alignment


def _build_optimizer(parameters, training_config: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = training_config.get("lr", 0.001)
    return torch.optim.Adam(
        parameters,
        lr=lr,
        amsgrad=training_config.get("amsgrad", True),
        betas=tuple(training_config.get("betas", (0.9, 0.999))),
        weight_decay=training_config.get("weight_decay", 0.0),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, training_config: Dict[str, Any]
):
    scheduler_cfg = training_config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "plateau")
    if scheduler_type == "none":
        return None, "none"

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            min_lr=scheduler_cfg.get("min_lr", 1e-6),
            threshold=scheduler_cfg.get("threshold", 1e-4),
            cooldown=scheduler_cfg.get("cooldown", 0),
        )
        return scheduler, "plateau"

    if scheduler_type == "cosine":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 3)
        total_epochs = math.ceil(training_config.get("epochs", 100))
        t_max = max(1, total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=scheduler_cfg.get("min_lr", 1e-6),
        )
        return scheduler, "cosine"

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class AlignAndTranslate(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(**kwargs.get("encoder", {}))
        self.decoder = Decoder(**kwargs.get("decoder", {}))

        # Training configuration
        training_config = kwargs.get("training", {})
        self.output_vocab_size = training_config.get("output_vocab_size", 100)
        self.criterion = training_config.get(
            "criterion",
            Loss(nn.NLLLoss(ignore_index=self.output_vocab_size - 1)),
        )
        self.optimizer = training_config.get("optimizer") or _build_optimizer(
            self.parameters(), training_config
        )
        self.base_lr = self.optimizer.param_groups[0]["lr"]
        scheduler_cfg = training_config.get("scheduler", {})
        self.warmup_epochs = scheduler_cfg.get("warmup_epochs", 3)
        self.scheduler, self.scheduler_type = _build_scheduler(
            self.optimizer, training_config
        )
        self.device = training_config.get("device", "cpu")
        self.epochs = training_config.get("epochs", 100)
        self.print_every = training_config.get("print_every", 100)
        self.save_every = training_config.get("save_every", 1000)
        self.best_val_loss = float("inf")
        self.source_vocab = training_config.get("english_vocab", [])
        self.target_vocab = training_config.get("french_vocab", [])
        self.load_last_checkpoints = training_config.get("load_last_model", False)
        self.beam_search_eval = training_config.get(
            "beam_search_eval", training_config.get("beam_search", True)
        )
        self.display_every_epochs = training_config.get("display_every_epochs", 5)
        self.grad_accum_steps = training_config.get("grad_accum_steps", 1)
        self.start_time = self.timestamp
        self.Tx = training_config["Tx"]
        self.Ty = training_config["Ty"]
        self.scaler = GradScaler() if self.device == "cuda" else None
        self._detokenizers: Dict[str, MosesDetokenizer] = {}

        self.train_losses = []
        self.val_losses = [1e10]
        os.makedirs(DATA_DIR / "trained_models/", exist_ok=True)

        self.start_time = self.timestamp
        if self.load_last_checkpoints and not self._try_load_latest_compatible_checkpoint():
            tqdm.write("No compatible checkpoint found; starting a fresh run.")

        self.create_folders(self.start_time)


    def create_folders(self, time):
        
        self.local_dir = DATA_DIR / ("trained_models/" + time)
        self.models_dir = self.local_dir / "checkpoints/"
        self.best_models_dir = self.local_dir / "best_models/"
        self.output_dir = self.local_dir / "outputs/"
        self.plot_dir = self.local_dir / "plots/"
        self.bleu_scores = [0.0]
        os.makedirs(DATA_DIR / "trained_models/", exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        return_alignments: bool = True,
    ) -> torch.Tensor:
        with maybe_autocast():
            encoder_output, _ = self.encoder(x)

            h_emb = self.decoder.alignment.nn_h(encoder_output)
            s_i = None
            y_i = None
            allignments = [] if return_alignments else None
            decoder_output = torch.zeros(
                (x.shape[0], self.Ty, self.decoder.relaxation_nn.output_size),
                device=self.device,
            )
            for t in range(self.Ty):
                if y is not None:
                    # Teacher forcing: feed ground-truth previous token y_t (y_0 = <sos>)
                    y_prev = y[:, t]
                elif y_i is not None:
                    # Inference: use hard token from previous step (not softmax distribution)
                    y_prev = torch.argmax(y_i, dim=-1)
                else:
                    y_prev = None

                y_i, s_i, a_i = self.decoder(t, encoder_output, h_emb, s_i, y_prev)
                if return_alignments:
                    allignments.append(a_i)
                decoder_output[:, t, :] = y_i

        if return_alignments:
            allignments = torch.stack(allignments, dim=1)
        return decoder_output, allignments

    def calc_loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # output[:, t] predicts target[:, t+1]; last step has no target
        return self.criterion(output[:, :-1, :], y[:, 1:])

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, *, optimizer_step: bool = True
    ) -> float:
        output, _ = self.forward(x, y, return_alignments=False)
        loss = self.calc_loss(output, y)
        scaled_loss = loss / self.grad_accum_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            if optimizer_step:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            scaled_loss.backward()
            if optimizer_step:
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

        return loss.item(), output, None

    def save_model(self, best: bool = False) -> None:
        # Create a directory with timestamp
        save_dir = os.path.join(
            self.models_dir if not best else self.best_models_dir, self.timestamp
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        save_path = os.path.join(save_dir, "model.pth")
        torch.save(self.state_dict(), save_path)
        tqdm.write(f"Model saved at {save_path}")

    def load_model(self, path: str) -> bool:
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
            tqdm.write(f"Model loaded from {path}")
            return True
        except (RuntimeError, OSError) as e:
            tqdm.write(f"Failed to load model from {path}: {e}")
            return False

    def _try_load_latest_compatible_checkpoint(self) -> bool:
        models_root = DATA_DIR / "trained_models"
        run_dirs = sorted(
            (p for p in models_root.iterdir() if p.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for run_dir in run_dirs:
            for subdir_name in ("best_models", "checkpoints"):
                subdir = run_dir / subdir_name
                if not subdir.is_dir():
                    continue
                checkpoint_dirs = sorted(
                    (p for p in subdir.iterdir() if p.is_dir()),
                    key=lambda path: path.name,
                    reverse=True,
                )
                for checkpoint_dir in checkpoint_dirs:
                    model_path = checkpoint_dir / "model.pth"
                    if model_path.is_file() and self.load_model(str(model_path)):
                        self.start_time = run_dir.name
                        return True
        return False

    def load_last_model(self) -> bool:
        return self._try_load_latest_compatible_checkpoint()

    @staticmethod
    def _current_lr(optimizer: torch.optim.Optimizer) -> float:
        return optimizer.param_groups[0]["lr"]

    def _apply_warmup_lr(self, epoch: int) -> None:
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return
        warmup_factor = (epoch + 1) / self.warmup_epochs
        lr = self.base_lr * warmup_factor
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _step_scheduler(self, epoch: int, val_loss: float) -> None:
        if self.scheduler is None:
            return
        if epoch < self.warmup_epochs:
            return

        if epoch == self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr

        prev_lr = self._current_lr(self.optimizer)
        if self.scheduler_type == "plateau":
            self.scheduler.step(val_loss)
        elif self.scheduler_type == "cosine":
            self.scheduler.step()

        new_lr = self._current_lr(self.optimizer)
        if new_lr < prev_lr - 1e-12:
            tqdm.write(f"Learning rate reduced: {prev_lr:.2e} -> {new_lr:.2e}")

    def train(self, train_loader, val_loader) -> None:
        epoch_bar = tqdm(
            range(math.ceil(self.epochs)),
            desc="Training",
            unit="epoch",
        )
        for epoch in epoch_bar:
            self._apply_warmup_lr(epoch)
            losses = []
            self.optimizer.zero_grad()
            batch_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{math.ceil(self.epochs)}",
                unit="batch",
                leave=False,
            )
            for i, train_sample in batch_bar:
                exact_epoch = epoch + i / len(train_loader)
                if exact_epoch > self.epochs:
                    break
                x, y = (
                    train_sample["english"]["idx"],
                    train_sample["french"]["idx"],
                )
                non_blocking = self.device == "cuda"
                x = x.to(self.device, non_blocking=non_blocking)
                y = y.to(self.device, non_blocking=non_blocking)
                optimizer_step = (
                    (i + 1) % self.grad_accum_steps == 0
                    or (i + 1) == len(train_loader)
                )
                loss, output, allignments = self.train_step(
                    x, y, optimizer_step=optimizer_step
                )
                if optimizer_step:
                    self.optimizer.zero_grad()
                losses.append(loss)

                batch_bar.set_postfix(
                    loss=f"{loss:.4f}",
                    mean=f"{sum(losses) / len(losses):.4f}",
                    refresh=False,
                )

                show_samples = (
                    i % self.print_every == 0
                    and i % (self.print_every * 10) == 0
                    and epoch % self.display_every_epochs == 0
                )
                if show_samples:
                    _, allignments = self.forward(x, y, return_alignments=True)
                    self.display(output, allignments, x, y, val=False)
                if i % self.save_every == 0:
                    self.save_model()

            with open(self.output_dir / "train_losses.txt", "a", encoding="utf-8") as f:
                f.writelines(f"{loss_val}\n" for loss_val in losses)

            val_losses = []
            with torch.no_grad():
                val_loss = self.evaluate(
                    val_loader,
                    display_samples=epoch % self.display_every_epochs == 0,
                )
                val_losses.append(val_loss)
                self.val_losses.append(val_loss)

            epoch_train_loss = sum(losses) / len(losses)
            self._step_scheduler(epoch, val_loss)
            current_lr = self._current_lr(self.optimizer)
            epoch_bar.set_postfix(
                train=f"{epoch_train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{self.best_val_loss:.4f}",
                lr=f"{current_lr:.2e}",
            )

            with open(self.output_dir / "val_losses.txt", "a", encoding="utf-8") as f:
                f.write(f"{val_loss}\n")

            self.train_losses.append(epoch_train_loss)

            with open(self.output_dir / "losses.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"{self.train_losses[-1]} {self.val_losses[-1]} "
                    f"{torch.mean(self.bleu_scores[-1]).half()}\n"
                )

            if sum(val_losses) / len(val_losses) < self.best_val_loss:
                self.best_val_loss = sum(val_losses) / len(val_losses)
                self.save_model(best=True)

    def display(
        self,
        output: torch.tensor,
        allignments: torch.tensor,
        x: torch.tensor,
        y: torch.tensor,
        val: bool = True,
    ):
        random_idx = torch.randint(0, len(x), (4,))
        if val and self.beam_search_eval:
            prediction_idx, _ = self.beam_search_decoder(x[random_idx])
        else:
            prediction = output[random_idx]
            prediction[:, :, -3] = torch.min(prediction)  # mask <unk>

            prediction_idx = self.greedy_search_batch(output[random_idx])


        sample = self.sample_translation(x[random_idx], prediction_idx, y[random_idx])

        self.bleu_scores.append(bleu_seq(sample[2], sample[1], n=4))

        name = "Validation" if val else "Training"
        translations = f"{name} samples:\n"
        for s in range(4):
            translations += f"\tSource: {sample[0][s]}\n"
            translations += f"\tPrediction: {sample[1][s]}\n"
            translations += f"\tTranslation: {sample[2][s]}\n"
            translations += "\n"
        with open(
            self.output_dir / (self.timestamp + ".txt"), "a", encoding="utf-8"
        ) as myfile:
            myfile.write(translations)
        tqdm.write(translations.rstrip())
        bleu_scores = self.bleu_scores[-1]
        self.plot_attention(sample[0], sample[1], allignments[:4], bleu_scores, val=val)

    def evaluate(self, val_loader, display_samples: bool = True) -> float:
        total_loss = 0
        val_bar = tqdm(
            val_loader,
            desc="Validating",
            unit="batch",
            leave=False,
        )
        non_blocking = self.device == "cuda"
        for i, val_sample in enumerate(val_bar):
            x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            output, allignments = self.forward(x, y, return_alignments=display_samples)

            loss = self.calc_loss(output, y)
            total_loss += loss.item()
            val_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
            if i == 0 and display_samples:
                self.display(output, allignments, x, y, val=True)

        return total_loss / len(val_loader)

    @property
    def timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def sample_translation(self, source, prediction, translation):
        source_sentences = []
        prediction_sentences = []
        translation_sentences = []
        for i in range(source.shape[0]):
            s, p, t = source[i], prediction[i], translation[i]
            # Sample a translation from the model
            source_sentences.append(
                self.idx_to_word(s, self.source_vocab, language="en")
            )
            translation_sentences.append(
                self.idx_to_word(t, self.target_vocab, language="fr")
            )
            prediction_sentences.append(
                self.idx_to_word(p, self.target_vocab, language="fr")
            )

        return [source_sentences, prediction_sentences, translation_sentences]

    def idx_to_word(self, idx: torch.Tensor, vocab: List, language="fr") -> str:
        idx = idx.cpu().detach().int().numpy()
        tokens = list(vocab[idx[(idx < len(vocab) - 1)]])
        if language not in self._detokenizers:
            self._detokenizers[language] = MosesDetokenizer(lang=language)
        phrase = self._detokenizers[language].detokenize(tokens, return_str=True)
        return phrase.replace("  ", " ")

    @staticmethod
    def _is_repetition(seq: torch.Tensor, pos: int, new_idx: int) -> bool:
        """Return True if appending new_idx at pos would repeat a prior subsequence."""
        new_token = torch.tensor([new_idx], dtype=torch.int64, device=seq.device)
        for length in range(pos):
            concatenated = torch.cat([seq[pos - length : pos], new_token])
            previous = seq[pos - 2 * length - 1 : pos - length]
            if len(previous) == len(concatenated) and (concatenated == previous).all():
                return True
        return False
    
    def beam_search_decoder(self, x: torch.Tensor, beam_size: int = 10) -> torch.Tensor:
        with torch.no_grad():
            encoder_output, _ = self.encoder(x.to(self.device))
            h_emb = self.decoder.alignment.nn_h(encoder_output)

            batch_size, vocab_size = x.shape[0], len(self.target_vocab) + 2
            sos_token, _unk_token, pad_token = vocab_size - 2, vocab_size - 3, vocab_size - 1

            output = torch.zeros(batch_size, self.Ty, device=self.device)
            alignments = torch.zeros(batch_size, self.Ty, self.Ty, device=self.device)

            for b in range(batch_size):
                first_candidate_seq = torch.full((self.Ty+2,), pad_token, device=self.device, dtype=torch.long)
                first_candidate_seq[0] = sos_token
                candidates = [(first_candidate_seq, 0.0, None, [])]

                for t in range(self.Ty+1):
                    new_candidates = []
                    for seq, score, s_i, align in candidates:
                        if seq[t] == pad_token:
                            new_candidates.append((seq, score, s_i, align))
                            continue
                        y_i, s_i, a_i = self.decoder(
                            t,
                            encoder_output[b : b + 1],
                            h_emb[b : b + 1],
                            s_i,
                            seq[t],
                        )
                        scores = F.log_softmax(y_i, dim=-1)
                        cumulative_scores = score + scores.squeeze()
                        s_i = s_i.view(1,-1)
                        top_k_scores, top_k_indices = torch.topk(cumulative_scores, beam_size)

                        for i in range(beam_size):
                            new_seq = seq.clone()
                            if not self._is_repetition(new_seq[:t], t, top_k_indices[i].item()):
                                new_seq[t + 1] = top_k_indices[i]
                                new_candidates.append(
                                    (new_seq, top_k_scores[i], s_i, align + [a_i])
                                )

                    candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                best_seq, _, _, best_align = max(candidates, key=lambda x: x[1])
                best_seq = best_seq[1: -1]
                #remove first alignment
                best_align = best_align[1:]
                #remove last 
                output[b] = best_seq
                try:
                    alignments[b][: len(best_align)] = torch.stack(
                        best_align, dim=0
                    ).squeeze()
                except (RuntimeError, ValueError):
                    pass

        return output, alignments

    

    def plot_attention(
        self, source, prediction, allignments, titles, val=True, path=None
    ):
        source_list = [s.split(" ") for s in source]
        prediction_list = [p.split(" ") for p in prediction]
        alls = []
        alignments = allignments.cpu().detach().numpy().swapaxes(1, 2)
        for i in range(alignments.shape[0]):
            attn = alignments[i, : len(prediction_list[i]), : len(source_list[i])]
            if self.encoder.reverse_source:
                attn = attn[:, ::-1]
            alls.append(attn)

        data = {
            f"phrase {i}: bleu-score of {int(titles[i]*100.0)}%": (
                source_list[i],
                prediction_list[i],
                alls[i],
            )
            for i in range(len(source_list))
        }
        return plot_alignment(
            data,
            save_path=self.plot_dir
            / (self.timestamp + "_{}".format("val" if val else "train") + ".png")
            if path is None
            else None,
        )

    def eval(self, dataloader, max_len):
        references = []
        hypotheses = []
        with torch.no_grad():
            for val_sample in tqdm(dataloader, desc="Evaluating BLEU", unit="batch"):
                x, y = val_sample["english"]["idx"], val_sample["french"]["idx"]
                x = x.to(self.device)
                if self.beam_search_eval:
                    prediction_idx, _ = self.beam_search_decoder(x)
                else:
                    output, _ = self.forward(x, return_alignments=False)
                    output[:, :, -3] = torch.min(output)
                    prediction_idx = self.greedy_search_batch(output)
                sample = self.sample_translation(x, prediction_idx, y)
                references.extend(sample[2])
                hypotheses.extend(sample[1])
        return bleu_seq(references, hypotheses, n=4).mean()

    def translate_sentence(self, sentences: List[Dict[Any, Any]]):
        tokenizer_en = MosesTokenizer(lang="en")
        tokenizer_fr = MosesTokenizer(lang="fr")
        tokenizer = TokenizerWrapper(tokenizer_en, tokenizer_fr)
        to_id = toIdTransform(self.source_vocab, self.target_vocab, torch)
        treated_sentences = []
        for sentence in sentences:
            sentence = tokenizer.tokenize_function(sentence)
            sentence = to_id(sentence)
            treated_sentences.append(sentence)

        idx_tensor_en, idx_tensor_fr = pad_sequences(
            treated_sentences,
            self.Tx,
            self.Ty,
            len(self.source_vocab),
            len(self.target_vocab),
            multiprocess=False,
        )

        output, alignment = self.forward(idx_tensor_en.to(self.device))

        output[:, :, -3] = torch.min(output)  # mask <unk> (vocab_size - 3)

        if self.beam_search_eval:
            prediction_idx, _ = self.beam_search_decoder(idx_tensor_en)
        else:
            prediction_idx = self.greedy_search_batch(output)
        sample = self.sample_translation(idx_tensor_en, prediction_idx, idx_tensor_fr)

        return sample, alignment

    def greedy_search_batch(self, tensors, avoid_repetition=True):
        batch_size, len_seq, _ = tensors.size()
        output = torch.zeros(batch_size, len_seq, dtype=torch.long, device=tensors.device)

        for b in range(batch_size):
            for i in range(len_seq):
                logits = tensors[b, i]
                if avoid_repetition:
                    for _ in range(logits.size(0)):
                        idx = torch.argmax(logits).item()
                        if not self._is_repetition(output[b], i, idx):
                            output[b, i] = idx
                            break
                        logits = logits.clone()
                        logits[idx] = float("-inf")
                    else:
                        output[b, i] = torch.argmax(tensors[b, i]).item()
                else:
                    output[b, i] = torch.argmax(logits).item()

        return output
