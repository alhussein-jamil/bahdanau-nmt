import argparse
import torch
import yaml

from data_preprocessing import load_data
from models.translation_models import AlignAndTranslate


torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Define command-line arguments
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # Parse command-line arguments
    parser.add_argument(
        "--train_len", type=int, default=100000, help="Number of training examples"
    )
    parser.add_argument(
        "--val_len", type=int, default=None, help="Number of validation examples"
    )
    parser.add_argument(
        "--Tx", type=int, default=10, help="Length of the input sequence"
    )
    parser.add_argument(
        "--Ty", type=int, default=10, help="Length of the output sequence"
    )
    parser.add_argument(
        "--hidden_size", "-n", type=int, default=256, help="Size of the hidden layers"
    )
    parser.add_argument(
        "--embedding_size", "-m", type=int, default=256, help="Size of the embedding"
    )
    parser.add_argument(
        "--max_out_units", "-l", type=int, default=64, help="Size of the hidden layers"
    )
    parser.add_argument(
        "--vocab_size_en",
        type=int,
        default=10000,
        help="Size of the English vocabulary",
    )
    parser.add_argument(
        "--vocab_size_fr", type=int, default=10000, help="Size of the French vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batches"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--vocab_source", type=str, default="train", help="Path to the vocabulary file"
    )
    parser.add_argument(
        "--load_last_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the last saved model checkpoint",
    )
    parser.add_argument(
        "--encoder_decoder",
        action="store_true",
        default=False,
        help="Use the encoder-decoder model",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="translation_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--ignore_config",
        action="store_true",
        default=False,
        help="Ignore the config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run the model in evaluation mode",
    )

    args = parser.parse_args()
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print(f"Using {device} device")
    # Load YAML config file
    with open(args.config_file, "r") as config_file:
        config = yaml.safe_load(config_file)

    if args.ignore_config:
        # Override config values with command-line arguments
        for arg in vars(args):
            if getattr(args, arg) is not None:
                config[arg] = getattr(args, arg)

    # Load data
    (
        (train_data, train_dataloader),
        (val_data, val_dataloader),
        (english_vocab, french_vocab),
    ) = load_data(
        train_len=config["train_len"],
        val_len=config["val_len"],
        kx=config["vocab_size_en"],
        ky=config["vocab_size_fr"],
        Tx=config["Tx"],
        Ty=config["Ty"],
        batch_size=config["batch_size"],
        vocab_source=config["vocab_source"],
        mp=config["multiprocessing"],
    )

    # Define configuration for the decoder
    config_rnn_decoder = dict(
        input_size=config["hidden_size"] * 2 + config["embedding_size"],
        hidden_size=config["hidden_size"],
        num_layers=1,
        device=device,
        type="LSTM",
        bidirectional=False,
    )

    alignment_cfg = dict(
        input_size=config["hidden_size"] * 3,
        device=device,
        dropout=0.0,
    )

    output_nn_cfg = dict(
        embedding_size=config["embedding_size"],
        max_out_units=config["max_out_units"],
        hidden_size=config["hidden_size"],
        vocab_size=len(french_vocab) + 2,
        device=device,
        dropout=0.0,
    )

    decoder_embedding_cfg = dict(
        embedding_size=config["embedding_size"],
        vocab_size=len(french_vocab) + 2,
        device=device,
    )

    config_decoder = dict(
        alignment=alignment_cfg,
        rnn=config_rnn_decoder,
        output_nn=output_nn_cfg,
        embedding=decoder_embedding_cfg,
        Ty=config["Ty"],
        traditional=config["encoder_decoder"],
    )

    # Define configuration for the encoder
    config_encoder = dict(
        rnn_hidden_size=config["hidden_size"],
        rnn_num_layers=1,
        rnn_device=device,
        vocab_size=len(english_vocab) + 2,
        rnn_type="LSTM",
        embedding_size=config["embedding_size"],
        reverse_source=True,
    )

    # Define training configuration
    training_cfg = dict(
        device=device,
        output_vocab_size=len(french_vocab) + 2,
        english_vocab=english_vocab,
        french_vocab=french_vocab,
        epochs=config["epochs"],
        load_last_model=config["load_last_model"],
        beam_search_eval=config.get("beam_search_eval", True),
        display_every_epochs=config.get("display_every_epochs", 5),
        grad_accum_steps=config.get("grad_accum_steps", 1),
        lr=config.get("lr", 0.001),
        scheduler=config.get("scheduler", {}),
        Tx=config["Tx"],
        Ty=config["Ty"],
    )

    # Define translator configuration
    translator_cfg = dict(
        encoder=config_encoder, decoder=config_decoder, training=training_cfg
    )

    # Create the model
    model = AlignAndTranslate(**translator_cfg).to(device)
    english_phrases = [
        "it should be noted that the marine environment is the least known of environments .",
        "The agreement on the European Economic Area was signed in August 1992 .",
        "Destruction of the equipment means that Syria can no longer produce new chemical weapons .",
        '" This will change my future with my family , " the man said .',
    ]
    french_translation = english_phrases

    to_translate = []
    for en, fr in zip(english_phrases, french_translation):
        to_translate.append(dict(translation=dict(en=en, fr=fr)))

    sample, alignment = model.translate_sentence(to_translate)
    for i, s in enumerate(sample):
        print(f"Sample {i+1}")
        en = s[0]
        fr = s[1]
        print(f"\tEnglish: {en}")
        print(f"\tFrench: {fr}")

    # Train the model
    if args.test:
        evaluation = model.eval(val_dataloader, max_len=config["Ty"])
        print(f"Evaluation BLEU scores by length: {evaluation}")
    else:
        model.train(train_loader=train_dataloader, val_loader=val_dataloader)
