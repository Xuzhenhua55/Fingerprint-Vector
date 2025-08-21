import os
import time
import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import argparse
import json


class FingerprintVector:
    """
    A utility class for extracting and adding fingerprint vectors (fp_vector).
    """

    @staticmethod
    def extract_fp_vector(base_model_path: str, finetuned_model_path: str,
                          output_path: str) -> None:
        """
        Extracts the fingerprint vector between a base model and a finetuned model,
        and saves it to the specified output path.
        """
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                          torch_dtype='auto')
        ft_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path,
                                                        torch_dtype='auto')

        fp_vector_params = {
            'fp_embed': base_model.get_input_embeddings().weight,
            'fp_lmhead': base_model.get_output_embeddings().weight,
            'fp_vector': {},
            'cfg': {
                'base_model_path': base_model_path,
                'finetuned_model_path': finetuned_model_path,
            }
        }
        start_time = time.time()
        for (n1, p1), (n2, p2) in zip(base_model.named_parameters(),
                                      ft_model.named_parameters()):
            fp_vector_params['fp_vector'][n1] = p2.data - p1.data
        elapsed_time = time.time() - start_time
        print(f"Time taken to process fp vectors: {elapsed_time:.2f} seconds")
        os.makedirs(output_path, exist_ok=True)
        torch.save(fp_vector_params,
                   os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def add_fp_vector(base: PreTrainedModel,
                      fp_vector_path: str,
                      ratio: float,
                      skip_embed: bool = False,
                      special_tokens_map: Optional[Dict[int, int]] = None):
        """
        Adds a fingerprint vector to a base model.
        """
        print("fp_vector_path:", f'{fp_vector_path}/pytorch_model.bin')
        fp_vector = torch.load(f'{fp_vector_path}/pytorch_model.bin')
        for n, p in base.named_parameters():
            if 'embed_tokens' in n or 'word_embeddings' in n:
                if not skip_embed:
                    assert p.data.shape == fp_vector['fp_vector'][
                        n].shape, "embeds_token shape mismatch. Use --skip_embed to skip embedding layers."
                    p.data += ratio * fp_vector['fp_vector'][n]
                elif special_tokens_map:
                    for k, v in special_tokens_map.items():
                        p.data[k] += ratio * fp_vector['fp_embed'][v]
            elif 'lm_head' in n:
                if not skip_embed:
                    p.data += ratio * fp_vector['fp_vector'][n]
                elif special_tokens_map:
                    for k, v in special_tokens_map.items():
                        p.data[k] += ratio * fp_vector['fp_lmhead'][v]
            else:
                p.data += torch.tensor(
                    ratio, dtype=torch.float16) * fp_vector['fp_vector'][n]
        return base, fp_vector['cfg']

    @staticmethod
    def merge_fp_vectors(
            base_model_path: str,
            fp_vector_paths: List[str],
            output_path: str,
            ratios: List[float],
            skip_embed: bool = False,
            special_tokens_map: Optional[Dict[int, int]] = None) -> None:
        """
        Merges one or more fingerprint vectors into a base model.
        """
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                          torch_dtype='auto')
        if special_tokens_map:
            for k, v in special_tokens_map.items():
                base_model.get_input_embeddings().weight.data[k] = torch.zeros(
                    base_model.config.hidden_size)
                base_model.get_output_embeddings(
                ).weight.data[k] = torch.zeros(base_model.config.hidden_size)
        start_time = time.time()
        for fp_path, r in zip(fp_vector_paths, ratios):
            base_model, cfg = FingerprintVector.add_fp_vector(
                base_model, fp_path, r, skip_embed, special_tokens_map)
        elapsed_time = time.time() - start_time
        print(f"Time taken to process fp vectors: {elapsed_time:.2f} seconds")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model.save_pretrained(output_path,
                                   safe_serialization=True,
                                   max_shard_size='8GB')
        tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Fingerprint Vector Tool: extract or add fp_vector.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-parser for extracting fingerprint vectors
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract fingerprint vector from a base and a finetuned model')
    extract_parser.add_argument('--base_model_path',
                                type=str,
                                required=True,
                                help='Path to the base model')
    extract_parser.add_argument('--finetuned_model_path',
                                type=str,
                                required=True,
                                help='Path to the finetuned model')
    extract_parser.add_argument('--output_path',
                                type=str,
                                required=True,
                                help='Path to save the fp_vector')

    # Sub-parser for adding/merging fingerprint vectors
    add_parser = subparsers.add_parser(
        'add', help='Add one or more fingerprint vectors to a base model')
    add_parser.add_argument('--base_model_path',
                            type=str,
                            required=True,
                            help='Path to the base model')
    add_parser.add_argument('--fp_vector_path',
                            type=str,
                            nargs='+',
                            required=True,
                            help='List of paths to the fp_vectors')
    add_parser.add_argument('--output_path',
                            type=str,
                            required=True,
                            help='Path to save the merged model')
    add_parser.add_argument('--ratio',
                            type=float,
                            nargs='+',
                            required=True,
                            help='List of ratios for merging fp_vectors')
    add_parser.add_argument('--skip_embed',
                            action='store_true',
                            help='Whether to skip embedding layers')
    add_parser.add_argument('--special_tokens_map',
                            type=str,
                            default=None,
                            help='Special tokens map as a JSON string')

    args = parser.parse_args()

    if args.command == 'extract':
        FingerprintVector.extract_fp_vector(args.base_model_path,
                                            args.finetuned_model_path,
                                            args.output_path)
    elif args.command == 'add':
        if len(args.fp_vector_path) != len(args.ratio):
            raise ValueError(
                "The number of fp_vector_path and ratio must be the same.")
        special_tokens_map = json.loads(
            args.special_tokens_map) if args.special_tokens_map else None
        FingerprintVector.merge_fp_vectors(
            base_model_path=args.base_model_path,
            fp_vector_paths=args.fp_vector_path,
            output_path=args.output_path,
            ratios=args.ratio,
            skip_embed=args.skip_embed,
            special_tokens_map=special_tokens_map)


if __name__ == '__main__':
    main()
