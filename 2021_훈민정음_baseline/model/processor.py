import pandas as pd
import numpy as np
import os
import nsml
from nsml import HAS_DATASET, DATASET_PATH
from glob import glob
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import torch
##from transformers.tokenization_bert import BertTokenizer
from tokenization import BertTokenizer

import logging
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.utils import Metric
from data.dataloader import get_loader
from model.transformer import Transformer
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import time
import argparse
import math
from typing import Dict, List, Optional
import sys
from torch import Tensor
from model.generate_util import NGramRepeatBlock, BeamSearch

logger = logging.getLogger(__name__)


class Preprocess:
    """
        Preprocess class는 baseline code 거의 그대로 사용.
        make_model_input 함수만 모델 input에 맞게 변경.
        ToDo 1: source 간 [SEP] token 두기 -> make_set_as_df 함수에서 변경 가능할듯.
        :EX) 어제 그 동묘 고양이사건전말 떳다 ㅋㅋㅋ헐모양근데 사진은 잇어서 내용 아무렇게 올린덧 ...
            어제 그 동묘 고양이사건전말 떳다 ㅋㅋㅋ [SEP] 헐모양근데 사진은 잇어서 내용 아무렇게 올린덧 [SEP] ...
    """

    @staticmethod
    def make_dataset_list(path_list):
        json_data_list = []

        for path in path_list:
            with open(path) as f:
                json_data_list.append(json.load(f))

        return json_data_list

    @staticmethod
    def make_set_as_df(train_set, is_train=True):

        if is_train:
            train_dialogue = []
            train_dialogue_id = []
            train_summary = []
            for topic in train_set:
                for data in topic['data']:
                    train_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    train_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
                    # type, topic, gender, residentialProvince

                    train_summary.append(data['body']['summary'])

            train_data = pd.DataFrame(
                {
                    'dialogueID': train_dialogue_id,
                    'dialogue': train_dialogue,
                    'summary': train_summary
                }
            )
            return train_data

        else:
            test_dialogue = []
            test_dialogue_id = []
            for topic in test_set:
                for data in topic['data']:
                    test_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    test_dialogue.append(''.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))

            test_data = pd.DataFrame(
                {
                    'dialogueID': test_dialogue_id,
                    'dialogue': test_dialogue
                }
            )
            return test_data

    @staticmethod
    def train_valid_split(train_set, split_point):
        train_data = train_set.iloc[:split_point, :]
        val_data = train_set.iloc[split_point:, :]

        return train_data, val_data

    @staticmethod
    def make_model_input(dataset, vs, data_list, is_valid=False, is_test=False):
        if is_test:
            source = []
            encoder_input = dataset['dialogue']

            for i, row in tqdm(encoder_input.iteritems()):
                source.append(row)

            data_list.append(source)

            return encoder_input, data_list

        elif is_valid:
            # mlm, summ
            """
            source = []
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            decoder_output = dataset['summary']

            for src, tgt in zip(encoder_input.iteritems(), decoder_output.iteritems()):
                src = src[1]
                tgt = tgt[1]
                #combine = src + " " + tgt
                #combine2 = tgt + " " + src

                source.append(src)
                source.append(tgt)
                #source.append(combine)
                #source.append(combine2)

            vs = "asdf"
            #for i, row in tqdm(encoder_input.iteritems()):
            #    source.append(row)
            #    vs += " " + row

            #for i, row in tqdm(decoder_output.iteritems()):
            #    source.append(row)
            #    vs += " " + row

            data_list.append(source)

            """
            source = []
            target = []
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            decoder_output = dataset['summary']

            for i, row in tqdm(encoder_input.iteritems()):
                source.append(row)
                vs += " " + row

            for i, row in tqdm(decoder_output.iteritems()):
                target.append(row)
                vs += " " + row

            data_list.append(source)
            data_list.append(target)

            return encoder_input, decoder_input, decoder_output, vs, data_list

        else:
            # mlm, summ
            """
            source = []
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary']
            decoder_output = dataset['summary']

            step = 0
            vs = "asdf"
            for src, tgt in zip(encoder_input.iteritems(), decoder_input.iteritems()):
                src = src[1]
                tgt = tgt[1]
                #combine = src + " " + tgt
                #combine2 = tgt + " " + src

                source.append(src)
                source.append(tgt)
                #source.append(combine)
                #source.append(combine2)

            #for i, row in tqdm(encoder_input.iteritems()):
            #    source.append(row)
            #    if i == 0:
            #        vs += row
            #    else:
            #        vs += " " + row
            #    step += 1
            #for i, row in tqdm(decoder_input.iteritems()):
            #    source.append(row)
            #    vs += " " + row
            #    step += 1

            data_list.append(source)
            """

            source = []
            target = []
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary']
            decoder_output = dataset['summary']

            step = 0
            for i, row in tqdm(encoder_input.iteritems()):
                source.append(row)
                if i == 0:
                    vs += row
                else:
                    vs += " " + row
                step += 1

            for i, row in tqdm(decoder_input.iteritems()):
                target.append(row)
                vs += " " + row
                step += 1

            data_list.append(source)
            data_list.append(target)

            return encoder_input, decoder_input, decoder_output, vs, data_list


def train_data_loader(root_path):
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes


def add_padding_data(inputs, args, pad_token_idx):
    if len(inputs) < args.max_len:
        pad = np.array([pad_token_idx] * (args.max_len - len(inputs)))
        inputs = np.concatenate([inputs, pad])
    else:
        inputs = inputs[:args.max_len]

    return torch.tensor(inputs, dtype=torch.long, device=args.device).unsqueeze(0)


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tokenizer,
        beam_size=1,
        max_len_a=0,
        max_len_b=256,
        max_len=256,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.model = models
        self.pad = 1
        self.unk = 3
        self.eos = 2 if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tokenizer)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = BeamSearch(tokenizer)
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
            enc_input: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.
        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(enc_input, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, enc_input: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(enc_input, **kwargs)

    def _generate(
        self,
        enc_input: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(1)
            ],
        )
        encoder_input = enc_input

        src_tokens = enc_input
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.encoder(encoder_input)

        # encoder_outs should be = l X b X d
        encoder_outs = encoder_outs.transpose(0, 1)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        encoder_outs = encoder_outs.index_select(1, new_order)
        encoder_input = encoder_input.index_select(0, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        avg_attn_scores = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                #self.reorder_incremental_state(incremental_states, reorder_state)
                outs = self.reorder_encoder_out(
                    encoder_input, encoder_outs, reorder_state
                )
                encoder_input = outs["encoder_input"]
                encoder_outs = outs["encoder_out"]
            with torch.autograd.profiler.record_function("EnsembleModel: forward_decoder"):
                decoder_outs, _ = self.model.decoder(encoder_input, tokens[:, :step + 1], encoder_outs.transpose(0, 1))
                lprobs = self.model.projection(decoder_outs)
                lprobs = lprobs[:,-1,:]

                """lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )"""

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.min(prefix_lprobs) - 1
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.
        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = bbsz_idx // beam_size
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def reorder_encoder_out(
            self, encoder_input: Optional[List[Dict[str, List[Tensor]]]], encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        assert encoder_outs is not None
        new_outs = self._reorder_encoder_out(encoder_input, encoder_outs, new_order)
        return new_outs

    def _reorder_encoder_out(self, encoder_input: Dict[str, List[Tensor]], encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = encoder_out.index_select(1, new_order)

        if len(encoder_input) == 0:
            encoder_input = []
        else:
            encoder_input = (encoder_input).index_select(0, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_input": encoder_input,  # B x T
        }


def bind_model(model, parser):
    def save(dir_name, *parser):

        """
            학습된 모델하고 tokenizer 저장.
            nsml.save 불릴 때마다 디렉토리가 랜덤으로 지정되기에 tokenizer도 함께 저장..
        """
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model')

        torch.save(model.state_dict(), save_dir)

        print("저장 완료!")

    def load(dir_name, *parser):
        """
            모델이 저장된 dir_name에서 모델과 tokenizer 같이 load.
        """

        save_dir = os.path.join(dir_name, 'model')
        model.load_state_dict(torch.load(save_dir))

        print("로딩 완료!")

    def infer(test_path, **kwparser):

        start = time.time()
        # tokenizer = BertTokenizer.from_pretrained('aihub_vocab.txt', do_lower_case=False)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        preprocessor = Preprocess()

        test_json_path = os.path.join(test_path, 'test_data', '*')

        test_path_list = glob(test_json_path)
        test_path_list.sort()

        test_json_list = preprocessor.make_dataset_list(test_path_list)
        test_data = preprocessor.make_set_as_df(test_json_list)

        test_data_list = []
        encoder_input_test, test_data_list = preprocessor.make_model_input(test_data,
                                                                           None,
                                                                           test_data_list,
                                                                           is_test=True)

        test_loader, _ = get_loader(parser, tokenizer, test_data_list[0], mode='test')

        total_data = len(encoder_input_test)
        print(f"==Total test data length {total_data}==")

        model.eval()
        summary = []

        # pad_token_idx = tokenizer.convert_tokens_to_ids('[PAD]')  # 1
        # eos_token_idx = tokenizer.convert_tokens_to_ids('[EOS]')  # 2

        test_id = test_data['dialogueID']
        print(len(test_id))
        index_ = 0

        generator = SequenceGenerator(
            model,
            tokenizer,
            beam_size=2,
            max_len_b=128,
            max_len=128,
            min_len=1,
            no_repeat_ngram_size=2,
        )

        with torch.no_grad():
            for i, inputs in enumerate(tqdm(test_loader)):

                bos = 0
                with torch.no_grad():
                    hypos = generator.generate(inputs[0]['input_ids'], bos_token=bos)

                # target = B X L
                for idx, hypo in enumerate(hypos):
                    val = hypo[0]["tokens"].tolist()

                    h = tokenizer.decode(val, skip_special_tokens=True)
                    h = h.strip()
                    print(h)
                    # h = h.replace(' ', '').replace('_', ' ')

                    """
                    a85b7d4b-0e2d-527a-b8e0-d9c23849edc1
                    <class 'str'>
                    """
                    summary.append((test_data['dialogueID'][index_], h))
                    index_ += 1

                    # time checking
                    ti = time.time() - start
                    if ti >= 3595:
                        break

                if ti >= 3595:
                    break

                print(len(summary))
                print("=========================")

        print("==FINISH==")
        print(len(summary))

        """
            DONOTCHANGE: They are reserved for nsml
            리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있음.
        """
        return summary

    """
        DONOTCHANGE: They are reserved for nsml
        nsml에서 지정한 함수에 접근할 수 있도록 하는 함수.
    """
    nsml.bind(save=save, load=load, infer=infer)


def makeVocab(vocab_sentence):
    vocab_name = "makeVocab.txt"
    with open(vocab_name, "w", encoding="utf-8") as f:
        f.write(str(vocab_sentence))

    return vocab_name


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}

    def run(self, inputs, mode=None):
        logits = self.config['model'](inputs[0]['input_ids'], inputs[0]['decoder_input_ids'])
        # logits = self.config['model'](inputs[1]['source'], inputs[0]['decoder_input_ids'])

        loss = self.config['criterion'](logits, inputs[0]['labels'].view(-1))

        return loss

    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']

        return loss

    def get_object(self, model, ignore_index):
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.args.warmup_ratio * train_total,
            num_training_steps=train_total)

        return scheduler

    def make_vocab_and_split_data(self, preprocessor, train_set, train_data_list, valid_set, valid_data_list):
        """
            BPE tokenizer 학습을 위한 함수.
            vocab_sentence: training, validation dataset의 source & target을 string 값으로 저장 -> for vocab.txt

            tokenizer: training, validation 단계에서의 tokenizer.
            tokenizer_for_save: testing 단계에서 tokenizer를 불러오기 위해 사용.

            train_data_list: train data loader를 만들기 위한 train data set을 갖고 있음 | [[source1, source2...], [target1, target2...]]
            valid_data_list: validation data loader를 만들기 위한 validation data set을 갖고 있음 | [[source1, source2...], [target1, target2...]]
        """

        vocab_sentence = ""
        encoder_input_train, decoder_input_train, decoder_output_train, vocab_sentence, train_data_list = preprocessor.make_model_input(
            train_set,
            vocab_sentence,
            train_data_list)

        encoder_input_val, decoder_input_val, decoder_output_val, vocab_sentence, valid_data_list = preprocessor.make_model_input(
            valid_set,
            vocab_sentence,
            valid_data_list,
            is_valid=True)

        # tokenizer = BertTokenizer.from_pretrained('aihub_vocab.txt', do_lower_case=False)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        return train_data_list, valid_data_list, tokenizer, None  # tokenizer, tokenizer_for_save

    def model_setting(self):
        if self.args.mode == 'test':
            # tokenizer = "tokenizer_init"
            # tokenizer = BertTokenizer.from_pretrained('aihub_vocab.txt', do_lower_case=False)
            tokenizer = AutoTokenizer.from_pretrained("roberta-large")

            model = Transformer(self.args, len(tokenizer))
            model.to(self.args.device)
            
            # 모델 바인딩.
            # bind_model(model=model, parser=self.args, tokenizer=tokenizer)
            bind_model(model=model, parser=self.args)
            if self.args.pause:
                nsml.paused(scope=locals())

        else:
            """
            train_data_list = []
            valid_data_list = []

            train_path_list = train_data_loader(DATASET_PATH)
            train_path_list.sort()

            preprocessor = Preprocess()

            train_json_list = preprocessor.make_dataset_list(train_path_list)
            train_data = preprocessor.make_set_as_df(train_json_list)

            # train valid -> 9:1
            split_point = int(len(train_data) * 0.95)

            train_set, valid_set = preprocessor.train_valid_split(train_data, split_point)
            train_data_list, valid_data_list, tokenizer, tokenizer_for_save = self.make_vocab_and_split_data(
                preprocessor,
                train_set,
                train_data_list,
                valid_set,
                valid_data_list)

            train_loader, _ = get_loader(self.args, tokenizer, train_data_list)
            valid_loader, _ = get_loader(self.args, tokenizer, valid_data_list)

            print("Train: ", len(train_loader))
            print("Valid: ", len(valid_loader))

            loader = {'train': train_loader, 'valid': valid_loader}
            """
            tokenizer = AutoTokenizer.from_pretrained("roberta-large")
            model = Transformer(self.args, len(tokenizer))
            model.to(self.args.device)

            # 모델 바인딩
            # bind_model(model=model, parser=self.args, tokenizer=tokenizer_for_save)
            bind_model(model=model, parser=self.args)
            if self.args.pause:
                nsml.paused(scope=locals())

            # nsml.load(checkpoint='9', session='nia2045/final_dialogue/169')
            """
            pad_id = tokenizer.convert_tokens_to_ids('<pad>')  # tokenizer.convert_tokens_to_ids('[PAD]')

            criterion, optimizer = self.get_object(model, ignore_index=pad_id)

            if self.args.mode == 'train':
                scheduler = self.get_scheduler(optimizer, loader['train'])
            else:
                scheduler = None

            config = {'loader': loader,
                      'optimizer': optimizer,
                      'criterion': criterion,
                      'scheduler': scheduler,
                      'tokenizer': tokenizer,
                      'args': self.args,
                      'model': model}

            self.config = config
            """
            # 다른 session에서 불러올 때 사용.
            nsml.load(checkpoint='6', session='nia2045/final_dialogue/170')
            nsml.save('banana')
            exit()

            return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs = batch
            loss = self.run(inputs, mode='train')

            loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            self.progress(loss.data)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):
                inputs = batch
                loss = self.run(inputs, mode='valid')

                self.progress(loss.data)

        return self.return_value()