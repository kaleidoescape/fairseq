# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import numpy as np


logger = logging.getLogger(__name__)


def uniform(dataset_sizes: List[int]):
    return [1.0] * len(dataset_sizes)
    
def make_ratio_sampling(ratios):
    def sampling_func(dataset_sizes):
        return ratios

    return sampling_func


def temperature_sampling(langs, data_params_list, dataset_sizes, temp):
    r"""
    Return a tuple of sampling ratios across the datasets, and sampling ratios
    across the languages of all the datasets (will be identical only when the
    datasets represent unique languages on both source and target side). 
    """
    lang_sizes = [0]*len(langs)
    for i, params in enumerate(data_params_list):
        src_lang, tgt_lang = params['src'], params['tgt']
        src_idx, tgt_idx = langs.index(src_lang), langs.index(tgt_lang)
        lang_sizes[src_idx] += dataset_sizes[i]
        lang_sizes[tgt_idx] += dataset_sizes[i]
    lang_probs = [(size / sum(lang_counts)) ** (1.0 / temp) for size in lang_sizes]
    probs = [(size / sum(dataset_sizes)) ** (1.0 / temp) for size in dataset_sizes]
    return probs, lang_probs

def make_temperature_sampling(temp=1.0):
    def sampling_func(langs, data_params_list, dataset_sizes):
        return temperature_sampling(dataset_sizes, temp)

    return sampling_func


def sinkhorn_temperature_sampling_distribution(
        langs, data_params_list, dataset_sizes, temp=1.0
    ):
    r"""
    Convert dataset sizes into a distribution which takes into account both the
    availability of a particular lang pair together, as well as the
    availability of a particular lang alone across the pairs. We use the
    Sinkhorn-Knopp algorithm to convert a matrix of lang pair counts into 
    a doubly stochastic matrix, which is then converted into the temperature 
    sampled probabilities. 

    Motivation (section 3.4): https://arxiv.org/abs/2010.11125
    Sinkhorn-Knopp paper: http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
    Our fork of skp: https://github.com/kaleidoescape/sinkhorn_knopp
    """
    from sinkhorn_knopp import sinkhorn_knopp as skp

    #fill a matrix with language pair counts across datasets
    slangs = sorted(langs)
    A = np.zeros((len(slangs), len(slangs))) #(src, tgt)
    for i, params in enumerate(data_params_list):
        src_lang, tgt_lang = params['src'], params['tgt']
        src_idx, tgt_idx = slangs.index(src_lang), slangs.index(tgt_lang)
        A[src_idx, tgt_idx] += dataset_sizes[i]
    logger.info(f"Data counts for {langs}: {A}")

    #if any row is fully 0, we have to remove the lang from both axes because
    #we need a square matrix with total support to perform sinkhorn-knopp
    #This will cause us to miss a lang that is ever only used as src or only
    #used as tgt, but for multiling models, we typically use both directions
    zero_rows = np.where(~A.any(axis=0))[0]
    if zero_rows.size > 0:
        [slangs.remove(langs[i]) for i in zero_rows]
        logger.warning(
            f"Ignoring all datasets for langs {[langs[i] for i in zero_rows]}"
            " because this lang is never used as the src")
    A = np.delete(A, zero_rows, 0)
    A = np.delete(A, zero_rows, 1)
    #also if any col is fully 0, we have to remove the lang from both axes 
    zero_cols = np.where(~A.any(axis=1))[0]
    if zero_cols.size > 0:
        [slangs.remove(langs[i]) for i in zero_cols]
        logger.warning(
            f"Ignoring all datasets for langs {[langs[i] for i in zero_cols]}"
            " because this lang is never used as the tgt")
    A = np.delete(A, zero_cols, 0)
    A = np.delete(A, zero_cols, 1)
    if zero_rows.size > 0 or zero_cols.size > 0:
        logger.info(f"Remaining data counts for {langs}: {A}")

    #make matrix doubly stochastic (rows and cols each sum to 1)
    #and convert that into a new probability distrib with temperature
    sk = skp.SinkhornKnopp()
    probs = sk.fit(A) ** (1 / temp)
    probs = probs / sum(probs)
    logger.info(f"Sinkhorn temperature sampled probs for {slangs}: {probs}")
    return probs, slangs

def sinkhorn_temperature_sampling(
        langs, data_params_list, dataset_sizes, temp=1.0
    ):
    r"""
    Return a tuple of sampling ratios across the datasets, and sampling ratios
    across the language pairs in all the datasets (will be identical only when
    the datasets represent unique language directions). 
    """
    #get the ratios back for each of the datasets (datasets with langs 
    #never used for translating into or translating out of get a ratio of 0)
    ratios = []
    probs, slangs = sinkhorn_temperature_sampling_distribution(
        langs, data_params_list, dataset_sizes, temp)
    for params in data_params_list:
        src_lang, tgt_lang = params['src'], params['tgt']
        if src_lang not in langs or tgt_lang not in langs:
            prob = 0 
        else:
            src_idx, tgt_idx = slangs.index(src_lang), slangs.index(tgt_lang)
            prob = probs[src_idx, tgt_idx]
        ratios.append(prob)

    return ratios, probs

def make_sinkhorn_temperature_sampling(temp=1.0):
    def sampling_func(langs, data_params_list, dataset_sizes):
        return sinkhorn_temperature_sampling(langs, data_params_list, dataset_sizes, temp)

    return sampling_func


class SamplingMethod:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--sampling-method",
            choices=[
                "uniform",
                "temperature",
                "concat",
                "RoundRobin", #see translation_multi_simple_epoch.get_batch_iterator
                "sinkhorn",
            ],
            type=str,
            default="concat",
            help="The method to sample data per language pairs",
        )
        parser.add_argument(
            "--sampling-temperature",
            default=1.5,
            type=float,
            help="only works with --sampling-method {temperature,sinkhorn}",
        )

    @staticmethod
    def build_sampler(args, task):
        return SamplingMethod(args, task)

    def __init__(self, args, task):
        self.args = args
        self.task = task

    def is_adaptive(self):
        return False

    def sampling_method_selector(self):
        args = self.args
        logger.info(f"selected sampler: {args.sampling_method}")
        if args.sampling_method == "uniform":
            return uniform
        elif args.sampling_method == "temperature" or self.is_adaptive():
            return make_temperature_sampling(
                float(args.sampling_temperature)
            )
        elif args.sampling_method == "sinkhorn":
            return make_sinkhorn_temperature_sampling(
                float(args.sampling_temperature)
            )
        else:
            return None #default to concating all data set together
