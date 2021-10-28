# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

import copy
import json
import math
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
# from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.streaming_utils import *
from nemo.utils import logging

can_gpu = torch.cuda.is_available()


def get_wer_feat(mfst, asr, frame_len, tokens_per_chunk, delay, vad_delay, preprocessor_cfg, model_stride_in_secs, device, 
                vad=None, threshold=0.4):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []
    total_durations_to_asr = []
    original_durations = []
    total_speech_segments = []
    total_streaming_vad_logits = []

    with open(mfst, "r") as mfst_f:
        for l in mfst_f:
            asr.reset()
            row = json.loads(l.strip())
            if vad:
                vad.reset()
                vad.read_audio_file(row['audio_filepath'], offset=0, duration=None, delay=vad_delay, model_stride_in_secs=1)          
                streaming_vad_logits, speech_segments = vad.decode(threshold=threshold)
                speech_segments = [list(i) for i in speech_segments]
                speech_segments.sort(key=lambda x: x[0])

                final_hyp = " "
                total_duration_to_asr = 0
                for i in range(len(speech_segments)): 
                    asr.reset()
                    speech_segment = speech_segments[i] 
                    offset = speech_segment[0] - frame_len * 4
                    duration = speech_segment[1] - speech_segment[0] + frame_len * 4

                    asr.read_audio_file(row['audio_filepath'], offset, duration, delay, model_stride_in_secs)
                    hyp = asr.transcribe(tokens_per_chunk, delay) + " "
                    # there should be some better method to merge the hyps of segments.
                    final_hyp += hyp
                    total_duration_to_asr += duration

                final_hyp = final_hyp[1:-1]
                # print(final_hyp)

                hyps.append(final_hyp)
                refs.append(row['text'])
                total_durations_to_asr.append(total_duration_to_asr)
                original_durations.append(row['duration'])
                total_speech_segments.append(speech_segments)
                total_streaming_vad_logits.append(streaming_vad_logits)

            else:
                asr.read_audio_file(row['audio_filepath'], offset=0, duration=None, 
                                    delay=delay, model_stride_in_secs=model_stride_in_secs)
                hyp = asr.transcribe(tokens_per_chunk, delay)
                # print(hyp)
                hyps.append(hyp)
                refs.append(row['text'])
                total_durations_to_asr.append(row['duration'])
                speech_segments = "ALL"
                total_speech_segments.append(speech_segments)

    wer = word_error_rate(hypotheses=hyps, references=refs)
    print(wer)
    if vad:
        print(f"VAD reduces total durations for ASR inference from {int(sum(original_durations))} seconds to {int(sum(total_durations_to_asr))} seconds, by filtering out some noise or musci")
    
    return hyps, refs, wer, total_durations_to_asr, total_speech_segments, total_streaming_vad_logits



def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    parser.add_argument(
        "--vad_model", type=str, required=False, help="Path to asr model .nemo file",
    )
    parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--total_buffer_in_secs",
        type=float,
        default=4.0,
        help="Length of buffer (chunk + left and right padding) in seconds ",
    )
    parser.add_argument(
        "--total_vad_buffer_in_secs",
        type=float,
        default=0.63,
        help="Used for streaming VAD, Length of buffer (chunk + left and right padding) in seconds ",
    )
    parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )

    parser.add_argument(
        "--vad_before_asr",
        help="Whether to perform VAD before ASR",
        action='store_true',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="threshold",
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    if args.vad_model:
        if args.vad_model.endswith('.nemo'):
            logging.info(f"Using local ASR model from {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.vad_model)
        elif args.vad_model.endswith('.ckpt'):
            logging.info(f"Using local ASR model from {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(args.vad_model)
        else:
            logging.info(f"Using NGC cloud ASR model {args.vad_model}")
            vad_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.vad_model)

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * args.model_stride

    total_buffer = args.total_buffer_in_secs
    total_vad_buffer = args.total_vad_buffer_in_secs

    chunk_len = args.chunk_len_in_ms / 1000

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)

    vad_mid_delay = math.ceil((chunk_len + (total_vad_buffer - chunk_len) / 2) / 1)

    frame_vad = None
    threshold = args.threshold
    if args.vad_before_asr and args.vad_model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vad_model.eval()
        vad_model = vad_model.to(vad_model.device)
        # note VAD model itself is none normalized
        frame_vad = FrameBatchVAD(
            vad_model=vad_model, 
            frame_len=chunk_len, 
            total_buffer=args.total_vad_buffer_in_secs, 
            batch_size=args.batch_size,
        )
    
    frame_asr = FrameBatchASR(
        asr_model=asr_model, 
        frame_len=chunk_len, 
        total_buffer=args.total_buffer_in_secs, 
        batch_size=args.batch_size,
    )

    hyps, refs, wer, total_durations_to_asr, total_speech_segments, total_streaming_vad_logits = get_wer_feat(
        args.test_manifest,
        frame_asr,
        chunk_len,
        tokens_per_chunk,
        mid_delay,
        vad_mid_delay,
        cfg.preprocessor, # ASR and VAD have same preprocessor except normalization during training
        model_stride_in_secs,
        asr_model.device,
        frame_vad,
        threshold
    )
    
    logging.info(f"WER is {round(wer, 2)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    if args.output_path is not None:

        fname = (
            os.path.splitext(os.path.basename(args.asr_model))[0]
            + "_"
            + os.path.splitext(os.path.basename(args.test_manifest))[0]
            + "_"
            + str(args.chunk_len_in_ms)
            + "_"
            + str(int(total_buffer * 1000))
            + ".json"
        )
        hyp_json = os.path.join(args.output_path, fname)
        os.makedirs(args.output_path, exist_ok=True)
        with open(hyp_json, "w") as out_f:
            for i, hyp in enumerate(hyps):
                if args.vad_before_asr:
                    record = {
                        "pred_text": hyp,
                        "text": refs[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                        "total_duration_to_asr": total_durations_to_asr[i],
                        "total_speech_segments": total_speech_segments[i],
                        "total_streaming_vad_logits": str(total_streaming_vad_logits[i]),
                    }
                else:
                    record = {
                        "pred_text": hyp,
                        "text": refs[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                        "total_duration_to_asr": total_durations_to_asr[i],
                        "total_speech_segments": total_speech_segments[i],
                    }
                out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
