import transformers.generation_tf_utils as GenUtils
from DecodingStrategies import WordInclusion
import tensorflow as tf


def _generate_no_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, temperature, top_k, top_p,
                             repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id,
                             batch_size, vocab_size,
                             encoder_outputs, attention_mask, use_cache, return_dict_in_generate,
                             repeitionPenaltyFunc,
                             **kwargs) -> \
        GenUtils.Union[
            GenUtils.TFGreedySearchOutput, GenUtils.TFSampleOutput, tf.Tensor]:
    """
    Generate sequences for each example without beam search (num_beams == 1). All returned sequences are generated
    independently.
    """

    # length of generated sentences / unfinished sentences
    unfinished_sents = tf.ones_like(input_ids[:, 0])
    sent_lengths = tf.ones_like(input_ids[:, 0]) * max_length

    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and kwargs["output_scores"]) else None
    decoder_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
    cross_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
    decoder_hidden_states = () if (return_dict_in_generate and kwargs["output_hidden_states"]) else None
    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if self.config.is_encoder_decoder:
        encoder_attentions = (
            kwargs["encoder_attentions"] if (return_dict_in_generate and kwargs["encoder_attentions"]) else None
        )
        encoder_hidden_states = (
            kwargs["encoder_hidden_states"]
            if (return_dict_in_generate and kwargs["encoder_hidden_states"])
            else None
        )

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **kwargs
        )
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=kwargs["output_attentions"],
            output_hidden_states=kwargs["output_hidden_states"],
        )
        next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if kwargs["output_scores"]:
                scores += (next_token_logits,)
            if kwargs["output_attentions"]:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if kwargs["output_hidden_states"]:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            next_token_logits_penalties = repeitionPenaltyFunc(
                input_ids, next_token_logits, repetition_penalty
            )
            next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_tokens = GenUtils.calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
            # create banned_tokens boolean mask
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                )

            next_token_logits = GenUtils.set_tensor_by_indices_to_value(
                next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
            )

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = GenUtils.calc_banned_bad_words_ids(input_ids, bad_words_ids)

            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                )

            next_token_logits = GenUtils.set_tensor_by_indices_to_value(
                next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            # create eos_token_id boolean mask
            is_token_logit_eos_token = tf.convert_to_tensor(
                [True if token is eos_token_id else False for token in range(vocab_size)], dtype=tf.bool
            )
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [batch_size, vocab_size])

            next_token_logits = GenUtils.set_tensor_by_indices_to_value(
                next_token_logits, eos_token_indices_mask, -float("inf")
            )

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logits = GenUtils.tf_top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            next_token = tf.squeeze(
                tf.random.categorical(next_token_logits, dtype=tf.int32, num_samples=1), axis=1
            )
        else:
            # Greedy decoding
            next_token = tf.math.argmax(next_token_logits, axis=-1, output_type=tf.int32)

        WordInclusion.nextTokenHook(next_token)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = tf.concat([input_ids, tf.expand_dims(tokens_to_add, -1)], 1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = tf.math.multiply(
                unfinished_sents, tf.cast(eos_in_sents, tf.int32)
            )
            sent_lengths = (
                    sent_lengths * (1 - is_sents_unfinished_and_token_to_add_is_eos)
                    + cur_len * is_sents_unfinished_and_token_to_add_is_eos
            )

            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents -= is_sents_unfinished_and_token_to_add_is_eos

        # stop when there is a </s> in each sentence, or if we exceed the maximum length
        if tf.math.reduce_max(unfinished_sents) == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = tf.concat(
                [attention_mask, tf.ones((GenUtils.shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
            )

    # if there are different sentences lengths in the batch, some batches have to be padded
    min_sent_length = tf.math.reduce_min(sent_lengths)
    max_sent_length = tf.math.reduce_max(sent_lengths)
    if min_sent_length != max_sent_length:
        assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
        # finished sents are filled with pad_token
        padding = tf.ones([batch_size, max_sent_length.numpy()], dtype=tf.int32) * pad_token_id

        # create length masks for tf.where operation
        broad_casted_sent_lengths = tf.broadcast_to(
            tf.expand_dims(sent_lengths, -1), [batch_size, max_sent_length]
        )
        broad_casted_range = tf.transpose(
            tf.broadcast_to(tf.expand_dims(tf.range(max_sent_length), -1), [max_sent_length, batch_size])
        )

        decoded = tf.where(broad_casted_range < broad_casted_sent_lengths, input_ids, padding)
    else:
        decoded = input_ids

    if return_dict_in_generate:
        if do_sample:
            if self.config.is_encoder_decoder:
                return GenUtils.TFSampleEncoderDecoderOutput(sequences=decoded, scores=scores,
                                                             encoder_attentions=encoder_attentions,
                                                             encoder_hidden_states=encoder_hidden_states,
                                                             decoder_attentions=decoder_attentions,
                                                             cross_attentions=cross_attentions,
                                                             decoder_hidden_states=decoder_hidden_states,
                                                             )
            else:
                return GenUtils.TFSampleDecoderOnlyOutput(sequences=decoded, scores=scores,
                                                          attentions=decoder_attentions,
                                                          hidden_states=decoder_hidden_states,
                                                          )
        else:
            if self.config.is_encoder_decoder:
                return GenUtils.TFGreedySearchEncoderDecoderOutput(sequences=decoded, scores=scores,
                                                                   encoder_attentions=encoder_attentions,
                                                                   encoder_hidden_states=encoder_hidden_states,
                                                                   decoder_attentions=decoder_attentions,
                                                                   cross_attentions=cross_attentions,
                                                                   decoder_hidden_states=decoder_hidden_states,
                                                                   )
            else:
                return GenUtils.TFGreedySearchDecoderOnlyOutput(sequences=decoded, scores=scores,
                                                                attentions=decoder_attentions,
                                                                hidden_states=decoder_hidden_states,
                                                                )
    else:
        return decoded


def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        forced_bos_token_id,
        forced_eos_token_id,
        return_dict_in_generate,
        **kwargs,
) -> GenUtils.Union[GenUtils.TFBeamSearchOutput, GenUtils.TFBeamSampleOutput, tf.Tensor]:
    """Generate sequences for each example with beam search."""

    # generated hypotheses
    generated_hyps = [
        GenUtils.BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores_begin = tf.zeros((batch_size, 1), dtype=tf.float32)
        beam_scores_end = tf.ones((batch_size, num_beams - 1), dtype=tf.float32) * (-1e9)
        beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)
    else:
        beam_scores = tf.zeros((batch_size, num_beams), dtype=tf.float32)

    beam_scores = tf.reshape(beam_scores, (batch_size * num_beams,))

    # cache compute states
    past = encoder_outputs
    # to stay similar to torch : past = (encoder_outputs, None) if encoder_outputs is not None else None

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and kwargs["output_scores"]) else None
    decoder_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
    cross_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
    decoder_hidden_states = () if (return_dict_in_generate and kwargs["output_hidden_states"]) else None
    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if self.config.is_encoder_decoder:
        encoder_attentions = (
            kwargs["encoder_attentions"] if (return_dict_in_generate and kwargs["encoder_attentions"]) else None
        )
        encoder_hidden_states = (
            kwargs["encoder_hidden_states"]
            if (return_dict_in_generate and kwargs["encoder_hidden_states"])
            else None
        )

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **kwargs
        )
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=kwargs["output_attentions"],
            output_hidden_states=kwargs["output_hidden_states"],
        )
        next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]

        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            next_token_logits_penalties = GenUtils._create_next_token_logits_penalties(
                input_ids, next_token_logits, repetition_penalty
            )
            next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)

        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if self.config.is_encoder_decoder and do_sample is False:
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits,
                cur_len=cur_len,
                max_length=max_length,
                forced_bos_token_id=forced_bos_token_id,
                forced_eos_token_id=forced_eos_token_id,
            )
        #             calculate log softmax score
        scores = tf.nn.log_softmax(next_token_logits, axis=-1)  # (batch_size * num_beams, vocab_size)

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            # create eos_token_id boolean mask
            num_batch_hypotheses = batch_size * num_beams

            is_token_logit_eos_token = tf.convert_to_tensor(
                [True if token is eos_token_id else False for token in range(vocab_size)], dtype=tf.bool
            )
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [num_batch_hypotheses, vocab_size])

            scores = GenUtils.set_tensor_by_indices_to_value(scores, eos_token_indices_mask, -float("inf"))

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            num_batch_hypotheses = batch_size * num_beams
            banned_tokens = GenUtils.calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            # create banned_tokens boolean mask
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                )

            scores = GenUtils.set_tensor_by_indices_to_value(
                scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
            )

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = GenUtils.calc_banned_bad_words_ids(input_ids, bad_words_ids)

            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                )

            scores = GenUtils.set_tensor_by_indices_to_value(
                scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
            )

        assert GenUtils.shape_list(scores) == [batch_size * num_beams, vocab_size]

        if do_sample:
            _scores = scores + tf.broadcast_to(
                beam_scores[:, None], (batch_size * num_beams, vocab_size)
            )  # (batch_size * num_beams, vocab_size)

            # Top-p/top-k filtering
            _scores = GenUtils.tf_top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            _scores = tf.reshape(_scores, (batch_size, num_beams * vocab_size))

            next_tokens = GenUtils.sample_without_replacement(
                _scores, num_samples=2 * num_beams
            )  # (batch_size, 2 * num_beams)
            # Compute next scores
            next_scores = tf.gather(_scores, next_tokens, batch_dims=1)  # (batch_size, 2 * num_beams)

            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
            next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
            next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
        else:
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            next_scores = scores + tf.broadcast_to(
                beam_scores[:, None], (batch_size * num_beams, vocab_size)
            )  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis across beams)
            next_scores = tf.reshape(
                next_scores, (batch_size, num_beams * vocab_size)
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = tf.math.top_k(next_scores, k=2 * num_beams, sorted=True)

        assert GenUtils.shape_list(next_scores) == GenUtils.shape_list(next_tokens) == [batch_size, 2 * num_beams]

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if kwargs["output_scores"]:
                scores += (next_token_logits,)
            if kwargs["output_attentions"]:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if kwargs["output_hidden_states"]:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), f"Batch can only be done if at least {num_beams} beams have been generated."
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence or last iteration
                if (eos_token_id is not None) and (token_id.numpy() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        tf.identity(input_ids[effective_beam_id]), beam_token_score.numpy()
                    )
                else:
                    # add next predicted token if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                tf.reduce_max(next_scores[batch_idx]).numpy(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
        beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
        beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)

        ############# Custom Hackz
        WordInclusion.nextTokenBeamsHook(next_batch_beam)

        # re-order batch and update current length
        input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
        input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
        cur_len = cur_len + 1

        # re-order internal states
        if past is not None:
            past = self._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = tf.concat(
                [attention_mask, tf.ones((GenUtils.shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
            )

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(batch_size):
        # Add all open beam hypothesis to generated_hyps
        if done[batch_idx]:
            continue
        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
                (token_id % vocab_size).numpy().item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            if not tf.reduce_all(
                    next_scores[batch_idx, :num_beams] == tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
            ):
                raise ValueError(
                    f"If batch_idx is not done, final next scores: {next_scores[:, :num_beams][batch_idx]} have "
                    "to equal to accumulated beam_scores: "
                    f"{tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]}"
                )
        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].numpy().item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths_list = []
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths_list.append(len(best_hyp))
            best.append(best_hyp)
    assert output_batch_size == len(
        best
    ), f"Output batch size {output_batch_size} must match output beam hypotheses {len(best)}"

    sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)

    # shorter batches are filled with pad_token
    if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths).numpy():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
        decoded_list = []

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            assert sent_lengths[i] == GenUtils.shape_list(hypo)[0]
            # if sent_length is max_len do not pad
            if sent_lengths[i] == sent_max_len:
                decoded_slice = hypo
            else:
                # else pad to sent_max_len
                num_pad_tokens = sent_max_len - sent_lengths[i]
                padding = pad_token_id * tf.ones((num_pad_tokens,), dtype=tf.int32)
                decoded_slice = tf.concat([hypo, padding], axis=-1)

                # finish sentence with EOS token
                if sent_lengths[i] < max_length:
                    decoded_slice = tf.where(
                        tf.range(sent_max_len, dtype=tf.int32) == sent_lengths[i],
                        eos_token_id * tf.ones((sent_max_len,), dtype=tf.int32),
                        decoded_slice,
                    )
            # add to list
            decoded_list.append(decoded_slice)

        decoded = tf.stack(decoded_list)
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = tf.stack(best)

    if return_dict_in_generate:
        if do_sample and self.config.is_encoder_decoder:
            return GenUtils.TFBeamSampleEncoderDecoderOutput(
                sequences=decoded,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        elif do_sample and not self.config.is_encoder_decoder:
            return GenUtils.TFBeamSampleDecoderOnlyOutput(
                sequences=decoded,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        elif self.config.is_encoder_decoder:
            return GenUtils.TFBeamSearchEncoderDecoderOutput(
                sequences=decoded,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GenUtils.TFBeamSearchDecoderOnlyOutput(
                sequences=decoded,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return decoded
