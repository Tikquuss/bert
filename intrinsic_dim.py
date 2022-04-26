from extract_features import read_examples, convert_examples_to_features, model_fn_builder, input_fn_builder

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

import pandas as pd  
import torch
from intrinsics_dimension import mle_id, twonn_pytorch

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def ID_from_json_file(output_file, layer_indexes):
    jsonObj = pd.read_json(path_or_buf=output_file, lines=True)
    features = jsonObj["features"]
    data = {li : [] for li in layer_indexes}
    for sentence in features :
        #print([token["token"] for token in sentence])
        for (layer, li) in enumerate(layer_indexes) :
            data[li].append(torch.FloatTensor([token['layers'][layer]['values'] for token in sentence]))
    IDs = {}
    for li in layer_indexes :
        h = torch.stack(data[li]) # (batch_size, seq_len, dim)
        batch_size = h.size(0)
        IDs[li] = round(mle_id(h.view(batch_size, -1), k=2), 3) 

    return IDs

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host)
    )

    examples = read_examples(FLAGS.input_file)

    start_with_pad = False
    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer, start_with_pad = start_with_pad)

    unique_id_to_feature = {}
    for feature in features: unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    
    estimate_on_the_fly = True
    IDs = {}
    with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file, "w")) as writer:
        if estimate_on_the_fly :
            data = {li : [] for li in layer_indexes}
        for result in estimator.predict(input_fn, yield_single_examples=True):
            # print("\n\n*******")
            # print(len(result["layer_output_0"]))
            # for k, v in result.items() :
            #     print(k, "\n *** \n" , v)
            # print("\n\n*******")
            if estimate_on_the_fly :
                for (layer, li) in enumerate(layer_indexes) :
                    data[li].append(torch.FloatTensor(result["layer_output_%d" % layer]))
            else :
                unique_id = int(result["unique_id"])
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                # for i in range(FLAGS.max_seq_length):
                #     try : token = feature.tokens[i]
                #     except IndexError : token = "[PAD]"
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = result["layer_output_%d" % j]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                        ]
                        all_layers.append(layers)
                    features = collections.OrderedDict()
                    features["token"] = token
                    features["layers"] = all_layers
                    all_features.append(features)

                include_pad = True
                if include_pad :
                    for i in range(len(feature.tokens), FLAGS.max_seq_length):
                        all_layers = []
                        token = "[PAD]"
                        for (j, layer_index) in enumerate(layer_indexes):
                            layer_output = result["layer_output_%d" % j]
                            layers = collections.OrderedDict()
                            layers["index"] = layer_index
                            layers["values"] = [
                                round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                            ]
                            all_layers.append(layers)
                        features = collections.OrderedDict()
                        features["token"] = token
                        features["layers"] = all_layers
                        all_features.append(features)

                output_json["features"] = all_features
                writer.write(json.dumps(output_json) + "\n")

        ## 
        for li in layer_indexes :
            h = torch.stack(data[li]) # (batch_size, seq_len, dim)
            batch_size = h.size(0)
            IDs[li] = round(mle_id(h.view(batch_size, -1), k=2), 3) 

    if not estimate_on_the_fly :
        IDs = ID_from_json_file(FLAGS.output_file, layer_indexes)
    
    print("\n", "*"*5, f"\n Intrinsic Dimension : {IDs} \n", "*"*5, "\n")
    
if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()