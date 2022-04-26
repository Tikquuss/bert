from extract_features import read_examples, convert_examples_to_features, model_fn_builder, input_fn_builder
from extract_features import InputExample

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

import os
import pandas as pd  
import tqdm
import torch
from intrinsics_dimension import mle_id, twonn_pytorch
import wandb
    
# # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# import logging
# tf.get_logger().setLevel(logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True # https://stackoverflow.com/a/57294815/11814682

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("group_texts", True, "group texts or not")
flags.DEFINE_bool("estimate_on_the_fly", True, "estimate on the fly, or group in a json file before estimation")
flags.DEFINE_bool("use_wandb", True, "")

dir_path = "/content/bert/seed_0"
cpt_list = [os.path.join(dir_path, rep, "bert.ckpt") for rep in os.listdir(dir_path)]

flags.DEFINE_multi_string(
#flags.DEFINE_list(
    name = "checkpoints_list", 
    default = None, 
    #default = cpt_list, 
    help = "list of checkpoints", 
    #flag_values=_flagvalues.FLAGS, 
    required=False
)

####################################
# See extract_features.py for arguments #
####################################

def group_texts_fn(examples, seq_length, tokenizer):
    """Concatenate all texts"""
    concatenated_examples = []
    for example in tqdm.tqdm(examples, desc="concatenate all texts ..."):
        concatenated_examples += tokenizer.tokenize(example.text_a)
        if example.text_b : concatenated_examples += tokenizer.tokenize(example.text_b)
    total_length = len(concatenated_examples)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // seq_length) * seq_length if total_length > seq_length else total_length
    # Split by chunks of max_len.
    #result = [concatenated_examples[i : i + seq_length] for i in range(0, total_length, seq_length)]
    result = [InputExample(unique_id=i, text_a=concatenated_examples[i : i + seq_length], text_b=None) for i in range(0, total_length, seq_length)]
    return result

def ID_from_json_file(output_file, layer_indexes):
    jsonObj = pd.read_json(path_or_buf=output_file, lines=True)
    features = jsonObj["features"]
    data = {li : [] for li in layer_indexes}
    for sentence in tqdm.tqdm(features, desc="sentences ...") :
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
    if FLAGS.group_texts : examples = group_texts_fn(examples, FLAGS.max_seq_length, tokenizer)

    start_with_pad = False
    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        already_tokenize=FLAGS.group_texts, start_with_pad = start_with_pad)

    unique_id_to_feature = {}
    for feature in features: unique_id_to_feature[feature.unique_id] = feature

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    checkpoints_list = FLAGS.checkpoints_list
    verbose = False
    if checkpoints_list is None :
        verbose = True
        checkpoints_list = [FLAGS.init_checkpoint]
    
    use_wandb = FLAGS.use_wandb and FLAGS.checkpoints_list is not None
    if use_wandb :
        wandb.init(
            project="phase_trantision_google_bert_checkpoints",
            entity="grokking_ppsp",
            group="multi_bert_checkpoint",
            notes="https://github.com/google-research/language/tree/master/language/multiberts"
            #resume=True
        )
        

    all_IDs = []
    for cpt in tqdm.tqdm(checkpoints_list, desc="checkpoints_list ...")  :
        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=cpt,
            layer_indexes=layer_indexes,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
            verbose = verbose
        )

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.batch_size)

        estimate_on_the_fly = FLAGS.estimate_on_the_fly
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

                    include_pad = not FLAGS.group_texts
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
        
        all_IDs.append(IDs)
        if use_wandb : wandb.log({f"ID_layer_{k}" : v for k, v in IDs.items()})
        print("\n", "*"*5, f"\n Intrinsic Dimension : {IDs} \n", "*"*5, "\n")

    torch.save(all_IDs, os.path.join(os.path.dirname(FLAGS.output_file), "IDs.pth"))
    
if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()