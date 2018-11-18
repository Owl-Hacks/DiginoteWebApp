import difflib
import math
import random
import string

from leven import levenshtein
import gluonnlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mxnet as mx
import numpy as np
from skimage import transform as skimage_tf
from utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from tqdm import tqdm
from pycontractions import Contractions

from utils.CTCDecoder.BeamSearch import ctcBeamSearch
from utils.CTCDecoder.LexiconSearch import ctcLexiconSearchWithNLTK
from utils.lexicon_search import LexiconSearch
from utils.expand_bounding_box import expand_bounding_box
from utils.sclite_helper import Sclite_helper
from utils.word_to_line import sort_bbs_line_by_line, crop_line_images

from paragraph_segmentation_dcnn import make_cnn as ParagraphSegmentationNet, paragraph_segmentation_transform
from word_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from handwriting_line_recognition import decode as decoder_handwriting

import cv2
import nltk
from pyScan import pyScan
import os

ctx = mx.gpu(0)
alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
ls = LexiconSearch()
contractions = Contractions('/home/jrmo/git/HandwrittenTextRecognition_MXNet/models/GoogleNews-vectors-negative300.bin.gz')
lm_model, vocab = gluonnlp.model.get_model('awd_lstm_lm_1150', pretrained=True, ctx=ctx)
'''Thanks to Thomas Delteil for creating this model'''

predictions = []


# Allows to pass initialized nets
def predict(image, psn=None, wsn=None, hlrn=None, min_c=0.01, overlap_thres=0.001, topk=400, model_prefix=''):
    if psn is None:
        paragraph_segmentation_net = ParagraphSegmentationNet()
        paragraph_segmentation_net.load_parameters(model_prefix + "../models/paragraph_segmentation2.params")
    else:
        # Assume params are loaded
        paragraph_segmentation_net = psn
    if wsn is None:
        word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
        word_segmentation_net.load_parameters(model_prefix + "../models/word_segmentation.params")
    else:
        # Assume params are loaded
        word_segmentation_net = wsn
    if hlrn is None:
        handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=128, rnn_layers=2, ctx=ctx)
        handwriting_line_recognition_net.load_parameters(model_prefix + "../models/handwriting_line_recognition5.params")
    else:
        # Assume params are loaded
        handwriting_line_recognition_net = hlrn

    form_size = (1120, 800)
    segmented_paragraph_size = (700, 700)
    line_image_size = (30, 400)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized_image = paragraph_segmentation_transform(image, form_size)
    print(type(image))
    print(image.shape)
    paragraph_bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))
    paragraph_bb_predicted = paragraph_bb_predicted[0].asnumpy()

    image = crop_handwriting_page(image, paragraph_bb_predicted, image_size=segmented_paragraph_size)

    words_predicted_bbs = predict_bounding_boxes(word_segmentation_net, image, min_c, overlap_thres, topk, ctx)
    line_bbs = sort_bbs_line_by_line(words_predicted_bbs)
    line_images = crop_line_images(image, line_bbs)
    form_character_probs = []

    for i, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_prob = handwriting_line_recognition_net(line_image.as_in_context(ctx))
        form_character_probs.append(line_character_prob)
    lines = []
    for i, line_character_probs in enumerate(form_character_probs):
        decoded_line = get_text_from_chracter_probs_with_language_model(line_character_probs)
        lines.append(decoded_line)
        print(decoded_line)
        predictions.append(decoded_line)
    return lines

def init_models(model_prefix=''):
    if model_prefix[-1] is not '/':
        model_prefix += '/'
    print(os.getcwd())
    psn = ParagraphSegmentationNet()
    psn.load_parameters("/home/jrmo/git/HandwrittenTextRecognition_MXNet/models/paragraph_segmentation2.params")
    wsn = WordSegmentationNet(2, ctx=ctx)
    wsn.load_params("/home/jrmo/git/HandwrittenTextRecognition_MXNet/models/word_segmentation.params")
    hrln = HandwritingRecognitionNet(rnn_hidden_states=128, rnn_layers=2, ctx=ctx)
    hrln.load_parameters("/home/jrmo/git/HandwrittenTextRecognition_MXNet/models/handwriting_line_recognition5.params")

    return psn, wsn, hrln


def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_perplexcity(line_string, length_normalisation_factor=0.7):
    '''
    Algorithm to calculate the perplexicity of the string.
    1) Tokenise the string into words
    2) One-hot encode the tokenised string
    3) Fed the vector a language model to obtain the output probability
    4) Calculate the perplexcity
    '''
    word_tokenize = nltk.word_tokenize(line_string)
    bos_ids = [vocab[ele] for ele in word_tokenize]  # One-hot encode the word tokens
    if len(bos_ids) == 1 or len(bos_ids) == 0:
        # if none of the words exist, return a large perplexcity.
        return 100000

    hidden = lm_model.begin_state(batch_size=1, func=mx.nd.zeros, ctx=ctx)  #
    input_data = mx.nd.array(bos_ids).as_in_context(ctx)
    input_data = input_data.expand_dims(1)
    output, _ = lm_model(input_data, hidden)
    output_np = output.softmax(axis=2).asnumpy()
    input_np = input_data.asnumpy().astype(int)

    # Calculate the perplexcity.
    ppl = 0
    for i, index in enumerate(input_np):
        if i == 0:
            continue
        ppl -= math.log(float(output_np[i - 1, 0, index]))
    length_normalisation = math.pow(i + 1, length_normalisation_factor)
    return math.exp(ppl / length_normalisation)


def fix_decontractions(input_line_string, decontracted_words):
    '''
    Fix correctly expanded texts made by pycontractions.
    e.g., were -> we are (not we're)
    '''

    def replace_words(wrong_line, input_line, target, replace_with):
        output_string = ""
        wrong_line_array = wrong_line.split(" ")
        input_line_array = input_line.split(" ")
        output_array = []
        skip_next = False
        for i0, i1 in zip(input_line_array, input_line_array[1:]):
            if skip_next:
                skip_next = False
            elif i0 == replace_with[0] and i1 == replace_with[1]:
                output_array.append(target)
                skip_next = True
            else:
                output_array.append(i0)
        output_array.append(input_line_array[-1])
        output_string = " ".join(output_array)
        return output_string

    decontracted_words = replace_words(input_line_string, decontracted_words, target="were", replace_with=["we", "are"])
    return decontracted_words


def replace_line_with_lexicons(input_line_string):
    '''
    Function to replace words in a line string with a lexicon search algorithm.
    '''
    # input_word_tokenize is used to identify which words were changed for contractions.
    input_word_tokenize = nltk.word_tokenize(input_line_string)

    # Decontract the words (and fix the decontractions)
    decontracted_words = list(contractions.expand_texts([input_line_string], precise=True))[0]
    decontracted_words = fix_decontractions(input_line_string, decontracted_words)
    word_tokenize = nltk.word_tokenize(decontracted_words)

    # Obtain the closest word given a noisy word.
    correct_sentence_array = []
    for i, misspelled_words in enumerate(word_tokenize):
        corrected_word = ls.minimumEditDistance_spell_corrector(misspelled_words)
        correct_sentence_array.append(corrected_word)

    # contract words to if it was originally contracted
    contracted_words = []
    for word, original_word in zip(correct_sentence_array, input_word_tokenize):
        is_contracted = False
        for diff in difflib.ndiff(word, original_word):
            if diff[0] == "+" and diff[-1] == "'":
                is_contracted = True
        if is_contracted:
            contracted_words.append(original_word)
        else:
            contracted_words.append(word)
    output = _untokenize(contracted_words)

    if len(input_line_string) == 0:
        return input_line_string
    else:
        # Replace line only if it was not too different. The lexicon search only
        # works well when the handwriting recognition is reasonable, otherwise the lexicon
        # search usually returns worse results.
        difference = levenshtein(input_line_string, output) / len(input_line_string)
        if difference > 0.1:
            return input_line_string
        else:
            return output


def _untokenize(tokens):
    '''
    Helper function to group the list of words into a string.
    '''
    output_string = ""
    for i, token in enumerate(tokens):
        if i == 0:
            output_string += token
        elif token == "quot":
            output_string += token
        elif "'" in token:
            output_string += token
        elif token not in string.punctuation:
            output_string += " " + token
        else:
            output_string += token
    return output_string


def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_lexicon_search(chracter_probabilities):
    '''
    The lexicon search algorithm that replaces words from the output of the greedy
    algorithm string.
    '''
    decoded_line = get_arg_max(chracter_probabilities)
    decoded_line = replace_line_with_lexicons(decoded_line)
    return decoded_line


def get_text_from_chracter_probs_with_language_model(probs):
    '''
    The beam search algorithm that provides multiple proposals of the handwriting
    network's output. The lexicon search algorithm was then applied to the string and
    it was fed into a language model to calculate the sentence perplexcity.
    '''
    prob = probs.softmax().asnumpy()
    line_string_proposals = ctcBeamSearch(prob[0], alphabet_encoding,
                                          None, k=4, beamWidth=25)
    lexicon_line_strings = []
    for line_string in line_string_proposals:
        lexicon_line_string = replace_line_with_lexicons(line_string)
        lexicon_line_strings.append(lexicon_line_string)

    perplexcities = []
    for lexicon_line_string in lexicon_line_strings:
        perplexcity = get_perplexcity(lexicon_line_string)
        # print("{} = {}".format(lexicon_line_string, perplexcity))
        perplexcities.append(perplexcity)
    lowest_perplexicty_index = np.argmin(perplexcities)
    output = lexicon_line_strings[lowest_perplexicty_index]
    return output


def finalfunc(imagepath, model_prefix=''):
    img = cv2.imread(imagepath)
    img = pyScan.process(img)
    models = init_models(model_prefix=model_prefix)
    return predict(img, *models, model_prefix=model_prefix)
