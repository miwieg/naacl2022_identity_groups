import spacy
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from flair.models import TextClassifier
from flair.data import Sentence
import os



######################################
##################### GLOBAL VARIABLES
infile = "input.sample.txt" #ADJUST TO YOUR OWN SETTINGS
outfile = "output.sample.txt" #ADJUST TO YOUR OWN SETTINGS
resource_dir = "./Resources" #ADJUST TO YOUR OWN SETTINGS
verb_file = os.path.join(resource_dir, "verb.allPresent.txt")
dictators_file = os.path.join(resource_dir, "dictators.txt")
protected_groups_file = os.path.join(resource_dir, "protectedGroups.txt")
terrorists_file = os.path.join(resource_dir, "terroristGroups.txt")
adjunct_cues_file = os.path.join(resource_dir, "adjunctCues.txt")
aspect_roberta_model_file = os.path.join(resource_dir, "aspect-roberta-model.pt")
perpetrating_evoking_verbs_file = os.path.join(resource_dir, "verbsWithPerpetrators.automaticExtension.csv")
agent_to_patient_verb_sentiment_file = os.path.join(resource_dir, "sentimentOfAgentTowardsPatient.automaticExtension.csv")
tweeteval_model_url = "cardiffnlp/twitter-roberta-base-sentiment"
tweeteval_mapping_link_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
spacy_model = "en_core_web_sm"
flair_sentiment_model = "en-sentiment"



###############################
##################### FUNCTIONS


##################### utility function
def get_words_from_file(file):
    """generic function that reads in lines of files (i.e. words) into a set"""
    # Vorschlag
    # return set([line.strip() for line in open(file) if not line.startswith('#')])
    words = set()
    with open(file) as f:
        for line in f:
            if not(line.startswith("#")):
                line = line.strip()
                words.add(line)
    return words


##################### functions for syntactic processing based on POS-output
def get_index_of_verb(sentence,verbs):
    """determines the index of the (first) main verb in sentence (spacy-object)"""
    # Vorschlag
    # for token in sentence:
        # if token.pos_ == 'VERB':
            # return token.i
    # # occasionally, spacy misses some verbs, we have a list of verb forms in order to find these verb
    # for token in sentence:
        # if token.lower_ in verbs:
            # return token.i
    # return -1
    
    index_of_verb = -1            
    for index in range(0, len(sentence)):
        token = sentence[index]
        if token.pos_ == "VERB":
            index_of_verb = index
            break
    if index_of_verb == -1:
        # occasionally, spacy misses some verbs, we have a list of verb forms in order to find these verb
        for index in range(0, len(sentence)):
            token = sentence[index]
            if token.text.lower() in verbs:
                index_of_verb = index
                break
    return index_of_verb


def remove_leading_preposition(phrase):
    """removes leading preposition of phrase (list of spacy-token-objects)"""
    new_phrase = phrase[:]
    if len(phrase) > 0:
        first_token = phrase[0]
        if first_token.pos_ == "ADP":
            new_phrase = phrase[1:]
    return new_phrase


def get_pure_lemma_of_phrase_as_string(phrase):
    """gets the lemmas for each token of phrase (list of spacy-token-objects)"""
    pure_lemmas = [el.lemma_.lower() for el in phrase]
    pure_lemmas_as_string = " ".join(pure_lemmas)
    return pure_lemmas_as_string
    # Vorschlag
    # return " ".join([el.lemma_.lower() for el in phrase])


def get_patient(sentence, index_of_verb, adjunct_cues):
    """determines the patient of a sentence (spacy-object) based on part-of-speech information"""
    patient_tokens = []
    if index_of_verb != -1:
        # Vorschlag
        # for token in sentence[index_of_verb + 1:]:
        for index in range(index_of_verb + 1, len(sentence)):
            token = sentence[index]
            # there are a number of prepositions that are more likely to introduce adjuncts rather than
            # complements (i.e. patients)
            if token.text.lower() in adjunct_cues:
                break
            # the following POS-tags mark other possible boundaries (end of patient)
            elif token.pos_ == "INTJ" or token.pos_ == "PUNCT" or token.pos_ == "X" or token.pos_ == "SPACE" or token.pos_ == "SCONJ":
                break
            patient_tokens.append(token)
    return patient_tokens



##################### aspect classifier (RoBERTa-model)
def classify_aspect(lines):
    """runs aspect-classifier (RoBERTa-classifier) on lines (each line is one sentence in plain text)"""
    classifier = TextClassifier.load(aspect_roberta_model_file)
    predictions = []
    for line in lines:
        sentence = Sentence(line)
        # predict class and print
        classifier.predict(sentence)
        pred_label = str(sentence.labels)
        if "nonEpisodic" in pred_label:
            predictions.append("NON_EPISODIC")
        elif "episodic" in pred_label:
            predictions.append("EPISODIC")
        else:
            raise Exception("ERROR: unknown label: " + pred_label)
    return predictions



##################### perpetrator-evoking verbs
def read_in_perpetrator_verbs(file):
    """read in list of perpetrator-evoking verbs"""
    perpetrator_verbs = set()
    with open(file) as f:
        for line in f:
            if not(line.startswith("#")):
                verb, label = line.strip().split("\t")
                if label == "PERPETRATOR":
                    perpetrator_verbs.add(verb)
    return perpetrator_verbs



##################### sentiment analysis
def get_prior_sentiment_according_wordlists(phrase, dictators, protected_groups, terrorists):
    """determines the prior sentiment of phrase (dictators and terrorists are NEGATIVE
    while protected_groups are POSITIVE)"""
    prior_sentiment = "?"
    for i in range(0,len(phrase)):
        exprs_to_test = []# these expressions may be unigrams, bigrams, trigrams etc.
        unigram = phrase[i].lemma_.lower()
        exprs_to_test.append(unigram)
        if i-1 >= 0:
            bigram = phrase[i-1].lemma_.lower() + " " + unigram
            exprs_to_test.append(bigram)
            if i-2 >= 0:
                trigram = phrase[i-2].lemma_.lower() + " " + bigram
                exprs_to_test.append(trigram)
                if i-3 >= 0:
                    fourgram = phrase[i-3].lemma_.lower() + " " + trigram
                    exprs_to_test.append(fourgram)
                    if i-4 >= 0:
                        fivegram = phrase[i-4].lemma_.lower() + " " + fourgram
                        exprs_to_test.append(fivegram)
                        if i-5 >= 0:
                            sixgram = phrase[i-5].lemma_.lower() + " " + fivegram
                            exprs_to_test.append(sixgram)
        for expr in exprs_to_test:
            if expr in protected_groups:
                prior_sentiment = "POSITIVE"
                break
            elif (expr in dictators) or (expr in terrorists):
                prior_sentiment = "NEGATIVE"
                break
    return prior_sentiment


def preprocess_for_tweeteval(text):
    """preprocess text (username and link placeholders)"""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_tweeteval_model_labels_and_tokenizer():
    """initialize model, labels and tokenizer from tweeteval"""
    tokenizer = AutoTokenizer.from_pretrained(tweeteval_model_url)
    # download label mapping
    with urllib.request.urlopen(tweeteval_mapping_link_url) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    model = AutoModelForSequenceClassification.from_pretrained(tweeteval_model_url)
    return (model,labels,tokenizer)


def map_tweeteval_label(tweeteval_label):
    """utility function for tweeteval sentiment-text-classifier"""
    converted_label = "POSITIVE"
    if tweeteval_label == "negative":
        converted_label = "NEGATIVE"
    return converted_label


def get_sentiment_for_text_from_tweeteval(instance,model,labels,tokenizer):
    """computes the sentiment of instance with sentiment-text-classifier from tweeteval"""
    text = preprocess_for_tweeteval(instance)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    sentiment_prediction = labels[ranking[0]]
    return map_tweeteval_label(sentiment_prediction)


def get_sentiment_for_text_from_flair(instance,flair_sentiment):
    """computes the sentiment of instance with sentiment-text-classifier from flair"""
    s = Sentence(instance)
    flair_sentiment.predict(s)
    total_sentiment = s.labels[0]
    return total_sentiment.value

def get_phrase_sentiment(phrase, phrase_str, protected_groups, dictators, terrorists, tweeteval_model, tweeteval_labels, tweeteval_tokenizer, flair_sentiment):
    phrase_sentiment = get_prior_sentiment_according_wordlists(phrase, dictators,
                                                               protected_groups, terrorists)
    """function to compute the sentiment of a phrase;
    it combines two sentiment text classifiers and looking up wordlists"""
    if (phrase_sentiment == "?") and (phrase_str != ""):  # if there was no match with wordlists
        phrase_sentiment = "POSITIVE"
        phrase_sentiment_tweeteval = get_sentiment_for_text_from_tweeteval(phrase_str, tweeteval_model,
                                                                           tweeteval_labels, tweeteval_tokenizer)
        phrase_sentiment_flair = get_sentiment_for_text_from_flair(phrase_str, flair_sentiment)
        # individual sentiment analyzers are too conservative with "NEGATIVE"-sentiment, therefore I combine two classifiers
        if (phrase_sentiment_tweeteval == "NEGATIVE") or (phrase_sentiment_flair  == "NEGATIVE"):
            phrase_sentiment = "NEGATIVE"
    return phrase_sentiment

def read_in_agent_to_patient_sentiment(file):
    """read in lexicon with the information about the sentiment of agent towards patient as evoked by a verb"""
    verb_to_sentiment = {}
    with open(file) as f:
        for line in f:
            if not(line.startswith("#")):
                verb, label = line.strip().split("\t")
                verb_to_sentiment[verb] = label
    return verb_to_sentiment


def modify_agent_to_patient_sentiment(patient_raw, senti_label):
    """the sentiment of the agent towards the patient sometimes needs to consider the
    preposition occurring before the patient (phrase);
    example: '[...]_agent discriminates against [women]_patient': irrespective of the semantics of the verb
    the agent will have negative sentiment towards 'women' because of 'against';
    currently the preposition is part of patient_raw"""

    # there are 3 prepositions that, if preceding the patient, determine the sentiment of the agent towards of
    # the patient, thy are 'for', 'to', 'into' and 'against'
    for token in patient_raw:
        if token.text.lower() == "for" or token.text.lower() == "to" or token.text.lower() == "into":
            senti_label = "POSITIVE"
            break
        elif token.text.lower() == "against":
            senti_label = "NEGATIVE"
            break
    return senti_label



##################### non-conformist views
def get_non_conformist_view_label(agent_to_patient_sentiment,patient_sentiment, patient):
    """there is a non-conformistic view if agent_to_patient_sentiment and patient_sentiment differ in sentiment"""
    view_label = "_"
    if len(patient) > 0:
        if agent_to_patient_sentiment == "POSITIVE" and patient_sentiment == "NEGATIVE":
            view_label = "NON-CONFORMISTIC"
        elif agent_to_patient_sentiment == "NEGATIVE" and patient_sentiment  == "POSITIVE":
            view_label = "NON-CONFORMISTIC"
    return view_label



##################### main-function
def main():
    sp = spacy.load(spacy_model)
    verbs = get_words_from_file(verb_file)
    dictators = get_words_from_file(dictators_file)
    protected_groups = get_words_from_file(protected_groups_file)
    terrorists = get_words_from_file(terrorists_file)
    adjunct_cues = get_words_from_file(adjunct_cues_file)
    perpetrator_evoking_verbs = read_in_perpetrator_verbs(perpetrating_evoking_verbs_file)
    verb_agent_to_patient_sentiment = read_in_agent_to_patient_sentiment(agent_to_patient_verb_sentiment_file)
    tweeteval_model, tweeteval_labels, tweeteval_tokenizer = get_tweeteval_model_labels_and_tokenizer()
    flair_sentiment = TextClassifier.load(flair_sentiment_model)

    with open(infile) as f:
        with open(outfile,"w") as fw:
            fw.write("#SENTENCE\tVERB_LEMMA\tASPECT\tAGENT_PERPETRATOR\tPATIENT\tSENTIMENT_OF_PATIENT\tSENTIMENT_OF_AGENT_TOWARDS_PATIENT\tAGENT_IS_NON-CONFORMISTIC\n")
            lines = f.readlines()
            aspect_predictions = classify_aspect(lines)

            for line, aspect_prediction in zip(lines, aspect_predictions):
                sentence = sp(line)
                index_of_verb = get_index_of_verb(sentence,verbs)
                verb_lemma = "UNKNOWN_VERB"
                if index_of_verb != -1:
                    verb_lemma = sentence[index_of_verb].lemma_.lower()

                perpetrator_label = "_" # default: assume no perpetrator-evoking verb
                if verb_lemma in perpetrator_evoking_verbs:
                    perpetrator_label = "PERPETRATOR"

                patient_raw = get_patient(sentence,index_of_verb, adjunct_cues)
                # patient_raw may still include a leading preposition which semantically belongs to the preceding main verb
                # example: "X discriminates against women." --> patient_raw "against women"
                #          but the verb basically is "discriminates against" and the patient is just "women"
                patient = remove_leading_preposition(patient_raw)
                patient_str = get_pure_lemma_of_phrase_as_string(patient)

                patient_sentiment = get_phrase_sentiment(patient, patient_str, protected_groups, dictators,
                                     terrorists, tweeteval_model, tweeteval_labels, tweeteval_tokenizer,
                                     flair_sentiment)

                agent_to_patient_sentiment = "NEGATIVE" # if the verb is not in lexical resource, we assume negative sentiment since it is much more frequent
                if verb_lemma in verb_agent_to_patient_sentiment.keys():
                    agent_to_patient_sentiment = verb_agent_to_patient_sentiment[verb_lemma]
                # the following function incorporates a possible leading preposition of patient_raw into the verb sentiment (agent to patient)
                agent_to_patient_sentiment = modify_agent_to_patient_sentiment(patient_raw,agent_to_patient_sentiment)

                non_conformist_view_label = get_non_conformist_view_label(agent_to_patient_sentiment, patient_sentiment, patient)

                if len(patient_raw) > 0:
                    fw.write(line.strip() + "\t" + verb_lemma + "\t" + aspect_prediction + "\t" + perpetrator_label + "\t" +\
                             patient_str + "\t" + patient_sentiment + "\t" + agent_to_patient_sentiment +\
                             "\t" + non_conformist_view_label + "\n")
                    # Vorschlag
                    # fw.write('\t'.join(line.strip(),
                                       # verb_lemma,
                                       # aspect_prediction,
                                       # perpetrator_label,
                                       # patient_str,
                                       # patient_sentiment,
                                       # agent_to_patient_sentiment,
                                       # non_conformist_view_label)
                                       # + "\n")
                else:
                    #there may be cases in which there is no patient
                    fw.write(
                        line.strip() + "\t" + verb_lemma + "\t" + aspect_prediction + "\t" + perpetrator_label + "\t" + \
                        "N/A\tN/A\tN/A\t_\n")



####################################
##################### MAIN PROGRAMME
if __name__ == '__main__':
    main()
