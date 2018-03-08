# -*- coding: utf-8 -*-
import io


import codecs
import numpy as np


def replace(w1, w2):
    for x in range(0, len(raw_sentences)):
        if (w1 in raw_sentences[x]):
            sentences.append(raw_sentences[x].replace(w1, w2))
        if (w2 in raw_sentences[x] and w2 != "fakultäten"):
            sentences.append(raw_sentences[x].replace(w2, w1))

def addDuesseldorf(sentence):
    sentence = str(sentence)
    if (" hochschule " in sentence):
        if (" hochschule düsseldorf" not in sentence):
            sentence = sentence.replace(" hochschule", " hochschule düsseldorf")
    if (" fachhochschule" in sentence and " fachhochschule düsseldorf" not in sentence):
        sentence = sentence.replace(" fachhochschule", " fachhochschule düsseldorf")
    if (" hs " in sentence and " hs düsseldorf" not in sentence):
        sentence = sentence.replace(" hs", " hs düsseldorf")
    return sentence

with io.open('../training_data/questions.txt', encoding="utf-8") as f:
    raw_sentences = f.readlines()
    for x in range(0, len(raw_sentences)):
        # First parameter is the replacement, second parameter is your input string
        raw_sentences[x] = raw_sentences[x].replace("-", " ")
        raw_sentences[x] = raw_sentences[x].replace("?", "")
        raw_sentences[x] = raw_sentences[x].replace(",", "")
        raw_sentences[x] = raw_sentences[x].replace(".", "")
        raw_sentences[x] = raw_sentences[x].replace("!", "")
        raw_sentences[x] = raw_sentences[x].replace("\n", "")
        raw_sentences[x] = raw_sentences[x].lower()
    sentences = raw_sentences

    currentSenLen = 0
    v = 0
    while (v < 5):
        v = v + 1
        print("Step: " , v)
        replace("adresse ", "anschrift ")
        replace("mensa ", "kantine ")
        replace("zu finden sind ", "zu finden seien ")
        replace(" können ", " könnten ")
        replace("mich interessiert ", "ich hätte gerne gewusst ")
        replace("mich interessiert ", "ich wollte fragen ")
        replace("sage mir bitte ", "ich wollte fragen ")
        replace("ich wollte fragen ", "sag mir bitte ")
        replace("kannst du mir sagen ", "ich wollte fragen ")
        replace("ich wollte fragen ", "ich will fragen ")
        replace("kannst du mir bitte sagen ", "ich wollte fragen ")
        replace("kannst du mir sagen ", "ich wollte dich fragen ")
        replace("kannst du bitte mir sagen ", "ich wollte dich fragen ")
        replace("kannst du mir bitte sagen ", "ich wollte dich fragen ")
        replace("kannst du mir sagen ", "ich will wissen ")
        replace("kannst du bitte mir sagen ", "ich will wissen ")
        replace("kannst du mir bitte sagen ", "ich will wissen ")
        replace("ich will wissen ", "ich würde gerne wissen ")
        replace("kannst du mir sagen ", "ich wollte sie fragen ")
        replace("kannst du mir sagen ", "ich will wissen ")
        replace("kannst du mir sagen ", "ich wollte wissen ")
        replace("können sie mir sagen ", "ich wollte sie fragen ")
        replace("können sie mir sagen ", "ich will sie fragen ")
        replace("können sie mir bitte sagen ", "ich will sie fragen ")
        replace("können sie mir bitte sagen ", "ich wollte sie fragen ")
        replace("können sie mir bitte sagen ", "ich wollte fragen ")
        replace("ich wollte fragen ", "ich wollte wissen ")
        replace("ich wollte fragen ", "ich will wissen ")
        replace("ich will wissen ", "mich interessiert ")
        replace("mich interessiert ", "ich würde gerne wissen ")
        replace("mich interessiert ", "ich hätte gerne gewusst ")
        replace("mich interessiert ", "ich hätte gern gewusst ")
        replace("ich hätte gerne gewusst ", "mich würde interessieren ")
        replace(" kannst ", " könntest ")
        replace(" hsd ", " hochschule ")
        replace(" hsd ", " fachhochschule ")
        replace(" hsd ", " hs ")
        replace(" hsd ", " uni ")
        replace("verschiedene ", "unterschiedliche ")
        replace("verschiedenen ", "unterschiedlichen ")
        replace("anzahl ", "menge ")
        replace("auto ", "fahrzeug ")
        replace("autos ", "wagen ")
        replace("autos ", "fahrzeuge ")
        replace(" man parken ", " man stellen ")
        replace(" man stellen ", " man abstellen ")
        replace("haben", "besitzen")
        replace("fachbereichen ", "fakultäten ")
        replace(" brauche ", " benötige ")
        replace(" benötige ", " bräuchte ")
        replace(" will ", " möchte ")
        replace(" geben ", " angeben ")
        replace("sage ", "sag ")
        replace(" garage ", " tiefgarage ")
        replace(" gerne ", " gern ")
        sentences = list(set(sentences))
    print(len(sentences))
    print("Add düsseldorf")

    sentences = np.array(sentences)
    # print(sentences.shape)
    for sentence in sentences:
        new_sen = addDuesseldorf(sentence)
        if (new_sen not in sentences):
            sentences = np.append(sentences, new_sen)


    file = codecs.open("../training_data/input_data.txt", "w", "utf-8")


    for sentence in sentences:
        if (len(str(sentence).split()) < 15):
            file.write(sentence + '\n')
    file.close()


    print(len(sentences))
