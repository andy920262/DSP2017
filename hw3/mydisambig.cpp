#include <istream>
#include <string>
#include "Ngram.h"
#include "File.h"
#include "Vocab.h"
#include "VocabMap.h"
#include "Trellis.cc"
using namespace std;

typedef const VocabIndex* VocabContext;

string lm_path;
string text_path;
string map_path;

int order = 2;
Vocab voc_zhuyin, voc_big5;
VocabMap map(voc_zhuyin, voc_big5);

void parse_arg(int argc, char* argv[]);
void disambiguateSentence(VocabIndex *wids, Ngram lm);
void disambiguateFile(File &file, Ngram lm);

int main(int argc, char* argv[])
{
    parse_arg(argc, argv);

    Ngram lm(voc_big5, order);
    {
        File lm_file(lm_path.c_str(), "r");
        lm.read(lm_file);
        lm_file.close();
        File map_file(map_path.c_str(), "r");
        map.read(map_file);
        map_file.close();
    }
    File text_file(text_path.c_str(), "r");
    disambiguateFile(text_file, lm);
    text_file.close();

    return 0;
}

void parse_arg(int argc, char* argv[])
{
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-lm")
            lm_path = argv[i + 1];
        if (arg == "-text")
            text_path = argv[i + 1];
        if (arg == "-map")
            map_path = argv[i + 1];
        if (arg == "-order")
            order = stoi(string(argv[i + 1]));
    }
}

void disambiguateSentence(VocabIndex *wids, Ngram lm)
{
    unsigned len = Vocab::length(wids);
    Trellis <VocabContext> trellis(len);

    {
        VocabMapIter iter(map, wids[0]);
        VocabIndex context[2];
        Prob prob;
        context[1] = Vocab_None;
        while(iter.next(context[0], prob))
            trellis.setProb(context, ProbToLogP(prob));
    }

    unsigned pos = 1;
    const VocabIndex emptyContext[] = {Vocab_None};
    while (wids[pos] != Vocab_None) {
        trellis.step();
        VocabMapIter currIter(map, wids[pos]);
        VocabIndex currWid;
        Prob currProb;

        while (currIter.next(currWid, currProb)) {
		    LogP localProb = ProbToLogP(currProb);
            LogP unigramProb = lm.wordProb(currWid, emptyContext);

            VocabIndex newContext[maxWordsPerLine + 2];
            newContext[0] = currWid;

            TrellisIter<VocabContext> prevIter(trellis, pos - 1);
            VocabContext prevContext;
            LogP prevProb;

            while (prevIter.next(prevContext, prevProb)) {
                LogP transProb = lm.wordProb(currWid, prevContext);
                if (transProb == LogP_Zero && unigramProb == LogP_Zero) {
                    transProb = -100;
                }
                unsigned i = 0;
                for (i = 0; i < maxWordsPerLine && prevContext[i] != Vocab_None; i++) {
                    newContext[i + 1] = prevContext[i];
                }
                newContext[i + 1] = Vocab_None;

                unsigned usedLength;
                lm.contextID(newContext, usedLength);
                newContext[usedLength > 0 ? usedLength : 1] = Vocab_None;
                trellis.update(prevContext, newContext, transProb + localProb);
            }
        }
        pos++;
    }
    VocabContext hiddenContexts[len + 1];
    trellis.viterbi(hiddenContexts, len);
    for(int i = 0; i < len; i++) {
        cout << map.vocab2.getWord(hiddenContexts[i][0]) << ((i == len - 1) ? "\n" : " ");
    }
    
}

void disambiguateFile(File &file, Ngram lm)
{
    char* line;
    while (line = file.getline()) {
        VocabString sentence[maxWordsPerLine];
        unsigned numWords = Vocab::parseWords(line, sentence, maxWordsPerLine);
        VocabIndex wids[maxWordsPerLine + 2];
        map.vocab1.getIndices(sentence, &wids[1], maxWordsPerLine, map.vocab1.unkIndex());
        wids[0] = map.vocab1.ssIndex(); // set <s>
        wids[numWords + 1] = map.vocab1.seIndex(); // set </s>
        wids[numWords + 2] = Vocab_None;
        disambiguateSentence(wids, lm);
    }
}
