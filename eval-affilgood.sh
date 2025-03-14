#!/usr/bin/bash

#echo "Evaluating Whoosh (no rerank) - Sample S2AFF"
#python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/sample_S2AFF.tsv --output eval/ner_el/output/sample_S2AFF_Whoosh_NoRerank.tsv
#echo

#echo "Evaluating Whoosh+qLLM - Sample S2AFF"
#python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/sample_S2AFF.tsv --output eval/ner_el/output/sample_S2AFF_Whoosh_qLLM.tsv
#echo

echo "Evaluating Whoosh (no rerank) - CORDIS"
python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/CORDIS.tsv --output eval/ner_el/output/CORDIS_Whoosh_NoRerank.tsv
echo

echo "Evaluating Whoosh+qLLM - CORDIS"
python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/CORDIS.tsv --output eval/ner_el/output/CORDIS_Whoosh_qLLM.tsv
echo

echo "Evaluating Whoosh (no rerank) - CWTS"
python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/CWTS-Non_Related_Organisations.tsv --output eval/ner_el/output/CWTS-Non_Related_Organisations_Whoosh_NoRerank.tsv
echo

echo "Evaluating Whoosh+qLLM - CWTS"
python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/CWTS-Non_Related_Organisations.tsv --output eval/ner_el/output/CWTS-Non_Related_Organisations_Whoosh_qLLM.tsv
echo

echo "Evaluating Whoosh (no rerank) - HFE"
python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/HFE-Hard_French_Examples.tsv --output eval/ner_el/output/HFE-Hard_French_Examples_Whoosh_NoRerank.tsv
echo

echo "Evaluating Whoosh+qLLM - HFE"
python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/HFE-Hard_French_Examples.tsv --output eval/ner_el/output/HFE-Hard_French_Examples_Whoosh_qLLM.tsv
echo

echo "Evaluating Whoosh (no rerank) - OpenAlex"
python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/OpenAlex.tsv --output eval/ner_el/output/OpenAlex_Whoosh_NoRerank.tsv
echo

echo "Evaluating Whoosh+qLLM - OpenAlex"
python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/OpenAlex.tsv --output eval/ner_el/output/OpenAlex_Whoosh_qLLM.tsv
echo

echo "Evaluating Whoosh (no rerank) - S2AFF"
python eval-affilgood.py --entity-linkers Whoosh_NoRerank --input eval/ner_el/input/S2AFF.tsv --output eval/ner_el/output/S2AFF_Whoosh_NoRerank.tsv
echo

echo "Evaluating Whoosh+qLLM - S2AFF"
python eval-affilgood.py --entity-linkers Whoosh --input eval/ner_el/input/S2AFF.tsv --output eval/ner_el/output/S2AFF_Whoosh_qLLM.tsv
echo


