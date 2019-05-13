#!/bin/bash
trainingFolder=../data/pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02
evaluationSetFolder=../data/pan18-cross-domain-authorship-attribution-test-dataset2-2018-04-20

 
echo "******************  Training ******************"
#python ../2018/pan18-cdaa-baseline.py -i "$trainingFolder" -o "outputs/svm_training" -n 4
python ../2018/pan18-cdaa-evaluator.py -a "outputs/svm_training" -i "$trainingFolder" -o "outputEvaluation/svm_training"
say "finished Training"

echo "******************  Evaluation ******************"
#python ../2018/pan18-cdaa-baseline.py -i "$evaluationSetFolder" -o "outputs/svm_evaluation" -n 4
python ../2018/pan18-cdaa-evaluator.py -a "outputs/svm_evaluation" -i "$evaluationSetFolder" -o "outputEvaluation/svm_evaluation"
say "finished Evaluation"