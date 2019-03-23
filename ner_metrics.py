from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys

TypedStringSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return ' '.join(self.tag_sequence)


class SpanBasedF1Measure:
    def __init__(self):
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: List[List[str]],
                 gold_labels: List[List[str]]):
        assert len(predictions) == len(gold_labels)
        # Iterate over timesteps in batch.
        size = len(gold_labels)
        for i in range(size):
            predicted_spans = self._bio_tags_to_spans(predictions[i])
            if len(predictions[i]) != len(gold_labels[i]):
                print('The length of line %d are not equal: %d %d' % (i, len(predictions[i]), len(gold_labels[i])))

            gold_spans = self._bio_tags_to_spans(gold_labels[i])

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        stable_num = 1e-13
        precision = float(true_positives) / float(true_positives + false_positives + stable_num)
        recall = float(true_positives) / float(true_positives + false_negatives + stable_num)
        f1_measure = 2. * ((precision * recall) / (precision + recall + stable_num))
        return precision, recall, f1_measure

    @staticmethod
    def _bio_tags_to_spans(tag_sequence: List[str]):
        """
        Given a sequence corresponding to BIO tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.

        Parameters
        ----------
        tag_sequence : List[str], required.
            The integer class labels for a sequence.

        Returns
        -------
        spans : List[TypedStringSpan]
            The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        spans: Set[Tuple[str, Tuple[int, int]]] = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, string_tag in enumerate(tag_sequence):
            # Actual BIO tag.
            bio_tag = string_tag[0]
            if bio_tag not in ["B", "I", "O"]:
                raise InvalidTagSequence(tag_sequence)
            conll_tag = string_tag[2:]
            if bio_tag == "O":
                # The span has ended.
                if active_conll_tag is not None:
                    if conll_tag == active_conll_tag:
                        spans.add((active_conll_tag, (span_start, span_end+1)))
                    else:
                        spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "B":
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) The span is an I tag for a different conll annotation.
                # We'll process the previous span if it exists, but also
                # include this span. This is important, because otherwise,
                # a model may get a perfect F1 score whilst still including
                # false positive ill-formed spans.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag is not None:
            spans.add((active_conll_tag, (span_start, span_end)))
        return list(spans)


if __name__ == '__main__':
    predictions, gold_labels = [], []
    for line in open(sys.argv[1]):
        predictions.append(line.strip().split())
    for line in open(sys.argv[2]):
        gold_labels.append(line.strip().split())
    measure = SpanBasedF1Measure()
    measure(predictions, gold_labels)
    metrics = measure.get_metric()
    print(metrics)
