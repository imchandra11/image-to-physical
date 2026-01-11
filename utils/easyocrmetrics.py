import jiwer

class EasyOCRMetricsManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_ground_truths = []
        self.all_predictions = []

    def update(self, ground_truths, predictions):
        assert len(ground_truths) == len(predictions), "Mismatch in batch size."
        self.all_ground_truths.extend(ground_truths)
        self.all_predictions.extend(predictions)

    def compute(self):
        """Compute WER and CER using jiwer.process_words() with positional args."""
        ground = " ".join(self.all_ground_truths)
        pred = " ".join(self.all_predictions)

        process = jiwer.process_words(ground, pred)  # ✅ positional arguments

        return {
            "wer": process.wer,
            "cer": jiwer.cer(ground, pred),
        }

    def summarize(self):
        return self.compute()









