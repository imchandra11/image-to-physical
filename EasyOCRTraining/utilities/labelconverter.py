import torch

class CTCLabelConverter(object):
    """
    Convert between text-label and text-index for CTC-based recognition.
    """
    def __init__(self, character):
        """
        Args:
            character (str): String of possible characters.
        """
        # Create mapping char -> index (1 to N); reserve 0 for blank
        dict_character = list(character)
        self.dict = {char: i+1 for i, char in enumerate(dict_character)}
        # List of characters, index 0 is blank token
        self.character = ['[blank]'] + dict_character



    def decode_greedy(self, text_index, length):
        """
        Greedy decoding for CTC: converts index sequences to text.
        Args:
            text_index (Tensor): flattened indices from model output [sum(text_lengths)]
            length (Tensor): lengths of each sequence in batch
        Returns:
            List[str]: decoded texts
        """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # skip blanks and repeats
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
            index += l
        return texts
   

    def encode(self, text_list, batch_max_length=25):
        """
        Args:
            text_list: list of strings (length = batch_size)
            batch_max_length: fixed maximum length for padding
        Returns:
            text_tensor: LongTensor of shape (batch_size, batch_max_length)
            lengths: IntTensor of original lengths
        """
        if isinstance(text_list, str):
            text_list = [text_list]
        lengths = [len(s) for s in text_list]
        batch_size = len(text_list)
        text_tensor = torch.LongTensor(batch_size, batch_max_length).fill_(0)
        for i, text in enumerate(text_list):
            text = list(text)
            text_indices = []
            for char in text:
                if char in self.dict:
                    text_indices.append(self.dict[char])
                else:
                    # skip unknown characters
                    continue
            length = len(text_indices)
            text_tensor[i][:length] = torch.LongTensor(text_indices)
        return text_tensor, torch.IntTensor(lengths)

    def decode(self, preds_indices, preds_sizes):
        """
        Args:
            preds_indices: Tensor (batch, max_length) of indices (after argmax).
            preds_sizes:  Tensor or list of lengths for each batch entry.
        Returns:
            List of decoded strings (with repeats/blanks removed).
        """
        preds_indices = preds_indices.detach().cpu().numpy()
        if hasattr(preds_sizes, 'detach'):
            preds_sizes = preds_sizes.detach().cpu().numpy()
        texts = []
        for i, length in enumerate(preds_sizes):
            pred = preds_indices[i]
            char_list = []
            for j in range(length):
                # skip blank (0) and duplicate chars
                if pred[j] != 0 and not (j > 0 and pred[j-1] == pred[j]):
                    char_list.append(self.character[pred[j]])
            texts.append(''.join(char_list))
        return texts

class AttnLabelConverter(object):
    """
    Convert between text-label and text-index for attention-based recognition (sequence-to-sequence).
    """
    def __init__(self, character):
        """
        Args:
            character (str): String of possible characters.
        """
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character
        self.dict = {char: i for i, char in enumerate(self.character)}

    def encode(self, text_list, batch_max_length=25):
        """
        Args:
            text_list: list of strings (length = batch_size)
            batch_max_length: maximum text length (not counting [GO])
        Returns:
            text_tensor: LongTensor of shape (batch_size, batch_max_length+1) 
                         with [GO] at index 0 and [s] at the end of each entry.
            lengths: IntTensor of (len(text)+1) for each (including [s]).
        """
        if isinstance(text_list, str):
            text_list = [text_list]
        lengths = [len(s) + 1 for s in text_list]  # +1 for [s]
        batch_size = len(text_list)
        max_length = batch_max_length + 1  # account for appended [s]
        text_tensor = torch.LongTensor(batch_size, max_length+1).fill_(0)
        for i, text in enumerate(text_list):
            text = list(text + '[s]')
            text_indices = []
            for char in text:
                if char in self.dict:
                    text_indices.append(self.dict[char])
                else:
                    continue
            length = len(text_indices)
            # Place after [GO] token (which is index 0)
            text_tensor[i][1:1+length] = torch.LongTensor(text_indices)
        return text_tensor, torch.IntTensor(lengths)

    def decode(self, preds_indices, preds_sizes=None):
        """
        Args:
            preds_indices: Tensor (batch, seq_length) of indices.
        Returns:
            List of decoded strings (may include [GO] and [s] tokens).
        """
        preds_indices = preds_indices.detach().cpu().numpy()
        texts = []
        for pred in preds_indices:
            chars = [self.character[idx] for idx in pred]
            texts.append(''.join(chars))
        return texts
