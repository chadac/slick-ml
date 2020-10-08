import torch
import numpy as np
from typing import List
from collections import defaultdict

import transformers
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup


class Dataset(torch.utils.data.Dataset):
    """Python iterable for batch processing 
    (https://pytorch.org/docs/stable/data.html)
    Parameters
    ----------
    text: str
        Name of input text column (e.g. review/text/etc.)
    target: str
        Name of input target colum (e.g. sentiment/etc.)
    tokenizer: object
        Tokenizer for encoding input text for modeling
    max_len: int
        Maximum length (in number of tokens) for the inputs to the transformer model
    """

    def __init__(self, text, target, tokenizer, max_len):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        batch_text = str(self.text[item])
        batch_target = self.target[item]

        encoding = self.tokenizer.encode_plus(
            batch_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {
            "text": batch_text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(batch_target, dtype=torch.long),
        }


class SentimentBert(nn.Module):
    """Subclass of PyTorch torch.nn.Module; this class inherits from the BERT
    PreTrainedModel, the bare Bert Model transformer outputting raw 
    hidden-states without any specific head on top.
    (https://huggingface.co/transformers/model_doc/bert.html#bertmodel)
    Parameters
    ----------
    n_classes: str, optional (default='text')
        Number input classes for target variable

    Attributes
    ----------
    forward: Tensor
        Return A 'BaseModelOutputWithPooling' (if return_dict=True)
        is passed or when config.return_dict=True) 
    """

    def __init__(self, n_classes):
        super(SentimentBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.10)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """BERT Model Forward Pass
        Parameters
        ----------
        input_ids: tensor,
            Indices of input sequence tokens in the vocabulary.
        attention_mask: tensor,
            Mask to avoid performing attention on padding token indices.
            1 for tokens that are not masked and 0 for tokens that are maked.
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


class SentimentTorch:
    """Sentiment Analysis using PyTorch and pre-trained BERT. This class 
    performs sentiment analysis on an input dataset using the pre-trained 
    BERT (Bidirectional Encoder Representations from Transformers) from the 
    Hugging Face Transformer library. Please see the docs for more information 
    on (https://huggingface.co/transformers/model_doc/bert.html)
    Parameters
    ----------
    text: str, optional (default='text')
        Name of input text column (e.g. review/text/etc.)
    target: str, optional (default='sentiment')
        Name of input target colum (e.g. sentiment/etc.)
    tokenizer: str, optional (default='bert-base-uncased')
        Tokenizer for encoding input text for modeling
    epochs: int, optional (default=None)
        Number of epochs for training 
    batch_size: int, optional (default=None)
        how many samples per batch to load
    max_len: int, optional (default=None)
        Maximum length (in number of tokens) for the inputs to the transformer model
    n_classes: int, optional (default=None)
        Number input classes for target variable
    n_workers: int, optional (default=4)
        how many subprocesses to use for data loading. 0 means that the data will be 
        loaded in the main processn

    Attributes
    ----------
    batch_train: dict()
        Returns a dict() of all feature importance based on
        importance_type at each fold of each iteration during
        selection process
    batch_evaluate: Pandas DataFrame()
        Returns a DataFrame() cosists of total frequency of
        each feature during the selection process
    fit: Pandas DataFrame()
        Returns a DataFrame() cosists of total frequency of
        each feature during the selection process
    predict: Pandas DataFrame()
        Returns a DataFrame() cosists of total frequency of
        each feature during the selection process
    """

    def __init__(
        self,
        text="text",
        target="sentiment",
        tokenizer="bert-base-uncased",
        epochs=10,
        batch_size=None,
        max_len=None,
        n_classes=None,
        n_workers=4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text = text
        self.target = target
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_workers = n_workers
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer)

        # results
        self._text = []
        self._pred = []
        self._proba = []
        self._labels = []

    def __repr__(self):
        name = self.__class__.__name__
        if torch.cuda.is_available():
            return f"{name} Model: training with GPUs."
        else:
            return f"{name} Model: training with CPUs."

    def batch_train(self, data: dict, n_correct: float = 0, _loss: List[int] = []):
        """
        Function to perform batch training
        Parameters
        ----------
        data: dict,
            Encoding dictionary containing input_ids,attention_mask,target
        n_correct: float, optional, (default=0)
            Tracks the number of correct predictions for each batch
        _loss: List[int], optional, (default=[])
            Tracks the loss for each batch
        """

        model = self.model.train()
        for idx, d in enumerate(data):

            # batch encodings
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            target = d["targets"].to(self.device)

            # model output
            output = model(input_ids, attention_mask)

            # loss
            loss = self.loss_func(output, target)
            _loss.append(loss.item())

            # prediction
            _, pred = torch.max(output, dim=1)
            n_correct += torch.sum(pred == target).item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return np.mean(_loss), n_correct / len(data)

    def batch_evaluate(self, data: dict, n_correct: float = 0, _loss: List[int] = []):
        """
        Function to perform batch evaluation on validation dataset
        Parameters
        ----------
        data: dict,
            Encoding dictionary containing input_ids,attention_mask,target
        n_correct: float, optional, (default=0)
            Tracks the number of correct predictions for each batch
        _loss: List[int], optional, (default=[])
            Tracks the loss for each batch
        """

        model = self.model.eval()
        with torch.no_grad():
            for idx, d in enumerate(data):

                # batch encodings
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                target = d["targets"].to(self.device)

                # model output
                output = model(input_ids, attention_mask)

                # loss
                loss = self.loss_func(output, target)
                _loss.append(loss.item())

                # prediction
                _, pred = torch.max(output, dim=1)
                n_correct += torch.sum(pred == target).item()

        return np.mean(_loss), n_correct / len(data)

    def fit(self, train, validation):
        """
        Function for model training/evaluating
        Parameters
        ----------
        train: pd.DataFrame,
            Input train dataframe
        validation: pd.DataFrame,
            Input validation dataframe
        """
        best_score = 0
        history = defaultdict(list)

        # train dataset
        train_gen = Dataset(
            text=train[self.text].to_numpy(),
            target=train[self.target].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        train_data = DataLoader(
            train_gen, batch_size=self.batch_size, num_workers=self.n_workers
        )

        # valid dataset
        val_gen = Dataset(
            text=validation[self.text].to_numpy(),
            target=validation[self.target].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        val_data = DataLoader(
            val_gen, batch_size=self.batch_size, num_workers=self.n_workers
        )

        # model params
        self.model = SentimentBert(self.n_classes)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)

        # schedule lr that decreases linearly from initial lr
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_data) * self.epochs,
        )
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epochs):
            print(f"{'='*20} Epoch: {epoch+1} {'='*20}")
            train_loss, train_acc = self.batch_train(train_data, epoch)
            val_loss, val_acc = self.batch_evaluate(val_data, epoch)
            print(f"--Train Loss: {train_loss:.2f} --Train Accuracy: {train_acc:.2f}")
            print(f"--Val Lss: {val_loss:.2f}   --Val Accuracy: {val_acc:.2f}")

            # update results
            history["train_acc"] = train_acc
            history["train_loss"] = train_loss
            history["val_acc"] = val_acc
            history["val_loss"] = val_loss

            # update best score
            if val_acc > best_score:
                best_score = val_acc

        return history

    def predict(self, test):
        """
        Function for evaluating test dataset
        Parameters
        ----------
        test: pd.DataFrame,
            Input test dataframe
        """

        # test dataset
        test_gen = Dataset(
            text=test[self.text].to_numpy(),
            target=test[self.target].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        test_data = DataLoader(
            test_gen, batch_size=self.batch_size, num_workers=self.n_workers
        )

        model = self.model.eval()
        with torch.no_grad():
            for d in test_data:

                # batch encodings
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                target = d["targets"].to(self.device)

                # model output
                output = model(input_ids, attention_mask)

                # prediction
                _, pred = torch.max(output, dim=1)

                # append results
                self._pred.extend(pred)
                self._proba.extend(output)
                self._labels.extend(target)

        pred = torch.stack(self._pred)
        proba = torch.stack(self._proba)
        labels = torch.stack(self._labels)
        return pred, proba, labels
