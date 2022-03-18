import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel

seed = 7777
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")


def set_device():
    """
    Returns current device
    Returns:
        device: current available device
    """
    if torch.cuda.is_available():  # if cuda is available
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:  # if cuda is unavailable
        device = torch.device("cpu")

    return device


def load_clean_data():
    """
    Load cleaned data (sample_df.csv)

    Returns:
        df: cleaned data
    """
    df = pd.read_csv("sample_df.csv")
    df = df.dropna(axis=0, how="any")
    return df


def label_evenly_balanced_dataset_sampler(df, sample_size: int, label_column: str):
    """
    Get sample balanced data from dataset

    Args:
        df(DataFrame): input dataset
        sample_size(int): a number of sample
        label_column(str): column name of label

    Returns:
        df : after sampling
    """
    df = df.groupby(label_column).sample(n=sample_size // 2, random_state=seed)
    return df


def custom_collate_fn(batch):
    """
    Convert sentences in a batch to tokenizing before tensorize.
    Apply dynamic padding.
    A number of token is the longest sentence in a batch.
    Args:
        batch(list of tuple): [(input_data, target_data)]

    Returns:
        (tensorized_input, tensorized_label): tuple of tensorized input and label
    """
    global tokenizer_bert

    input_list, target_list = [], []

    for input, target in batch:
        input_list.append(input)
        target_list.append(target)

    tensorized_input = tokenizer_bert(
        input_list,
        add_special_tokens=True,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )

    tensorized_label = torch.tensor(target_list)
    return tensorized_input, tensorized_label


class CustomDataset(Dataset):
    """
    Custom Dataset using pytorch Dataset

    Args:
        input_data(list(str)): input data from dataset
        target_data(list(int)): target data from dataset
    Attributes:
        X (list(str)): store input_data
        Y (list(int)): store target_data
    """

    def __init__(self, input_data: list, target_data: list) -> None:
        self.X = input_data
        self.Y = target_data

    def __len__(self):
        """Return length of Dataset"""
        return len(self.Y)

    def __getitem__(self, index):
        """
        Get item by index

        Args:
            index(int): index of dataset
        Returns:
            (item in X, item in Y)
        """
        return self.X[index], self.Y[index]


class CustomClassifier(nn.Module):
    """
    Custom Dataset using pytorch Dataset

    Args:
        hidden_size(int): a value of hidden size
        n_label(int): a number of label
        freeze_base(bool): BERT model freeze or not
    Attributes:
        bert: store BERT pretrained model
        classifier: customized sequential layer
    """

    def __init__(self, hidden_size: int, n_label: int, freeze_base: bool):
        super(CustomClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("klue/bert-base")

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = not freeze_base

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, linear_layer_hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(linear_layer_hidden_size, n_label),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        """
        Train model

        Args:
            input_ids
            attention_mask
            token_type_ids
        Returns:
            logits: predicted result
        """
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        cls_token_last_hidden_states = outputs["pooler_output"]
        logits = self.classifier(cls_token_last_hidden_states)

        return logits
