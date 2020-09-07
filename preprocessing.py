from collections import defaultdict

from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils import data
import pandas as pd
import transformers
import torch
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW


class_names = ['spam', 'ham']
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
df = pd.read_csv("./spam_small.csv")


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])


# Create a Dataset
class torchData(data.Dataset):
    def __init__(self, review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        encoding = tokenizer.encode_plus(review, max_length=64, add_special_tokens=True, pad_to_max_length=True,
                                         return_attention_mask=True, return_tensors='pt')

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(self.target[item], dtype=torch.long)
        }
# CONSTANTS-
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 20

#Spliting data-
df_train, df_test = train_test_split(df, test_size=0.1)
df_val, df_test = train_test_split(df_test, test_size=0.5)

#Dataloader-
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = torchData(
    review=df.review.to_numpy(),
    target=df.target.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )
    return data.DataLoader(ds,batch_size=batch_size, num_workers=4)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data =next(iter(train_data_loader))

data['input_ids'] = torch.squeeze(data['input_ids'])
data['attention_mask'] = torch.squeeze(data['attention_mask'])
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#
# last_hidden_state, pooled_output =bert_model(input_ids=encoding['input_ids'])

#Building a classifier
class Classfier(nn.Module):
    def __init__(self, n_classes):
        super(Classfier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        data['input_ids'] = torch.squeeze(data['input_ids'])
        data['attention_mask'] = torch.squeeze(data['attention_mask'])
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask = attention_mask)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)

model = Classfier(len(class_names))

input_ids = data['input_ids']
attention_mask = data['attention_mask']

# print(model(input_ids,attention_mask))

optimizer = AdamW(model.parameters(),lr = 2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for i in data_loader:
        input_ids = torch.squeeze(i['input_ids'])
        attention_mask=torch.squeeze(i['attention_mask'])
        targets = i['targets']
        outputs=model(input_ids = input_ids,attention_mask=attention_mask)

        _ , preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions +=torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, n_examples):
    model=model.eval()

    losses=[]
    correct_predictions = 0
    with torch.no_grad():
        for i in data_loader:
            input_ids = torch.squeeze(i['input_ids'])
            attention_mask = torch.squeeze(i['attention_mask'])
            targets = i['targets']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'EPOCHS {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,train_data_loader, loss_fn,optimizer, scheduler,len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_val))

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)

    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model,'model.pth')
        best_accuracy = val_acc
