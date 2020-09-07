optimizer = AdamW(model.parameter(),lr = 2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    mpdel= model.train()
    losses = []
    correct_predictions = 0

    for i in data_loader:
        input_ids = i['input_ids']
        attention_mask=i['attention_mask']
        targets = i['targets']

        outputs=model(input_ids = input_ids,attention_mask=attention_mask)

        _ , pred = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions +=torch.sum(preds == targets)
        losses.append(loss.item())

        loass.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_prediction.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model