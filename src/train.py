import time, torch, csv
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

def train_one_epoch(model, loader, optimizer, loss_fn, grad_clip=None, device="cpu"):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn, device="cpu"):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            total_loss += loss.item() * xb.size(0)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds_bin = [1 if p>=0.5 else 0 for p in preds]
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin, average="macro")
    return total_loss / len(loader.dataset), acc, f1

def train_and_evaluate(model_class, model_name, loaders, activation, optimizer_name,
                       seq_len, grad_clip, epochs, batch_size, output_dir, device="cpu"):
    model = model_class(vocab_size=10000, activation=activation).to(device)
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader, test_loader = loaders
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, grad_clip, device)
        val_loss, acc, f1 = evaluate(model, test_loader, loss_fn, device)
        print(f"[{model_name}] epoch {epoch+1}: loss={val_loss:.4f} acc={acc:.4f} f1={f1:.4f}")
    total_time = time.time() - start
    model_path = f"{output_dir}/model_{model_name}_act{activation}_opt{optimizer_name}_seq{seq_len}.pt"
    torch.save(model.state_dict(), model_path)
    with open(f"{output_dir}/metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, activation, optimizer_name, seq_len,
                         "Yes" if grad_clip else "No", acc, f1, total_time, model_path])
