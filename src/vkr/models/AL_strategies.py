import numpy as np
import torch


def qbc(committee, X_sample):

    tx = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
    preds = []
    for model, likelihood in committee:
        model.eval()
        likelihood.eval()
        x_query_t = tx.reshape(1, -1)
        mean = model(x_query_t).mean
        preds.append(mean)

    preds_tensor = torch.stack(preds)  
    f_avg = preds_tensor.mean(dim=0)
    loss = torch.var(preds_tensor - f_avg, dim=0).mean()
    tx.grad = None  
    loss.backward()

    gradients = (
        tx.grad.detach().numpy() if tx.grad is not None else np.zeros_like(X_sample)
    )
    return loss.item(), gradients

# Стратегия отбора NA-QBC
def NA_query_strategy(comittee, X_sample, X_pool):

    _, grads = qbc(comittee, X_sample)  
    grads = [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in grads]
    grads = torch.tensor(grads, dtype=torch.float32).numpy()
    sign = np.sign(grads)

    for ind in range(len(sign)):
        step = 0.01
        x_gen = X_sample[ind] + step * sign[ind]
        x_gen = np.clip(x_gen, 0.0, 1.0)
        dists = np.linalg.norm(X_pool - x_gen, axis=1)
        idx = np.argmin(dists)
    return idx, grads

#Функция возвращает только значение qbc
def qbc_for_pool(committee, X_sample):

    tx = torch.tensor(X_sample, dtype=torch.float32)
    preds = []
    for model, likelihood in committee:
        model.eval()
        likelihood.eval()
        x_query_t = tx.reshape(1, -1)
        mean = model(x_query_t).mean
        preds.append(mean)

    preds_tensor = torch.stack(preds)  
    f_avg = preds_tensor.mean(dim=0)
    loss = torch.var(preds_tensor - f_avg, dim=0).mean()
    return loss


# Стратегия отбора типа pool-based
def pool_based_qbc(committee, X_pool, n_instances=7):
    train_idx = np.random.choice(range(X_pool.shape[0]), size=21, replace=False)
    X_init = X_pool[train_idx]
    uncertainties = []

    for x in X_init:
        loss = qbc_for_pool(committee, np.array([x]))
        uncertainties.append(loss.detach().numpy())  
    
    uncertainties_np = np.array(uncertainties)
    sorted_indices = np.argsort(uncertainties_np)[::-1]
    selected_init_indices = sorted_indices[:n_instances]
    selected_pool_indices = train_idx[selected_init_indices]
    
    return selected_pool_indices, X_pool[selected_pool_indices]