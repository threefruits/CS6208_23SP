import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import argparse
from models import NEGA
from utils import save_model, load_model, generate_graphs_and_states

parser = argparse.ArgumentParser()
parser.add_argument("--graph_type", type=str, default='erdos_renyi', help="graph_type")
parser.add_argument("--input_dim", type=int, default=1, help="input_dim")
parser.add_argument("--hidden_dim", type=int, default=32, help="hidden_dim")
parser.add_argument("--lr", type=float, default=0.0005, help="lr")
parser.add_argument("--layers", type=int, default=1, help="layers")
parser.add_argument("--epochs", type=int, default=30, help="epochs")
parser.add_argument("--n_samples_train", type=int, default=100, help="n_samples_train")
parser.add_argument("--n_nodes_train", type=int, default=20, help="n_nodes_train")
parser.add_argument("--n_samples_val", type=int, default=5, help="n_samples_val")
parser.add_argument("--n_nodes_val", type=int, default=20, help="n_nodes_val")
parser.add_argument("--n_samples_test", type=int, default=5, help="n_samples_test")
parser.add_argument("--n_nodes_test", type=int, default=100, help="n_nodes_test")

args = parser.parse_args()


if __name__ == '__main__':
    print('Preparing data....')
    start_time = time.time()
    train_graphs, train_states = generate_graphs_and_states(args.n_samples_train, args.n_nodes_train ,args.graph_type)
    val_graphs, val_states = generate_graphs_and_states(args.n_samples_val, args.n_nodes_val ,args.graph_type)
    test_graphs, test_states = generate_graphs_and_states(args.n_samples_test, args.n_nodes_test ,args.graph_type)

    end_time = time.time()
    print(f'Prepared in {end_time - start_time:.2f} seconds')
    print()

    if torch.cuda.is_available():
        print('use gpu')
        device = torch.device("cuda")
    else:
        print('use cpu')
        device = torch.device("cpu")

    model = NEGA(args.input_dim, args.hidden_dim, args.layers).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    schedualr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criteria = nn.BCELoss()

    best_model = None
    best_accu = 0
    # PATH='./best_model.pt'
    patience = 0
    MAX_PATIENCE=15


    for i in range(args.epochs):
        t_losses = []
        model.train()
        for graph, states in zip(train_graphs,train_states):
            x, edge_index = graph.x.to(device), graph.edge_index.to(device)
            states = states.to(device)
            N = x.shape[0]
            h = torch.zeros(N, args.hidden_dim).to(device)
            num_nodes = len(x)
            x=x.unsqueeze(dim=1)
            losses = []
            for j in range(num_nodes):
                optimizer.zero_grad()
                out,t, h = model(x, h, edge_index)
                stop = 0
                if len(states) - 1 == j:
                    stop = 1
                x = (out > .5).to(int)
                target= states[j]
                out = out.squeeze(1)
                classification_loss = criteria(out,target)
                termination_target = torch.tensor(stop).to(device).unsqueeze(0)
                
                termination_loss = criteria(t,termination_target.float())
                
                loss = classification_loss + termination_loss
            
                loss.backward()
                
                optimizer.step()
                schedualr.step()
                h = h.detach()
                x = x.detach()
                losses.append(loss.item())
                if stop ==1:
                    break
            mean_loss = np.mean(losses)
            t_losses.append(mean_loss)


        all_accu = []

        last = []
        model.eval()
        with torch.no_grad():
            for graph, states in zip(val_graphs, val_states):
                x, edge_index = graph.x.to(device), graph.edge_index.to(device)
                states = states.to(device)
                N = x.shape[0]
                h = torch.zeros(N, args.hidden_dim).to(device)
                v = len(x)
                x=x.unsqueeze(dim=1)
                accus = []
                for k in range(v):
                    out,t, h = model(x, h, edge_index)
                    stop = 0
                    if len(states)-1 == k:
                        stop = 1
                    x = ( out > .5).to(int)
                    target= states[k]
                    out =( out > .5).to(int)
                    out = out.squeeze(1)
                    accu = torch.sum(out == target) / len(out)
                    accus.append(accu.item())
                    if stop ==1:
                        break
                accu = np.mean(accus)
                all_accu.append(accu)
                last.append(accus[-1])
        average_accu=np.mean(all_accu)
        last_accu = np.mean(last)

        
        loss = np.mean(t_losses)


        print(f"Epoch {i+1} | loss (Training) {loss:.2f} | Average Accuracy (Val) {average_accu*100:.2f} | Last Accuracy (Val) {last_accu*100:.2f} | Best {best_accu*100:.2f}")
        if best_accu < average_accu:
            patience=0
            model_name = 'model.pt'
            save_model(model,model_name)
            best_accu = average_accu
        else:
            patience +=1
            if patience == MAX_PATIENCE:
                break

