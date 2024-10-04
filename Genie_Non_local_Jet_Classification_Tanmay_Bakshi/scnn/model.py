import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scnn
import chebyshev
import json
import logging
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score

class MySCNN(nn.Module):
    def __init__(self, colors=1):
        super(MySCNN, self).__init__()

        assert(colors > 0)
        self.colors = colors

        num_filters = 10  # 30
        variance = 0.1  # 0.001

        # Degree 0 convolutions.
        self.C0_1 = scnn.SimplicialConvolution(2, C_in=4, C_out=16, enable_bias= True, variance = variance)
        self.C0_2 = scnn.SimplicialConvolution(2, C_in=16, C_out=32, enable_bias = False, variance=variance)
        self.C0_3 = scnn.SimplicialConvolution(2, C_in=32, C_out=64, enable_bias = False, variance=variance)
        self.C0_4 = scnn.SimplicialConvolution(2, C_in=64, C_out=128, enable_bias = False, variance=variance)
        self.C0_5 = scnn.SimplicialConvolution(2, C_in=128, C_out=256, enable_bias = False, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.SimplicialConvolution(2, C_in=4, C_out=16, enable_bias = True, variance=variance)
        self.C1_2 = scnn.SimplicialConvolution(2, C_in=16, C_out=32, enable_bias = False, variance=variance)
        self.C1_3 = scnn.SimplicialConvolution(2, C_in=32, C_out=64, enable_bias = False, variance=variance)
        self.C1_4 = scnn.SimplicialConvolution(2, C_in=64, C_out=128, enable_bias=False, variance=variance)
        self.C1_5 = scnn.SimplicialConvolution(2, C_in=128, C_out=256, enable_bias=False, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.SimplicialConvolution(2, C_in=4, C_out=16, enable_bias=True, variance = variance)
        self.C2_2 = scnn.SimplicialConvolution(2, C_in=16, C_out=32, enable_bias = False, variance=variance)
        self.C2_3 = scnn.SimplicialConvolution(2, C_in=32, C_out=64, enable_bias = False, variance=variance)
        self.C2_4 = scnn.SimplicialConvolution(2, C_in=64, C_out=128, enable_bias=False, variance=variance)
        self.C2_5 = scnn.SimplicialConvolution(2, C_in=128, C_out=256, enable_bias=False, variance=variance)

        # Binary classification layer
        self.fc = nn.Sequential(
            nn.Linear(3 * 32, 64),  # Linear (3*64, 128)
            nn.Dropout(p=0.5),       # Dropout (p=0.5 can be adjusted as needed)
            #nn.Linear(256, 128),      # Linear (128, 64)
            #nn.Dropout(p=0.5),
            #nn.Linear(128, 64),       # Linear (64, 32)
            #nn.Dropout(p=0.5),
            nn.Linear(64, 32),       # Linear (32, 16)
            nn.Dropout(p=0.5),
            nn.Linear(32, 16),
            nn.Dropout(p=0.5),
            nn.Linear(16, 2)# Linear (16, 2)
        ) # Concatenated out0_3, out1_3, out2_3
        self.sigmoid = nn.Sigmoid()

    def forward(self, Ls, Ds, adDs, xs):
        assert(len(xs) == 3)  # The three degrees are fed together as a list.

        out0_1 = self.C0_1(Ls[0], xs[0])
        out1_1 = self.C1_1(Ls[1], xs[1])
        out2_1 = self.C2_1(Ls[2], xs[2])


        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1))
        out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1))
        out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1))

        #out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2))
        #out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2))
        #out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2))

        #out0_4 = self.C0_4(Ls[0], nn.LeakyReLU()(out0_3))
        #out1_4 = self.C1_4(Ls[1], nn.LeakyReLU()(out1_3))
        #out2_4 = self.C2_4(Ls[2], nn.LeakyReLU()(out2_3))

        xs_update = [out0_2, out1_2, out2_2]
        out0_4_agg = torch.mean(out0_2, dim=2, keepdim=True)
        out1_4_agg = torch.mean(out1_2, dim=2, keepdim=True)
        out2_4_agg = torch.mean(out2_2, dim=2, keepdim=True)

        concatenated_out = torch.cat([out0_4_agg, out1_4_agg, out2_4_agg], dim=1)
        concatenated_out = torch.flatten(concatenated_out, start_dim=1)

        logits = self.fc(concatenated_out)
        # probs = self.sigmoid(logits)
        return logits, xs_update  # Return the binary classification probabilities


def load_data(lapl_file, boundary_file, label_file):
    laplacians = np.load(lapl_file, allow_pickle=True)
    boundaries = np.load(boundary_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)

    all_lapl = [laplacians[f'arr_{i}'] for i in range(len(laplacians))]
    all_bounds = [boundaries[f'arr_{i}'] for i in range(len(boundaries))]
    all_labels = [labels[f'arr_{i}'] for i in range(len(labels))]
    # formatted_labels = [1 if label[0] == 1 else 0 for label in all_labels]

    del laplacians, boundaries, labels
    return all_lapl, all_bounds, all_labels


def load_chunked_data(lapl_prefix, boundary_prefix, label_prefix):
    """
    Loads and concatenates data from multiple chunked .npz files.

    Parameters:
    - lapl_prefix (str): The prefix of the laplacians .npz files.
    - boundary_prefix (str): The prefix of the boundaries .npz files.
    - label_prefix (str): The prefix of the labels .npz files.

    Returns:
    - all_lapl (list): List of all laplacians from the loaded chunks.
    - all_bounds (list): List of all boundaries from the loaded chunks.
    - all_labels (list): List of all labels from the loaded chunks.
    """
    all_lapl, all_bounds, all_labels = [], [], []
    chunk_idx = 1

    # Continue loading chunked files until no more are found
    while chunk_idx < 4:
        lapl_file = f"{lapl_prefix}_{chunk_idx}.npz"
        boundary_file = f"{boundary_prefix}_{chunk_idx}.npz"
        label_file = f"{label_prefix}_{chunk_idx}.npz"

        if not os.path.exists(lapl_file) or not os.path.exists(boundary_file) or not os.path.exists(label_file):
            break  # No more chunks to load

        logging.info(f"Loading chunk {chunk_idx}")

        # Load data from the current chunk
        laplacians = np.load(lapl_file, allow_pickle=True)
        boundaries = np.load(boundary_file, allow_pickle=True)
        labels = np.load(label_file, allow_pickle=True)

        # Append data from the current chunk to the lists
        all_lapl.extend([laplacians[f'arr_{i}'] for i in range(len(laplacians))])
        all_bounds.extend([boundaries[f'arr_{i}'] for i in range(len(boundaries))])
        all_labels.extend([labels[f'arr_{i}'] for i in range(len(labels))])

        del laplacians, boundaries, labels  # Free memory
        chunk_idx += 1

    return all_lapl, all_bounds, all_labels


def prepare_inputs(all_lapl, all_bounds, batch_size, all_labels):
    xs, Ls_all, Ds_all, adDs_all, labels_all = [], [], [], [], []

    for i in range(len(all_lapl)):
        lap = all_lapl[i]
        boundary = all_bounds[i]
        num_nodes = all_lapl[i][0].shape[0]  # Number of nodes (0-simplices)

        if len(all_bounds[i]) == 2:
            topdim = 2
            num_edges = all_bounds[i][0].shape[1]  # Number of edges (1-simplices)
            num_faces = all_bounds[i][1].shape[1]  # Number of faces (2-simplices)
            # print(lap.shape)
            # input()
            try:
                xs_temp = [
                    torch.rand((batch_size, 4, num_nodes)),  # Degree 0 input (nodes)
                    torch.rand((batch_size, 4, num_edges)),  # Degree 1 input (edges)
                    torch.rand((batch_size, 4, num_faces))  # Degree 2 input (faces)
                ]
                Ls = [scnn.coo2tensor(scnn.chebyshev.normalize(lap[k])) for k in range(topdim + 1)]
                Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
                adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]
            except Exception as e:
                logging.warning(f"Error while processing laplacian at index {i}: {e}")
                del all_labels[i]
                continue  # Going to the next iterations...

        elif len(all_bounds[i]) == 1:
            topdim = 1
            num_edges = all_bounds[i][0].shape[1]
            num_faces = 1  # No faces exist in the chosen filtration

            try:
                Ls = [scnn.coo2tensor(scnn.chebyshev.normalize(lap[k])) for k in range(topdim + 1)]
                Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
                adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]

                xs_temp = [
                    torch.rand((batch_size, 4, num_nodes)),
                    torch.rand((batch_size, 4, num_edges)),
                    torch.zeros((batch_size, 4, num_faces))
                ]

            except Exception as e:
                logging.warning(f"Error while processing laplacian at index {i}: {e}")
                del all_labels[i]

                continue

        xs.append(xs_temp)
        Ls_all.append(Ls)
        Ds_all.append(Ds)
        adDs_all.append(adDs)

    return xs, Ls_all, Ds_all, adDs_all, all_labels


def compute_accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")



def save_metrics(train_acc, test_acc, train_loss, test_loss):
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("Training metrics saved to training_metrics.pkl")

def initialize():
    os.chdir(os.path.abspath('bounds_and_laps/'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Load training data

    batch_size = 1

    train_laplacians, train_boundaries, train_labels = load_chunked_data('train_laplacians', 'train_boundaries',
                                                                         'train_labels')
    train_xs, train_Ls, train_Ds, train_adDs, train_labels = prepare_inputs(train_laplacians, train_boundaries, batch_size, train_labels)

    test_laplacians, test_boundaries, test_labels = load_chunked_data('test_laplacians', 'test_boundaries',
                                                                      'test_labels')
    test_xs, test_Ls, test_Ds, test_adDs, test_labels = prepare_inputs(test_laplacians, test_boundaries, batch_size, test_labels)

    torch.save(train_xs, 'train_xs.pt')
    torch.save(train_Ls, 'train_Ls.pt')
    torch.save(train_Ds, 'train_Ds.pt')
    torch.save(train_adDs, 'train_adDs.pt')
    torch.save(train_labels, 'train_labels.pt')

    torch.save(test_xs, 'test_xs.pt')
    torch.save(test_Ls, 'test_Ls.pt')
    torch.save(test_Ds, 'test_Ds.pt')
    torch.save(test_adDs, 'test_adDs.pt')
    torch.save(test_labels, 'test_labels.pt')
    return None


def main():
    # torch.manual_seed(1337)
    # np.random.seed(1337)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    batch_size = 1
    num_epochs = 50
    learning_rate = 0.001

    os.chdir('bounds_and_laps/')

    train_xs = torch.load('train_xs.pt')
    train_Ls = torch.load('train_Ls.pt')
    train_Ds = torch.load('train_Ds.pt')
    train_adDs = torch.load('train_adDs.pt')
    train_labels = torch.load('train_labels.pt')
    test_xs = torch.load('test_xs.pt')
    test_Ls = torch.load('test_Ls.pt')
    test_Ds = torch.load('test_Ds.pt')
    test_adDs = torch.load('test_adDs.pt')
    test_labels = torch.load('test_labels.pt')

    network = MySCNN(colors=1)
    # total_params = sum(p.numel() for p in network.parameters())
    # print(f"Number of parameters: {total_params}")
    # input()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)
    # criterion = nn.BCELoss()

    # stats = {'train_acc': [], 'test_acc': []}
    epoch_train_acc = {}
    epoch_test_acc = {}
    epoch_train_loss = {}
    epoch_test_loss = {}
    epoch_train_f1 = {}   # To store F1 scores for training
    epoch_test_f1 = {}


    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_train_labels = []
        all_train_preds = []
        all_test_labels = []
        all_test_preds = []

        network.train()
        # Training loop with progress bar
        with tqdm(total=len(train_xs), desc=f"Epoch {epoch + 1}/{num_epochs} - Training") as tepoch:
            for j in range((len(train_xs)-1)):
                try :
                    optimizer.zero_grad()
                    labels_grnd = torch.tensor(train_labels[j], dtype=torch.float).unsqueeze(0)

                    logits, xs_upd = network(train_Ls[j], train_Ds[j], train_adDs[j], train_xs[j])
                    # print(logits)
                    # print(labels_grnd)
                    loss = F.binary_cross_entropy_with_logits(logits, labels_grnd)
                    loss.backward()
                    optimizer.step()

                    # Accumulate loss and accuracy
                    train_loss += loss.item()
                    # print(torch.sigmoid(logits))
                    # print((torch.sigmoid(logits) > 0.5).float())

                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    # print((predictions == labels_grnd).sum().item())
                    # input()
                    all_train_preds.append(predictions)
                    all_train_labels.append(labels_grnd)
                    # print(predictions)
                    # input()
                    train_correct += int((predictions == labels_grnd).sum().item()/2.)
                    # print(train_correct)
                    train_total += 1

                    # Update progress bar
                    tepoch.set_postfix(loss=loss.item())
                    tepoch.update(1)

                except Exception as e:
                    # del train_labels[j], train_Ls[j], train_Ds[j], train_adDs[j], train_xs[j]
                    logging.warning(f'Error while propagating entry at index {j}: {e}')
                    continue

        # train_f1 = f1_score(all_train_labels, all_train_preds)
        # Compute average train loss and accuracy
        avg_train_loss = train_loss / (len(train_xs))
        train_accuracy = train_correct / train_total

        epoch_train_loss[epoch + 1] = avg_train_loss
        epoch_train_acc[epoch + 1] = train_accuracy
        # epoch_train_f1[epoch + 1] = train_f1

        # Testing loop
        network.eval()
        with torch.no_grad():
            with tqdm(total=len(test_xs), desc=f"Epoch {epoch + 1}/{num_epochs} - Testing") as tepoch:
                for j in range(len(test_xs)):
                    try:
                        labels_test = torch.tensor(test_labels[j], dtype=torch.float).unsqueeze(0)
                        logits, _ = network(test_Ls[j], test_Ds[j], test_adDs[j], test_xs[j])

                        loss = F.binary_cross_entropy_with_logits(logits, labels_grnd)
                        test_loss += loss.item()

                        predictions = (torch.sigmoid(logits) > 0.5).float()  # Threshold for binary classification
                        test_correct += int((predictions == labels_test).sum().item()/2)
                        test_total += 1

                        all_test_preds.append(predictions)
                        all_test_labels.append(labels_test)

                        # Update progress bar
                        tepoch.set_postfix(loss=loss.item())
                        tepoch.update(1)
                    except Exception as e:
                        # del test_xs[j], test_labels[j], test_Ls[j], test_Ds[j], test_adDs[j]
                        logging.warning(f'Error while propagating entry at index {j}: {e}')
                        continue

        # test_f1 = f1_score(all_test_labels, all_test_preds)
        # Compute average test loss and accuracy
        avg_test_loss = test_loss / (len(test_xs))
        test_accuracy = test_correct / test_total

        epoch_test_loss[epoch + 1] = avg_test_loss
        epoch_test_acc[epoch + 1] = test_accuracy
        # epoch_test_f1[epoch + 1] = test_f1

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - '
              f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    save_model(network, 'trained_model.pth')
    save_metrics(epoch_train_acc, epoch_test_acc, epoch_train_loss, epoch_test_loss)


if __name__ == "__main__":
    initialize()
    # main()





