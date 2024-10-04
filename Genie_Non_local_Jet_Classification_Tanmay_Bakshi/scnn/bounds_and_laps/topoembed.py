import topoembedx as tex
import toponetx as tnx
import numpy as np

if __name__ == '__main__':
    simpl = np.load("simpl.npz", allow_pickle=True)
    all_simpl = []
    for i in range(len(simpl)):
        all_simpl.append(simpl[f'arr_{i}'])
    del simpl

    for i in range(10):
        print(all_simpl[i])
        print('\n')

    sc = tnx.classes.SimplicialComplex(all_simpl[0], ranks = 2)
    model = tex.Cell2Vec()

    model.fit(sc, neighborhood_type = "adj", neighborhood_dim = {"rank : 2"})

    embeddings = model.get_embedding()
    print(embeddings)