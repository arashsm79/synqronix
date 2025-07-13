# ----------------- graph_level_loader.py -----------------
import os, glob, numpy as np, torch, scipy.io
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler, LabelEncoder

from synqronix.dataproc.utils import process_mat, create_neuron_info_df


class SessionGraphDataLoader:
    """
    Build *one* Data object per recording session.
    Node features = all neurons in the session.
    Graph label   = majority BF category (categorical, graph-level).
    """
    def __init__(self, data_dir, connectivity_th=0.5,
                 batch_size=8, cache=True):
        self.data_dir   = data_dir
        self.th         = connectivity_th
        self.batch_size = batch_size
        self.cache      = cache

        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ---------- helpers --------------------------------------------------
    def _session_paths(self):
        return glob.glob(os.path.join(self.data_dir, "**", "*.mat"),
                         recursive=True)

    def _build_graph_from_session(self, mat_path: str) -> Data:
        mat = scipy.io.loadmat(mat_path)
        sess  = process_mat(mat)
        ninfo = create_neuron_info_df(sess,
                                      mat_path.replace(".mat", "_info.pkl"))
        N     = len(ninfo)

        # ---------- node features ----------
        feats, labels = [], []
        for row in ninfo.itertuples():
            pc1, pc2 = row.PC[0], row.PC[1]
            x, y, z  = row.x, row.y, row.z
            bf_resp  = row.BFresp
            avg_act  = np.mean(row.activity)
            feats.append([pc1, pc2, x, y, z, bf_resp, avg_act])
            labels.append(row.BFval)

        x = torch.tensor(feats,  dtype=torch.float)   # [N, 7]

        # ---------- edges ----------
        ei, ew = [], []
        corr   = np.vstack(ninfo.global_corr.values)  # (N,N)
        for i in range(N):
            for j in range(i + 1, N):                # upper-triangle
                c = abs(corr[i, j])
                if c >= self.th:
                    ei.append([i, j]); ew.append(c)

        if not ei:                                   # fully disconnected
            ei, ew = [[0, 0]], [0.1]                 # self-loop

        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(ew, dtype=torch.float)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        # ---------- graph label ----------
        maj_label  = int(np.bincount(labels).argmax())   # majority BFval
        y          = torch.tensor([maj_label], dtype=torch.long)  # [1]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # ---------- public API ----------------------------------------------
    def load_all_graphs(self):
        cache_file = os.path.join(self.data_dir, "_graph_level.pt")
        if self.cache and os.path.exists(cache_file):
            graphs, self.scaler, self.label_encoder = torch.load(cache_file)
            return graphs

        graphs, all_feats, all_y = [], [], []
        for p in self._session_paths():
            if "diffxy" in p:     # keep your skip rule
                continue
            g = self._build_graph_from_session(p)
            graphs.append(g)
            all_feats.append(g.x.numpy())
            all_y.append(g.y.item())

        # ---------- fit scalers / encoders ----------
        self.scaler.fit(np.vstack(all_feats))
        self.label_encoder.fit(all_y)

        # scale & encode in-place
        for g in graphs:
            g.x = torch.tensor(self.scaler.transform(g.x), dtype=torch.float)
            g.y = torch.tensor([self.label_encoder.transform(g.y)[0]],
                               dtype=torch.long)

        if self.cache:
            torch.save((graphs, self.scaler, self.label_encoder), cache_file)

        return graphs

    def split(self, graphs, train=0.7, val=0.15):
        idx = np.random.permutation(len(graphs))
        t   = int(train*len(idx)); v = int((train+val)*len(idx))
        return [graphs[i] for i in idx[:t]], \
               [graphs[i] for i in idx[t:v]], \
               [graphs[i] for i in idx[v:]]

    def dataloaders(self):
        g_all = self.load_all_graphs()
        g_tr, g_va, g_te = self.split(g_all)
        return (DataLoader(g_tr, batch_size=self.batch_size, shuffle=True),
                DataLoader(g_va, batch_size=self.batch_size),
                DataLoader(g_te, batch_size=self.batch_size))

    # convenience for model-building
    def num_node_features(self):  return 7
    def num_graph_classes(self):  return len(self.label_encoder.classes_)

if __name__ == "__main__":
    # -------------- usage --------------
    loader = SessionGraphDataLoader("path/to/mat/dir",
                                    connectivity_th=0.5,
                                    batch_size=4)
    train_loader, val_loader, test_loader = loader.dataloaders()

    print("num node features :", loader.num_node_features())
    print("num graph classes  :", loader.num_graph_classes())
