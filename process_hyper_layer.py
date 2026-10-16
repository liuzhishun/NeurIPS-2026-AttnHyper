import os
import argparse

import torch
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MIN_MEMBERS_IN_HE = 2        # require at least this many members per hyperedge
INCLUDE_CENTER_TOKEN = True  # add the center token i to its own hyperedge


def process_single_file(args):
    fname, input_dir, out_dir, tau, layer_id = args
    path = os.path.join(input_dir, fname)
    if os.path.exists(os.path.join(out_dir, f"hypergraph_{fname}")):
        return fname
    sample = torch.load(path, weights_only=False)

    token_ids = sample["token_ids"]
    hallucination_labels = sample["hallucination_labels"]
    response_idx = int(sample["response_idx"])

    attention = sample["attention"].float()  # [L, H, seq_len, seq_len]
    L, H, seq_len, _ = attention.shape
    num_heads = L * H
    attention_flat = attention.reshape(num_heads, seq_len, seq_len)

    # node features: self-attention values on the diagonal
    diag = torch.arange(seq_len)
    self_att = attention_flat[:, diag, diag].transpose(0, 1)  # [seq_len, num_heads]
    X_V = torch.clamp(self_att, 0.0, 1.0)

    # causal mask, with attention inside the prompt disabled
    tri_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1)
    prompt_mask = torch.arange(seq_len) < response_idx
    valid_mask = tri_mask.clone()
    valid_mask[:response_idx, :response_idx] = False

    he_node_list = []
    he_id_list = []
    he_attr_list = []
    he_mark_list = []
    he_member_counts_list = []
    he_counter = 0

    for head in range(num_heads):
        if head // H != layer_id:
            continue
        att = attention_flat[head]  # [seq_len, seq_len]
        att_masked = att * valid_mask

        for i in range(seq_len):
            # build hyperedges only for response tokens
            if i < response_idx:
                continue
            scores = att_masked[i]

            members = torch.nonzero(scores > tau, as_tuple=False).view(-1)
            if INCLUDE_CENTER_TOKEN:
                members = torch.cat([members, members.new_tensor([i])])
                members = torch.unique(members, sorted=True)
            m_len = members.numel()
            if m_len < MIN_MEMBERS_IN_HE:
                continue

            he_node_list.append(members)
            he_id_list.append(torch.full((m_len,), he_counter, dtype=torch.long))

            # edge attributes from the raw (unmasked) attention: [mean, max, normalized_head]
            att_vals = att[i, members]
            mean_val = float(att_vals.mean().item())
            max_val = float(att_vals.max().item())
            normalized_head = float(head) / max(1.0, num_heads - 1)
            mean_val = max(0.0, min(mean_val, 1.0))
            max_val = max(0.0, min(max_val, 1.0))
            normalized_head = max(0.0, min(normalized_head, 1.0))
            he_attr_list.append(torch.tensor([mean_val, max_val, normalized_head], dtype=torch.float32))

            # he_mark: [cross prompt->response, within response]
            has_prompt_member = (prompt_mask[members]).any()
            if has_prompt_member and i >= response_idx:
                he_mark_list.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
            else:
                he_mark_list.append(torch.tensor([0.0, 1.0], dtype=torch.float32))

            he_member_counts_list.append(m_len)
            he_counter += 1

    if len(he_node_list) > 0:
        all_nodes = torch.cat(he_node_list)
        all_hids = torch.cat(he_id_list)
        he_incidence_index = torch.stack([all_nodes, all_hids], dim=0)  # [2, num_incidence]
        he_attr = torch.stack(he_attr_list)   # [E, 3]
        he_mark = torch.stack(he_mark_list)   # [E, 2]
        he_member_counts = torch.tensor(he_member_counts_list, dtype=torch.float32)
    else:
        he_incidence_index = torch.zeros((2, 0), dtype=torch.long)
        he_attr = torch.zeros((0, 3), dtype=torch.float32)
        he_mark = torch.zeros((0, 2), dtype=torch.float32)
        he_member_counts = torch.zeros((0,), dtype=torch.float32)

    obj = {
        "response_idx": response_idx,
        "token_ids": token_ids,
        "x": X_V,
        "he_incidence_index": he_incidence_index,
        "he_attr": he_attr,
        "he_mark": he_mark,
        "he_member_counts": he_member_counts,
        "y_token": torch.tensor(hallucination_labels, dtype=torch.float32),
    }

    torch.save(obj, os.path.join(out_dir, f"hypergraph_{fname}"))
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hypergraphs from a single attention layer")
    parser.add_argument("--model", type=str, default="llama-2-7b-chat-hf",
                        help="must match the model directory saved by get_attention.py")
    parser.add_argument("--task_type", type=str, default="Data2txt", choices=["QA", "Summary", "Data2txt"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--layer_id", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--in_root", type=str, default="attention_hidden_data/RAGtruth")
    parser.add_argument("--out_root", type=str, default="hypergraphs/RAGtruth")
    cfg = parser.parse_args()
    set_start_method("spawn", force=True)

    input_dir = f"{cfg.in_root}/{cfg.model}/{cfg.split}_{cfg.task_type}"
    out_dir = f"{cfg.out_root}/{cfg.model}/{cfg.split}_{cfg.task_type}_{cfg.tau}_{cfg.layer_id}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing {input_dir} ......")

    files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    args = [(f, input_dir, out_dir, cfg.tau, cfg.layer_id) for f in files]
    NUM_WORKERS = min(cfg.num_workers, os.cpu_count())

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, args), total=len(files)):
            pass

    print(f"{cfg.layer_id}_Preprocessing Done!")
