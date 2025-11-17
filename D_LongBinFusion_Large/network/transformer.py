import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryAttention(nn.Module):
    def __init__(self, cam_channels):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv3d(1 + cam_channels, cam_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(cam_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.attn_conv(x)  # (B, 1, Z, Y, X)
        return attn


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, value_emb=False):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)  # Query
        self.k_proj = nn.Linear(input_dim, hidden_dim)  # Key
        self.v_proj = nn.Linear(input_dim, hidden_dim)  # Value
        self.value_emb = value_emb
        
    def forward(self, q, k, v, chunk_size): 
        
        k = k.to(q.device)
        #q_emb = self.q_proj(q) # (b, bin*h*w, 3) -> (b, bin*h*w, 64)
        k_emb = self.k_proj(k) # (b, z*y*x, 3) -> (b, z*y*x, 64)
        
        if self.value_emb:
            v = self.v_proj(v)
        
        outputs = []
        for i in range(0, q.shape[1], chunk_size):
            q_chunk = q[:, i:i+chunk_size, :]  # (b, chunk_size, 3)
            q_emb = self.q_proj(q_chunk)       # (B, chunk, C)
            attn_scores = (q_emb @ k_emb.transpose(-1, -2)) # (b, bin*h*w, z*y*x)
            attn_probs = attn_scores.softmax(dim=-1)
            out = attn_probs @ v # (b, bin*h*w, z*y*x) @ (b, z*y*x, 64) => (b, bin*h*w, 64)
            outputs.append(out)
            
            # 메모리 절약용 참조 해제 (autograd에는 영향 없음)
            del q_emb, q_chunk, attn_scores, attn_probs
            
        outputs = torch.cat(outputs, dim=1)  # (b, bin*h*w, 64)    
        return outputs

class SparseAttention(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, value_emb=False):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)  # Query
        self.k_proj = nn.Linear(input_dim, hidden_dim)  # Key
        self.v_proj = nn.Linear(input_dim, hidden_dim)  # Value
        self.value_emb = value_emb
    
    def batched_index_select(self, inputs, index):
        """
        inputs: (B, N, C)
        index: (B, Nq, K)
        return: (B, Nq, K, C)
        """
        B, N, C = inputs.shape
        _, Nq, K = index.shape

        index = index.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, Nq, K, C)

        # gather from dim=1 (N dimension)
        output = torch.gather(inputs.unsqueeze(1).expand(-1, Nq, -1, -1), dim=2, index=index)  # (B, Nq, K, C)
        return output

    
    def forward(self, q, k, v, top_k, chunk_size): 
        # Nq = bin*h*w, Nk = z*y*x
        k = k.to(q.device)
        #dists = torch.cdist(q, k)  # (B, Nq, Nk)
        #_, knn_idx = dists.topk(k=top_k, dim=-1, largest=False)  # (B, Nq, K)
        
        outputs = []
        for i in range(0, q.shape[1], chunk_size):
            q_chunk = q[:, i:i+chunk_size, :]        # (B, chunk, C)
            dists = torch.cdist(q_chunk, k)          # (B, chunk, Nk)
            _, knn_idx = dists.topk(k=top_k, dim=-1, largest=False)  # (B, chunk, topk)

            k_knn = self.batched_index_select(k, knn_idx)  # (B, chunk, K, C)
            v_knn = self.batched_index_select(v, knn_idx)  # (B, chunk, K, C)

            q_emb = self.q_proj(q_chunk)             # (B, chunk, C)
            k_emb = self.k_proj(k_knn)               # (B, chunk, K, C)
            
            if self.value_emb:
                v_emb = self.v_proj(v_knn)           # (B, chunk, K, C)
            else:
                v_emb = v_knn

            attn_scores = torch.einsum("bnc,bnkc->bnk", q_emb, k_emb)  # (B, chunk, K)
            attn = attn_scores.softmax(dim=-1)
            out = torch.einsum("bnk,bnkc->bnc", attn, v_emb)           # (B, chunk, C)

            outputs.append(out)
            
            # 💡 중간 변수 삭제로 메모리 확보
            del q_chunk, dists, knn_idx
            del k_knn, v_knn, q_emb, k_emb, v_emb
            del attn_scores, attn
            torch.cuda.empty_cache()  # optional, 메모리 파편화 방지
            print(out.retain_grad())

        # concat chunk-wise outputs
        out = torch.cat(outputs, dim=1)  # (B, Nq, C)
        return out # [1, 192000, 64], [1, 384000, 64]

class BEVAttention(nn.Module): # 36175MiB
    def __init__(self, input_dim=2, hidden_dim=64, use_value_emb=False):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)  # Query
        self.k_proj = nn.Linear(input_dim, hidden_dim)  # Key
        self.v_proj = nn.Linear(input_dim, hidden_dim)  # Value
        self.use_value_emb = use_value_emb

    
    def forward(self, q, k, v): 
        # Nq = bin*h*w, Nk = z*y*x
        k = k.to(q.device)

        q_emb = self.q_proj(q)       # (B, Nq, C)
        k_emb = self.k_proj(k)   # (B, Nq, K, C)
            
        if self.use_value_emb:
            v_emb = self.v_proj(v) 
        else:
            v_emb = v
            
        attn_scores = (q_emb @ k_emb.transpose(-1, -2))  # (B, Nq, K)
        attn = attn_scores.softmax(dim=-1)
        attn_probs = attn_scores.softmax(dim=-1)
        out = attn_probs @ v_emb
        
        return out 

class SparseBEVAttention(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, use_value_emb=False):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)  # Query
        self.k_proj = nn.Linear(input_dim, hidden_dim)  # Key
        self.v_proj = nn.Linear(input_dim, hidden_dim)  # Value
        self.use_value_emb = use_value_emb
    
    def batched_index_select(self, inputs, index):
        """
        inputs: (B, Nk, C)
        index: (B, Nq, K)
        return: (B, Nq, K, C)
        """
        B, N, C = inputs.shape
        _, Nq, K = index.shape

        # 1. 플랫하게 만들기
        flat_inputs = inputs.reshape(B * N, C)  # (B*Nk, C)

        # 2. 오프셋 계산, batch까지 flat 하기 위해서
        offset = (torch.arange(B, device=inputs.device) * N).view(B, 1, 1)  # batch=1 -> [[[0]]], (B, 1, 1)
        flat_index = (index + offset).reshape(-1)  # (B, Nq, K) -> (B*Nq*K)

        # 3. 인덱싱
        '''
        Example) 파이토치는 1차원 인덱스 텐서로 다차원 텐서의 첫 번째 축(0번째 dim)을 선택할 수 있게 설계
        flat_inputs:
        tensor([[10., 11., 12.],
                [20., 21., 22.],
                [30., 31., 32.],
                [40., 41., 42.],
                [50., 51., 52.],
                [60., 61., 62.],
                [70., 71., 72.],
                [80., 81., 82.]])

        flat_index: tensor([3, 0, 6, 7])

        selected:
        tensor([[40., 41., 42.],  # flat_inputs[3]
                [10., 11., 12.],  # flat_inputs[0]
                [70., 71., 72.],  # flat_inputs[6]
                [80., 81., 82.]]) # flat_inputs[7]

        '''
        selected = flat_inputs[flat_index]  # (B*Nq*K, C)
        output = selected.view(B, Nq, K, C)

        return output

    
    def forward(self, q, k, v, top_k): 
        # Nq = bin*h*w, Nk = z*y*x
        k = k.to(q.device)
        #q_bev = q[..., :2]  # (B, Nq, 2)
        
        '''
        a -> (3,2)
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        
        b -> (2,2)
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
                
        torch.cdist(a, b, p=2) -> (3,2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
        '''
        
        dists = torch.cdist(q, k)  # (B, Nq, Nk)
        _, knn_idx = dists.topk(k=top_k, dim=-1, largest=False)  # (B, Nq, K), query와 차이가 작은 key 값의 k개의 인덱스를 추출 (key 기준 인덱스)
        
        k_knn = self.batched_index_select(k, knn_idx)  # (B, Nq, K, C)
        v_knn = self.batched_index_select(v, knn_idx)  # (B, Nq, K, C)

        q_emb = self.q_proj(q)       # (B, Nq, C)
        k_emb = self.k_proj(k_knn)   # (B, Nq, K, C)
            
        if self.use_value_emb:
            v_emb = self.v_proj(v_knn) 
        else:
            v_emb = v_knn

        #attn_scores = torch.einsum("bnc,bnkc->bnk", q_emb, k_emb)
        attn_scores = torch.matmul(q_emb.unsqueeze(2), k_emb.transpose(2, 3)).squeeze(2)  # (B, Nq, K)
        attn = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn.unsqueeze(2), v_emb).squeeze(2)  # (B, Nq, 1, C) → (B, Nq, C)
        #out = torch.einsum("bnk,bnkc->bnc", attn, v_emb)
        
        return out 

class MultiheadTransformer(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(MultiheadTransformer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.linear = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embed_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.cross_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        attended_values = v + self.dropout1(attn_output)
        out = self.norm1(attended_values)
        '''
        src2 = self.linear2(self.dropout(F.relu(self.linear1(attended_values))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        '''
        return out