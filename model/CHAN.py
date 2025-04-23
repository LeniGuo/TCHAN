import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding Module: Adds positional encoding to model the temporal order of frames
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size * max_seg_num, max_seg_length, d_model]
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention Mechanism: Replaces the original single-head attention to enhance feature extraction capability
    """
    def __init__(self, Q_in_features_dim, K_in_features_dim, attention_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim // num_heads
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        self.linear_q = nn.Linear(Q_in_features_dim, attention_dim, bias=False)
        self.linear_k = nn.Linear(K_in_features_dim, attention_dim, bias=False)
        self.linear_v = nn.Linear(K_in_features_dim, attention_dim, bias=False)
        self.linear_out = nn.Linear(attention_dim, attention_dim, bias=False)
        self.scale = self.attention_dim ** 0.5

    def forward(self, Q, K, mask):
        # Q: [batch_size * max_seg_num, seq_len_q, 1, Q_in_features_dim]
        # K: [batch_size * max_seg_num, 1, seq_len_k, K_in_features_dim]
        # mask: [batch_size * max_seg_num, 1, seq_len_k]
        batch_size = Q.size(0)
        seq_len_q = Q.size(1)
        seq_len_k = K.size(2)
        
        # Remove extra dimension
        Q = Q.squeeze(2)  # [batch_size, seq_len_q, Q_in_features_dim]
        K = K.squeeze(1)  # [batch_size, seq_len_k, K_in_features_dim]
        
        # Linear transformation
        Q = self.linear_q(Q)  # [batch_size, seq_len_q, attention_dim]
        K = self.linear_k(K)  # [batch_size, seq_len_k, attention_dim]
        V = self.linear_v(K)  # [batch_size, seq_len_k, attention_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.attention_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.attention_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.attention_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, attention_dim]

        # Scaled dot-product attention
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attention_score: [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_score = attention_score.masked_fill(~mask.unsqueeze(1), -1e10)
        attention_score = F.softmax(attention_score, dim=-1)
        
        # Apply attention to V
        out = torch.matmul(attention_score, V)  # [batch_size, num_heads, seq_len_q, attention_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.num_heads * self.attention_dim)
        out = self.linear_out(out)  # [batch_size, seq_len_q, attention_dim]
        
        return attention_score, out


class TCHAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(self.config["in_channel"])

        # Encoder Block
        self.conv1d_1 = nn.Conv1d(self.config["in_channel"], self.config["conv1_channel"], kernel_size=5, stride=1, padding=2)
        self.max_pooling_1 = nn.MaxPool1d(2, stride=2, padding=0)
        self.conv1d_2 = nn.Conv1d(self.config["conv1_channel"], self.config["conv2_channel"], kernel_size=5, stride=1, padding=2)
        self.max_pooling_2 = nn.MaxPool1d(2, stride=2, padding=0)
        
        # Residual Connection
        self.residual_conv = nn.Conv1d(self.config["in_channel"], self.config["conv2_channel"], kernel_size=1, stride=4, padding=0)

        # Information Fusion Block with Attetion Mechanism
        self.self_attention = MultiHeadAttention(
            self.config["conv2_channel"], self.config["conv2_channel"], self.config["conv2_channel"], num_heads=8
        )
        self.concept_attention = MultiHeadAttention(
            self.config["concept_dim"], self.config["conv2_channel"], self.config["conv2_channel"], num_heads=8
        )

        # Decoder Block
        self.upsample1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv_after_upsample1 = nn.Conv1d(
            4 * self.config["conv2_channel"], self.config["deconv1_channel"], kernel_size=3, stride=1, padding=1
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv_after_upsample2 = nn.Conv1d(
            self.config["deconv1_channel"], self.config["deconv2_channel"], kernel_size=3, stride=1, padding=1
        )

        # Skip Connection
        self.skip_conv = nn.Conv1d(self.config["conv1_channel"], self.config["deconv1_channel"], kernel_size=1)

        # Similarity Score Computation
        self.similarity_linear1 = nn.Linear(self.config["deconv2_channel"], self.config["similarity_dim"], bias=False)
        self.similarity_linear2 = nn.Linear(self.config["concept_dim"], self.config["similarity_dim"], bias=False)

        self.MLP = nn.Linear(self.config["similarity_dim"], 1)

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        max_seg_num = batch.size(1)
        max_seg_length = batch.size(2)

        # Positional Encoding
        batch_reshaped = batch.view(batch_size * max_seg_num, max_seg_length, -1)
        batch_reshaped = self.positional_encoding(batch_reshaped)
        batch_reshaped = batch_reshaped.transpose(1, 2)  # [batch_size * max_seg_num, in_channel, max_seg_length]

        # Encoder Block
        residual = self.residual_conv(batch_reshaped)  # [batch_size * max_seg_num, conv2_channel, max_seg_length/4]
        tmp1 = self.conv1d_1(batch_reshaped)  # [batch_size * max_seg_num, conv1_channel, max_seg_length]
        tmp1_pooled = self.max_pooling_1(tmp1)  # [batch_size * max_seg_num, conv1_channel, max_seg_length/2]
        tmp2 = self.conv1d_2(tmp1_pooled)  # [batch_size * max_seg_num, conv2_channel, max_seg_length/2]
        tmp2_pooled = self.max_pooling_2(tmp2)  # [batch_size * max_seg_num, conv2_channel, max_seg_length/4]
        
        # Residual Connection
        tmp2_pooled = tmp2_pooled + residual  # [batch_size * max_seg_num, conv2_channel, max_seg_length/4]
        tmp2 = tmp2_pooled.transpose(1, 2)  # [batch_size * max_seg_num, max_seg_length/4, conv2_channel]

        print(f"tmp2 shape: {tmp2.shape}")
        print(f"batch_reshaped shape: {batch_reshaped.shape}")
        print(f"residual shape: {residual.shape}")

        # Attention Mask Generation
        attention_mask = torch.zeros(batch_size, max_seg_num, int(max_seg_length/4), dtype=torch.bool).cuda()
        for i in range(batch_size):
            for j in range(len(seg_len[i])):
                for k in range(math.ceil(seg_len[i][j]/4.0)):
                    attention_mask[i][j][k] = 1
        attention_mask = attention_mask.view(batch_size * max_seg_num, -1).unsqueeze(1)  # [batch_size * max_seg_num, 1, max_seg_length/4]

        # Self-Attention
        K = tmp2.unsqueeze(1)  # [batch_size * max_seg_num, 1, max_seg_length/4, conv2_channel]
        Q = tmp2.unsqueeze(2)  # [batch_size * max_seg_num, max_seg_length/4, 1, conv2_channel]
        print(f"K shape: {K.shape}, Q shape: {Q.shape}")
        self_attention_score, self_attention_result = self.self_attention(Q, K, attention_mask)
        # Remove sum(-2) to preserve sequence dimension
        # self_attention_result: [batch_size * max_seg_num, max_seg_length/4, conv2_channel]

        # Concept-Guided Attention
        concept1_expanded = concept1.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            batch_size, max_seg_num, int(self.config["max_frame_num"]/4), 1, self.config["concept_dim"]
        ).contiguous().view(batch_size * max_seg_num, int(self.config["max_frame_num"]/4), 1, self.config["concept_dim"])
        concept2_expanded = concept2.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            batch_size, max_seg_num, int(self.config["max_frame_num"]/4), 1, self.config["concept_dim"]
        ).contiguous().view(batch_size * max_seg_num, int(self.config["max_frame_num"]/4), 1, self.config["concept_dim"])

        concept1_attention_score, concept1_attention_result = self.concept_attention(concept1_expanded, K, attention_mask)
        # Remove sum(-2)
        # concept1_attention_result: [batch_size * max_seg_num, max_seg_length/4, conv2_channel]
        
        concept2_attention_score, concept2_attention_result = self.concept_attention(concept2_expanded, K, attention_mask)
        # Remove sum(-2)
        # concept2_attention_result: [batch_size * max_seg_num, max_seg_length/4, conv2_channel]

        # Information Fusion
        print(f"self_attention_result shape: {self_attention_result.shape}")
        print(f"concept1_attention_result shape: {concept1_attention_result.shape}")
        print(f"concept2_attention_result shape: {concept2_attention_result.shape}")
        attention_result = torch.cat(
            (tmp2, self_attention_result, concept1_attention_result, concept2_attention_result), dim=-1
        )  # [batch_size * max_seg_num, max_seg_length/4, 4*conv2_channel]

        # Decoder Block
        result = attention_result.transpose(1, 2)  # [batch_size * max_seg_num, 4*conv2_channel, max_seg_length/4]
        result = self.upsample1(result)  # [batch_size * max_seg_num, 4*conv2_channel, max_seg_length/2]
        result = self.conv_after_upsample1(result)  # [batch_size * max_seg_num, deconv1_channel, max_seg_length/2]
        
        # Skip Connection
        skip = self.skip_conv(tmp1_pooled)  # [batch_size * max_seg_num, deconv1_channel, max_seg_length/2]
        result = result + skip  # [batch_size * max_seg_num, deconv1_channel, max_seg_length/2]

        result = self.upsample2(result)  # [batch_size * max_seg_num, deconv1_channel, max_seg_length]
        result = self.conv_after_upsample2(result)  # [batch_size * max_seg_num, deconv2_channel, max_seg_length]
        result = result.transpose(1, 2).contiguous().view(
            batch_size, max_seg_num * max_seg_length, -1
        )  # [batch_size, max_seg_num*max_seg_length, deconv2_channel]

        # Similarity Score Computation
        similar_1 = self.similarity_linear1(result)  # [batch_size, max_seg_num*max_seg_length, similarity_dim]
        concept1_similar = self.similarity_linear2(concept1)  # [batch_size, similarity_dim]
        concept2_similar = self.similarity_linear2(concept2)  # [batch_size, similarity_dim]

        concept1_similar = concept1_similar.unsqueeze(1) * similar_1  # [batch_size, max_seg_num*max_seg_length, similarity_dim]
        concept2_similar = concept2_similar.unsqueeze(1) * similar_1  # [batch_size, max_seg_num*max_seg_length, similarity_dim]

        concept1_score = self.MLP(concept1_similar)  # [batch_size, max_seg_num*max_seg_length, 1]
        concept2_score = self.MLP(concept2_similar)  # [batch_size, max_seg_num*max_seg_length, 1]

        concept1_score = torch.sigmoid(concept1_score).squeeze(-1).view(batch_size, max_seg_num, max_seg_length)
        concept2_score = torch.sigmoid(concept2_score).squeeze(-1).view(batch_size, max_seg_num, max_seg_length)

        return concept1_score, concept2_score