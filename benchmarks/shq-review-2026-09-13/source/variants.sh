cd /home/mixer/gufo
export HSA_OVERRIDE_GFX_VERSION=11.5.1
Q=/tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/llama-build2/bin/llama-quantize
P=/tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/llama-build-hip/bin/llama-perplexity
COMMON="--allow-requantize --tensor-type indexer\.attn_q_b=f16 --tensor-type indexer=f16 --tensor-type compressor=f16 --tensor-type gate_inp=f16 --tensor-type hc_=f16 --tensor-type ffn_gate_exps=iq2_xxs --tensor-type ffn_up_exps=iq2_xxs --tensor-type ffn_down_exps=q2_k --tensor-type attn_q_a=q8_0 --tensor-type attn_kv=q8_0 --token-embedding-type f16"
run() {
  name=$1; shift
  OUT=/persist/models/DeepSeek-V4-Flash-$name-experiment.gguf
  $Q $COMMON "$@" /persist/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf $OUT Q4_K_M 24 > /tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/requant-$name.log 2>&1; echo "$name requant_exit=$?"
  nix develop -c $P -m $OUT -f /tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 24 -ngl 99 -b 2048 -ub 2048 -t 16 --kl-divergence --kl-divergence-base /tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/base-logits.bin > /tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/kl-$name.log 2>&1; echo "$name kl_exit=$?"
  rm -f $OUT
}
run denseQ6K --tensor-type attn_q_b=q6_k --tensor-type attn_output_a=q6_k --tensor-type attn_output_b=q6_k --tensor-type shexp=q6_k --output-tensor-type q6_k
run attnQ4K-shexpQ8 --tensor-type attn_q_b=q4_k --tensor-type attn_output_a=q4_k --tensor-type attn_output_b=q4_k --tensor-type shexp=q8_0 --output-tensor-type q8_0
run attnQ5K --tensor-type attn_q_b=q5_k --tensor-type attn_output_a=q5_k --tensor-type attn_output_b=q5_k --tensor-type shexp=q5_k --output-tensor-type q6_k
echo VARIANTS_DONE

