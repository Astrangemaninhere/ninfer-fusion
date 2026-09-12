p = '/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/text_context_impl.h'
s = open(p).read()
old = '''void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {'''
new = '''void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    const int layer_begin = active_layer_first_;
    const int layer_end   = active_layer_last_ < 0
                                ? kCfg.n_layers
                                : std::min(kCfg.n_layers, active_layer_last_ + 1);
    for (int layer = layer_begin; layer < layer_end; ++layer) {'''
assert old in s
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('run_layers windowed')
