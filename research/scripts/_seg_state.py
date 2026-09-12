p = '/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/text_context.h'
s = open(p).read()
old = '''    // Optional layer window for segmented graph capture. last < first disables.
    std::int32_t active_layer_first_                                               = 0;
    std::int32_t active_layer_last_                                                = -1;'''
new = '''    // Optional layer window for segmented graph capture. last < first disables.
    std::int32_t active_layer_first_                                               = 0;
    std::int32_t active_layer_last_                                                = -1;
    // Segmented decode state: when active_graph_segments_ > 1 each
    // ordinary_decode_batch call executes only its slice of the forward pass.
    std::int32_t active_graph_segment_                                             = 0;
    std::int32_t active_graph_segments_                                            = 1;'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''    void run_layers(Tensor& x, Phase phase);'''
new2 = '''    void set_graph_segment(std::int32_t segment, std::int32_t segments) noexcept {
        active_graph_segment_  = segment;
        active_graph_segments_ = segments;
    }
    void run_layers(Tensor& x, Phase phase);'''
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, 'w').write(s)
print('text_context.h segment state added')
