p = '/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/text_context.h'
s = open(p).read()
# remove private set_graph_segment and add a public one near the first public method area
old = '''    void set_graph_segment(std::int32_t segment, std::int32_t segments) noexcept {
        active_graph_segment_  = segment;
        active_graph_segments_ = segments;
    }
    void run_layers(Tensor& x, Phase phase);'''
assert old in s
s = s.replace(old, '    void run_layers(Tensor& x, Phase phase);', 1)

# add public accessor right after 'public:' at line ~154
old2 = '''public:'''
new2 = '''public:
    void set_graph_segment(std::int32_t segment, std::int32_t segments) noexcept {
        active_graph_segment_  = segment;
        active_graph_segments_ = segments;
    }'''
assert s.count(old2) >= 1
s = s.replace(old2, new2, 1)
open(p, 'w').write(s)
print('accessor moved to public')
