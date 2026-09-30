# -*- coding: utf-8 -*-
"""i18n_misc.py — zh/en strings for convert_gui.py / rag_gui.py / model_import.py /
gui_tips.py (the "misc" GUI set, i.e. everything except the serve console).

Contract (same as every other i18n_*.py table):
    STRINGS = {key: {'zh': ..., 'en': ...}}
`gui_i18n.py` merges all i18n_*.py tables at import; callers use `t('key', **kw)`
with `str.format` placeholders. EVERY key here must carry BOTH languages —
`gui_i18n_check.py` fails the build on a zh-only entry.

Key areas:
  misc.*    language switch, shared chrome
  conv.*    convert_gui.py (convert page + /api/convert messages)
  rag.*     rag_gui.py (retrieval page + /api/query messages)
  imp.*     model_import.py (scan / verdict / env check / archkit gap report)
  tips.*    gui_tips.py TIPS   (keys = HTML control id, see that file)
  tipg.*    gui_tips.py GROUPS (parameter group captions)
  tipgl.*   gui_tips.py GLOSSARY ({id}.term / {id}.text)
  imp.gap.* per-`need`-prefix rendering of the archkit manifest gap actions
"""
from __future__ import annotations

STRINGS = {
    # ------------------------------------------------------------------ misc
    'misc.lang_label': {'zh': '语言', 'en': 'Language'},
    # Language names stay endonyms in both languages (a switcher you cannot read
    # is useless).
    'misc.lang_zh': {'zh': '中文', 'en': '中文'},
    'misc.lang_en': {'zh': 'English', 'en': 'English'},

    # ------------------------------------------------------- convert_gui.py
    'conv.tab_title': {'zh': 'NInfer 模型转换', 'en': 'NInfer convert'},
    'conv.title': {'zh': 'NInfer 模型转换', 'en': 'NInfer model convert'},
    'conv.sub': {'zh': 'safetensors（HF / vLLM）→ .ninfer — 滑块简约版',
                 'en': 'safetensors (HF / vLLM) → .ninfer — slider-minimal'},
    'conv.lbl_model': {'zh': '模型目录', 'en': 'Model folder'},
    'conv.ph_model': {'zh': '/path/to/Qwen3.8-27B', 'en': '/path/to/Qwen3.8-27B'},
    'conv.lbl_qmodel': {'zh': '量化源目录', 'en': 'Quantised source folder'},
    'conv.ph_qmodel': {'zh': '（NVFP4 转换时填写）', 'en': '(fill in for an NVFP4 conversion)'},
    'conv.lbl_out': {'zh': '输出路径', 'en': 'Output path'},
    'conv.ph_out': {'zh': 'out/qwen3_8_27b.ninfer', 'en': 'out/qwen3_8_27b.ninfer'},
    'conv.lbl_quant': {'zh': '量化档位', 'en': 'Quantisation'},
    'conv.lbl_kv': {'zh': 'KV 精度档', 'en': 'KV precision'},
    'conv.lbl_ctx': {'zh': '上下文上限', 'en': 'Context ceiling'},
    'conv.lbl_batch': {'zh': '批量宽度', 'en': 'Batch width'},
    'conv.btn_go': {'zh': '开始转换', 'en': 'Start conversion'},
    'conv.log_ready': {'zh': '就绪。', 'en': 'Ready.'},
    'conv.status_need_paths': {'zh': '填模型和输出路径', 'en': 'Fill in the model and output paths'},
    'conv.status_running': {'zh': '转换中…', 'en': 'Converting…'},
    'conv.status_done': {'zh': '完成 ✓', 'en': 'Done ✓'},
    'conv.status_failed': {'zh': '失败: {err}', 'en': 'Failed: {err}'},
    'conv.status_req_failed': {'zh': '请求失败', 'en': 'Request failed'},
    'conv.err_need_qmodel': {'zh': 'NVFP4 需要量化源目录',
                             'en': 'NVFP4 conversion needs the quantised source folder'},
    'conv.err_exit': {'zh': '退出码 {code}', 'en': 'exit code {code}'},
    'conv.console_ready': {'zh': 'NInfer 转换 GUI: http://127.0.0.1:{port}  (转换脚本目录: {dir})',
                           'en': 'NInfer Convert GUI: http://127.0.0.1:{port}  '
                                 '(convert dir: {dir})'},
    # slider labels (also read by JS through the injected table)
    'conv.quant_0': {'zh': 'BF16', 'en': 'BF16'},
    'conv.quant_1': {'zh': 'FP8', 'en': 'FP8'},
    'conv.quant_2': {'zh': 'NVFP4', 'en': 'NVFP4'},
    'conv.quant_3': {'zh': 'Groupwise', 'en': 'Groupwise'},
    'conv.kv_0': {'zh': 'BF16', 'en': 'BF16'},
    'conv.kv_1': {'zh': 'INT8', 'en': 'INT8'},
    'conv.kv_2': {'zh': 'FP8', 'en': 'FP8'},
    'conv.kv_3': {'zh': 'NVFP4', 'en': 'NVFP4'},
    'conv.kv_4': {'zh': 'Rk4v4 混合', 'en': 'Rk4v4 mix'},

    # ----------------------------------------------------------- rag_gui.py
    'rag.tab_title': {'zh': 'NInfer RAG 检索', 'en': 'NInfer RAG search'},
    'rag.title': {'zh': 'NInfer RAG 检索', 'en': 'NInfer RAG search'},
    'rag.sub': {'zh': '本地向量检索（bge-small-zh · Chroma · wenxin_test）— 滑块简约版',
                'en': 'Local vector search (bge-small-zh · Chroma · wenxin_test) — slider-minimal'},
    'rag.lbl_topk': {'zh': 'top-k', 'en': 'top-k'},
    'rag.lbl_floor': {'zh': '相似度下限', 'en': 'Similarity floor'},
    'rag.lbl_snippet': {'zh': '片段扩展', 'en': 'Snippet window'},
    'rag.seg_suffix': {'zh': ' 段', 'en': ' seg'},
    'rag.ph_query': {'zh': '输入查询…', 'en': 'Type a query…'},
    'rag.btn_search': {'zh': '检索', 'en': 'Search'},
    'rag.status_searching': {'zh': '检索中…', 'en': 'Searching…'},
    'rag.status_results': {'zh': '{n} 条结果 · {ms} ms', 'en': '{n} hits · {ms} ms'},
    'rag.status_error': {'zh': '错误: {err}', 'en': 'Error: {err}'},
    'rag.status_req_failed': {'zh': '请求失败', 'en': 'Request failed'},
    'rag.score': {'zh': '相似 {score}', 'en': 'sim {score}'},
    'rag.page': {'zh': '第 {p} 页', 'en': 'page {p}'},
    'rag.console_ready': {'zh': 'NInfer RAG GUI: http://127.0.0.1:{port}',
                          'en': 'NInfer RAG GUI: http://127.0.0.1:{port}'},

    # ------------------------------------------------------ model_import.py
    # scan / friendly format blurbs
    'imp.kind.ninfer': {'zh': 'ninfer 自研容器', 'en': 'ninfer container'},
    'imp.kind.gguf': {'zh': 'GGUF', 'en': 'GGUF'},
    'imp.kind.hf': {'zh': 'HuggingFace / safetensors', 'en': 'HuggingFace / safetensors'},
    'imp.kind.lora': {'zh': 'LoRA 微调包', 'en': 'LoRA adapter'},
    'imp.kind.unknown': {'zh': '无法识别', 'en': 'unrecognised'},
    'imp.err_not_found': {'zh': '路径不存在', 'en': 'path does not exist'},
    'imp.err_unknown_file': {'zh': '无法识别的文件类型', 'en': 'unrecognised file type'},
    'imp.f_unknown_file': {'zh': '这是一个文件,但不是 ninfer 或 GGUF 模型格式。',
                           'en': 'This is a file, but not a ninfer or GGUF model.'},
    'imp.err_no_model_in_dir': {'zh': '目录内无已知模型文件',
                                'en': 'no known model file in this folder'},
    'imp.f_empty_dir': {'zh': '这个文件夹里没有找到模型文件。请确认里面应该有 config.json + '
                              '.safetensors, 或一个 .gguf / .ninfer 文件。',
                        'en': 'No model file found in this folder. It should contain '
                              'config.json + .safetensors, or a single .gguf / .ninfer file.'},
    'imp.err_scan_failed': {'zh': '扫描出错: {err}', 'en': 'scan error: {err}'},
    'imp.f_ninfer': {'zh': '这是 ninfer 自家格式,可以直接运行。',
                     'en': 'This is ninfer\'s own format; it runs as-is.'},
    'imp.q_ninfer': {'zh': 'NVFP4/自研量化', 'en': 'NVFP4 / ninfer quantisation'},
    'imp.err_read_failed': {'zh': '文件读取失败', 'en': 'could not read the file'},
    # LoRA
    'imp.f_lora': {'zh': '这是 LoRA 微调包(只含增量权重,不是完整模型)。',
                   'en': 'This is a LoRA adapter (only the delta weights, not a full model).'},
    'imp.lora_rank': {'zh': 'LoRA r={r}', 'en': 'LoRA r={r}'},
    'imp.lora_base': {'zh': 'PEFT {peft}; 基座: {base}', 'en': 'PEFT {peft}; base model: {base}'},
    'imp.lora_base_missing': {'zh': 'PEFT {peft}; 基座信息缺失',
                              'en': 'PEFT {peft}; base-model info missing'},
    'imp.err_lora_cfg': {'zh': 'adapter_config.json 读取失败: {err}',
                         'en': 'cannot read adapter_config.json: {err}'},
    'imp.issue_lora_needs_base': {
        'zh': 'LoRA 需对应基座模型 + 离线融合(见 tools/convert/lora/_LORA.md)',
        'en': 'a LoRA needs its base model plus an offline merge (see '
              'tools/convert/lora/_LORA.md)'},
    # HF folder
    'imp.f_hf': {'zh': '这是 HuggingFace/safetensors 格式(vLLM 也用它)。',
                 'en': 'This is HuggingFace/safetensors format (vLLM uses it too).'},
    'imp.arch_known': {'zh': '架构: {archs} ({mt})', 'en': 'architecture: {archs} ({mt})'},
    'imp.arch_unknown': {'zh': '架构: 未知', 'en': 'architecture: unknown'},
    'imp.arch_not_supported': {'zh': ' —— 这个架构不在本软件支持列表里。',
                               'en': ' — this architecture is not on the supported list.'},
    'imp.arch_qwen_candidate': {'zh': ' —— Qwen3 家族, 但不是软件支持的规格',
                                'en': ' — Qwen3 family, but not the supported spec'},
    'imp.q_nvfp4_pre': {'zh': 'NVFP4 预量化(推荐, 转换后体积小)',
                        'en': 'NVFP4 pre-quantised (recommended: the converted file is small)'},
    'imp.q_bitsandbytes': {'zh': 'bitsandbytes 量化', 'en': 'bitsandbytes quantisation'},
    'imp.q_bf16': {'zh': 'BF16/FP16 全精度(转换后体积大)',
                   'en': 'BF16/FP16 full precision (the converted file is big)'},
    # GGUF
    'imp.f_gguf': {'zh': '这是 GGUF 格式(llama.cpp 生态)。',
                   'en': 'This is GGUF format (the llama.cpp ecosystem).'},
    'imp.f_gguf_versioned': {'zh': 'GGUF 格式 (版本 {ver}, {n} 个张量)',
                             'en': 'GGUF format (version {ver}, {n} tensors)'},
    'imp.err_gguf_header': {'zh': 'GGUF 文件头不完整', 'en': 'truncated GGUF header'},
    'imp.issue_kquant': {'zh': '包含 K 系列量化(K-quant),本仓库的读者还解不了',
                         'en': "contains K-series K-quant tensors the reader in this "
                               'repo cannot decode yet'},
    'imp.issue_quant_other': {
        'zh': '包含引擎不能直接读的量化类型(只认 F32/F16/BF16)',
        'en': 'contains tensor types the engine cannot read (F32/F16/BF16 only)'},
    # 层数记账: block_count 把 nextn/草稿块也算进去, 而 max(blk.N)+1 数的是"块数"。
    # 两个来源(元数据 / 张量表)对不上时不挑一个用 —— 挑错就是层数差一层且不响。
    'imp.issue_gguf_layers': {
        'zh': '层数记账对不上: 元数据说 {declared} 个 nextn/草稿块, 张量表里有 '
              '{observed} 个块带着 .nextn.* 张量 ({index}); 文件共 {blocks} 块。'
              '按元数据记账, 但这两个数不一致本身就是问题(少算把草稿块当解码层, '
              '多算把解码层当草稿块)',
        'en': 'layer accounting disagrees: the metadata declares {declared} nextn/draft '
              'block(s) while the tensor table has {observed} block(s) carrying '
              '`.nextn.` tensors ({index}); the file holds {blocks} blocks. The '
              'metadata is used for the count, but the two numbers disagreeing is '
              'itself the problem (under-counting books the draft block as a decoder '
              'layer, over-counting books a decoder layer as the draft block)'},
    'imp.qn.q4_2': {'zh': 'Q4_2', 'en': 'Q4_2'},
    'imp.qn.q4_3': {'zh': 'Q4_3', 'en': 'Q4_3'},
    'imp.qn.i8': {'zh': 'I8 整型', 'en': 'I8'},
    'imp.qn.i16': {'zh': 'I16 整型', 'en': 'I16'},
    'imp.qn.i32': {'zh': 'I32 整型', 'en': 'I32'},
    'imp.qn.i64': {'zh': 'I64 整型', 'en': 'I64'},
    'imp.qn.f64': {'zh': 'F64 浮点', 'en': 'F64'},
    'imp.err_gguf_parse': {'zh': 'GGUF 解析失败: {err}', 'en': 'cannot parse the GGUF: {err}'},
    'imp.arch_unknown_from_name': {'zh': '无法从文件名确认架构家族',
                                   'en': 'cannot tell the architecture family from the file name'},
    # GGUF tensor type names (ids from the gguf spec)
    'imp.qn.f32': {'zh': 'F32 浮点', 'en': 'F32 float'},
    'imp.qn.f16': {'zh': 'F16 半精度', 'en': 'F16 half'},
    'imp.qn.bf16': {'zh': 'BF16 半精度', 'en': 'BF16 half'},
    'imp.qn.q4_0': {'zh': 'Q4_0', 'en': 'Q4_0'},
    'imp.qn.q4_1': {'zh': 'Q4_1', 'en': 'Q4_1'},
    'imp.qn.q5_0': {'zh': 'Q5_0', 'en': 'Q5_0'},
    'imp.qn.q5_1': {'zh': 'Q5_1', 'en': 'Q5_1'},
    'imp.qn.q8_0': {'zh': 'Q8_0', 'en': 'Q8_0'},
    'imp.qn.q8_1': {'zh': 'Q8_1', 'en': 'Q8_1'},
    'imp.qn.q2_k': {'zh': 'Q2_K', 'en': 'Q2_K'},
    'imp.qn.q3_k': {'zh': 'Q3_K', 'en': 'Q3_K'},
    'imp.qn.q4_k': {'zh': 'Q4_K', 'en': 'Q4_K'},
    'imp.qn.q5_k': {'zh': 'Q5_K', 'en': 'Q5_K'},
    'imp.qn.q6_k': {'zh': 'Q6_K', 'en': 'Q6_K'},
    'imp.qn.q8_k': {'zh': 'Q8_K', 'en': 'Q8_K'},
    'imp.qn.iq': {'zh': 'IQ 系列', 'en': 'IQ series'},
    'imp.qn.unknown': {'zh': '未知({id})', 'en': 'unknown({id})'},
    # verdict: ninfer
    'imp.v_ninfer_run_title': {'zh': '这个模型可以直接运行!',
                               'en': 'This model can run as-is!'},
    'imp.v_ninfer_run_detail': {'zh': '这是 ninfer 自家的模型文件({quant}),不用转换。',
                                'en': 'This is a ninfer model file ({quant}) — no conversion '
                                      'needed.'},
    'imp.v_ninfer_bad_family_title': {'zh': '这个模型文件本软件跑不了',
                                      'en': 'This model file will not run here'},
    'imp.v_ninfer_bad_family_detail': {
        'zh': '文件格式是对的,但它的模型家族({family})不在支持列表里。',
        'en': 'The file format is right, but its model family ({family}) is not on the '
              'supported list.'},
    # verdict: VRAM
    'imp.v_vram_title': {'zh': '显存可能不够,建议先试试小上下文',
                         'en': 'VRAM may be too small — try a shorter context first'},
    'imp.v_vram_detail': {
        'zh': '模型本体约 {gb} GB,你的显卡 {gpu} 只有 {vram} GB 显存,'
              '开 {ctx} 上下文可能放不下。可以先减小上下文(比如 2048)再试。',
        'en': 'The model alone is about {gb} GB and your {gpu} has only {vram} GB of VRAM, '
              'so a {ctx}-token context may not fit. Try a smaller context (2048) first.'},
    'imp.tip_ctx_smaller': {'zh': '把上下文长度调小(2048~4096)',
                            'en': 'turn the context length down (2048-4096)'},
    'imp.tip_smaller_model': {'zh': '换更小的模型', 'en': 'switch to a smaller model'},
    'imp.tip_bigger_gpu': {'zh': '购买更大显存的显卡(本软件最低建议 16GB+)',
                           'en': 'get a GPU with more VRAM (16GB+ is the recommended minimum)'},
    'imp.tip_no_nvidia': {'zh': '没有检测到 NVIDIA 显卡:本软件需要 NVIDIA 显卡(CUDA)才能跑。',
                          'en': 'No NVIDIA GPU detected: this software needs an NVIDIA GPU '
                                '(CUDA) to run.'},
    'imp.tip_install_driver': {'zh': '驱动安装: {url}', 'en': 'driver download: {url}'},
    # verdict: HF
    'imp.v_hf_candidate_title': {'zh': '这是 Qwen3 家族,但不是软件支持的规格',
                                 'en': 'Qwen3 family, but not a supported spec'},
    'imp.v_hf_candidate_detail': {
        'zh': '本软件目前只支持 5120 隐藏宽度、64 层的 27B 版 Qwen3。'
              '你这份模型({layers} 层 / {hidden} 隐藏)对不上,转换出来也跑不了,'
              '所以就不浪费时间转换了。',
        'en': 'This software supports only the 27B Qwen3 with hidden width 5120 and 64 '
              'layers. Your model ({layers} layers / hidden {hidden}) does not match, so a '
              'conversion would not run either — we skip it instead of wasting your time.'},
    'imp.tip_hf_download_27b': {
        'zh': '去 {url} 下载 Qwen3.8-27B(27B, 64 层)或 Qwen3.6-27B。',
        'en': 'Download Qwen3.8-27B (27B, 64 layers) or Qwen3.6-27B from {url}.'},
    'imp.v_hf_unsupported_title': {'zh': '这个模型本软件跑不了',
                                   'en': 'This model will not run here'},
    'imp.v_hf_unsupported_detail': {
        'zh': '你下载的模型架构不在本软件支持列表里。目前只支持 Qwen3 家族的 27B/64 层版本。'
              '支持列表外的模型需要专门的适配(每个架构的推理引擎都要重新实现),'
              '不是简单转换能解决的。',
        'en': 'The architecture of the model you downloaded is not on the supported list. '
              'Today only the Qwen3 family at 27B / 64 layers is supported. Anything else '
              'needs dedicated adaptation (every architecture needs its own inference '
              'engine), which a plain conversion cannot deliver.'},
    'imp.tip_hf_search': {'zh': '去 {url} 搜索支持列表里的模型下载。',
                          'en': 'Search {url} for a model from the supported list.'},
    'imp.v_hf_convert_title': {'zh': '可以自动转换成 ninfer 格式再运行',
                               'en': 'Can be converted to ninfer format automatically'},
    'imp.v_hf_convert_detail': {
        'zh': '这个模型({family})属于支持家族,可以转换。转换需要一些时间(纯 CPU 也能跑)。',
        'en': 'This model ({family}) belongs to a supported family, so it can be converted. '
              'The conversion takes a while (it runs on CPU alone).'},
    'imp.v_convert_vram_title': {'zh': '转换完也装不下:显存不够',
                                 'en': 'It will not fit even after converting: not enough VRAM'},
    'imp.v_convert_vram_detail': {
        'zh': '这个模型 {quant} 约 {params} 亿参数,转换后仍需 {need} GB 左右显存,'
              '你的显卡只有 {vram} GB。建议换小一号的模型,或换大显存显卡。',
        'en': 'This model is {quant}, about {params}0 billion parameters, and still needs '
              'roughly {need} GB of VRAM after conversion while your GPU has only {vram} GB. '
              'Go one size down, or get a GPU with more VRAM.'},
    'imp.tip_vram_scales': {'zh': '模型越大越吃显存: 27B 级别建议 24GB+ 显卡',
                            'en': 'bigger models eat more VRAM: 24GB+ is recommended for 27B'},
    'imp.tip_qwen_small': {'zh': 'Qwen 系列有小号版本可下载',
                           'en': 'smaller Qwen versions are available for download'},
    'imp.tip_driver_update': {'zh': '安装/更新驱动: {url}',
                              'en': 'install/update the driver: {url}'},
    'imp.tip_prefer_nvfp4': {'zh': '提示:全精度版转换后体积大;'
                                   '有 NVFP4 预量化版的话下载那个更快。',
                             'en': 'Tip: a full-precision source converts into a large file; '
                                   'if an NVFP4 pre-quantised release exists, that one is '
                                   'faster to bring up.'},
    # verdict: GGUF
    'imp.v_gguf_kquant_title': {'zh': '这个 GGUF 的量化类型本仓库解不了',
                                'en': 'This GGUF uses a quantisation this repo cannot decode'},
    'imp.v_gguf_kquant_detail': {
        'zh': 'K 系列量化是 llama.cpp 的压缩格式,本软件不能直接读。'
              '请到模型页面下载 F16 或 BF16(未量化)版本,文件名通常带 '
              "'f16' 或 'bf16' 字样,后缀仍是 .gguf。",
        'en': 'K-quant is a llama.cpp compression format this software cannot read '
              'directly. Download the F16 or BF16 (unquantised) release from the model '
              'page instead — the file name usually says \'f16\' or \'bf16\', and the '
              'suffix is still .gguf.'},
    'imp.v_gguf_unsupported_title': {'zh': '这个模型本软件跑不了',
                                     'en': 'This model will not run here'},
    'imp.v_gguf_unsupported_detail': {
        'zh': '这个 GGUF 对应的模型家族不在支持列表里(只支持 Qwen3 的 27B/64 层版本)。',
        'en': 'The family behind this GGUF is not on the supported list (only the Qwen3 '
              '27B / 64-layer spec is).'},
    'imp.v_gguf_convert_title': {'zh': '可以转换成 ninfer 格式再运行',
                                 'en': 'Can be converted to ninfer format'},
    'imp.v_gguf_convert_detail': {
        'zh': 'GGUF 会先解包成半精度,再转成 ninfer 格式,需要一些时间。',
        'en': 'The GGUF is first unpacked into half precision and then converted to ninfer '
              'format; that takes a while.'},
    'imp.tip_gguf_f16_only': {'zh': '只支持 F16/BF16/F32 的 GGUF; K-quant 版本请换源。',
                              'en': 'only F16/BF16/F32 GGUFs are supported; for a K-quant '
                                    'file switch to another source.'},
    # ---- actionable refusals (what is missing, not just "unsupported") ----
    'imp.err_gguf_short_header': {
        'zh': 'GGUF 文件头只有 {got} 字节,格式要求 {need} 字节'
              '(magic 4 + 版本 4 + 张量数 8 + 元数据条数 8) —— 文件被截断',
        'en': 'the GGUF header is only {got} bytes; the format requires {need} '
              '(magic 4 + version 4 + tensor_count 8 + metadata_kv_count 8) — the file '
              'is truncated'},
    'imp.err_gguf_implausible': {
        'zh': 'GGUF 头声明 {tensors} 个张量 / {kv} 条元数据,数量不可信 —— 文件损坏'
              '或根本不是 GGUF',
        'en': 'the GGUF header claims {tensors} tensors / {kv} metadata entries, which is '
              'not plausible — the file is corrupt or is not a GGUF'},
    'imp.err_gguf_bad_ndims': {
        'zh': '张量 {name} 声明了 {n} 维(上限 8)—— 张量表已错位或文件损坏',
        'en': 'tensor {name} claims {n} dimensions (limit 8) — the tensor table is '
              'misaligned or the file is corrupt'},
    'imp.err_gguf_string_too_long': {
        'zh': 'GGUF 里出现长度为 {n} 字节的字符串(上限 {cap})—— 元数据块已错位或文件损坏',
        'en': 'the GGUF contains a {n}-byte string (ceiling {cap}) — the metadata block '
              'is misaligned or the file is corrupt'},
    'imp.issue_gguf_unmapped': {
        'zh': '有 {n} 个张量没有 GGUF→HF 名称映射(如 {names}),转换器会漏掉它们',
        'en': '{n} tensors have no GGUF→HF name mapping (e.g. {names}); a converter '
              'would drop them'},
    'imp.arch_gguf_geometry': {
        'zh': '(张量表实测 hidden={hidden}, {layers} 层)',
        'en': ' (tensor table measures hidden={hidden}, {layers} layers)'},
    'imp.v_gguf_broken_title': {'zh': '这个 GGUF 读不出来(文件不完整或已损坏)',
                                'en': 'This GGUF cannot be read (incomplete or corrupt)'},
    'imp.v_gguf_broken_detail': {
        'zh': '读 {path} 时失败:{err}。这次失败与模型家族无关 —— 文件本身没有读通,'
              '所以还不能判断能不能跑。缺的是:完整可读的 GGUF 文件。'
              '补救:重新下载(或校验下载完整性),再拖进来。',
        'en': 'Reading {path} failed: {err}. This says nothing about the model family — '
              'the file itself did not parse, so its runnability is unknown. What is '
              'missing: a complete, readable GGUF. Re-download it (or verify the '
              'download) and drag it in again.'},
    'imp.v_gguf_kquant_detail2': {
        'zh': '这个文件里的 {quant} 本仓库的 GGUF 读者解不了(它能解 {support},'
              '见 tools/convert/gguf_kquant.py 的 GGUF_LAYOUT/DEQUANTIZERS)。补救:'
              '到模型页面下载 F16/BF16 版(文件名通常带 '
              "'f16'/'bf16',后缀仍是 .gguf);或用量化工具自行解量化成 F16 再拖入。"
              '注意:认出格式 != 引擎能跑 —— 即使字节层能读,引擎侧仍然需要该架构的 '
              'target;逐项缺件用 `python3 tools/convert/import_model.py <file>` 看。',
        'en': 'The {quant} tensors in this file cannot be decoded by this repo'
              "'s GGUF reader (it decodes {support}; see GGUF_LAYOUT/DEQUANTIZERS "
              'in tools/convert/gguf_kquant.py). Remedy: download the F16/BF16 release '
              "(the file name usually says 'f16'/'bf16' and the suffix stays .gguf), "
              'or dequantise it to F16 first. Note: recognising the format is not the '
              'same as the engine being able to run it — even when the bytes are '
              'readable, the engine still needs a target for that architecture; run '
              '`python3 tools/convert/import_model.py <file>` for the itemised gaps.'},
    'imp.v_gguf_unsupported_detail2': {
        'zh': '这个 GGUF 自己声明 general.architecture={arch}。缺的部分:'
              '① 没有任何注册 target 覆盖它 —— 引擎的注册 target 是 {targets},'
              '它们接受的解码器族只有 {families};'
              '② 该族的权重布局/口味(注意力、norm、头)与前端 tokenizer 侧车在引擎里'
              '都还没有实现,这不是"转换一下"能解决的。'
              '补救:先跑接入规格流水线生成缺口清单,再按清单落地新 target。',
        'en': 'This GGUF declares general.architecture={arch}. What is missing: '
              '(1) no registered target covers it — the engine registers {targets}, and '
              'the decoder families they accept are {families} only; (2) that family\'s '
              'weight layout and flavour (attention, norms, head) and its frontend '
              'tokenizer sidecar are not implemented in the engine either, and no '
              'conversion step can deliver them. '
              'Remedy: run the access-spec pipeline to get the gap list, then land the '
              'new target it asks for.'},
    'imp.v_hf_unsupported_detail2': {
        'zh': '{facts}缺的部分:① 没有任何注册 target 接受这个族 —— 注册 target 是 '
              '{targets},它们接受的解码器族只有 {families};② 该族的权重布局/口味'
              '(注意力、norm、头)在引擎里没有实现。这不是"转换一下"能解决的,'
              '每个架构都要有自己的推理实现。补救:跑接入规格流水线生成缺口清单。',
        'en': '{facts}What is missing: (1) no registered target accepts this family — the '
              'engine registers {targets}, whose decoder families are {families} only; '
              '(2) that family\'s weight layout and flavour (attention, norms, head) is '
              'not implemented in the engine. A plain conversion cannot deliver this; '
              'every architecture needs its own inference implementation. Remedy: run the '
              'access-spec pipeline to get the gap list.'},
    'imp.v_cfg_broken_title': {'zh': 'config.json 读不出来(文件不完整或已损坏)',
                               'en': 'config.json cannot be read (incomplete or corrupt)'},
    'imp.v_cfg_broken_detail': {
        'zh': '{path}/config.json 解析失败:{err}。这份配置读不通,所以架构根本无法判断 ——'
              '并不是"架构不支持"。缺的是:一份可解析的 config.json。'
              '补救:重新下载 config.json(常见于下载中断)。',
        'en': 'Parsing {path}/config.json failed: {err}. The config cannot be read, so the '
              'architecture cannot be judged at all — this is not an "unsupported '
              'architecture". What is missing: a parseable config.json. Re-download it '
              '(an interrupted download is the usual cause).'},
    'imp.v_hf_incomplete_title': {'zh': '这个文件夹缺权重文件', 'en': 'This folder has no weights'},
    'imp.v_hf_incomplete_detail': {
        'zh': '{path}:{err}。缺的是:.safetensors 权重分片(config.json 已经在)。'
              '补救:重新下载/续传权重文件,或把路径指向真正存放权重的目录。',
        'en': '{path}: {err}. What is missing: the .safetensors weight shards (config.json '
              'is already there). Re-download or resume the weights, or point the path at '
              'the directory that actually holds them.'},
    'imp.f_hf_noconfig_weights': {
        'zh': '这个文件夹有 config.json,但没有任何 .safetensors 权重。',
        'en': 'This folder has config.json but no .safetensors weights.'},
    'imp.err_cfg_no_weights': {
        'zh': '有 config.json,但没有 .safetensors 权重分片', 'en': 'no .safetensors weights'},
    'imp.cfg_missing': {'zh': '目录内没有 config.json', 'en': 'no config.json in the folder'},
    'imp.cfg_unparseable': {'zh': 'config.json 不是合法 JSON: {err}',
                            'en': 'config.json is not valid JSON: {err}'},
    'imp.cfg_not_object': {'zh': 'config.json 顶层不是对象(是 {kind})',
                           'en': 'config.json is not a JSON object (it is {kind})'},
    'imp.tip_redownload': {'zh': '重新下载该文件(常见于下载中断/未校验)',
                           'en': 're-download the file (interrupted downloads are the '
                                 'usual cause)'},
    'imp.tip_redownload_config': {'zh': '重新下载 config.json 后重试',
                                  'en': 're-download config.json and try again'},
    'imp.tip_adapt_all': {'zh': '接入规格流水线(生成缺口清单): python3 {tools} "{path}"',
                          'en': 'access-spec pipeline (produces the gap list): '
                                'python3 {tools} "{path}"'},
    'imp.tip_spec_exists': {'zh': '本仓库已有这个族的接入规格: {specs} —— 直接拿它跑上面的'
                                  '流水线,不用从零抽 spec。',
                            'en': 'this repo already holds an access spec for that family: '
                                  '{specs} — feed it to the pipeline above instead of '
                                  'extracting a spec from scratch.'},
    'imp.tip_gguf_unmapped': {'zh': '无名称映射的张量(会被转换器丢弃,是缺件的直接证据): '
                                     '{names}',
                              'en': 'tensors with no name mapping (a converter would drop '
                                    'them — direct evidence of the missing piece): {names}'},
    'imp.fact_layer_types': {'zh': '层型 = {mix};', 'en': 'layer types = {mix}; '},
    'imp.fact_rope': {'zh': 'rope = {rope};', 'en': 'rope = {rope}; '},
    'imp.fact_towers': {'zh': '多模态塔 = {towers};', 'en': 'extra towers = {towers}; '},
    'imp.fact_quant': {'zh': '量化 = {method};', 'en': 'quantisation = {method}; '},
    # verdict: LoRA
    'imp.v_lora_title': {'zh': '这是 LoRA 微调包,还需要一步融合',
                         'en': 'This is a LoRA adapter and needs one more merge step'},
    'imp.v_lora_detail': {
        'zh': 'LoRA 只含增量权重(rank {rank}),不能单独运行。需要: '
              '① 对应的基座模型(见 adapter_config 里的 base_model), '
              '② 用融合工具把它离线合进基座,生成一个新的 .ninfer。'
              '融合是逐层 解量化→加增量→重量化,需要一些时间。',
        'en': 'A LoRA holds only the delta weights (rank {rank}) and cannot run on its own. '
              'You need: (1) the matching base model (see base_model in adapter_config), '
              '(2) an offline merge of the adapter into that base, producing a new .ninfer. '
              'The merge runs layer by layer — dequantise, add the delta, requantise — and '
              'takes a while.'},
    'imp.tip_lora_tools': {'zh': '工具与流程: tools/convert/lora/lora_merge.py + _LORA.md',
                           'en': 'tools and workflow: tools/convert/lora/lora_merge.py + '
                                 '_LORA.md'},
    # verdict: unknown
    'imp.v_unknown_title': {'zh': '没能认出这个模型', 'en': 'Could not recognise this model'},
    'imp.v_unknown_detail': {
        'zh': '请确认给的是一个 .ninfer / .gguf 文件,或一个含 config.json 的模型文件夹。',
        'en': 'Point this at a .ninfer / .gguf file, or at a model folder containing '
              'config.json.'},
    # environment self-check
    'imp.env.gpu': {'zh': 'NVIDIA 显卡', 'en': 'NVIDIA GPU'},
    'imp.env.python': {'zh': 'Python', 'en': 'Python'},
    'imp.env.engine': {'zh': '推理引擎 ninfer-serve', 'en': 'inference engine ninfer-serve'},
    'imp.env.gpu_ok': {'zh': '{name}, 显存 {vram} GB, 驱动 {driver}',
                       'en': '{name}, {vram} GB VRAM, driver {driver}'},
    'imp.env.gpu_missing': {'zh': '没有检测到。本软件需要 NVIDIA 显卡 + 驱动。',
                            'en': 'not detected. This software needs an NVIDIA GPU plus its '
                                  'driver.'},
    'imp.env.gpu_fix': {'zh': '安装驱动: {url}', 'en': 'install the driver: {url}'},
    'imp.env.engine_missing': {'zh': '未找到(Windows 版打包后自动内置)',
                               'en': 'not found (the packaged Windows build bundles it)'},
    'imp.env.engine_fix': {'zh': '开发环境需先在 WSL 构建',
                           'en': 'in a dev checkout, build it inside WSL first'},
    # CLI self-test (python model_import.py <path>)
    'imp.cli.path': {'zh': '== {path}', 'en': '== {path}'},
    'imp.cli.format': {'zh': '  格式: {kind} | {friendly}',
                       'en': '  format: {kind} | {friendly}'},
    'imp.cli.family': {'zh': '  家族: {family}', 'en': '  family: {family}'},
    'imp.cli.size': {'zh': '  体积: {gb} GB | {quant}',
                     'en': '  size: {gb} GB | {quant}'},
    'imp.cli.issues': {'zh': '  注意: {issues}', 'en': '  notes: {issues}'},
    'imp.cli.arch': {'zh': '  架构: {note}', 'en': '  architecture: {note}'},
    'imp.cli.unmapped': {'zh': '  无名称映射的张量({n} 个): {names}',
                         'en': '  tensors with no name mapping ({n}): {names}'},
    'imp.cli.verdict': {'zh': '  结论[{level}]: {title}',
                        'en': '  verdict[{level}]: {title}'},
    'imp.cli.bullet': {'zh': '  - {text}', 'en': '  - {text}'},
    'imp.cli.desc': {
        'zh': '模型识别/导入裁决引擎 (被导入向导调用, 也可单独跑自检)',
        'en': 'model recognition / import-verdict engine (driven by the import wizard, '
              'also runnable standalone as a self-test)'},
    'imp.cli.paths_help': {
        'zh': '一个或多个模型文件/目录路径; 逐个给出格式、能否运行、下一步',
        'en': 'one or more model file/folder paths; each gets a format, a can-it-run '
              'verdict and a next step'},
    'imp.cli.gaps_help': {
        'zh': '只看某个模型的 archkit 缺口报告 (tools/archkit/out/<id>/manifest.json; '
              '也认旧名 arch_manifest.json —— 不带 gaps 列表的清单报"无法判定")。'
              '退出码就是结论: 0 可以通过 / 1 读不到清单 / 2 无法判定 / 3 被拦截',
        'en': 'print only the archkit gap report for one model '
              '(tools/archkit/out/<id>/manifest.json; the legacy name arch_manifest.json '
              'is accepted too -- a manifest with no gaps list reports UNKNOWN). '
              'The exit code IS the verdict: 0 clear / 1 unreadable / 2 unknown / '
              '3 blocked'},
    'imp.cli.require_servable_help': {
        'zh': '把退出码当门用: 只有"可以通过"才返回 0, 其余(含"无法判定")返回 3, '
              '并在 stderr 打一行原因。不加这个旗标时退出码区分四种结论',
        'en': 'treat the exit code as a gate: 0 only for CLEAR, everything else that was '
              'read (including UNKNOWN) exits 3, with a one-line reason on stderr. '
              'Without it the exit code names which of the four outcomes it was'},
    'imp.cli.gaps_gate_refused': {
        'zh': '门: 结论={verdict} (--require-servable) ⇒ 退出码 {rc}',
        'en': 'gate: verdict={verdict} (--require-servable) => exit {rc}'},

    # ------------------------- archkit gap report (tools/archkit/out/<id>) ----
    'imp.gaps.title': {'zh': '模型导入缺口报告 (archkit manifest)',
                       'en': 'Model import gap report (archkit manifest)'},
    'imp.gaps.usage': {
        'zh': '用法: python model_import.py --gaps <model-id> [--lang zh|en] '
              '[--require-servable] (退出码 0 通过 / 1 读不到 / 2 无法判定 / 3 被拦截)',
        'en': 'usage: python model_import.py --gaps <model-id> [--lang zh|en] '
              '[--require-servable] (exit 0 clear / 1 unreadable / 2 unknown / '
              '3 blocked)'},
    'imp.gaps.line_model': {'zh': '模型: {model}', 'en': 'model: {model}'},
    'imp.gaps.line_manifest': {'zh': '清单: {path}', 'en': 'manifest: {path}'},
    'imp.gaps.line_summary': {'zh': '缺口: {total} 项 (hook {hook} / new_op {new_op} / '
                                    'post {post})',
                              'en': 'gaps: {total} (hook {hook} / new_op {new_op} / '
                                    'post {post})'},
    'imp.gaps.no_manifest': {
        'zh': '没有找到 {id} 的 archkit manifest。先跑: '
              'python tools/archkit/adapt.py <模型目录>',
        'en': 'No archkit manifest for {id}. Run: '
              'python tools/archkit/adapt.py <model folder> first'},
    'imp.gaps.bad_manifest': {'zh': 'manifest 读取失败: {err}',
                              'en': 'cannot read the manifest: {err}'},
    # File-name neutral on purpose: gen_target.py writes `manifest.json` too since the
    # writer/reader name split was closed, so naming a file here would be a statement
    # about a writer that no longer exists. What is load-bearing is the SHAPE.
    'imp.gaps.not_a_gap_report': {
        'zh': '{path} 不是缺口清单: 里面没有 gaps 列表,所以无法据此判断能不能跑。'
              '这通常是一份"仅记录 spec 抽取成功"的清单(gen_target.py 产出),'
              '不是 adapt.py 的缺口清单。{hint}'
              '缺少的部分: 一份带 gaps 列表的清单(gaps: hook/new_op/post)。'
              '补救: python3 tools/archkit/adapt_all.py <模型目录> 生成真正的缺口清单。',
        'en': '{path} is not a gap report: it carries no `gaps` list, so nothing can be '
              'concluded about runnability. It is usually a manifest that only records a '
              'successful spec extraction (written by gen_target.py), not adapt.py\'s gap '
              'list. {hint}'
              'What is missing: a manifest carrying a gaps list (gaps: hook/new_op/post). '
              'Remedy: python3 tools/archkit/adapt_all.py <model folder>.'},
    'imp.gaps.no_gaps': {'zh': '清单里没有缺口(引擎可直接吃)。',
                         'en': 'The manifest lists no gaps (the engine can take it as-is).'},
    # verdict lines
    # "the header is withheld as config.h.BLOCKED" used to be here and was FALSE: no
    # writer in this tree emits that name, and a new_op gap does not withhold the
    # header (measured: out/ornith-1.5-9b-q4-k-m/ has a new_op gap AND a config.h).
    # The verdict says what was concluded about the model; the header file is named by
    # its own line, read off the directory.
    'imp.gaps.verdict_blocked': {
        'zh': '结论: 拦截 —— 不能进入转换 (清单里有 new_op 缺口: 须先有一次新内核落地)',
        'en': 'VERDICT: BLOCKED — not fit to convert (the manifest carries a new_op '
              'gap: one new kernel has to land first)'},
    'imp.gaps.verdict_ok': {
        'zh': '结论: 可以通过 —— 已生成 config.h, 没有 new_op 缺口',
        'en': 'VERDICT: CLEAR — config.h generated, no new_op gaps'},
    # A measured list whose entries are all `covered`. This is the arm the two
    # true-green controls pin; an open `hook` is engine work adapt.py has only
    # written as patch text, and it does NOT reach this sentence any more.
    'imp.gaps.verdict_hooks': {
        'zh': '结论: 未接线 —— 没有 new_op 缺口, 但有 {n} 个引擎钩子未接线, 接线并编译'
              '通过之前不能 serve: {hooks}',
        'en': 'VERDICT: HOOKS OPEN — no new_op gap, but {n} engine hook(s) are not '
              'wired; not servable until they are applied and the tree compiles: '
              '{hooks}'},
    # ---- UNMEASURED. A manifest with no `gaps` LIST is not a gap-free manifest. ----
    # The reader (model_import.manifest_gap_state) routes EVERY shape of the absence
    # here -- no key at all, an explicit null, a non-list, or a gaps list that the
    # writer itself denies with "gaps_measured": false. `"gaps": null` used to fall
    # through to CLEAR because the guard tested `"gaps" not in data`; these four
    # entries are what makes "未测量" sayable out loud instead of silently green.
    'imp.gaps.verdict_unmeasured': {
        'zh': '结论: 无法判定 —— 这份清单里没有 gaps 列表, 缺口未测量。'
              '"未测量"不等于"没有缺口", 所以不能据此说引擎能吃。',
        'en': 'VERDICT: UNKNOWN — this manifest carries no gaps list, so the gaps were '
              'never measured. NOT MEASURED is not the same as NO GAPS, so this says '
              'nothing about whether the engine can take the model.'},
    'imp.gaps.header_unmeasured': {
        'zh': '头文件: 未知 (清单没测缺口, 所以无法判断生成的 config.h 能不能用)',
        'en': 'header file: unknown (with no measured gaps there is no way to tell '
              'whether the generated config.h is usable)'},
    'imp.gaps.gaps_unmeasured': {
        'zh': '清单 {path} 里没有 gaps 列表({shape}), 所以缺口是"未测量"而不是"没有"。{hint}{claims}',
        'en': 'The manifest {path} carries no gaps list ({shape}), so the gaps are '
              'UNMEASURED rather than absent. {hint}{claims}'},
    'imp.gaps.gaps_unmeasured_how': {
        'zh': '要看缺口, 先跑能报缺口的生成器: python3 tools/archkit/adapt_all.py <模型目录> '
              '(它写的清单带 gaps 列表); gen_target.py 只做确定性翻译, 没有算子目录。',
        'en': 'for gaps, run the generator that reports them: '
              'python3 tools/archkit/adapt_all.py <model folder> (its manifest carries the '
              'gaps list); gen_target.py only does the deterministic translation and has '
              'no operator catalogue.'},
    'imp.gaps.shape_missing': {
        'zh': '清单里根本没有 gaps 字段',
        'en': 'the manifest has no `gaps` field at all'},
    'imp.gaps.shape_null': {
        'zh': 'gaps 字段存在但值是 null (显式空值也是"没测过", 不是"没有缺口")',
        'en': 'the `gaps` field exists but its value is null (an explicit null is still '
              'unmeasured, not "no gaps")'},
    'imp.gaps.shape_not_a_list': {
        'zh': 'gaps 字段存在但不是列表 (是 {kind})',
        'en': 'the `gaps` field exists but is not a list (it is a {kind})'},
    'imp.gaps.shape_flag_false': {
        'zh': '清单写着 "gaps_measured": false (写者自己的声明为准, 所以这份清单的缺口'
              '仍算未测量; gaps 字段的实际内容: {kind})',
        'en': 'the manifest says "gaps_measured": false (the writer\'s own declaration '
              'wins, so the gaps still count as unmeasured; the `gaps` field actually '
              'holds: {kind})'},
    # ---- the two shapes that used to be handled by crashing ----
    # `measured` is not a key anything in this tree reads; the flag is `gaps_measured`.
    # A manifest carrying the wrong spelling is not guessed at: an unhonoured flag looks
    # like a check is running, and `{"gaps": [], "measured": false}` (a writer saying it
    # did NOT measure) used to be rendered as "the engine can take it as-is".
    'imp.gaps.shape_alias_key': {
        'zh': '清单带着一个没人读的旗标键 `measured` —— 本树唯一有效的旗标键是 '
              '"gaps_measured", 所以这份清单的缺口算"未测量"。把它改名成 '
              '"gaps_measured" (或删掉) 再来报缺口。',
        'en': 'the manifest carries a `measured` key, which NOTHING in this tree reads '
              '(the only honoured flag key is "gaps_measured"), so its gaps count as '
              'unmeasured. Rename it to "gaps_measured" (or drop it) to have the gaps '
              'count.'},
    # A `gaps` list whose element is not an object used to raise AttributeError out of
    # import_gap_report() -> a traceback and rc 1, i.e. the same code as "no manifest".
    'imp.gaps.shape_bad_element': {
        'zh': 'gaps 是列表, 但第 {index} 个元素不是对象 (是 {kind}) —— 读不出它的 '
              'need/tier/action, 所以这份清单不能算"测过缺口"。',
        'en': 'the `gaps` field is a list, but element {index} is not an object (it is a '
              '{kind}), so its need/tier/action cannot be read -- this is not a measured '
              'gap list.'},
    # A row decides whether the manifest blocks THROUGH ITS TIER, and the vocabulary is
    # what adapt.py writes (hook / new_op / covered / post). A row whose tier is absent or
    # spelled differently does not block, so without this guard such a list rendered
    # "no new_op gaps / CLEAR" while its own body listed a `new_op:` row -- a report that
    # contradicted itself and exited 0, `--require-servable` included.
    'imp.gaps.shape_bad_tier': {
        'zh': 'gaps 是列表, 但第 {index} 个元素的 tier 不是本树的层型词汇 (读到 "{tier}"; '
              'adapt.py 只会写 hook / new_op / covered / post)。层型读不出就等于"不算'
              '拦截", 所以这份清单的结论只能是"未测量"。',
        'en': 'the `gaps` field is a list, but element {index} carries tier "{tier}", '
              'which is not one of the tiers this tree writes (adapt.py writes hook / '
              'new_op / covered / post). A row whose tier cannot be read cannot block, '
              'so this list can only be reported as UNMEASURED, never as CLEAR.'},
    'imp.gaps.blocking': {'zh': '拦截项 (new_op): {items}',
                          'en': 'blocking gaps (new_op): {items}'},
    'imp.gaps.header_ok': {
        'zh': '头文件: 清单目录里有 {name}',
        'en': 'header file: {name} is present next to the manifest'},
    # No header in the directory. adapt.py's refusal path leaves none behind (it writes
    # the reason into the manifest instead), and a file left over from an earlier run is
    # the trap here: it is indistinguishable from a fresh one by listing the folder.
    'imp.gaps.header_missing': {
        'zh': '头文件: 清单目录里没有 config.h —— 这一次没有生成头文件 (adapt.py 拒写时'
              '会把原因写进 manifest)。若该目录里还留着上一次运行的 config.h, 那份头文件'
              '的层型表属于另一次输入, 不要用它',
        'en': 'header file: no config.h next to the manifest -- no header was produced '
              'for this run (adapt.py records the reason in the manifest when it '
              'refuses). If a config.h from an EARLIER run is still there, its '
              'layer-kind table belongs to that run; do not consume it'},
    'imp.gaps.next_step_head': {
        'zh': '下一步 (在转换期落地, 不需要改引擎内核):',
        'en': 'Next step (solved at conversion time; no engine kernel change needed):'},
    'imp.gaps.tie_step': {
        'zh': '物化 lm_head = embed^T: 转换时把转置嵌入矩阵写成独立 lm_head 对象'
              '(约 {mb} MB 额外权重)',
        'en': 'materialise lm_head = embed^T — at conversion time write the transposed '
              'embedding out as a separate lm_head object (~{mb} MB of extra weights)'},
    'imp.gaps.blocking_engine': {
        'zh': '引擎侧必须先把这个算子做出来: {need} (转换期无解, 只能改引擎内核)',
        'en': 'the engine side has to implement this operator first: {need} '
              '(there is no conversion-time workaround; the kernel must change)'},
    'imp.gaps.tie_note': {
        'zh': '注意: 转换器当前硬写 tie_word_embeddings: False (tie=False), '
              '所以必须先改转换侧; 否则产物缺 head, 跑起来直接错。',
        'en': 'Note: the converter currently hardcodes tie_word_embeddings: False, so the '
              'conversion side must change first — otherwise the artifact has no head and '
              'fails at load.'},
    'imp.gaps.tier_head': {'zh': '[{tier}] {n} 项', 'en': '[{tier}] {n} items'},
    'imp.gaps.row': {'zh': '  - {need}', 'en': '  - {need}'},
    'imp.gaps.row_action': {'zh': '      {action}', 'en': '      {action}'},
    'imp.gaps.row_raw': {'zh': '      (清单原文: {raw})', 'en': '      (manifest text: {raw})'},
    'imp.gaps.fallback': {
        'zh': '【译表未收录此缺口类型; {tier} 的原样处置】{action}',
        'en': '[gap kind not in the translation table; tier {tier}, original action] '
              '{action}'},
    # tier words
    'imp.tier.hook': {'zh': '引擎钩子 (constexpr 开关)',
                      'en': 'engine hook (constexpr wiring)'},
    'imp.tier.new_op': {'zh': '新算子 (需一次内核实现)',
                        'en': 'new operator (one kernel to implement)'},
    'imp.tier.post': {'zh': '转换后校验', 'en': 'post-conversion check'},
    'imp.tier.other': {'zh': '其它等级', 'en': 'other tier'},
  'imp.tier.covered': {'zh': '已覆盖（无需引擎改动）',
                        'en': 'covered (no engine change needed)'},
  'imp.gap.head_tied_true_covered': {
      'zh': 'tied head：转换期把嵌入键物化为独立 head 对象（约 {mb} MB），引擎无需新算子；'
            '转换器已实现（tools/convert/common/source_map.py）',
      'en': 'tied head: materialise the embedding as a separate head object at conversion '
            'time (~{mb} MB); no new engine operator is needed and the converter now '
            'implements this (tools/convert/common/source_map.py)'},
  'imp.gap.head_tied_true_covered.nosize': {
      'zh': 'tied head：转换期把嵌入键物化为独立 head 对象（额外权重，体积见 manifest），'
            '引擎无需新算子；转换器已实现（tools/convert/common/source_map.py）',
      'en': 'tied head: materialise the embedding as a separate head object at conversion '
            'time (extra weights; see the manifest for the size); no new engine operator '
            'is needed and the converter now implements this '
            '(tools/convert/common/source_map.py)'},
    # per-need action renderings (keyed by the `need` prefix emitted by adapt.py)
    'imp.gap.attention_gqa_full': {
        'zh': '全量 GQA 注意力 (v3 生成器覆盖)',
        'en': 'full GQA attention (covered by the v3 generator)'},
    'imp.gap.attention_gdn': {
        'zh': '线性注意力 (gdn) 叶子 (v3 生成; 27b 引擎主路径已在用)',
        'en': 'linear-attention (GDN) leaf (v3 generated; the 27b main path already uses it)'},
    'imp.gap.attention_swa': {
        'zh': '滑动窗口注意力: 引擎侧窗口掩码 constexpr + 生成叶子 (窗口 = {window} token)',
        'en': 'sliding-window attention: engine-side window-mask constexpr plus a generated '
              'leaf (window = {window} tokens)'},
    'imp.gap.attn_headwise_gate': {
        'zh': '逐头注意力输出门控 ({mode}): linear + sigmoid_gate_mul 都已存在, '
              '需要在叶子里接线',
        'en': 'headwise attention output gate ({mode}): the linear + sigmoid_gate_mul '
              'pieces already exist; the leaf needs wiring'},
    'imp.gap.attn_qk_norm': {'zh': 'qk_norm', 'en': 'qk_norm'},
    # The three detector shapes that had no row (see model_import._GAP_ACTION_ROWS).
    'imp.gap.attn_qk_norm_absent': {
        'zh': '该家族**无条件**对 q/k 做 rmsnorm, 而这个模型没有 qk_norm 权重 => '
              '需要引擎侧 qk_norm_enabled() 门 (默认关), 由 config.h 打开。',
        'en': 'this family rmsnorms q/k UNCONDITIONALLY, and this model carries no '
              'qk_norm weights => the engine needs a qk_norm_enabled() gate (default '
              'off) that config.h turns on.'},
    'imp.gap.attn_head_geometry': {
        'zh': '头几何 {q}q/{kv}kv@{hd}: 引擎的 gqa 分派表已注册 => 无需新算子; 未注册则'
              '按 q_heads 反推 KV 头, 会落进别的实例 (属 new_op)。',
        'en': 'head geometry {q}q/{kv}kv@{hd}: covered when the engine\'s gqa dispatch '
              'table registers it; otherwise KV heads are inferred from q_heads and it '
              'lands in a different instance (that is a new_op).'},
    'imp.gap.mlp_act_gated': {
        'zh': 'gated MLP 激活 {act}({mode}): 对应的 <act>_mul 共享算子已存在并登记进构建 '
              '(ops/wrapper/), 在叶子里接线即可。',
        'en': 'gated MLP activation {act}({mode}): the matching <act>_mul shared operator '
              'already exists and is in the build (ops/wrapper/); the leaf just has to '
              'call it.'},
    'imp.gap.mlp_act_gated_new_op': {
        'zh': 'gated MLP 激活 {act}({mode}): 引擎树里没有对应的 <act>_mul 算子, 需要一次内核实现。',
        'en': 'gated MLP activation {act}({mode}): the engine has no <act>_mul operator '
              'for it, so this needs one kernel to land.'},
    'imp.gap.attn_hybrid_hd': {
        'zh': '异构注意力头几何: global head_dim={ghd} (kv={gkv}) vs local head_dim={lhd}',
        'en': 'heterogeneous attention head geometry: global head_dim={ghd} (kv={gkv}) vs '
              'local head_dim={lhd}'},
    'imp.gap.attn_scale': {'zh': 'qk scale 由叶子透传 ({scale})',
                           'en': 'pass the qk scale through the leaf ({scale})'},
    'imp.gap.mlp_act': {
        'zh': 'MLP 激活 {act}: ops::{act} 已存在, 前向路径需要按激活选择 (不是新内核)',
        'en': 'MLP activation {act}: ops::{act} already exists, the MLP forward path just '
              'has to select it (no new kernel)'},
    'imp.gap.mlp_act_new_op': {
        'zh': 'MLP 激活 {act} 不在现有算子集里, 需要新激活内核',
        'en': 'MLP activation {act} is not in the op set yet; it needs a new activation '
              'kernel'},
    'imp.gap.head_tied_true': {
        'zh': 'tied head: 转换期物化 lm_head = embed^T (约 {mb} MB) 或让引擎 flavor 复用嵌入; '
              '引擎现在按独立 lm_head 对象加载',
        'en': 'tied head: materialise lm_head = embed^T at conversion time (~{mb} MB) or have '
              'an engine flavor reuse the embedding; the engine loads lm_head as a separate '
              'object today'},
    # same gap, but adapt.py did not quote a size next to it
    'imp.gap.head_tied_true.nosize': {
        'zh': 'tied head: 转换期物化 lm_head = embed^T (额外权重, 体积见 manifest) 或让引擎 '
              'flavor 复用嵌入; 引擎现在按独立 lm_head 对象加载',
        'en': 'tied head: materialise lm_head = embed^T at conversion time (extra weights; '
              'the manifest quotes the size) or have an engine flavor reuse the embedding; '
              'the engine loads lm_head as a separate object today'},
    'imp.gap.head_tied_false': {'zh': '独立 lm_head 对象 (已支持)',
                                'en': 'separate lm_head object (already supported)'},
    'imp.gap.head_logit_softcap': {'zh': 'logits 后处理: tanh 软帽 {cap}',
                                   'en': 'logits post-processing: tanh softcap {cap}'},
    'imp.gap.token_domain': {
        'zh': 'FrontendOptions.token_domain + official_specials 覆盖 '
              '(词表 {vocab} ≠ 家族 {family})',
        'en': 'FrontendOptions.token_domain plus official_specials override (vocab {vocab} '
              'does not match the family domain {family})'},
    'imp.gap.layers_over_cap': {
        'zh': 'per-layer 数组容量要按家族上限扩展: {layers} 层 > 家族上限 {cap}',
        'en': 'per-layer arrays and layouts must grow to the family cap: {layers} layers > '
              'family cap {cap}'},
    'imp.gap.layers_audit': {
        'zh': 'per-layer 数组容量审计 (cold_slots 一类): {layers} 层 > {cap} 槽; '
              '家族上限已到 64, 新家族仍须核对',
        'en': 'audit per-layer array capacity (cold_slots and friends): {layers} layers > '
              '{cap} slots; the family cap is already 64, a new family still needs the audit'},
    'imp.gap.layer_scale': {'zh': '逐层输出乘系数 {mult} (run_layers 尾部)',
                            'en': 'per-layer output multiplier {mult} (tail of run_layers)'},
    'imp.gap.rope_per_layer_theta': {
        'zh': '逐层 theta 表 (n={n}, 其中 {zero} 层是 0/NoPE)',
        'en': 'per-layer theta table (n={n}, {zero} of them 0/NoPE)'},
    'imp.gap.rope_per_type_theta': {
        'zh': '按层类型的 theta 表, 在 rope 叶子里按层型选表 ({table})',
        'en': 'per-layer-type theta table, picked by layer kind inside the rope leaf '
              '({table})'},
    'imp.gap.rope_partial_kind': {
        'zh': '按层类型的部分旋转: rotary_dim = head_dim × 系数 ({table}); '
              'ops::rope 已收 rotary_dim, 属接线',
        'en': 'per-layer-type partial rotary: rotary_dim = head_dim x factor ({table}); '
              'ops::rope already takes rotary_dim, so this is wiring'},
    'imp.gap.rope_partial_scalar': {
        'zh': '部分旋转: rotary_dim = head_dim × {factor} (ops::rope 已收 rotary_dim)',
        'en': 'partial rotary: rotary_dim = head_dim x {factor} (ops::rope already takes '
              'rotary_dim)'},
    'imp.gap.layer_kinds': {
        'zh': '未归类层型 ({kinds}): 短卷积可复用 causal_conv1d_silu 叶子, '
              'Mamba/SSD 类需新算子',
        'en': 'unclassified layer kinds ({kinds}): short convolutions can reuse the '
              'causal_conv1d_silu leaf, Mamba/SSD kinds need new operators'},
    'imp.gap.state_space': {
        'zh': '检出 SSM/Mamba 配置键 ({keys}) 却没有 layer_types: 层混合未建模, '
              '需按 hybrid pattern 展开',
        'en': 'SSM/Mamba config keys ({keys}) with no layer_types: the layer mix is '
              'unmodelled and must be expanded from the hybrid pattern'},
    'imp.gap.moe': {
        'zh': 'MoE 路由专家 ({experts} 个专家, top-{top}): 需要专家内核 + 路由 + '
              '专家并行/卸载',
        'en': 'MoE routed experts ({experts} experts, top-{top}): needs the expert kernel '
              'plus routing and expert parallel/offload'},
    'imp.gap.vision': {'zh': '视觉塔 (只跑文本可以先跳过)',
                       'en': 'vision tower (a text-only run can skip it for now)'},
    'imp.gap.quant_geometry': {
        'zh': '转换后校验: fp8/nvfp4 权重形状对照几何注册表 (A16 起步)',
        'en': 'post-conversion check: verify fp8/nvfp4 weight shapes against the geometry '
              'registry (starting at A16)'},

    # -------------------------------------------------------- gui_tips.py TIPS
    'tips.steps': {
        'zh': "总共训练多少步。可以把训练想成'让模型做练习题'——步数就是做题数量。\n"
              "做太少学不会(几万步量级才算入门);做太多会'背题'反而变笨(过拟合)。\n"
              "一般先跑小步数(如 500)看损失是否下降,再决定加不加。",
        'en': "How many training steps in total. Think of training as letting the model do "
              "practice problems — a step is one problem. Too few and it never learns "
              "(tens of thousands of steps is where it starts to count); too many and it "
              "memorises the answers and gets worse (overfitting).\n"
              "Run a small number first (say 500), watch whether the loss comes down, then "
              "decide whether to raise it."},
    'tips.batch': {
        'zh': "每步同时'做题'的句子数量。调大 = 每次看得更多、学得更稳,但更吃显存。\n"
              "显存不够就调小;显存够就调大,训练会更快收敛。",
        'en': "How many sentences the model works through at once per step. Bigger = it sees "
              "more each time and learns more steadily, but it eats more VRAM.\n"
              "Short on VRAM? Turn it down. VRAM to spare? Turn it up; training converges "
              "faster."},
    'tips.anchors': {
        'zh': "每句话里抽出多少个'接龙起点'。模型要学的就是'看到上文,接出下文'。\n"
              "越多 = 每个句子被利用得越充分,但每步计算量越大。",
        'en': "How many 'continuation starting points' get pulled out of each sentence. "
              "What the model learns is exactly 'see the prefix, produce the continuation'.\n"
              "More = each sentence is used more fully, but every step costs more compute."},
    'tips.ctx': {
        'zh': "模型'回忆'上文的最大长度(按 token 计,1 个汉字约 1~2 个 token)。\n"
              "越长越懂上下文,但显存和计算量越大,训练越慢。",
        'en': "The longest stretch of context the model can 'remember' (counted in tokens; "
              "one Chinese character is about 1-2 tokens).\n"
              "Longer means it follows the context better, but VRAM use, compute and "
              "training time all go up."},
    'tips.lr': {
        'zh': "学习率 = 每次调整权重的'步子大小'。步子太大容易原地乱跳(损失震荡);\n"
              "太小则学得慢。常见范围 1e-4 ~ 1e-3。不确定就保持默认。",
        'en': "Learning rate = how big each step is when adjusting the weights. Too big and "
              "it jumps around without settling (the loss oscillates); too small and it "
              "learns slowly.\n"
              "The usual range is 1e-4 to 1e-3. Not sure? Keep the default."},
    'tips.resume': {
        'zh': "断点续训:填一个之前保存的 step_XXXX.pt 文件路径,就从那里接着练,"
              "而不是从头开始。留空 = 从头训练。",
        'en': "Resume training: give the path of a previously saved step_XXXX.pt file and "
              "training continues from there instead of starting over. Empty = train from "
              "scratch."},
    'tips.ddtree': {
        'zh': "树状思维训练开关:让草稿模型不只学'最可能的下一句',还学'第二、第三可能'。\n"
              "开了之后,推理时'猜多个分支'的成功率会更高(投机解码提速)。",
        'en': "Tree-of-thought training switch: the draft model learns not only the most "
              "likely next token but also the second and third options.\n"
              "With it on, guessing several branches during inference succeeds more often, "
              "which speeds up speculative decoding."},
    'tips.c_rag': {
        'zh': "从百科/维基类知识库采集的'知识问答'条数。给模型喂常识。",
        'en': "How many knowledge Q&A items to collect from encyclopaedia/Wikipedia-style "
              "stores. Feeds the model general knowledge."},
    'tips.c_qa': {
        'zh': "从通用问答语料采集的条数。给模型喂'怎么好好回答'。",
        'en': "How many items to collect from general Q&A corpora. Teaches the model how to "
              "answer well."},
    'tips.c_code': {
        'zh': "从代码语料采集的条数。让模型懂代码(换行缩进括号)。",
        'en': "How many items to collect from code corpora. Teaches the model code (line "
              "breaks, indentation, brackets)."},
    'tips.c_out': {
        'zh': "采集结果存放的目录名。留默认即可,训练脚本会自动去读它。",
        'en': "Folder name where the collected data lands. Leave it at the default; the "
              "training script reads it automatically."},
    'tips.smodel': {
        'zh': "要启动推理的模型文件(.ninfer 格式)。列表来自模型目录扫描;\n"
              "刚转换好的模型点'刷新'就会出现。",
        'en': "The model file to serve (.ninfer format). The list comes from scanning the "
              "model folder;\n"
              "a freshly converted model appears after you hit 'refresh'."},
    'tips.sctx': {
        'zh': "对话上下文窗口长度:模型能'记住'的最近内容量。\n"
              "越长越能聊长篇,但显存占用直线上升。显存不够就调小(2048~4096)。",
        'en': "Chat context window: how much recent content the model can 'remember'.\n"
              "Longer means longer conversations, but VRAM use climbs steeply. Short on "
              "VRAM? Turn it down (2048-4096)."},
    'tips.sspec': {
        'zh': "投机解码方案 = '让一个又快又小的小助手先猜答案,大模型只负责检查'。\n"
              "none=不猜(慢但省事);mtp/dflash/dflash2=不同的小助手。\n"
              "模型文件自带哪个就用哪个(选错会报错,换 none 即可)。",
        'en': "Speculative decoding scheme = 'a small fast helper guesses the answer first, "
              "the big model only checks it'.\n"
              "none = no guessing (slower but simpler); mtp/dflash/dflash2 = different "
              "helpers.\n"
              "Use whichever the model file ships with; picking the wrong one errors out — "
              "switch back to none."},
    'tips.sdt': {
        'zh': "小助手每次提前猜几个 token(字)。3~8 之间常见。\n"
              "猜得越多越可能错,大模型检查后要重来,反而变慢——所以不是越大越好。",
        'en': "How many tokens the helper guesses ahead each time. 3-8 is the usual range.\n"
              "Guessing more makes wrong guesses likelier, and the big model has to redo the "
              "work — so more is not better."},
    'tips.slabd': {
        'zh': "查表加速(LABD/ngram 链):'翻历史笔记'式的草稿——\n"
              "对话里经常整句复述或改写前文(代码补全、复述、RAG 问答),\n"
              "直接从历史里找相似句子当草稿,命中率极高。\n"
              "引擎下一版接入; 届时与 ngram-SSD 外挂一起出现。",
        'en': "Table-lookup acceleration (LABD/ngram chain): a 'flip back through your "
              "notes' draft —\n"
              "conversations often restate or lightly rewrite earlier text (code completion, "
              "repeats, RAG questions),\n"
              "so taking a similar sentence from history as the draft hits very often.\n"
              "The engine picks this up in the next version, together with the ngram-SSD "
              "add-on."},
    'tips.slabd2': {
        'zh': "同上:查表加速。规划中,引擎接入后此选项自动可用。",
        'en': "Same as above: table-lookup acceleration. Planned; the option becomes "
              "available once the engine wires it in."},
    'tips.scold': {
        'zh': "冷存储 = '把暂时用不到的东西挪到硬盘(SSD)'。\n"
              "显存不够时,让引擎把旧对话/冷数据放 SSD,要用时再取回。\n"
              "off=全放显存(最快,最吃显存); disk=放 SSD(省显存,慢一点);\n"
              "window=只保留最近一段。\n"
              "ngram 查表与 MoE 专家的 SSD 外挂将来都走这条链。",
        'en': "Cold storage = 'move what you do not need right now onto the SSD'.\n"
              "When VRAM is tight, the engine parks old conversation/cold data on the SSD "
              "and fetches it back on demand.\n"
              "off = everything in VRAM (fastest, hungriest); disk = on the SSD (saves "
              "VRAM, a little slower);\n"
              "window = keep only the most recent stretch.\n"
              "The ngram lookup and the MoE expert SSD offload will both use this chain "
              "later."},
    'tips.scold2': {
        'zh': "同上:冷存储策略。",
        'en': "Same as above: cold-storage policy."},
    'tips.scoldpath': {
        'zh': "SSD 缓存目录。留空用引擎默认(通常在系统临时目录)。\n"
              "建议填一个空间大的盘,例如 D:\\ninfer-cold。",
        'en': "SSD cache folder. Empty = the engine default (usually the system temp "
              "folder).\n"
              "Better to point it at a roomy drive, e.g. D:\\ninfer-cold."},
    'tips.scoldgb': {
        'zh': "SSD 缓存上限(GB)。0 = 自动。填小一点可防止缓存把盘塞满。",
        'en': "SSD cache cap in GB. 0 = automatic. A smaller number keeps the cache from "
              "filling the disk."},
    'tips.scoldgb2': {
        'zh': "同上:SSD 缓存上限(GB)。",
        'en': "Same as above: SSD cache cap in GB."},
    'tips.sktiers': {
        'zh': "KV 缓存三层各自的量化精度, 逗号分隔, 组合自由:\n"
              "  热=正在生成的位置 (必须高精度, 只许 bf16/fp16/int8)\n"
              "  尾=最近一段 (精度尾, 不能低于热层)\n"
              "  冷=更老的内容 (可压到 int4/iso4/iso4e/rk4v4 省显存)\n"
              "例: hot=bf16,cold=iso4e (tail= 引擎按名拒绝: 无尾层)。留空 = 引擎默认。\n"
              "契约已定, 引擎接入此参数后自动生效。",
        'en': "Quantisation precision for each of the three KV-cache tiers, comma "
              "separated, any combination:\n"
              "  hot = the positions being generated (must stay precise: only "
              "bf16/fp16/int8)\n"
              "  tail = the most recent stretch (precision at most one step below hot)\n"
              "  cold = older content (can be squeezed to int4/iso4/iso4e/rk4v4 to save VRAM)\n"
              "Example: hot=bf16,cold=iso4e (tail= is refused by name: no tail tier). Empty = engine default.\n"
              "The contract is settled; it takes effect once the engine wires this "
              "parameter in."},
    'tips.snvfp4mode': {
        'zh': "NVFP4 权重模式:\n"
              "  fusion = 融合版全部特性 (Rk4v4/冷池/尾窗/iso 全格式可用)\n"
              "  pure   = 纯 NVFP4 基线 (只许经典格式), 用来对照'融合到底"
              "带来了多少'",
        'en': "NVFP4 weight mode:\n"
              "  fusion = every feature of the fused build (Rk4v4 / cold pool / tail window / "
              "all iso formats available)\n"
              "  pure   = plain NVFP4 baseline (classic formats only), the reference point "
              "for measuring how much the fusion actually bought you"},
    'tips.import_path': {
        'zh': "模型所在的文件夹或文件。支持三种:\n"
              " 1. .ninfer 文件(本软件自家格式,可直接跑)\n"
              " 2. GGUF 文件(很多网站在传的格式,会自动转换)\n"
              " 3. 含 config.json 的模型文件夹(HuggingFace 格式)\n"
              "在网页里没法弹文件选择框,请复制完整路径粘贴进来,\n"
              "例如 C:\\models\\qwen3.8-27b.gguf",
        'en': "The folder or file the model lives in. Three kinds are supported:\n"
              " 1. a .ninfer file (this software's own format, runs directly)\n"
              " 2. a GGUF file (the format many sites hand out; converted automatically)\n"
              " 3. a model folder with config.json (HuggingFace layout)\n"
              "A web page cannot open a file picker, so copy the full path and paste it "
              "in,\n"
              "for example C:\\models\\qwen3.8-27b.gguf"},
    'tips.import_run_ctx': {
        'zh': "运行时的上下文长度。显存紧张就填 2048~4096,宽裕再往上加。",
        'en': "Runtime context length. 2048-4096 when VRAM is tight, higher when you have "
              "room."},
    'tips.wizard_note': {
        'zh': "本软件 ≠ 通用的'什么模型都能跑'。\n"
              "每个模型架构都要专门的推理引擎,目前支持 Qwen3 家族 27B/64 层规格。\n"
              "其它架构(如 DeepSeek、Kimi)的适配框架正在开发中,请关注后续版本。",
        'en': "This software is not a universal 'runs any model' tool.\n"
              "Every architecture needs its own inference engine; today the supported spec "
              "is the Qwen3 family at 27B / 64 layers.\n"
              "The adaptation framework for other architectures (DeepSeek, Kimi, ...) is "
              "under development — watch for a later release."},

    # ------------------------------------------------------ gui_tips.py GROUPS
    'tipg.train': {'zh': '微调训练参数 —— 不懂就保持默认',
                   'en': 'Fine-tuning parameters — keep the defaults if unsure'},
    'tipg.collect': {'zh': '语料采集参数 —— 数量越多训练资料越足',
                     'en': 'Data-collection parameters — more items means more training '
                           'material'},
    'tipg.serve': {'zh': '推理服务参数 —— 不懂就保持默认',
                   'en': 'Serving parameters — keep the defaults if unsure'},
    'tipg.import': {'zh': '模型导入 —— 全程有提示,照着做即可',
                    'en': 'Model import — the wizard walks you through it, just follow along'},

    # ---------------------------------------------------- gui_tips.py GLOSSARY
    'tipgl.vram.term': {'zh': '显存(VRAM)', 'en': 'VRAM'},
    'tipgl.vram.text': {
        'zh': "显卡自带的内存,决定模型能不能装下、能聊多长的天。\n"
              "就像手机内存:App 太大、同时开太多就会卡/闪退。",
        'en': "The memory that comes with the graphics card; it decides whether the model "
              "fits and how long a conversation can get.\n"
              "Like phone RAM: an app that is too big, or too many at once, and things "
              "stutter or crash."},
    'tipgl.token.term': {'zh': 'token', 'en': 'token'},
    'tipgl.token.text': {'zh': '模型读文本的最小单位,大致等于半个到一个汉字。',
                         'en': 'The smallest unit of text the model reads; roughly half to '
                               'one Chinese character.'},
    'tipgl.quant.term': {'zh': '量化', 'en': 'quantisation'},
    'tipgl.quant.text': {
        'zh': "把模型权重'压缩'的技术,体积和显存占用变小,精度略降。\n"
              "就像把高清照片压成 JPEG:小很多,肉眼看几乎没差。",
        'en': "The technique of 'compressing' model weights: smaller files and less VRAM, "
              "slightly lower precision.\n"
              "Like saving a photo as JPEG — much smaller, and you can barely tell."},
    'tipgl.nvfp4.term': {'zh': 'NVFP4', 'en': 'NVFP4'},
    'tipgl.nvfp4.text': {'zh': '本软件采用的一种量化格式,27B 模型压到约 14GB。',
                         'en': 'A quantisation format this software uses; it squeezes a 27B '
                               'model down to about 14 GB.'},
    'tipgl.context.term': {'zh': '上下文', 'en': 'context'},
    'tipgl.context.text': {'zh': "模型能看到的最近对话内容,超出部分会被'忘记'。",
                           'en': "The recent conversation the model can see; anything older "
                                 "gets 'forgotten'."},
    'tipgl.spec_decode.term': {'zh': '投机解码', 'en': 'speculative decoding'},
    'tipgl.spec_decode.text': {
        'zh': "让一个小助手先快速猜一串答案,大模型一次检查一串——\n"
              "猜对了就白赚速度,猜错了只损失一点点时间。",
        'en': "A small helper guesses a run of tokens quickly and the big model checks the "
              "whole run at once —\n"
              "right guesses buy speed for free, wrong ones cost only a little time."},
    'tipgl.gguf.term': {'zh': 'GGUF', 'en': 'GGUF'},
    'tipgl.gguf.text': {'zh': 'llama.cpp 生态通用的模型打包格式,网上下载的模型大多是它。',
                        'en': 'The common model packaging format of the llama.cpp '
                              'ecosystem; most models you download use it.'},
    'tipgl.safetensors.term': {'zh': 'safetensors', 'en': 'safetensors'},
    'tipgl.safetensors.text': {'zh': 'HuggingFace 生态的模型格式,通常是一整个文件夹。',
                               'en': 'The model format of the HuggingFace ecosystem; '
                                     'usually a whole folder.'},
}
