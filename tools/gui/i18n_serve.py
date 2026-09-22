# -*- coding: utf-8 -*-
"""i18n_serve.py — serve_gui.py 的界面词条 (zh/en).

gui_i18n.py 会自动合并本文件; 键统一用 'serve.' 前缀, 不与其它 GUI 冲突.
标点/术语约定: 用户看到的是"启动一个推理服务", 不是"实例化运行时".
"""
STRINGS = {
    # ---- 页头 / 通用 ----
    'serve.title':        {'zh': 'NInfer Serve 控制台', 'en': 'NInfer Serve console'},
    'serve.subtitle':     {'zh': 'RTX 5090 · ninfer-serve 控制台 — 滑块简约版',
                           'en': 'RTX 5090 · ninfer-serve console — slider-minimal edition'},
    'serve.badge_idle':   {'zh': '未启动', 'en': 'Stopped'},
    'serve.badge_run':    {'zh': '运行中', 'en': 'Running'},
    'serve.language':     {'zh': '语言', 'en': 'Language'},
    'serve.lang_zh':      {'zh': '中文', 'en': '中文'},
    'serve.lang_en':      {'zh': 'English', 'en': 'English'},
    'serve.console_ready': {'zh': 'NInfer Serve 界面: http://127.0.0.1:{port}  (引擎: {bin})',
                            'en': 'NInfer Serve GUI: http://127.0.0.1:{port}  (serve: {bin})'},

    # ---- 滑块面板 (新手模式) ----
    'serve.lbl_model':    {'zh': '模型', 'en': 'Model'},
    'serve.lbl_kv':       {'zh': 'KV 精度档', 'en': 'KV precision'},
    'serve.lbl_ctx':      {'zh': '上下文', 'en': 'Context'},
    'serve.lbl_spec':     {'zh': '投机档位', 'en': 'Speculation'},
    'serve.lbl_draft':    {'zh': '草稿数', 'en': 'Draft tokens'},
    'serve.lbl_conc':     {'zh': '并发', 'en': 'Concurrency'},
    'serve.lbl_vis':      {'zh': '多模态', 'en': 'Vision'},
    'serve.lbl_cold':     {'zh': '冷策略', 'en': 'Cold policy'},
    'serve.lbl_hotwin':   {'zh': '热窗 tokens', 'en': 'Hot window tokens'},
    'serve.hotwin_note':  {'zh': '0=引擎默认128', 'en': '0 = engine default 128'},
    'serve.lbl_kvlayer':  {'zh': '逐层 KV', 'en': 'Per-layer KV'},
    'serve.ph_kvlayer':   {'zh': '0-15:rk4v4,32-63:iso4e (空=默认)',
                           'en': '0-15:rk4v4,32-63:iso4e (empty = default)'},
    'serve.lbl_graph':    {'zh': 'CUDA Graph', 'en': 'CUDA Graph'},
    'serve.lbl_port':     {'zh': '端口', 'en': 'Port'},
    'serve.btn_start':    {'zh': '启动', 'en': 'Start'},
    'serve.btn_stop':     {'zh': '停止', 'en': 'Stop'},

    # 滑块档位显示值
    'serve.kv_0':         {'zh': 'BF16', 'en': 'BF16'},
    'serve.kv_1':         {'zh': 'INT8', 'en': 'INT8'},
    'serve.kv_2':         {'zh': 'FP8', 'en': 'FP8'},
    'serve.kv_3':         {'zh': 'NVFP4', 'en': 'NVFP4'},
    'serve.kv_4':         {'zh': 'Rk4v4 混合', 'en': 'Rk4v4 mixed'},
    'serve.spec_0':       {'zh': '无', 'en': 'None'},
    'serve.spec_1':       {'zh': 'MTP', 'en': 'MTP'},
    'serve.spec_2':       {'zh': 'DFlash', 'en': 'DFlash'},
    'serve.spec_3':       {'zh': 'DFlash2', 'en': 'DFlash2'},
    'serve.cold_0':       {'zh': 'off', 'en': 'off'},
    'serve.cold_1':       {'zh': 'window', 'en': 'window'},
    'serve.cold_2':       {'zh': 'host', 'en': 'host'},
    'serve.cold_3':       {'zh': 'disk', 'en': 'disk'},
    'serve.val_on':       {'zh': '开', 'en': 'On'},
    'serve.val_off':      {'zh': '关', 'en': 'Off'},
    'serve.hot_default':  {'zh': '默认', 'en': 'Default'},

    # ---- 投机档位 / 接受率 / 位置剖面 ----
    'serve.spec_title':     {'zh': '投机档位与接受率（当前会话）',
                             'en': 'Speculation tier and acceptance (this session)'},
    'serve.spec_none':      {'zh': '未启用', 'en': 'Not enabled'},
    'serve.spec_configured': {'zh': '已配置，等待实测', 'en': 'Configured, no run yet'},
    'serve.spec_measured':  {'zh': '实测中', 'en': 'Measured'},
    'serve.spec_na':        {'zh': '—', 'en': '—'},
    'serve.spec_tier':      {'zh': '档位 / 草稿后端', 'en': 'Tier / draft backend'},
    'serve.spec_rate':      {'zh': '接受率（accepted/drafted）', 'en': 'Acceptance (accepted/drafted)'},
    'serve.spec_len':       {'zh': '接受长度', 'en': 'Acceptance length'},
    'serve.spec_len_unit':  {'zh': 'tok/轮', 'en': 'tok/round'},
    'serve.spec_drafted':   {'zh': '草稿 token', 'en': 'Drafted tokens'},
    'serve.spec_accepted':  {'zh': '被接受 token', 'en': 'Accepted tokens'},
    'serve.spec_rounds':    {'zh': '草稿轮数', 'en': 'Drafting rounds'},
    'serve.spec_rounds_word': {'zh': '轮', 'en': 'rounds'},
    'serve.spec_fallback':  {'zh': '空转步（未草稿）', 'en': 'Fallback steps (no draft)'},
    'serve.spec_window':    {'zh': '草稿窗', 'en': 'Draft window'},
    'serve.spec_pos':       {'zh': '位置剖面', 'en': 'Accepted by position'},
    'serve.spec_pos_none':  {'zh': '（本轮日志里没有位置直方图）',
                             'en': '(no per-position histogram in this log)'},
    'serve.spec_src':       {'zh': '来源', 'en': 'source'},
    'serve.spec_per_round': {'zh': '/轮', 'en': '/round'},
    'serve.spec_hint':      {'zh': '接受率 = spec_accepted / spec_drafted（token 级）；位置剖面的每一根柱子是该位置被接受的次数，'
                                   '分母是"草稿轮数"。柱高按同一张图内的最大值归一；鼠标悬停显示"次数 / 每轮概率"。',
                             'en': 'Acceptance = spec_accepted / spec_drafted (token level). Each bar is how many times '
                                   'that position was accepted, over the drafting-round count. Bar height is normalised '
                                   'to the largest value in the same chart; hover shows "count / per-round rate".'},

    # ---- 全参数面板 ----
    'serve.adv_summary':  {'zh': '全参数（来自引擎注册表）',
                           'en': 'All parameters (from the engine registry)'},
    'serve.adv_hint':     {'zh': '留空 = 不传给引擎。只有填过的参数才会拼进启动命令，'
                                 '所以完全不动它就是原来的行为。',
                           'en': 'Leave a field empty to omit it. Only the ones you fill in '
                                 'are appended to the command line, so an untouched panel '
                                 'behaves exactly as before.'},
    'serve.adv_controlled': {'zh': '由上方滑块控制', 'en': 'set by the sliders above'},
    'serve.adv_registry': {'zh': '参数 {n} 项（来自引擎注册表）',
                           'en': '{n} options from the engine registry'},
    'serve.adv_no_registry': {'zh': '没有找到 serve_params.json，先运行 extract_serve_params.py',
                              'en': 'serve_params.json not found — run extract_serve_params.py first'},
    'serve.adv_placeholder': {'zh': '引擎默认', 'en': 'engine default'},

    'serve.grp_core':        {'zh': '核心', 'en': 'Core'},
    'serve.grp_kv':          {'zh': 'KV 缓存', 'en': 'KV cache'},
    'serve.grp_speculative': {'zh': '投机解码', 'en': 'Speculation'},
    'serve.grp_sampling':    {'zh': '采样', 'en': 'Sampling'},
    'serve.grp_thinking':    {'zh': '思考', 'en': 'Thinking'},
    'serve.grp_media':       {'zh': '多模态', 'en': 'Media'},
    'serve.grp_perf':        {'zh': '性能', 'en': 'Performance'},
    'serve.grp_network':     {'zh': '网络', 'en': 'Network'},
    'serve.grp_diagnostics': {'zh': '诊断', 'en': 'Diagnostics'},
    'serve.grp_advanced':    {'zh': '高级', 'en': 'Advanced'},
    'serve.grp_ungrouped':   {'zh': '其它', 'en': 'Other'},

    # ---- 日志 ----
    'serve.log_ready':    {'zh': '就绪。', 'en': 'Ready.'},
    'serve.log_sim_scrubbed': {
        'zh': '\u5f15\u64ce\u73af\u5883: \u5df2\u4e22\u5f03\u6d4b\u8bd5\u7528\u67b6\u6784\u6a21\u62df\u952e {keys} '
              '(\u672c GUI \u4e0d\u6a21\u62df\u663e\u5361; \u58f0\u79f0\u652f\u6301\u9700\u8981\u771f\u673a probe)',
        'en': 'engine env: dropped test-only simulator key(s) {keys} '
              '(this GUI does not simulate a card; support requires a probe on real hardware)'},
    'serve.log_started':  {'zh': '启动: ', 'en': 'Launching: '},
    'serve.log_stopped':  {'zh': '[已停止]', 'en': '[stopped]'},
    'serve.log_none':     {'zh': '(无日志)', 'en': '(no log)'},

    # ---- 错误 ----
    'serve.err_bad_request': {'zh': '请求格式不对（JSON 解析失败）',
                              'en': 'Malformed request body (JSON parse failed)'},
    'serve.err_not_found':   {'zh': '没有这个接口', 'en': 'No such endpoint'},
    'serve.err_stop_failed': {'zh': '停不下来: {err}', 'en': 'Stop failed: {err}'},
}
