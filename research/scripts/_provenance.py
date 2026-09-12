import json
import pathlib

print("=== q3nvfp4（我们的 target 源）身份线索 ===")
c = json.loads(pathlib.Path("/home/user/models/q3nvfp4/config.json").read_text())
for k in ("_name_or_path", "transformers_version", "model_type", "architectures",
          "tie_word_embeddings"):
    if k in c:
        print(f"  {k} = {str(c[k])[:120]}")
t = c.get("text_config") or {}
for k in ("_name_or_path", "num_hidden_layers", "hidden_size", "vocab_size",
          "tie_word_embeddings", "max_position_embeddings", "dtype"):
    if k in t:
        print(f"  text_config.{k} = {t[k]}")

g = pathlib.Path("/home/user/models/q3nvfp4/generation_config.json")
if g.exists():
    d = json.loads(g.read_text())
    print("  generation_config:", {k: d.get(k) for k in
                                  ("_from_model_config", "transformers_version", "eos_token_id",
                                   "repetition_penalty", "temperature") if k in d})

tc = pathlib.Path("/home/user/models/q3nvfp4/tokenizer_config.json")
if tc.exists():
    d = json.loads(tc.read_text())
    print("  tokenizer_class =", d.get("tokenizer_class"),
          " chat_template 长度 =", len(str(d.get("chat_template", ""))))
    ct = str(d.get("chat_template", ""))
    print("  chat_template 前 160 字：", ct[:160].replace("\n", " "))

print()
print("=== dflash2 参考草稿的出处 ===")
c2 = json.loads(pathlib.Path("/home/user/models/draft_dflash2_ref/config.json").read_text())
for k in ("_name_or_path", "architectures", "model_type"):
    print(f"  {k} = {str(c2.get(k))[:140]}")
print("  dflash_config =", json.dumps(c2.get("dflash_config") or {}, ensure_ascii=False)[:320])
tc2 = c2.get("text_config") or {}
print("  text_config 关键项:", {k: tc2.get(k) for k in
                            ("num_hidden_layers", "hidden_size", "vocab_size",
                             "max_position_embeddings") if k in tc2})

print()
print("=== 两份 config 的架构摘要是否一致（target vs draft 期望的 target）===")
print("  target: layers=%s hidden=%s vocab=%s" % (t.get("num_hidden_layers"), t.get("hidden_size"), t.get("vocab_size")))
print("  draft  : layers=%s hidden=%s vocab=%s" % (tc2.get("num_hidden_layers"), tc2.get("hidden_size"), tc2.get("vocab_size")))
