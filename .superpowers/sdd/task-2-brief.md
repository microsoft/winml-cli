### Task 2: Add SwinV2-tiny QNN recipe

**Files:**
- Create: `examples\recipes\microsoft_swinv2-tiny-patch4-window16-256\qnn\image-classification_fp16_opset21_matmul-transpose_config.json`

**Interfaces:**
- Consumes existing SwinV2-tiny fp16 recipe JSON schema.
- Produces a parseable `WinMLBuildConfig` JSON config with `opset_version=21`, fp16/no-quant behavior, and `matmul_transpose_fusion=true`.

- [ ] Create the `qnn` directory.
- [ ] Copy the existing SwinV2-tiny fp16 recipe into the `qnn` subfolder.
- [ ] Update only `export.opset_version` and `optim`.
- [ ] Validate the file with `WinMLBuildConfig.from_dict` and assertions for task, opset, `quant is None`, and optim flag.
