### Task 1: Add DINOv2 QNN recipes

**Files:**
- Create: `examples\recipes\facebook_dinov2-small\qnn\image-feature-extraction_w8a16_opset21_bias-softmax_config.json`
- Create: `examples\recipes\facebook_dinov2-base\qnn\image-feature-extraction_w8a16_opset21_bias-softmax_config.json`

**Interfaces:**
- Consumes existing DINOv2 W8A16 recipe JSON schema.
- Produces parseable `WinMLBuildConfig` JSON configs with `opset_version=21` and `bias_softmax_fusion=true`.

- [ ] Create both `qnn` directories.
- [ ] Copy each existing DINOv2 W8A16 recipe into its `qnn` subfolder.
- [ ] Update only `export.opset_version` and `optim`.
- [ ] Validate both files with `WinMLBuildConfig.from_dict` and assertions for model id, opset, quant type, and optim flag.
