@echo off
setlocal enabledelayedexpansion

call C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit\.venv\Scripts\activate.bat
cd /d C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit

set BASE=C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit\amd_test_results_round2

call :run_one "microsoft/swin-large-patch4-window7-224" "image-classification" "microsoft_swin-large-patch4-window7-224_image-classification"
call :run_one "BAAI/bge-large-en-v1.5" "sentence-similarity" "BAAI_bge-large-en-v1.5_sentence-similarity"
call :run_one "cardiffnlp/twitter-roberta-base-sentiment-latest" "text-classification" "cardiffnlp_twitter-roberta-base-sentiment-latest_text-classification"
call :run_one "dbmdz/bert-large-cased-finetuned-conll03-english" "token-classification" "dbmdz_bert-large-cased-finetuned-conll03-english_token-classification"
call :run_one "deepset/bert-large-uncased-whole-word-masking-squad2" "question-answering" "deepset_bert-large-uncased-whole-word-masking-squad2_question-answering"
call :run_one "deepset/roberta-base-squad2" "question-answering" "deepset_roberta-base-squad2_question-answering"
call :run_one "deepset/tinyroberta-squad2" "question-answering" "deepset_tinyroberta-squad2_question-answering"
call :run_one "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad" "question-answering" "google-bert_bert-large-uncased-whole-word-masking-finetuned-squad_question-answering"
call :run_one "google/vit-base-patch16-224" "image-classification" "google_vit-base-patch16-224_image-classification"
call :run_one "mattmdjaga/segformer_b2_clothes" "image-segmentation" "mattmdjaga_segformer_b2_clothes_image-segmentation"
call :run_one "microsoft/resnet-50" "image-classification" "microsoft_resnet-50_image-classification"
call :run_one "nvidia/segformer-b1-finetuned-ade-512-512" "image-segmentation" "nvidia_segformer-b1-finetuned-ade-512-512_image-segmentation"
call :run_one "nvidia/segformer-b2-finetuned-ade-512-512" "image-segmentation" "nvidia_segformer-b2-finetuned-ade-512-512_image-segmentation"
call :run_one "nvidia/segformer-b5-finetuned-ade-640-640" "image-segmentation" "nvidia_segformer-b5-finetuned-ade-640-640_image-segmentation"
call :run_one "openai/clip-vit-base-patch16" "feature-extraction" "openai_clip-vit-base-patch16_feature-extraction"
call :run_one "openai/clip-vit-base-patch32" "feature-extraction" "openai_clip-vit-base-patch32_feature-extraction"
call :run_one "rizvandwiki/gender-classification" "image-classification" "rizvandwiki_gender-classification_image-classification"
call :run_one "sentence-transformers/all-MiniLM-L6-v2" "feature-extraction" "sentence-transformers_all-MiniLM-L6-v2_feature-extraction"
call :run_one "sentence-transformers/all-MiniLM-L6-v2" "sentence-similarity" "sentence-transformers_all-MiniLM-L6-v2_sentence-similarity"
call :run_one "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" "sentence-similarity" "sentence-transformers_paraphrase-multilingual-mpnet-base-v2_sentence-similarity"
call :run_one "w11wo/indonesian-roberta-base-posp-tagger" "token-classification" "w11wo_indonesian-roberta-base-posp-tagger_token-classification"

echo.
echo ========== ALL DONE ==========
goto :eof

:run_one
set "MODEL_ID=%~1"
set "TASK=%~2"
set "FOLDER=%~3"

for %%p in (w8a8 w8a16 fp16) do (
    call :run_precision "%MODEL_ID%" "%TASK%" "%FOLDER%" "%%p"
)
goto :eof

:run_precision
set "MODEL_ID=%~1"
set "TASK=%~2"
set "FOLDER=%~3"
set "PREC=%~4"
set "DIR=%BASE%\%FOLDER%\%PREC%"
set "BUILDDIR=%DIR%\build"

if not exist "%BUILDDIR%" mkdir "%BUILDDIR%"

echo.
echo ========================================
echo MODEL: %MODEL_ID% ^| TASK: %TASK% ^| PRECISION: %PREC%
echo ========================================

REM Write commands
> "%DIR%\commands.txt" (
    echo # Config
    echo winml config -m %MODEL_ID% --task %TASK% --ep vitisai --precision %PREC% -o "%DIR%\config.json"
    echo.
    echo # Build
    echo winml build -c "%DIR%\config.json" -m %MODEL_ID% -o "%BUILDDIR%"
    echo.
    echo # Perf
    echo winml perf -m [best_onnx] --ep vitisai --iterations 100 -o "%DIR%\perf.json"
    echo.
    echo # Eval
    echo winml eval -m [best_onnx] --model-id %MODEL_ID% --task %TASK% --device npu -o "%DIR%\eval.json"
)

REM Config
echo [CONFIG] Starting...
winml config -m %MODEL_ID% --task %TASK% --ep vitisai --precision %PREC% -o "%DIR%\config.json" > "%DIR%\config_log.txt" 2>&1
if not exist "%DIR%\config.json" (
    echo [CONFIG] FAILED
    > "%DIR%\status.txt" echo CONFIG FAILED
    goto :eof
)
echo [CONFIG] PASS

REM Build
echo [BUILD] Starting...
winml build -c "%DIR%\config.json" -m %MODEL_ID% -o "%BUILDDIR%" > "%DIR%\build_log.txt" 2>&1

REM Find best onnx
set "BEST_ONNX="
set "BUILD_NOTE="
if exist "%BUILDDIR%\model.onnx" (
    set "BEST_ONNX=%BUILDDIR%\model.onnx"
    set "BUILD_NOTE=compiled"
)
if "%BEST_ONNX%"=="" if exist "%BUILDDIR%\quantized.onnx" (
    set "BEST_ONNX=%BUILDDIR%\quantized.onnx"
    set "BUILD_NOTE=quantized(compile_failed)"
)
if "%BEST_ONNX%"=="" if exist "%BUILDDIR%\optimized.onnx" (
    set "BEST_ONNX=%BUILDDIR%\optimized.onnx"
    set "BUILD_NOTE=optimized(quant_failed)"
)
if "%BEST_ONNX%"=="" if exist "%BUILDDIR%\export.onnx" (
    set "BEST_ONNX=%BUILDDIR%\export.onnx"
    set "BUILD_NOTE=export_only"
)

if "%BEST_ONNX%"=="" (
    echo [BUILD] FAILED - no onnx
    > "%DIR%\status.txt" echo CONFIG PASS, BUILD FAILED
    goto :eof
)
echo [BUILD] %BUILD_NOTE% - %BEST_ONNX%

REM Perf
echo [PERF] Starting...
winml perf -m "%BEST_ONNX%" --ep vitisai --iterations 100 -o "%DIR%\perf.json" > "%DIR%\perf_log.txt" 2>&1
if exist "%DIR%\perf.json" (
    set "PERF_STATUS=PASS"
    echo [PERF] PASS
) else (
    set "PERF_STATUS=FAIL"
    echo [PERF] FAIL
)

REM Eval
echo [EVAL] Starting...
winml eval -m "%BEST_ONNX%" --model-id %MODEL_ID% --task %TASK% --device npu -o "%DIR%\eval.json" > "%DIR%\eval_log.txt" 2>&1
if exist "%DIR%\eval.json" (
    set "EVAL_STATUS=PASS"
    echo [EVAL] PASS
) else (
    set "EVAL_STATUS=FAIL"
    echo [EVAL] FAIL
)

> "%DIR%\status.txt" echo CONFIG PASS, BUILD %BUILD_NOTE%, PERF %PERF_STATUS%, EVAL %EVAL_STATUS%
goto :eof
