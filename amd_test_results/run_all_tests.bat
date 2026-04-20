@echo off
setlocal enabledelayedexpansion

call C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit\.venv\Scripts\activate.bat
cd /d C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit

set BASE=C:\Users\shzhen\AppData\Local\repo\WinML-ModelKit\amd_test_results

call :run_one "BAAI/bge-base-en-v1.5" "feature-extraction" "BAAI_bge-base-en-v1.5_feature-extraction"
call :run_one "BAAI/bge-base-en-v1.5" "sentence-similarity" "BAAI_bge-base-en-v1.5_sentence-similarity"
call :run_one "BAAI/bge-small-en-v1.5" "feature-extraction" "BAAI_bge-small-en-v1.5_feature-extraction"
call :run_one "BAAI/bge-small-en-v1.5" "sentence-similarity" "BAAI_bge-small-en-v1.5_sentence-similarity"
call :run_one "Babelscape/wikineural-multilingual-ner" "token-classification" "Babelscape_wikineural-multilingual-ner_token-classification"
call :run_one "dslim/bert-base-NER" "token-classification" "dslim_bert-base-NER_token-classification"
call :run_one "facebook/convnext-tiny-224" "image-classification" "facebook_convnext-tiny-224_image-classification"
call :run_one "google-bert/bert-base-multilingual-cased" "feature-extraction" "google-bert_bert-base-multilingual-cased_feature-extraction"
call :run_one "Intel/bert-base-uncased-mrpc" "feature-extraction" "Intel_bert-base-uncased-mrpc_feature-extraction"
call :run_one "Intel/bert-base-uncased-mrpc" "text-classification" "Intel_bert-base-uncased-mrpc_text-classification"
call :run_one "microsoft/table-transformer-detection" "object-detection" "microsoft_table-transformer-detection_object-detection"
call :run_one "ProsusAI/finbert" "text-classification" "ProsusAI_finbert_text-classification"
call :run_one "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "feature-extraction" "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_feature-extraction"
call :run_one "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "sentence-similarity" "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_sentence-similarity"

echo.
echo ========== ALL DONE ==========
echo Check status files in each folder for results.
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
