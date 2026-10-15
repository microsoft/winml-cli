// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Native DXCGC benchmark host. No Windows ML Runtime API or ORT dependency.
#define NOMINMAX
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <d3d12.h>
#include <dxcgc.h>
#include <dxgi1_6.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <windows.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static void check(HRESULT hr) {
  if (FAILED(hr)) {
    std::ostringstream s;
    s << "HRESULT 0x" << std::hex << (unsigned long)hr;
    throw std::runtime_error(s.str());
  }
}
static std::string utf8(const std::wstring &w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0,
                              nullptr, nullptr);
  std::string s(n, 0);
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr,
                      nullptr);
  return s;
}
static std::vector<char> read(const fs::path &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  auto n = f.tellg();
  if (n < 0)
    throw std::runtime_error("Invalid file size");
  std::vector<char> b((size_t)n);
  f.seekg(0);
  if (n && !f.read(b.data(), n))
    throw std::runtime_error("Short file read");
  return b;
}
static void write(const fs::path &p, const void *data, size_t n) {
  std::ofstream f(p, std::ios::binary);
  if (!f.write((const char *)data, n))
    throw std::runtime_error("Cannot write output");
}
struct Buffer {
  ComPtr<ID3D12Resource> resource;
  UINT64 offset = 0, size = 0;
};
struct Host {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  UINT64 serial = 0;
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ~Host() {
    if (event)
      CloseHandle(event);
  }
  void init(IDXGIAdapter *adapter, ID3D12DeviceFactory *factory) {
    check(factory->CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC q = {};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
    check(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator)));
    check(device->CreateCommandList(0, q.Type, allocator.Get(), nullptr,
                                    IID_PPV_ARGS(&list)));
    check(list->Close());
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    if (!event)
      throw std::runtime_error("CreateEvent failed");
  }
  void begin() {
    check(allocator->Reset());
    check(list->Reset(allocator.Get(), nullptr));
  }
  void submit() {
    check(list->Close());
    ID3D12CommandList *lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    check(queue->Signal(fence.Get(), ++serial));
    check(fence->SetEventOnCompletion(serial, event));
    if (WaitForSingleObject(event, 120000) != WAIT_OBJECT_0)
      throw std::runtime_error("GPU fence timeout");
    check(device->GetDeviceRemovedReason());
  }
  Buffer buffer(UINT64 size, D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT) {
    D3D12_HEAP_PROPERTIES h = {};
    h.Type = type;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = std::max<UINT64>(size, 256);
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = type == D3D12_HEAP_TYPE_DEFAULT
                  ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                  : D3D12_RESOURCE_FLAG_NONE;
    Buffer b;
    b.size = size;
    check(device->CreateCommittedResource(
        &h, D3D12_HEAP_FLAG_NONE, &d,
        type == D3D12_HEAP_TYPE_DEFAULT  ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        : type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                         : D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&b.resource)));
    return b;
  }
  void transition(const Buffer &b, D3D12_RESOURCE_STATES before,
                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER v = {};
    v.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    v.Transition = {b.resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    before, after};
    list->ResourceBarrier(1, &v);
  }
  void upload(const Buffer &b, const void *data, size_t size) {
    if (size > b.size)
      throw std::runtime_error("Input exceeds binding");
    auto staging = buffer(size, D3D12_HEAP_TYPE_UPLOAD);
    void *p;
    check(staging.resource->Map(0, nullptr, &p));
    memcpy(p, data, size);
    staging.resource->Unmap(0, nullptr);
    begin();
    transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(b.resource.Get(), b.offset, staging.resource.Get(),
                           0, size);
    transition(b, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    submit();
  }
  void download(const Buffer &b, const fs::path &path) {
    auto staging = buffer(b.size, D3D12_HEAP_TYPE_READBACK);
    begin();
    transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(staging.resource.Get(), 0, b.resource.Get(),
                           b.offset, b.size);
    transition(b, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    submit();
    void *p;
    check(staging.resource->Map(0, nullptr, &p));
    write(path, p, (size_t)b.size);
    staging.resource->Unmap(0, nullptr);
  }
};
static std::string tensorName(IDXCGCModule *m, DXCGC_MODULE_TENSOR_ID id) {
  SIZE_T n = 0;
  check(m->GetTensorName(id, &n, nullptr));
  std::string s(n, 0);
  if (n)
    check(m->GetTensorName(id, &n, s.data()));
  while (!s.empty() && !s.back())
    s.pop_back();
  return s;
}
static std::string resourceName(IDXCGCExecutableState *s,
                                DXCGC_EXECUTABLE_RESOURCE_BIND_POINT b) {
  SIZE_T n = 0;
  check(s->GetResourceName(b, &n, nullptr));
  std::string v(n, 0);
  if (n)
    check(s->GetResourceName(b, &n, v.data()));
  while (!v.empty() && !v.back())
    v.pop_back();
  return v;
}
static void diagnostic(DXCGC_MESSAGE_SEVERITY, LPCSTR message, void *) {
  std::cerr << message << std::endl;
}
struct Executable {
  ComPtr<IDXCGCExecutable> code;
  ComPtr<IDXCGCExecutableState> state;
  std::vector<Buffer> args, constants;
  std::vector<ComPtr<ID3D12DescriptorHeap>> heaps;
};
static void record(Host &h, Executable &e) {
  std::vector<ID3D12DescriptorHeap *> heaps;
  for (auto &x : e.heaps)
    heaps.push_back(x.Get());
  if (!heaps.empty())
    h.list->SetDescriptorHeaps((UINT)heaps.size(), heaps.data());
  check(e.code->Record(h.list.Get(), e.state.Get()));
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  h.list->ResourceBarrier(1, &b);
}
static Executable create(Host &h, IDXCGCDevice *d, IDXCGCModule *m,
                         DXCGC_MODULE_SYMBOL_ID id,
                         std::map<std::string, Buffer> &processed) {
  Executable e;
  check(d->CreateExecutable(m, id, nullptr, DXCGC_EXECUTABLE_FLAG_NONE,
                            IID_PPV_ARGS(&e.code)));
  std::vector<DXCGC_DESCRIPTOR_RANGE> ranges;
  for (auto type : {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER}) {
    UINT count = e.code->GetRequiredDescriptorCount(type);
    if (!count)
      continue;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {
        type, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(h.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
    ranges.push_back({type, heap->GetCPUDescriptorHandleForHeapStart(),
                      heap->GetGPUDescriptorHandleForHeapStart(), count});
    e.heaps.push_back(heap);
  }
  check(d->CreateExecutableState(e.code.Get(), DXCGC_EXECUTABLE_STATE_FLAG_NONE,
                                 ranges.size(), ranges.data(),
                                 IID_PPV_ARGS(&e.state)));
  for (auto type : {DXCGC_EXECUTABLE_RESOURCE_TYPE_ARGUMENT,
                    DXCGC_EXECUTABLE_RESOURCE_TYPE_CONSTANT})
    for (SIZE_T i = 0; i < e.state->GetResourceCount(type); ++i) {
      DXCGC_EXECUTABLE_RESOURCE_BIND_POINT bp = {type, i};
      D3D12_RESOURCE_DESC desc;
      check(e.state->GetResourceDesc(bp, &desc));
      if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        throw std::runtime_error("Only buffer bindings supported");
      Buffer b;
      if (type == DXCGC_EXECUTABLE_RESOURCE_TYPE_CONSTANT) {
        auto name = resourceName(e.state.Get(), bp);
        auto found = processed.find(name);
        if (found != processed.end()) {
          b = found->second;
          if (b.size < desc.Width)
            throw std::runtime_error("Initializer binding too small");
        } else {
          DXCGC_MODULE_RESOURCE_ID rid;
          SIZE_T n = 0;
          const void *p = nullptr;
          check(m->GetResourceId(name.c_str(), &rid));
          check(m->GetResourceData(rid, &n, &p));
          if (!p || n > desc.Width)
            throw std::runtime_error("Invalid embedded constant: " + name);
          b = h.buffer(desc.Width);
          h.upload(b, p, n);
        }
      } else
        b = h.buffer(desc.Width);
      check(e.state->BindResource(bp, b.resource.Get(), b.offset));
      (type == DXCGC_EXECUTABLE_RESOURCE_TYPE_ARGUMENT ? e.args : e.constants)
          .push_back(b);
    }
  return e;
}
static Buffer view(IDXCGCModule *m, DXCGC_MODULE_TENSOR_ID id, Executable &e) {
  DXCGC_MODULE_TENSOR_SOURCE source;
  check(m->GetTensorSource(id, &source));
  if (source.id.type != DXCGC_MODULE_TENSOR_TYPE_ARGUMENT ||
      source.id.index >= e.args.size())
    throw std::runtime_error("Unsupported tensor view source");
  auto b = e.args[source.id.index];
  if (source.offsetInBytes > b.size ||
      source.sizeInBytes > b.size - source.offsetInBytes)
    throw std::runtime_error("View outside binding");
  b.offset += source.offsetInBytes;
  b.size = source.sizeInBytes;
  return b;
}
int wmain(int argc, wchar_t **argv) {
  try {
    if (argc != 9 && argc != 10)
      throw std::runtime_error(
          "Expected: model dll-dir sdk-dir sdk-version "
          "adapter-index iterations warmup data-dir [--disable-d3d-mlir-programs]");
    const bool disableD3dMlir = argc == 10;
    if (disableD3dMlir && std::wstring(argv[9]) != L"--disable-d3d-mlir-programs")
      throw std::runtime_error("Unknown native host option");
    fs::path model = argv[1], dll = argv[2], sdk = argv[3], data = argv[8];
    int iterations = std::stoi(argv[6]), warmup = std::stoi(argv[7]);
    if (iterations < 1 || warmup < 0)
      throw std::runtime_error("Invalid iteration count");
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                             LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (!AddDllDirectory(dll.c_str()))
      throw std::runtime_error("Invalid DLL directory");
    auto library = LoadLibraryExW((dll / L"dxcgc.dll").c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                      LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library)
      throw std::runtime_error("Cannot load dxcgc.dll or its dependencies");
    auto make = (decltype(&DXCGCCreateDevice))GetProcAddress(
        library, "DXCGCCreateDevice");
    if (!make)
      throw std::runtime_error("Missing DXCGCCreateDevice");
    ComPtr<ID3D12SDKConfiguration1> config;
    check(
        D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&config)));
    ComPtr<ID3D12DeviceFactory> factory;
    auto sdkPath = utf8(sdk.wstring()) + "/";
    check(config->CreateDeviceFactory(std::stoi(argv[4]), sdkPath.c_str(),
                                      IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIFactory6> dxgi;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)));
    ComPtr<IDXGIAdapter1> adapter;
    check(dxgi->EnumAdapterByGpuPreference(std::stoi(argv[5]),
          DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
    DXGI_ADAPTER_DESC1 ad;
    check(adapter->GetDesc1(&ad));
    if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
      throw std::runtime_error("Software adapter not supported");
    Host h;
    h.init(adapter.Get(), factory.Get());
    ComPtr<IDXCGCDevice> device;
    check(make(h.device.Get(), DXCGC_DEVICE_FLAG_NONE, IID_PPV_ARGS(&device)));
    DWORD cookie;
    check(device->RegisterMessageCallback(diagnostic, nullptr, &cookie));
    auto start = Clock::now();
    ComPtr<IDXCGCModule> input, module;
    check(device->CreateModuleFromFile(model.c_str(), IID_PPV_ARGS(&input)));
    if (input->GetType() == DXCGC_MODULE_IR_TYPE_INPUT) {
      ComPtr<IDXCGCCompilerParameters> params;
      if (disableD3dMlir) {
        check(device->CreateCompilerParameters(IID_PPV_ARGS(&params)));
        DXCGC_INTERNAL_OPTION option{"disable-d3d-mlir-programs", "true"};
        HRESULT hr = params->SetOption(DXCGC_COMPILER_OPTION_INTERNAL, &option, sizeof(option));
        if (FAILED(hr)) {
          std::cerr << "DXCGC rejected --disable-d3d-mlir-programs: ";
          check(hr);
        }
      }
      ComPtr<IDXCGCCompiler> compiler;
      check(device->CreateCompiler(params.Get(), IID_PPV_ARGS(&compiler)));
      check(compiler->Compile(input.Get(), 0, nullptr, DXCGC_COMPILE_FLAG_NONE,
                              nullptr, IID_PPV_ARGS(&module)));
    } else if (input->GetType() == DXCGC_MODULE_IR_TYPE_OUTPUT) {
      if (disableD3dMlir)
        throw std::runtime_error("--disable-d3d-mlir-programs requires Input IR; Output IR targets are already compiled");
      module = input;
    } else
      throw std::runtime_error("Unsupported IR type");
    double compileMs =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (module->GetSymbolCount(DXCGC_MODULE_SYMBOL_TYPE_PROGRAM) != 1)
      throw std::runtime_error("Exactly one program required");
    std::map<std::string, Buffer> processed;
    DXCGC_MODULE_SYMBOL_ID program = {DXCGC_MODULE_SYMBOL_TYPE_PROGRAM, 0};
    SIZE_T count = 0;
    check(module->ResolveSymbolDependencies(
        1, &program, DXCGC_MODULE_SYMBOL_TYPE_INITIALIZER, &count, nullptr));
    std::vector<DXCGC_MODULE_SYMBOL_ID> dependencies(count);
    if (count)
      check(module->ResolveSymbolDependencies(
          1, &program, DXCGC_MODULE_SYMBOL_TYPE_INITIALIZER, &count,
          dependencies.data()));
    for (auto id : dependencies) {
      auto e = create(h, device.Get(), module.Get(), id, processed);
      h.begin();
      record(h, e);
      h.submit();
      SIZE_T n = 0;
      check(module->GetTensorCount(
          id, DXCGC_MODULE_TENSOR_TYPE_CONSTANT_EXPORT_VIEW, &n));
      for (SIZE_T i = 0; i < n; ++i) {
        DXCGC_MODULE_TENSOR_ID tid = {
            id, DXCGC_MODULE_TENSOR_TYPE_CONSTANT_EXPORT_VIEW, i};
        processed[tensorName(module.Get(), tid)] = view(module.Get(), tid, e);
      }
    }
    auto e = create(h, device.Get(), module.Get(), program, processed);
    SIZE_T inputs = 0, outputs = 0;
    check(module->GetTensorCount(program, DXCGC_MODULE_TENSOR_TYPE_INPUT_VIEW,
                                 &inputs));
    check(module->GetTensorCount(program, DXCGC_MODULE_TENSOR_TYPE_OUTPUT_VIEW,
                                 &outputs));
    std::ifstream inputCount(data / "input-count.txt");
    SIZE_T supplied = 0;
    inputCount >> supplied;
    if (supplied != inputs)
      throw std::runtime_error("Input count does not match model");
    std::vector<Buffer> inputViews, uploads, outputViews, readbacks;
    std::vector<std::vector<char>> inputBytes, outputBytes;
    std::ofstream metadata(data / "io.tsv");
    for (auto type : {DXCGC_MODULE_TENSOR_TYPE_INPUT_VIEW,
                      DXCGC_MODULE_TENSOR_TYPE_OUTPUT_VIEW})
      for (SIZE_T i = 0;
           i < (type == DXCGC_MODULE_TENSOR_TYPE_INPUT_VIEW ? inputs : outputs);
           ++i) {
        DXCGC_MODULE_TENSOR_ID tid = {program, type, i};
        auto b = view(module.Get(), tid, e);
        DXCGC_DATA_TYPE dtype;
        check(module->GetTensorDataType(tid, &dtype));
        SIZE_T rank;
        check(module->GetTensorRank(tid, &rank));
        std::vector<INT64> dims(rank);
        check(module->GetTensorDimensions(tid, rank, dims.data()));
        metadata << (type == DXCGC_MODULE_TENSOR_TYPE_INPUT_VIEW ? "input"
                                                                 : "output")
                 << '	' << i << '	' << (UINT)dtype << '	';
        for (auto dim : dims)
          metadata << dim << ',';
        metadata << '	' << tensorName(module.Get(), tid) << std::endl;
        if (type == DXCGC_MODULE_TENSOR_TYPE_INPUT_VIEW) {
          auto bytes = read(data / ("input-" + std::to_string(i) + ".bin"));
          if (bytes.size() != b.size)
            throw std::runtime_error("Input byte size mismatch");
          std::ifstream spec(data / ("input-" + std::to_string(i) + ".txt"));
          UINT suppliedType = 0;
          SIZE_T suppliedRank = 0;
          spec >> suppliedType >> suppliedRank;
          if (suppliedType != (UINT)dtype || suppliedRank != rank)
            throw std::runtime_error("Input dtype/rank mismatch");
          for (auto dim : dims) {
            INT64 got = -1;
            spec >> got;
            if (got != dim)
              throw std::runtime_error("Input shape mismatch");
          }
          if (!spec)
            throw std::runtime_error("Invalid input metadata");
          h.upload(b, bytes.data(), bytes.size());
          inputViews.push_back(b);
          inputBytes.push_back(std::move(bytes));
          uploads.push_back(h.buffer(b.size, D3D12_HEAP_TYPE_UPLOAD));
        } else {
          outputViews.push_back(b);
          readbacks.push_back(h.buffer(b.size, D3D12_HEAP_TYPE_READBACK));
        }
      }
    metadata.close();
    ComPtr<ID3D12QueryHeap> queries;
    D3D12_QUERY_HEAP_DESC q = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    check(h.device->CreateQueryHeap(&q, IID_PPV_ARGS(&queries)));
    auto times = h.buffer(16, D3D12_HEAP_TYPE_READBACK);
    UINT64 frequency;
    check(h.queue->GetTimestampFrequency(&frequency));
    std::ofstream perf(data / "samples.csv");
    perf << "gpu_ms,host_ms\n";
    perf.precision(12);
    for (int i = 0; i < warmup + iterations; ++i) {
      auto begin = Clock::now();
      h.begin();
      {
        for (size_t j = 0; j < inputViews.size(); ++j) {
          void *mapped;
          check(uploads[j].resource->Map(0, nullptr, &mapped));
          memcpy(mapped, inputBytes[j].data(), inputBytes[j].size());
          uploads[j].resource->Unmap(0, nullptr);
          h.transition(inputViews[j], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_DEST);
          h.list->CopyBufferRegion(
              inputViews[j].resource.Get(), inputViews[j].offset,
              uploads[j].resource.Get(), 0, inputViews[j].size);
          h.transition(inputViews[j], D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
      }
      h.list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
      record(h, e);
      h.list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
      h.list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                               times.resource.Get(), 0);
      {
        for (size_t j = 0; j < outputViews.size(); ++j) {
          h.transition(outputViews[j], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
          h.list->CopyBufferRegion(readbacks[j].resource.Get(), 0,
                                   outputViews[j].resource.Get(),
                                   outputViews[j].offset, outputViews[j].size);
          h.transition(outputViews[j], D3D12_RESOURCE_STATE_COPY_SOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
      }
      h.submit();
      {
        outputBytes.clear();
        for (const auto &b : readbacks) {
          void *mapped;
          check(b.resource->Map(0, nullptr, &mapped));
          outputBytes.emplace_back((const char *)mapped,
                                   (const char *)mapped + b.size);
          b.resource->Unmap(0, nullptr);
        }
      }
      double host =
          std::chrono::duration<double, std::milli>(Clock::now() - begin)
              .count();
      UINT64 *ticks;
      check(times.resource->Map(0, nullptr, (void **)&ticks));
      double gpu = (ticks[1] - ticks[0]) * 1000.0 / frequency;
      times.resource->Unmap(0, nullptr);
      if (i >= warmup)
        perf << gpu << ',' << host << std::endl;
    }
    perf.close();
    for (SIZE_T i = 0; i < outputs; ++i) {
      write(data / ("output-" + std::to_string(i) + ".bin"),
            outputBytes[i].data(), outputBytes[i].size());
    }
    std::ofstream info(data / "device.tsv");
    info << utf8(ad.Description) << std::endl
         << compileMs << std::endl
         << ad.AdapterLuid.HighPart << ':' << ad.AdapterLuid.LowPart
         << std::endl;
    check(device->UnregisterMessageCallback(cookie));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
