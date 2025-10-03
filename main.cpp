#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
wgpu::Instance g_instance;
wgpu::Adapter g_adapter;
wgpu::Device g_device;
wgpu::Surface g_surface;
wgpu::TextureFormat g_format;

// ------------------------------------------------------------
// --- input state ---
// ------------------------------------------------------------
GLFWwindow* g_window = nullptr;
double g_mouseX = 0.0, g_mouseY = 0.0;
bool g_mouseDown = false;
uint32_t g_paintValue = 1;               // 1 = make alive, 0 = kill
int g_brushRadius = 0;                   // 0 = single cell, >0 = disc radius in cells
int g_LastCellX = -1, g_LastCellY = -1;   // avoid repainting same cell

// ------------------------------------------------------------
// Game of Life config
// ------------------------------------------------------------
const uint32_t WIDTH = 512;  //pixels
const uint32_t HEIGHT = 512; //pixels
const uint32_t GRID_SIZE = 32;
const uint32_t WORKGROUP_SIZE = 8;
const float UPDATE_INTERVAL_MS = 200.0f; // 5 Hz, matches JS

// ------------------------------------------------------------
// Simulation/render state
// ------------------------------------------------------------
wgpu::BindGroupLayout g_bindGroupLayout;
wgpu::PipelineLayout g_pipelineLayout;

wgpu::ShaderModule g_renderModule;
wgpu::ShaderModule g_computeModule;

wgpu::RenderPipeline g_renderPipeline;
wgpu::ComputePipeline g_computePipeline;

wgpu::Buffer g_vertexBuffer;
uint32_t g_vertexCount = 0;

wgpu::Buffer g_uniformBuffer;
wgpu::Buffer g_cellStateStorage[2];
wgpu::BindGroup g_bindGroups[2];

uint64_t g_stepIndex = 0;

// simple timer for UPDATE_INTERVAL_MS
std::chrono::steady_clock::time_point g_lastTick;
double g_accumulatorMs = 0.0;

// ------------------------------------------------------------

void ConfigureSurface() {
  wgpu::SurfaceCapabilities capabilities;
  g_surface.GetCapabilities(g_adapter, &capabilities);
  g_format = capabilities.formats[0];

  wgpu::SurfaceConfiguration g_config{};   // zero-initialise
  g_config.device = g_device;
  g_config.format = g_format;
  g_config.usage = wgpu::TextureUsage::RenderAttachment;
#if defined(__EMSCRIPTEN__)
  g_config.alphaMode = wgpu::CompositeAlphaMode::Premultiplied; // typical for canvas
#else
  g_config.alphaMode = wgpu::CompositeAlphaMode::Auto;
#endif
  g_config.width = WIDTH;
  g_config.height = HEIGHT;
  g_config.presentMode = wgpu::PresentMode::Fifo;

  g_surface.Configure(&g_config);
}

void Init() {
  static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instanceDesc{.requiredFeatureCount = 1,
                                        .requiredFeatures = &kTimedWaitAny};
  g_instance = wgpu::CreateInstance(&instanceDesc);

  wgpu::Future f1 = g_instance.RequestAdapter(
      nullptr, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::RequestAdapterStatus status, wgpu::Adapter a,
         wgpu::StringView message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          std::cout << "RequestAdapter: " << message << "\n";
          std::exit(1);
        }
        g_adapter = std::move(a);
      });
  g_instance.WaitAny(f1, UINT64_MAX);

  wgpu::DeviceDescriptor desc{};

  desc.SetUncapturedErrorCallback([](const wgpu::Device&,
                                     wgpu::ErrorType errorType,
                                     wgpu::StringView message) {
    std::cout << "Error: " << errorType << " - message: " << message << "\n";
  });

  desc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous, [](const wgpu::Device&,
                                wgpu::DeviceLostReason reason,
                                wgpu::StringView message) {
    std::cout << "Device lost: " << static_cast<int>(reason) << " - message: " << message << "\n";
  });

  wgpu::Future f2 = g_adapter.RequestDevice(
      &desc, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::RequestDeviceStatus status, wgpu::Device d,
         wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::cout << "RequestDevice: " << message << "\n";
          std::exit(1);
        }
        g_device = std::move(d);
      });
  g_instance.WaitAny(f2, UINT64_MAX);
}

// ------------------------------------------------------------
// WGSL shaders - taken from the original GoL JS Tutorial
// ------------------------------------------------------------

static const char kCellRenderWGSL[] = R"(
    struct VertexInput {
        @location(0) pos : vec2f,
        @builtin(instance_index) instance : u32,
    };

    struct VertexOutput {
        @builtin(position) pos : vec4f,
        @location(0) cell : vec2f,
    };

    @group(0) @binding(0) var<uniform> grid : vec2f;
    @group(0) @binding(1) var<storage> cellState : array<u32>;

    @vertex
    fn vertexMain(input : VertexInput) -> VertexOutput {
        let i = f32(input.instance);
        let cell = vec2f(i % grid.x, floor(i / grid.x));
        let state = f32(cellState[input.instance]);

        let cellOffset = cell / grid * 2.0;
        let gridPos = (input.pos * state + 1.0) / grid - 1.0 + cellOffset;

        var out : VertexOutput;
        out.pos = vec4f(gridPos, 0.0, 1.0);
        out.cell = cell;
        return out;
    }

    @fragment
    fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {
        let c = input.cell / grid;
        return vec4f(c, 1.0 - c.x, 1.0);
    }
)";

static const char kLifeComputeWGSL[] = R"(
    @group(0) @binding(0) var<uniform> grid : vec2f;
    @group(0) @binding(1) var<storage> cellStateIn : array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut : array<u32>;

    fn cellIndex(cell : vec2u) -> u32 {
        return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
    }

    fn cellActive(x : u32, y : u32) -> u32 {
        return cellStateIn[cellIndex(vec2(x, y))];
    }

    @compute @workgroup_size(8, 8)
    fn computeMain(@builtin(global_invocation_id) cell : vec3u) {
        let activeNeighbors =
            cellActive(cell.x + 1u, cell.y + 1u) +
            cellActive(cell.x + 1u, cell.y + 0u) +
            cellActive(cell.x + 1u, cell.y - 1u) +
            cellActive(cell.x + 0u, cell.y - 1u) +
            cellActive(cell.x - 1u, cell.y - 1u) +
            cellActive(cell.x - 1u, cell.y + 0u) +
            cellActive(cell.x - 1u, cell.y + 1u) +
            cellActive(cell.x + 0u, cell.y + 1u);

        let i = cellIndex(cell.xy);

        switch activeNeighbors {
          case 2u: { cellStateOut[i] = cellStateIn[i]; }
          case 3u: { cellStateOut[i] = 1u; }
          default: { cellStateOut[i] = 0u; }
        }
    }
)";

// ------------------------------------------------------------

wgpu::ShaderModule CreateWGSL(const char* code) {
  wgpu::ShaderSourceWGSL wgsl;
  wgsl.code = code;
  wgpu::ShaderModuleDescriptor desc{ .nextInChain = &wgsl };
  return g_device.CreateShaderModule(&desc);
}

void CreateBindGroupLayoutAndPipelineLayout() {
  // binding 0: uniform grid vec2f
  wgpu::BindGroupLayoutEntry b0{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment | wgpu::ShaderStage::Compute,
      .buffer = { .type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = false, .minBindingSize = sizeof(float) * 2 }
  };
  // binding 1: read-only storage in
  wgpu::BindGroupLayoutEntry b1{
      .binding = 1,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Compute,
      .buffer = { .type = wgpu::BufferBindingType::ReadOnlyStorage, .hasDynamicOffset = false, .minBindingSize = 0 }
  };
  // binding 2: storage out
  wgpu::BindGroupLayoutEntry b2{
      .binding = 2,
      .visibility = wgpu::ShaderStage::Compute,
      .buffer = { .type = wgpu::BufferBindingType::Storage, .hasDynamicOffset = false, .minBindingSize = 0 }
  };

  std::array<wgpu::BindGroupLayoutEntry,3> entries{b0,b1,b2};
  wgpu::BindGroupLayoutDescriptor bglDesc{ .entryCount = (uint32_t)entries.size(), .entries = entries.data() };
  g_bindGroupLayout = g_device.CreateBindGroupLayout(&bglDesc);

  wgpu::PipelineLayoutDescriptor plDesc{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_bindGroupLayout
  };
  g_pipelineLayout = g_device.CreatePipelineLayout(&plDesc);
}

void CreateRenderPipelineAndVertexBuffer() {
  g_renderModule = CreateWGSL(kCellRenderWGSL);

  // Vertex buffer: two triangles forming a quad in local space, same as JS
  // [-0.8,-0.8], [0.8,-0.8], [0.8,0.8], [-0.8,-0.8], [0.8,0.8], [-0.8,0.8]
  std::vector<float> verts = {
      -0.8f, -0.8f,
       0.8f, -0.8f,
       0.8f,  0.8f,
      -0.8f, -0.8f,
       0.8f,  0.8f,
      -0.8f,  0.8f
  };
  g_vertexCount = static_cast<uint32_t>(verts.size() / 2);

  wgpu::BufferDescriptor vbDesc{
      .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
      .size = sizeof(float) * verts.size(),
      .mappedAtCreation = false
  };
  g_vertexBuffer = g_device.CreateBuffer(&vbDesc);
  g_device.GetQueue().WriteBuffer(g_vertexBuffer, 0, verts.data(), vbDesc.size);

  // Vertex layout
  wgpu::VertexAttribute attrib{};
  attrib.format = wgpu::VertexFormat::Float32x2;
  attrib.offset = 0;
  attrib.shaderLocation = 0;

  wgpu::VertexBufferLayout vbl{};
  vbl.arrayStride = sizeof(float) * 2;
  vbl.stepMode = wgpu::VertexStepMode::Vertex;
  vbl.attributeCount = 1;
  vbl.attributes = &attrib;

  wgpu::ColorTargetState colorTarget{ .format = g_format };
  wgpu::FragmentState frag{
      .module = g_renderModule,
      .entryPoint = "fragmentMain",
      .targetCount = 1,
      .targets = &colorTarget
  };

  wgpu::RenderPipelineDescriptor rpDesc{};
  rpDesc.layout = g_pipelineLayout;

  rpDesc.vertex.module = g_renderModule;
  rpDesc.vertex.entryPoint = "vertexMain";
  rpDesc.vertex.bufferCount = 1;
  rpDesc.vertex.buffers = &vbl;

  rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;

  rpDesc.fragment = &frag;

  g_renderPipeline = g_device.CreateRenderPipeline(&rpDesc);
}

void CreateComputePipeline() {
  g_computeModule = CreateWGSL(kLifeComputeWGSL);

  wgpu::ComputePipelineDescriptor cpDesc{
      .layout = g_pipelineLayout,
      .compute = { .module = g_computeModule, .entryPoint = "computeMain" }
  };
  g_computePipeline = g_device.CreateComputePipeline(&cpDesc);
}

void CreateBuffersAndBindGroups() {
  // Uniform buffer with grid size [GRID_SIZE, GRID_SIZE]
  const std::array<float,2> gridData{ float(GRID_SIZE), float(GRID_SIZE) };
  wgpu::BufferDescriptor uboDesc{
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = sizeof(gridData),
      .mappedAtCreation = false
  };
  g_uniformBuffer = g_device.CreateBuffer(&uboDesc);
  g_device.GetQueue().WriteBuffer(g_uniformBuffer, 0, gridData.data(), sizeof(gridData));

  // Two storage buffers for ping pong
  const uint32_t cellCount = GRID_SIZE * GRID_SIZE;
  const uint64_t cellBytes = sizeof(uint32_t) * cellCount;

  for (int i = 0; i < 2; ++i) {
    wgpu::BufferDescriptor sbDesc{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = cellBytes,
        .mappedAtCreation = false
    };
    g_cellStateStorage[i] = g_device.CreateBuffer(&sbDesc);
  }

  // Initialise buffer 0 with random states like the JS
  {
    std::vector<uint32_t> cells(cellCount);
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < cellCount; ++i) {
      cells[i] = dist(rng) > 0.6f ? 1u : 0u;
    }
    g_device.GetQueue().WriteBuffer(g_cellStateStorage[0], 0, cells.data(), cellBytes);
  }
  // Initialise buffer 1 with a striped pattern like the JS second grid
  {
    std::vector<uint32_t> cells(cellCount);
    for (uint32_t i = 0; i < cellCount; ++i) {
      cells[i] = (i % 2u);
    }
    g_device.GetQueue().WriteBuffer(g_cellStateStorage[1], 0, cells.data(), cellBytes);
  }

  // Create two bind groups that swap input/output
  // A: in = 0, out = 1
  {
    std::array<wgpu::BindGroupEntry,3> entries{};
    entries[0] = wgpu::BindGroupEntry{ .binding = 0, .buffer = g_uniformBuffer, .offset = 0, .size = sizeof(float)*2 };
    entries[1] = wgpu::BindGroupEntry{ .binding = 1, .buffer = g_cellStateStorage[0], .offset = 0, .size = cellBytes };
    entries[2] = wgpu::BindGroupEntry{ .binding = 2, .buffer = g_cellStateStorage[1], .offset = 0, .size = cellBytes };

    wgpu::BindGroupDescriptor bgDesc{
        .layout = g_bindGroupLayout,
        .entryCount = (uint32_t)entries.size(),
        .entries = entries.data()
    };
    g_bindGroups[0] = g_device.CreateBindGroup(&bgDesc);
  }
  // B: in = 1, out = 0
  {
    std::array<wgpu::BindGroupEntry,3> entries{};
    entries[0] = wgpu::BindGroupEntry{ .binding = 0, .buffer = g_uniformBuffer, .offset = 0, .size = sizeof(float)*2 };
    entries[1] = wgpu::BindGroupEntry{ .binding = 1, .buffer = g_cellStateStorage[1], .offset = 0, .size = cellBytes };
    entries[2] = wgpu::BindGroupEntry{ .binding = 2, .buffer = g_cellStateStorage[0], .offset = 0, .size = cellBytes };

    wgpu::BindGroupDescriptor bgDesc{
        .layout = g_bindGroupLayout,
        .entryCount = 3,
        .entries = entries.data()
    };
    g_bindGroups[1] = g_device.CreateBindGroup(&bgDesc);
  }
}
static bool ScreenToCell(double sx, double sy, uint32_t& cx, uint32_t& cy) {
  if (sx < 0 || sy < 0 || sx >= WIDTH || sy >= HEIGHT) return false;

  cx = static_cast<uint32_t>(sx / double(WIDTH)  * GRID_SIZE);

  // GLFW reports y from top. Our grid.y increases upward in the shader, so invert.
  uint32_t yTop = static_cast<uint32_t>(sy / double(HEIGHT) * GRID_SIZE);
  cy = GRID_SIZE - 1 - yTop;

  if (cx >= GRID_SIZE) cx = GRID_SIZE - 1;
  if (cy >= GRID_SIZE) cy = GRID_SIZE - 1;
  return true;
}

static void PaintCell(uint32_t cx, uint32_t cy, uint32_t value) {
  const uint32_t idx = cy * GRID_SIZE + cx;
  const uint64_t byteOffset = sizeof(uint32_t) * uint64_t(idx);
  for (int b = 0; b < 2; ++b) {
    g_device.GetQueue().WriteBuffer(g_cellStateStorage[b], byteOffset, &value, sizeof(value));
  }
}

static void PaintAtCursor() {
  uint32_t cx, cy;
  if (!ScreenToCell(g_mouseX, g_mouseY, cx, cy)) return;

  if (g_brushRadius == 0) {
    if (int(cx) == g_LastCellX && int(cy) == g_LastCellY) return; // no duplicate writes
    PaintCell(cx, cy, g_paintValue);
  } else {
    // small disc brush
    for (int dy = -g_brushRadius; dy <= g_brushRadius; ++dy)
      for (int dx = -g_brushRadius; dx <= g_brushRadius; ++dx) {
        if (dx*dx + dy*dy > g_brushRadius*g_brushRadius) continue;
        int nx = int(cx) + dx, ny = int(cy) + dy;
        if (nx < 0 || ny < 0 || nx >= int(GRID_SIZE) || ny >= int(GRID_SIZE)) continue;
        PaintCell(uint32_t(nx), uint32_t(ny), g_paintValue);
      }
  }
  g_LastCellX = int(cx); g_LastCellY = int(cy);
}

static void CursorPosCallback(GLFWwindow* w, double x, double y) {
  g_mouseX = x; g_mouseY = y;
  if (g_mouseDown) PaintAtCursor(); // smooth painting while dragging
}

static void MouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
    if (action == GLFW_PRESS) {
      g_mouseDown = true;
      // Left to draw, Right to erase. Shift+Left also erases to avoid right click issues on web.
      g_paintValue = (button == GLFW_MOUSE_BUTTON_RIGHT || (mods & GLFW_MOD_SHIFT)) ? 0u : 1u;
      g_LastCellX = g_LastCellY = -1;
      PaintAtCursor(); // immediate feedback on click
    } else if (action == GLFW_RELEASE) {
      g_mouseDown = false;
    }
  }
}

/*
 * Execute a single frame
 */
void Frame() {

  #if defined(__EMSCRIPTEN__)
    // Web: fixed timestep driven by a monotonic browser clock
    static double lastMs = 0.0;
    const double nowMs = emscripten_get_now();        // or: glfwGetTime()*1000.0

    if (lastMs == 0.0)
        lastMs = nowMs;                // initialise on first call

    const double dtMs = nowMs - lastMs;
    lastMs = nowMs;

    accumulatorMs += dtMs;
    bool doUpdateNow = (accumulatorMs >= UPDATE_INTERVAL_MS);

    if (doUpdateNow)
        accumulatorMs -= UPDATE_INTERVAL_MS;  // drain only when we compute
  #else
    const auto now = std::chrono::steady_clock::now();
    const double dtMs = std::chrono::duration<double, std::milli>(now - g_lastTick).count();

    g_lastTick = now;
    g_accumulatorMs += dtMs;

    bool doUpdateNow = (g_accumulatorMs >= UPDATE_INTERVAL_MS);

    if (doUpdateNow)
        g_accumulatorMs -= UPDATE_INTERVAL_MS;   // only drain when we will compute
  #endif

    if (g_mouseDown)
        PaintAtCursor();

    wgpu::SurfaceTexture surfaceTexture;
      g_surface.GetCurrentTexture(&surfaceTexture);
      if (surfaceTexture.status == wgpu::SurfaceGetCurrentTextureStatus::Timeout) {
        return; // skip cleanly on resize
      }

      wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();

      // 1) Compute (only when scheduled). NOTE: stepIndex not incremented here.
      bool didCompute = false;
      if (doUpdateNow) {
        wgpu::ComputePassEncoder cpass = encoder.BeginComputePass();
        cpass.SetPipeline(g_computePipeline);
        cpass.SetBindGroup(0, g_bindGroups[g_stepIndex % 2]);  // in=step%2, out=other
        const uint32_t workgroups = (GRID_SIZE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        cpass.DispatchWorkgroups(workgroups, workgroups, 1);
        cpass.End();
        didCompute = true;
      }

      // 2) Render every frame, sampling the output buffer of the current step.
      {
        wgpu::RenderPassColorAttachment color{
          .view = surfaceTexture.texture.CreateView(),
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = {0.0f, 0.0f, 0.4f, 1.0f}
        };
        wgpu::RenderPassDescriptor rdesc{ .colorAttachmentCount = 1, .colorAttachments = &color };
        auto pass = encoder.BeginRenderPass(&rdesc);
        pass.SetPipeline(g_renderPipeline);
        pass.SetBindGroup(0, g_bindGroups[(g_stepIndex + 1) % 2]); // read freshly written side
        pass.SetVertexBuffer(0, g_vertexBuffer, 0, WGPU_WHOLE_SIZE);
        pass.Draw(g_vertexCount, GRID_SIZE * GRID_SIZE, 0, 0);
        pass.End();
      }

      wgpu::CommandBuffer commandBuffer = encoder.Finish();

      g_device.GetQueue().Submit(1, &commandBuffer);

    #ifndef __EMSCRIPTEN__
      g_surface.Present();
      g_instance.ProcessEvents();
    #endif

      // 3) Advance AFTER submit, and only if we computed this frame.
      if (didCompute)
          ++g_stepIndex;
}

void Start() {
  if (!glfwInit()) {
    return;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWwindow* window =
      glfwCreateWindow(WIDTH, HEIGHT, "WebGPU Life (Dawn + GLFW)", nullptr, nullptr);

  g_window = window;

  glfwSetCursorPosCallback(window, CursorPosCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);

  g_surface = wgpu::glfw::CreateSurfaceForWindow(g_instance, window);


  ConfigureSurface();

  CreateBindGroupLayoutAndPipelineLayout();

  CreateRenderPipelineAndVertexBuffer();

  CreateComputePipeline();

  CreateBuffersAndBindGroups();

  g_lastTick = std::chrono::steady_clock::now();
  g_accumulatorMs = 0.0;

#if defined(__EMSCRIPTEN__)
  emscripten_set_main_loop(Frame, 0, false);
#else
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    Frame();
  }
#endif
}

int main() {
  Init();
  Start();
}
