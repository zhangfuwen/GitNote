<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#development-tools">Development Tools</a>
<ul class="sectlevel1">
<li><a href="#_vulkan_layers">1. Vulkan Layers</a>
<ul class="sectlevel2">
<li><a href="#_khronos_layers">1.1. Khronos Layers</a></li>
<li><a href="#_vulkan_sdk_layers">1.2. Vulkan SDK layers</a></li>
<li><a href="#_vulkan_third_party_layers">1.3. Vulkan Third-party layers</a></li>
</ul>
</li>
<li><a href="#_debugging">2. Debugging</a></li>
<li><a href="#_profiling">3. Profiling</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/development_tools.html
---</p>
</div>
<h1 id="development-tools" class="sect0">Development Tools</h1>
<div class="paragraph">
<p>The Vulkan ecosystem consists of many tools for development. This is <strong>not</strong> a full list and this is offered as a good starting point for many developers. Please continue to do your own research and search for other tools as the development ecosystem is much larger than what can reasonably fit on a single Markdown page.</p>
</div>
<div class="paragraph">
<p>Khronos hosts <a href="https://github.com/KhronosGroup/Vulkan-Samples">Vulkan Samples</a>, a collection of code and tutorials that demonstrates API usage and explains the implementation of performance best practices.</p>
</div>
<div class="paragraph">
<p>LunarG is privately sponsored to develop and maintain Vulkan ecosystem components and is currently the curator for the <a href="https://github.com/KhronosGroup/Vulkan-Loader">Vulkan Loader</a> and <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers">Vulkan Validation Layers</a> Khronos Group repositories. In addition, LunarG delivers the <a href="https://vulkan.lunarg.com/">Vulkan SDK</a> and develops other key tools such as the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vkconfig.html">Vulkan Configurator</a> and <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/capture_tools.html">GFXReconstruct</a>.</p>
</div>
<div class="sect1">
<h2 id="_vulkan_layers">1. Vulkan Layers</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Layers are optional components that augment the Vulkan system. They can intercept, evaluate, and modify existing Vulkan functions on their way from the application down to the hardware. Layers are implemented as libraries that can be enabled and configured using <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vkconfig.html">Vulkan Configurator</a>.</p>
</div>
<div class="sect2">
<h3 id="_khronos_layers">1.1. Khronos Layers</h3>
<div class="ulist">
<ul>
<li>
<p><a href="validation_overview.html#khronos-validation-layer"><code>VK_LAYER_KHRONOS_validation</code></a>, the Khronos Validation Layer.
It is every developer&#8217;s first layer of defense when debugging their Vulkan application and this is the reason it is at the top of this list. Read the <a href="validation_overview.html#validation-overview">Validation Overview chapter</a> for more details.
The validation layer included multiple features:</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/synchronization_usage.html">Synchronization Validation</a>: Identify resource access conflicts due to missing or incorrect synchronization operations between actions (Draw, Copy, Dispatch, Blit) reading or writing the same regions of memory.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/gpu_validation.html">GPU-Assisted Validation</a>: Instrument shader code to perform run-time checks for error conditions produced during shader execution.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/debug_printf.html">Shader printf</a>: Debug shader code by &#8220;printing&#8221; any values of interest to the debug callback or stdout.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/best_practices.html">Best Practices Warnings</a>: Highlights potential performance issues, questionable usage patterns, common mistakes.</p>
</li>
</ul>
</div>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/view/latest/windows/synchronization2_layer.html"><code>VK_LAYER_KHRONOS_synchronization2</code></a>, the Khronos Synchronization2 layer.
The <code>VK_LAYER_KHRONOS_synchronization2</code> layer implements the <code>VK_KHR_synchronization2</code> extension. By default, it will disable itself if the underlying driver provides the extension.</p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_vulkan_sdk_layers">1.2. Vulkan SDK layers</h3>
<div class="paragraph">
<p>Besides the Khronos Layers, the Vulkan SDK included additional useful platform independent layers.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/api_dump_layer.html"><code>VK_LAYER_LUNARG_api_dump</code></a>, a layer to log Vulkan API calls.
The API dump layer prints API calls, parameters, and values to the identified output stream.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/capture_tools.html"><code>VK_LAYER_LUNARG_gfxreconstruct</code></a>, a layer for capturing frames created with Vulkan.
This layer is a part of GFXReconstruct, a software for capturing and replaying Vulkan API calls. Full Android support is also available at <a href="https://github.com/LunarG/gfxreconstruct" class="bare">https://github.com/LunarG/gfxreconstruct</a></p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/device_simulation_layer.html"><code>VK_LAYER_LUNARG_device_simulation</code></a>, a layer to test Vulkan applications portability.
The device simulation layer can be used to test whether a Vulkan application would run on a Vulkan device with lower capabilities.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/screenshot_layer.html"><code>VK_LAYER_LUNARG_screenshot</code></a>, a screenshot layer.
Captures the rendered image of a Vulkan application to a viewable image.</p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/monitor_layer.html"><code>VK_LAYER_LUNARG_monitor</code></a>, a framerate monitor layer.
Display the Vulkan application FPS in the window title bar to give a hint about the performance.</p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_vulkan_third_party_layers">1.3. Vulkan Third-party layers</h3>
<div class="paragraph">
<p>There are also other publicly available layers that can be used to help in development.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://github.com/ARM-software/perfdoc"><code>VK_LAYER_ARM_mali_perf_doc</code></a>, the ARM PerfDoc layer.
Checks Vulkan applications for best practices on Arm Mali devices.</p>
</li>
<li>
<p><a href="https://github.com/powervr-graphics/perfdoc"><code>VK_LAYER_IMG_powervr_perf_doc</code></a>, the PowerVR PerfDoc layer.
Checks Vulkan applications for best practices on Imagination Technologies PowerVR devices.</p>
</li>
<li>
<p><a href="https://developer.qualcomm.com/software/adreno-gpu-sdk/tools"><code>VK_LAYER_adreno</code></a>, the Vulkan Adreno Layer.
Checks Vulkan applications for best practices on Qualcomm Adreno devices.</p>
</li>
</ul>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_debugging">2. Debugging</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Debugging something running on a GPU can be incredibly hard, luckily there are tools out there to help.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://developer.arm.com/tools-and-software/graphics-and-gaming/arm-mobile-studio/components/graphics-analyzer">Arm Graphics Analyzer</a></p>
</li>
<li>
<p><a href="https://github.com/google/gapid">GAPID</a></p>
</li>
<li>
<p><a href="https://developer.nvidia.com/nsight-graphics">NVIDIA Nsight</a></p>
</li>
<li>
<p><a href="https://developer.imaginationtech.com">PVRCarbon</a></p>
</li>
<li>
<p><a href="https://renderdoc.org/">RenderDoc</a></p>
</li>
<li>
<p><a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/capture_tools.html">GFXReconstruct</a></p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_profiling">3. Profiling</h2>
<div class="sectionbody">
<div class="paragraph">
<p>With anything related to a GPU it is best to not assume and profile when possible. Here is a list of known profilers to aid in your development.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://gpuopen.com/rgp/">AMD Radeon GPU Profiler</a> - Low-level performance analysis tool for AMD Radeon GPUs.</p>
</li>
<li>
<p><a href="https://developer.arm.com/tools-and-software/graphics-and-gaming/arm-mobile-studio/components/streamline-performance-analyzer">Arm Streamline Performance Analyzer</a> - Visualize the performance of mobile games and applications for a broad range of devices, using Arm Mobile Studio.</p>
</li>
<li>
<p><a href="https://www.intel.com/content/www/us/en/developer/tools/graphics-performance-analyzers/overview.html">Intel&#174; GPA</a> - Intel&#8217;s Graphics Performance Analyzers that supports capturing and analyzing multi-frame streams of Vulkan apps.</p>
</li>
<li>
<p><a href="https://github.com/GPUOpen-Tools/OCAT">OCAT</a> - The Open Capture and Analytics Tool (OCAT) provides an FPS overlay and performance measurement for D3D11, D3D12, and Vulkan.</p>
</li>
<li>
<p><a href="https://developer.imaginationtech.com">PVRTune</a></p>
</li>
<li>
<p><a href="https://developer.qualcomm.com/software/snapdragon-profiler">Qualcomm Snapdragon Profiler</a> - Profiling tool targeting Adreno GPU.</p>
</li>
<li>
<p><a href="https://www.vktracer.com">VKtracer</a> - Cross-vendor and cross-platform profiler.</p>
</li>
</ul>
</div>
</div>
</div>