<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#hlsl-in-vulkan">HLSL in Vulkan</a>
<ul class="sectlevel1">
<li><a href="#applications-pov">1. From the application&#8217;s point-of-view</a></li>
<li><a href="#hlsl-spirv-mapping-manual">2. HLSL to SPIR-V feature mapping manual</a></li>
<li><a href="#vk-namespace">3. The Vulkan HLSL namespace</a></li>
<li><a href="#syntax-comparison">4. Syntax comparison</a>
<ul class="sectlevel2">
<li><a href="#_glsl">4.1. GLSL</a></li>
<li><a href="#_hlsl">4.2. HLSL</a></li>
</ul>
</li>
<li><a href="#DirectXShaderCompiler">5. DirectXShaderCompiler (DXC)</a>
<ul class="sectlevel2">
<li><a href="#_where_to_get">5.1. Where to get</a></li>
<li><a href="#_offline_compilation_using_the_stand_alone_compiler">5.2. Offline compilation using the stand-alone compiler</a></li>
<li><a href="#_runtime_compilation_using_the_library">5.3. Runtime compilation using the library</a></li>
<li><a href="#_vulkan_shader_stage_to_hlsl_target_shader_profile_mapping">5.4. Vulkan shader stage to HLSL target shader profile mapping</a></li>
</ul>
</li>
<li><a href="#_shader_model_coverage">6. Shader model coverage</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/hlsl.html
---</p>
</div>
<h1 id="hlsl-in-vulkan" class="sect0">HLSL in Vulkan</h1>
<div class="paragraph">
<p>Vulkan does not directly consume shaders in a human-readable text format, but instead uses <a href="what_is_spirv.html">SPIR-V</a> as an intermediate representation. This opens the option to use shader languages other than e.g. GLSL, as long as they can target the Vulkan SPIR-V environment.</p>
</div>
<div class="paragraph">
<p>One such language is the High Level Shading Language (HLSL) by Microsoft, used by DirectX. Thanks to <a href="https://www.khronos.org/blog/hlsl-first-class-vulkan-shading-language">recent additions to Vulkan 1.2</a> it is now considered a first class shading language for Vulkan that can be used just as easily as GLSL.</p>
</div>
<div class="paragraph">
<p>With <a href="https://github.com/microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#unsupported-hlsl-features">a few exceptions</a>, all Vulkan features and shader stages available with GLSL can be used with HLSL too, including recent Vulkan additions like hardware accelerated ray tracing. On the other hand, HLSL to SPIR-V supports Vulkan exclusive features that are not (yet) available in DirectX.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/what_is_spirv_dxc.png" alt="what_is_spriv_dxc.png">
</div>
</div>
<div class="sect1">
<h2 id="applications-pov">1. From the application&#8217;s point-of-view</h2>
<div class="sectionbody">
<div class="paragraph">
<p>From the application&#8217;s point-of-view, using HLSL is exactly the same as using GLSL. As the application always consumes shaders in the SPIR-V format, the only difference is in the tooling to generate the SPIR-V shaders from the desired shading language.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="hlsl-spirv-mapping-manual">2. HLSL to SPIR-V feature mapping manual</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A great starting point on using HLSL in Vulkan via SPIR-V is the <a href="https://github.com/microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst">HLSL to SPIR-V feature mapping manual</a>. It contains detailed information on semantics, syntax, supported features and extensions and much more and is a must-read. The <a href="decoder_ring.html">decoder ring</a> also has a translation table for concepts and terms used in Vulkan an DirectX.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="vk-namespace">3. The Vulkan HLSL namespace</h2>
<div class="sectionbody">
<div class="paragraph">
<p>To make HLSL compatible with Vulkan, an <a href="https://github.com/microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#the-implicit-vk-namespace)">implicit namespace</a> has been introduced that provides an interface for for Vulkan-specific features.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="syntax-comparison">4. Syntax comparison</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Similar to regular programming languages, HLSL and GLSL differ in their syntax. While GLSL is more procedural (like C), HLSL is more object-oriented (like C++).</p>
</div>
<div class="paragraph">
<p>Here is the same shader written in both languages to give quick comparison on how they basically differ, including the aforementioned namespace that e.g. adds explicit locations:</p>
</div>
<div class="sect2">
<h3 id="_glsl">4.1. GLSL</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inColor;

layout (binding = 0) uniform UBO
{
	mat4 projectionMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
} ubo;

layout (location = 0) out vec3 outColor;

void main()
{
	outColor = inColor * float(gl_VertexIndex);
	gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPosition.xyz, 1.0);
}</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_hlsl">4.2. HLSL</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-hlsl" data-lang="hlsl">struct VSInput
{
[[vk::location(0)]] float3 Position : POSITION0;
[[vk::location(1)]] float3 Color : COLOR0;
};

struct UBO
{
	float4x4 projectionMatrix;
	float4x4 modelMatrix;
	float4x4 viewMatrix;
};

cbuffer ubo : register(b0, space0) { UBO ubo; }

struct VSOutput
{
	float4 Pos : SV_POSITION;
[[vk::location(0)]] float3 Color : COLOR0;
};

VSOutput main(VSInput input, uint VertexIndex : SV_VertexID)
{
	VSOutput output = (VSOutput)0;
	output.Color = input.Color * float(VertexIndex);
	output.Position = mul(ubo.projectionMatrix, mul(ubo.viewMatrix, mul(ubo.modelMatrix, float4(input.Position.xyz, 1.0))));
	return output;
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>Aside from the syntax differences, built-ins use HLSL names. E.g. <code>gl_vertex</code> becomes <code>VertexIndex</code> in HLSL. A list of GLSL to HLSL built-in mappings can be found <a href="https://anteru.net/blog/2016/mapping-between-HLSL-and-GLSL/">here</a>.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="DirectXShaderCompiler">5. DirectXShaderCompiler (DXC)</h2>
<div class="sectionbody">
<div class="paragraph">
<p>As is the case with GLSL to SPIR-V, to use HLSL with Vulkan, a shader compiler is required. Whereas <a href="https://github.com/KhronosGroup/glslang">glslang</a> is the reference GLSL to SPIR-V compiler, the <a href="https://github.com/microsoft/DirectXShaderCompiler">DirectXShaderCompiler</a> (DXC) is the reference HLSL to SPIR-V compiler. Thanks to open source contributions, the SPIR-V backend of DXC is now supported and enabled in official release builds and can be used out-of-the box. While other shader compiling tools like <a href="https://github.com/KhronosGroup/glslang/wiki/HLSL-FAQ">glslang</a> also offer HLSL support, DXC has the most complete and up-to-date support and is the recommended way of generating SPIR-V from HLSL.</p>
</div>
<div class="sect2">
<h3 id="_where_to_get">5.1. Where to get</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/">LunarG Vulkan SDK</a> includes pre-compiled DXC binaries, libraries and headers to get you started. If you&#8217;re looking for the latest releases, check the <a href="https://github.com/microsoft/DirectXShaderCompiler/releases">official DXC repository</a>.</p>
</div>
</div>
<div class="sect2">
<h3 id="_offline_compilation_using_the_stand_alone_compiler">5.2. Offline compilation using the stand-alone compiler</h3>
<div class="paragraph">
<p>Compiling a shader offline via the pre-compiled dxc binary is similar to compiling with glslang:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code>dxc.exe -spirv -T vs_6_0 -E main .\triangle.vert -Fo .\triangle.vert.spv</code></pre>
</div>
</div>
<div class="paragraph">
<p><code>-T</code> selects the profile to compile the shader against (<code>vs_6_0</code> = Vertex shader model 6, <code>ps_6_0</code> = Pixel/fragment shader model 6, etc.).</p>
</div>
<div class="paragraph">
<p><code>-E</code> selects the main entry point for the shader.</p>
</div>
<div class="paragraph">
<p>Extensions are implicitly enabled based on feature usage, but can also be explicitly specified:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code>dxc.exe -spirv -T vs_6_1 -E main .\input.vert -Fo .\output.vert.spv -fspv-extension=SPV_EXT_descriptor_indexing</code></pre>
</div>
</div>
<div class="paragraph">
<p>The resulting SPIR-V can then be directly loaded, same as SPIR-V generated from GLSL.</p>
</div>
</div>
<div class="sect2">
<h3 id="_runtime_compilation_using_the_library">5.3. Runtime compilation using the library</h3>
<div class="paragraph">
<p>DXC can also be integrated into a Vulkan application using the DirectX Compiler API. This allows for runtime compilation of shaders. Doing so requires you to include the <code>dxcapi.h</code> header and link against the <code>dxcompiler</code> library. The easiest way is using the dynamic library and distributing it with your application (e.g. <code>dxcompiler.dll</code> on Windows).</p>
</div>
<div class="paragraph">
<p>Compiling HLSL to SPIR-V at runtime then is pretty straight-forward:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">#include "include/dxc/dxcapi.h"

...

HRESULT hres;

// Initialize DXC library
CComPtr&lt;IDxcLibrary&gt; library;
hres = DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&amp;library));
if (FAILED(hres)) {
	throw std::runtime_error("Could not init DXC Library");
}

// Initialize the DXC compiler
CComPtr&lt;IDxcCompiler&gt; compiler;
hres = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&amp;compiler));
if (FAILED(hres)) {
	throw std::runtime_error("Could not init DXC Compiler");
}

// Load the HLSL text shader from disk
uint32_t codePage = CP_UTF8;
CComPtr&lt;IDxcBlobEncoding&gt; sourceBlob;
hres = library-&gt;CreateBlobFromFile(filename.c_str(), &amp;codePage, &amp;sourceBlob);
if (FAILED(hres)) {
	throw std::runtime_error("Could not load shader file");
}

// Set up arguments to be passed to the shader compiler

// Tell the compiler to output SPIR-V
std::vector&lt;LPCWSTR&gt; arguments;
arguments.push_back(L"-spirv");

// Select target profile based on shader file extension
LPCWSTR targetProfile{};
size_t idx = filename.rfind('.');
if (idx != std::string::npos) {
	std::wstring extension = filename.substr(idx + 1);
	if (extension == L"vert") {
		targetProfile = L"vs_6_1";
	}
	if (extension == L"frag") {
		targetProfile = L"ps_6_1";
	}
	// Mapping for other file types go here (cs_x_y, lib_x_y, etc.)
}

// Compile shader
CComPtr&lt;IDxcOperationResult&gt; resultOp;
hres = compiler-&gt;Compile(
	sourceBlob,
	nullptr,
	L"main",
	targetProfile,
	arguments.data(),
	(uint32_t)arguments.size(),
	nullptr,
	0,
	nullptr,
	&amp;resultOp);

if (SUCCEEDED(hres)) {
	resultOp-&gt;GetStatus(&amp;hres);
}

// Output error if compilation failed
if (FAILED(hres) &amp;&amp; (resultOp)) {
	CComPtr&lt;IDxcBlobEncoding&gt; errorBlob;
	hres = resultOp-&gt;GetErrorBuffer(&amp;errorBlob);
	if (SUCCEEDED(hres) &amp;&amp; errorBlob) {
		std::cerr &lt;&lt; "Shader compilation failed :\n\n" &lt;&lt; (const char*)errorBlob-&gt;GetBufferPointer();
		throw std::runtime_error("Compilation failed");
	}
}

// Get compilation result
CComPtr&lt;IDxcBlob&gt; code;
resultOp-&gt;GetResult(&amp;code);

// Create a Vulkan shader module from the compilation result
VkShaderModuleCreateInfo shaderModuleCI{};
shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
shaderModuleCI.codeSize = code-&gt;GetBufferSize();
shaderModuleCI.pCode = (uint32_t*)code-&gt;GetBufferPointer();
VkShaderModule shaderModule;
vkCreateShaderModule(device, &amp;shaderModuleCI, nullptr, &amp;shaderModule);</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_vulkan_shader_stage_to_hlsl_target_shader_profile_mapping">5.4. Vulkan shader stage to HLSL target shader profile mapping</h3>
<div class="paragraph">
<p>When compiling HLSL with DXC you need to select a target shader profile. The name for a profile consists of the shader type and the desired shader model.</p>
</div>
<table class="tableblock frame-all grid-all stretch">
<colgroup>
<col style="width: 33.3333%;">
<col style="width: 33.3333%;">
<col style="width: 33.3334%;">
</colgroup>
<thead>
<tr>
<th class="tableblock halign-left valign-top">Vulkan shader stage</th>
<th class="tableblock halign-left valign-top">HLSL target shader profile</th>
<th class="tableblock halign-left valign-top">Remarks</th>
</tr>
</thead>
<tbody>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_VERTEX_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>vs</code></p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>hs</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Hull shader in HLSL terminology</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>ds</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Domain shader in HLSL terminology</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_GEOMETRY_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>gs</code></p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_FRAGMENT_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>ps</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Pixel shader in HLSL terminology</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_COMPUTE_BIT</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>cs</code></p></td>
<td class="tableblock halign-left valign-top"></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_RAYGEN_BIT_KHR</code>,
<code>VK_SHADER_STAGE_ANY_HIT_BIT_KHR</code>,
<code>VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR</code>,
<code>VK_SHADER_STAGE_MISS_BIT_KHR</code>,
<code>VK_SHADER_STAGE_INTERSECTION_BIT_KHR</code>,
<code>VK_SHADER_STAGE_CALLABLE_BIT_KHR</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>lib</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">All raytracing related shaders are built using the <code>lib</code> shader target profile and must use at least shader model 6.3 (e.g. <code>lib_6_3</code>).</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_TASK_BIT_NV</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>as</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Amplification shader in HLSL terminology. Must use at least shader model 6.5 (e.g. <code>as_6_5</code>).</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VK_SHADER_STAGE_MESH_BIT_NV</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>ms</code></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Must use at least shader model 6.5 (e.g. <code>ms_6_5</code>).</p></td>
</tr>
</tbody>
</table>
<div class="paragraph">
<p>So if you for example you want to compile a compute shader targeting shader model 6.6 features, the target shader profile would be <code>cs_6_6</code>. For a ray tracing any hit shader it would be <code>lib_6_3</code>.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_shader_model_coverage">6. Shader model coverage</h2>
<div class="sectionbody">
<div class="paragraph">
<p>DirectX and HLSL use a fixed shader model notion to describe the supported feature set. This is different from Vulkan and SPIR-V&#8217;s flexible extension based way of adding features to shaders. The following table tries to list Vulkan&#8217;s coverage for the HLSL shader models without guarantee of completeness:</p>
</div>
<table class="tableblock frame-all grid-all stretch">
<caption class="title">Table 1. Shader models</caption>
<colgroup>
<col style="width: 33.3333%;">
<col style="width: 33.3333%;">
<col style="width: 33.3334%;">
</colgroup>
<thead>
<tr>
<th class="tableblock halign-left valign-top">Shader Model</th>
<th class="tableblock halign-left valign-top">Supported</th>
<th class="tableblock halign-left valign-top">Remarks</th>
</tr>
</thead>
<tbody>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock">Shader Model 5.1 and below</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Excluding features without Vulkan equivalent</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.0">Shader Model 6.0</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Wave intrinsics, 64-bit integers</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.1">Shader Model 6.1</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">SV_ViewID, SV_Barycentrics</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.2">Shader Model 6.2</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">16-bit types, Denorm mode</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.3">Shader Model 6.3</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Hardware accelerated ray tracing</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.4">Shader Model 6.4</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">✔</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">Shader integer dot product, SV_ShadingRate</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.5">Shader Model 6.5</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">❌ (partially)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">DXR1.1 (KHR ray tracing), Mesh and Amplification shaders, additional Wave intrinsics</p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://github.com/microsoft/DirectXShaderCompiler/wiki/Shader-Model-6.6">Shader Model 6.6</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">❌ (partially)</p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock">VK_NV_compute_shader_derivatives, VK_KHR_shader_atomic_int64</p></td>
</tr>
</tbody>
</table>
</div>
</div>