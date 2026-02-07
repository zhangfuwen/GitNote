<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#shader-features">Shader Features</a>
<ul class="sectlevel1">
<li><a href="#VK_KHR_spirv_1_4">1. VK_KHR_spirv_1_4</a></li>
<li><a href="#VK_KHR_16bit_storage">2. VK_KHR_8bit_storage and VK_KHR_16bit_storage</a></li>
<li><a href="#VK_KHR_shader_float16_int8">3. VK_KHR_shader_float16_int8</a></li>
<li><a href="#VK_KHR_shader_float_controls">4. VK_KHR_shader_float_controls</a></li>
<li><a href="#VK_KHR_storage_buffer_storage_class">5. VK_KHR_storage_buffer_storage_class</a></li>
<li><a href="#VK_KHR_variable_pointers">6. VK_KHR_variable_pointers</a></li>
<li><a href="#VK_KHR_vulkan_memory_model">7. VK_KHR_vulkan_memory_model</a></li>
<li><a href="#VK_EXT_shader_viewport_index_layer">8. VK_EXT_shader_viewport_index_layer</a></li>
<li><a href="#VK_KHR_shader_draw_parameters">9. VK_KHR_shader_draw_parameters</a></li>
<li><a href="#VK_EXT_shader_stencil_export">10. VK_EXT_shader_stencil_export</a></li>
<li><a href="#VK_EXT_shader_demote_to_helper_invocation">11. VK_EXT_shader_demote_to_helper_invocation</a></li>
<li><a href="#VK_KHR_shader_clock">12. VK_KHR_shader_clock</a></li>
<li><a href="#VK_KHR_shader_non_semantic_info">13. VK_KHR_shader_non_semantic_info</a></li>
<li><a href="#VK_KHR_shader_terminate_invocation">14. VK_KHR_shader_terminate_invocation</a></li>
<li><a href="#VK_KHR_workgroup_memory_explicit_layout">15. VK_KHR_workgroup_memory_explicit_layout</a></li>
<li><a href="#VK_KHR_zero_initialize_workgroup_memory">16. VK_KHR_zero_initialize_workgroup_memory</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/shader_features.html
layout: default
---</p>
</div>
<h1 id="shader-features" class="sect0">Shader Features</h1>
<div class="paragraph">
<p>There are various reasons why every part of SPIR-V was not exposed to Vulkan 1.0. Over time the Vulkan Working Group has identified use cases where it makes sense to expose a new SPIR-V feature.</p>
</div>
<div class="paragraph">
<p>Some of the following extensions were added alongside a SPIR-V extension. For example, the <code>VK_KHR_8bit_storage</code> extension was created in parallel with <code>SPV_KHR_8bit_storage</code>. The Vulkan extension only purpose is to allow an application to query for SPIR-V support in the implementation. The SPIR-V extension is there to define the changes made to the SPIR-V intermediate representation.</p>
</div>
<div class="paragraph">
<p>For details how to use SPIR-V extension please read the <a href="../spirv_extensions.html">dedicated Vulkan Guide chapter</a>.</p>
</div>
<div class="sect1">
<h2 id="VK_KHR_spirv_1_4">1. VK_KHR_spirv_1_4</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension is designed for a Vulkan 1.1 implementations to expose the SPIR-V 1.4 feature set. Vulkan 1.1 only requires SPIR-V 1.3 and some use cases were found where an implementation might not upgrade to Vulkan 1.2, but still want to offer SPIR-V 1.4 features.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_16bit_storage">2. VK_KHR_8bit_storage and VK_KHR_16bit_storage</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_8bit_storage.html">SPV_KHR_8bit_storage</a></p>
</div>
<div class="paragraph">
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_16bit_storage.html">SPV_KHR_16bit_storage</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt">GLSL - GL_EXT_shader_16bit_storage</a> defines both</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Both <code>VK_KHR_8bit_storage</code> (promoted in Vulkan 1.2) and <code>VK_KHR_16bit_storage</code> (promoted in Vulkan 1.1) were added to allow the ability to use small values as input or output to a SPIR-V storage object. Prior to these extensions, all UBO, SSBO, and push constants needed to consume at least 4 bytes. With this, an application can now use 8-bit or 16-bit values directly from a buffer. It is also commonly paired with the use of <code>VK_KHR_shader_float16_int8</code> as this extension only deals with the storage interfaces.</p>
</div>
<div class="paragraph">
<p>The following is an example of using <code>SPV_KHR_8bit_storage</code> with the GLSL extension:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450

// Without 8-bit storage each block variable has to be 32-byte wide
layout (set = 0, binding = 0) readonly buffer StorageBuffer {
    uint data; // 0x0000AABB
} ssbo;

void main() {
    uint a = ssbo.data &amp; 0x0000FF00;
    uint b = ssbo.data &amp; 0x000000FF;
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>With the extension</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version 450
#extension GL_EXT_shader_8bit_storage : enable

layout (set = 0, binding = 0) readonly buffer StorageBuffer {
    uint8_t dataA; // 0xAA
    uint8_t dataB; // 0xBB
} ssbo;

void main() {
    uint a = uint(ssbo.dataA);
    uint b = uint(ssbo.dataB);
}</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_float16_int8">3. VK_KHR_shader_float16_int8</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt">GLSL - GL_EXT_shader_explicit_arithmetic_types</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows the use of 8-bit integer types or 16-bit floating-point types for arithmetic operations. This does not allow for 8-bit integer types or 16-bit floating-point types in any shader input and output interfaces and therefore is commonly paired with the use of <code>VK_KHR_8bit_storage</code> and <code>VK_KHR_16bit_storage</code>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_float_controls">4. VK_KHR_shader_float_controls</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
<div class="paragraph">
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_float_controls.html">SPV_KHR_float_controls</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows the ability to set how rounding of floats are handled. The <code>VkPhysicalDeviceFloatControlsProperties</code> shows the full list of features that can be queried. This is useful when converting OpenCL kernels to Vulkan.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_storage_buffer_storage_class">5. VK_KHR_storage_buffer_storage_class</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.1</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_storage_buffer_storage_class.html">SPV_KHR_storage_buffer_storage_class</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Originally SPIR-V combined both UBO and SSBO into the 'Uniform' storage classes and differentiated them only through extra decorations. Because some hardware treats UBO an SSBO as two different storage objects, the SPIR-V wanted to reflect that. This extension serves the purpose of extending SPIR-V to have a new <code>StorageBuffer</code> class.</p>
</div>
<div class="paragraph">
<p>An example of this can be seen if you take the following GLSL shader snippet:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) buffer ssbo {
    int x;
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>If you target Vulkan 1.0 (which requires SPIR-V 1.0), using glslang <code>--target-env vulkan1.0</code>, you will get something like:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">    Decorate 7(ssbo) BufferBlock
8:  TypePointer Uniform 7(ssbo)
9:  8(ptr) Variable Uniform
12: TypePointer Uniform 6(int)</code></pre>
</div>
</div>
<div class="paragraph">
<p>Since <code>SPV_KHR_storage_buffer_storage_class</code> was added to SPIR-V 1.3, if you target Vulkan 1.1 (which requires SPIR-V 1.3) ,using glslang <code>--target-env vulkan1.1</code>, it will make use of the new <code>StorageBuffer</code> class.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">    Decorate 7(ssbo) Block
8:  TypePointer StorageBuffer 7(ssbo)
9:  8(ptr) Variable StorageBuffer
12: TypePointer StorageBuffer 6(int)</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_variable_pointers">6. VK_KHR_variable_pointers</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.1</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_variable_pointers.html">SPV_KHR_variable_pointers</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>A <code>Variable pointer</code> is defined in SPIR-V as</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>A pointer of logical pointer type that results from one of the following instructions: <code>OpSelect</code>, <code>OpPhi</code>, <code>OpFunctionCall</code>, <code>OpPtrAccessChain</code>, <code>OpLoad</code>, or <code>OpConstantNull</code></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>When this extension is enabled, invocation-private pointers can be dynamic and non-uniform. Without this extension a variable pointer must be selected from pointers pointing into the same structure or be <code>OpConstantNull</code>.</p>
</div>
<div class="paragraph">
<p>This extension has two levels to it. The first is the <code>variablePointersStorageBuffer</code> feature bit which allows implementations to support the use of variable pointers into a SSBO only. The <code>variablePointers</code> feature bit allows the use of variable pointers outside the SSBO as well.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_vulkan_memory_model">7. VK_KHR_vulkan_memory_model</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
<div class="paragraph">
<p><a href="https://www.khronos.org/blog/comparing-the-vulkan-spir-v-memory-model-to-cs/">Comparing the Vulkan SPIR-V memory model to C&#8217;s</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#memory-model">Vulkan Memory Model</a> formally defines how to synchronize memory accesses to the same memory locations performed by multiple shader invocations and this extension exposes a boolean to let implementations to indicate support for it. This is important because with many things targeting Vulkan/SPIR-V it is important that any memory transfer operations an application might attempt to optimize doesn&#8217;t break across implementations.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_viewport_index_layer">8. VK_EXT_shader_viewport_index_layer</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_viewport_index_layer.html">SPV_EXT_shader_viewport_index_layer</a></p>
</div>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_viewport_layer_array.txt">GLSL - GL_ARB_shader_viewport_layer_array</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension adds the <code>ViewportIndex</code>, <code>Layer</code> built-in for exporting from vertex or tessellation shaders.</p>
</div>
<div class="paragraph">
<p>In GLSL these are represented by <code>gl_ViewportIndex</code> and <code>gl_Layer</code> built-ins.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_draw_parameters">9. VK_KHR_shader_draw_parameters</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.1</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_shader_draw_parameters.html">SPV_KHR_shader_draw_parameters</a></p>
</div>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_draw_parameters.txt">GLSL - GL_ARB_shader_draw_parameters</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension adds the <code>BaseInstance</code>, <code>BaseVertex</code>, and <code>DrawIndex</code> built-in for vertex shaders. This was added as there are legitimate use cases for both inclusion and exclusion of the <code>BaseVertex</code> or <code>BaseInstance</code> parameters in <code>VertexId</code> and <code>InstanceId</code>, respectively.</p>
</div>
<div class="paragraph">
<p>In GLSL these are represented by <code>gl_BaseInstanceARB</code>, <code>gl_BaseVertexARB</code> and <code>gl_BaseInstanceARB</code> built-ins.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_stencil_export">10. VK_EXT_shader_stencil_export</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_stencil_export.html">SPV_EXT_shader_stencil_export</a></p>
</div>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_stencil_export.txt">GLSL - GL_ARB_shader_stencil_export</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows a shader to generate the stencil reference value per invocation. When stencil testing is enabled, this allows the test to be performed against the value generated in the shader.</p>
</div>
<div class="paragraph">
<p>In GLSL this is represented by a <code>out int gl_FragStencilRefARB</code> built-in.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_demote_to_helper_invocation">11. VK_EXT_shader_demote_to_helper_invocation</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.3</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_demote_to_helper_invocation.html">SPV_EXT_demote_to_helper_invocation</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_demote_to_helper_invocation.txt">GLSL - GL_EXT_demote_to_helper_invocation</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension was created to help with matching the HLSL <code>discard</code> instruction in SPIR-V by adding a <code>demote</code> keyword. When using <code>demote</code> in a fragment shader invocation it becomes a helper invocation. Any stores to memory after this instruction are suppressed and the fragment does not write outputs to the framebuffer.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_clock">12. VK_KHR_shader_clock</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_shader_clock.html">SPV_KHR_shader_clock</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_realtime_clock.txt">GLSL - GL_EXT_shader_realtime_clock</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows the shader to read the value of a monotonically incrementing counter provided by the implementation. This can be used as one possible method for debugging by tracking the order of when an invocation executes the instruction. It is worth noting that the addition of the <code>OpReadClockKHR</code> alters the shader one might want to debug. This means there is a certain level of accuracy representing the order as if the instructions did not exists.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_non_semantic_info">13. VK_KHR_shader_non_semantic_info</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.3</p>
</div>
<div class="paragraph">
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_shader_clock.html">SPV_KHR_non_semantic_info</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension exposes <a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_non_semantic_info.html">SPV_KHR_non_semantic_info</a> which <a href="https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/nonsemantic.md">adds the ability</a> to declare extended instruction sets that have no semantic impact and can be safely removed from a module.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_terminate_invocation">14. VK_KHR_shader_terminate_invocation</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.3</p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_terminate_invocation.html">SPV_KHR_terminate_invocation</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension adds the new instruction <code>OpTerminateInvocation</code> to provide a disambiguated functionality compared to the <code>OpKill</code> instruction.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_workgroup_memory_explicit_layout">15. VK_KHR_workgroup_memory_explicit_layout</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_workgroup_memory_explicit_layout.html">SPV_KHR_workgroup_memory_explicit_layout</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shared_memory_block.txt">GLSL - GL_EXT_shared_memory_block</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension provides a way for the shader to define the layout of <code>Workgroup</code> <code>Storage Class</code> memory. <code>Workgroup</code> variables can be declared in blocks, and then use the same explicit layout decorations (e.g. <code>Offset</code>, <code>ArrayStride</code>) as other storage classes.</p>
</div>
<div class="paragraph">
<p>One use case is to do large vector copies (e.g. <code>uvec4</code> at at a time) from buffer memory into shared memory, even if the shared memory is really a different type (e.g. <code>scalar fp16</code>).</p>
</div>
<div class="paragraph">
<p>Another use case is a developers could potentially use this to reuse shared memory and reduce the total shared memory consumption using something such as the following:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code>pass1 - write shmem using type A
barrier()
pass2 - read shmem using type A
barrier()
pass3 - write shmem using type B
barrier()
pass4 - read shmem using type B</code></pre>
</div>
</div>
<div class="paragraph">
<p>The explicit layout support and some form of aliasing is also required for layering OpenCL on top of Vulkan.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_zero_initialize_workgroup_memory">16. VK_KHR_zero_initialize_workgroup_memory</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.3</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows <code>OpVariable</code> with a <code>Workgroup</code> <code>Storage Class</code> to use the <code>Initializer</code> operand.</p>
</div>
<div class="paragraph">
<p>For security reasons, applications running untrusted content (e.g. web browsers) need to be able to zero-initialize workgroup memory at the start of workgroup execution. Adding instructions to set all workgroup variables to zero would be less efficient than what some hardware is capable of, due to poor access patterns.</p>
</div>
</div>
</div>