<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#atomics">Atomics</a>
<ul class="sectlevel1">
<li><a href="#_variations_of_atomics">1. Variations of Atomics</a></li>
<li><a href="#_baseline_support">2. Baseline Support</a>
<ul class="sectlevel2">
<li><a href="#_atomic_counters">2.1. Atomic Counters</a></li>
<li><a href="#_expanding_atomic_support">2.2. Expanding Atomic support</a></li>
</ul>
</li>
<li><a href="#VK_KHR_shader_atomic_int64">3. VK_KHR_shader_atomic_int64</a></li>
<li><a href="#VK_EXT_shader_image_atomic_int64">4. VK_EXT_shader_image_atomic_int64</a>
<ul class="sectlevel2">
<li><a href="#_image_vs_sparse_image_support">4.1. Image vs Sparse Image support</a></li>
</ul>
</li>
<li><a href="#VK_EXT_shader_atomic_float">5. VK_EXT_shader_atomic_float</a></li>
<li><a href="#VK_EXT_shader_atomic_float2">6. VK_EXT_shader_atomic_float2</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/atomics.html
---</p>
</div>
<h1 id="atomics" class="sect0">Atomics</h1>
<div class="paragraph">
<p>The purpose of this chapter is to help users understand the various features Vulkan exposes for atomic operations.</p>
</div>
<div class="sect1">
<h2 id="_variations_of_atomics">1. Variations of Atomics</h2>
<div class="sectionbody">
<div class="paragraph">
<p>To better understand the different extensions, it is first important to be aware of the various types of atomics exposed.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Type</p>
<div class="ulist">
<ul>
<li>
<p><code>float</code></p>
</li>
<li>
<p><code>int</code></p>
</li>
</ul>
</div>
</li>
<li>
<p>Width</p>
<div class="ulist">
<ul>
<li>
<p><code>16 bit</code></p>
</li>
<li>
<p><code>32 bit</code></p>
</li>
<li>
<p><code>64 bit</code></p>
</li>
</ul>
</div>
</li>
<li>
<p>Operations</p>
<div class="ulist">
<ul>
<li>
<p>loads</p>
</li>
<li>
<p>stores</p>
</li>
<li>
<p>exchange</p>
</li>
<li>
<p>add</p>
</li>
<li>
<p>min</p>
</li>
<li>
<p>max</p>
</li>
<li>
<p>etc.</p>
</li>
</ul>
</div>
</li>
<li>
<p>Storage Class</p>
<div class="ulist">
<ul>
<li>
<p><code>StorageBuffer</code> or <code>Uniform</code> (buffer)</p>
</li>
<li>
<p><code>Workgroup</code> (shared memory)</p>
</li>
<li>
<p><code>Image</code> (image or sparse image)</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_baseline_support">2. Baseline Support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>With Vulkan 1.0 and no extensions, an application is allowed to use <code>32-bit int</code> type for atomics. This can be used for all supported SPIR-V operations (load, store, exchange, etc). SPIR-V contains some atomic operations that are guarded with the <code>Kernel</code> capability and are not currently allowed in Vulkan.</p>
</div>
<div class="sect2">
<h3 id="_atomic_counters">2.1. Atomic Counters</h3>
<div class="paragraph">
<p>While both GLSL and SPIR-V support the use of atomic counters, Vulkan does <strong>not</strong> expose the <code>AtomicStorage</code> SPIR-V capability needed to use the <code>AtomicCounter</code> storage class. It was decided that an app can just use <code>OpAtomicIAdd</code> and <code>OpAtomicISub</code> with the value <code>1</code> to achieve the same results.</p>
</div>
</div>
<div class="sect2">
<h3 id="_expanding_atomic_support">2.2. Expanding Atomic support</h3>
<div class="paragraph">
<p>The current extensions that expose additional support for atomics are:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_atomic_int64.html">VK_KHR_shader_atomic_int64</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_image_atomic_int64.html">VK_EXT_shader_image_atomic_int64</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_atomic_float.html">VK_EXT_shader_atomic_float</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_atomic_float2.html">VK_EXT_shader_atomic_float2</a></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Each explained in more details below.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_atomic_int64">3. VK_KHR_shader_atomic_int64</h2>
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
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_atomic_int64.txt">GLSL - GL_EXT_shader_atomic_int64</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows for <code>64-bit int</code> atomic operations for <strong>buffers</strong> and <strong>shared memory</strong>. If the <code>Int64Atomics</code> SPIR-V capability is declared, all supported SPIR-V operations can be used with <code>64-bit int</code>.</p>
</div>
<div class="paragraph">
<p>The two feature bits, <code>shaderBufferInt64Atomics</code> and <code>shaderSharedInt64Atomics</code>, are used to query what storage classes are supported for <code>64-bit int</code> atomics.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferInt64Atomics</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedInt64Atomics</code> - shared memory</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>The <code>shaderBufferInt64Atomics</code> is always guaranteed to be supported if using Vulkan 1.2+ or the extension is exposed.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_image_atomic_int64">4. VK_EXT_shader_image_atomic_int64</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_image_int64.html">SPV_EXT_shader_image_int64</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_image_int64.txt">GLSL_EXT_shader_image_int64</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows for <code>64-bit int</code> atomic operations for <strong>images</strong> and <strong>sparse images</strong>. If the <code>Int64Atomics</code> and <code>Int64ImageEXT</code> SPIR-V capability is declared, all supported SPIR-V operations can be used with <code>64-bit int</code> on images.</p>
</div>
<div class="sect2">
<h3 id="_image_vs_sparse_image_support">4.1. Image vs Sparse Image support</h3>
<div class="paragraph">
<p>This extension exposes both a <code>shaderImageInt64Atomics</code> and <code>sparseImageInt64Atomics</code> feature bit. The <code>sparseImage*</code> feature is an additional feature bit and is only allowed to be used if the <code>shaderImage*</code> bit is enabled as well. Some hardware has a hard time doing atomics on images with <a href="sparse_resources.html#sparse-resources">sparse resources</a>, therefor the atomic feature is split up to allow <strong>sparse images</strong> as an additional feature an implementation can expose.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_atomic_float">5. VK_EXT_shader_atomic_float</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_atomic_float_add.html">SPV_EXT_shader_atomic_float_add</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_atomic_float.txt">GLSL_EXT_shader_atomic_float</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows for <code>float</code> atomic operations for <strong>buffers</strong>, <strong>shared memory</strong>, <strong>images</strong>, and <strong>sparse images</strong>. Only a subset of operations is supported for <code>float</code> types with this extension.</p>
</div>
<div class="paragraph">
<p>The extension lists many feature bits. One way to group them is by <code>*Float*Atomics</code> and <code>*Float*AtomicAdd</code>:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>The <code>*Float*Atomics</code> features allow for the use of <code>OpAtomicStore</code>, <code>OpAtomicLoad</code>, and <code>OpAtomicExchange</code> for <code>float</code> types.</p>
<div class="ulist">
<ul>
<li>
<p>Note the <code>OpAtomicCompareExchange</code> &#8220;exchange&#8221; operation is not included as the SPIR-V spec only allows <code>int</code> types for it.</p>
</li>
</ul>
</div>
</li>
<li>
<p>The <code>*Float*AtomicAdd</code> features allow the use of the two extended SPIR-V operations <code>AtomicFloat32AddEXT</code> and <code>AtomicFloat64AddEXT</code>.</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>From here the rest of the permutations of features fall into the grouping of <code>32-bit float</code> support:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat32*</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat32*</code> - shared memory</p>
</li>
<li>
<p><code>shaderImageFloat32*</code> - images</p>
</li>
<li>
<p><code>sparseImageFloat32*</code> - sparse images</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>and <code>64-bit float</code> support:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat64*</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat64*</code> - shared memory</p>
</li>
</ul>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>OpenGLES <a href="https://www.khronos.org/registry/OpenGL/extensions/OES/OES_shader_image_atomic.txt">OES_shader_image_atomic</a> allowed the use of atomics on <code>r32f</code> for <code>imageAtomicExchange</code>. For porting, an application will want to check for <code>shaderImageFloat32Atomics</code> support to be able to do the same in Vulkan.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_atomic_float2">6. VK_EXT_shader_atomic_float2</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_atomic_float_min_max.html">SPV_EXT_shader_atomic_float_min_max</a></p>
</div>
<div class="paragraph">
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_atomic_float16_add.html">SPV_EXT_shader_atomic_float16_add</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_atomic_float.txt">GLSL_EXT_shader_atomic_float</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension adds 2 additional sets of features missing in <code>VK_EXT_shader_atomic_float</code></p>
</div>
<div class="paragraph">
<p>First, it adds <code>16-bit floats</code> for both <strong>buffers</strong> and <strong>shared memory</strong> in the same fashion as found above for <code>VK_EXT_shader_atomic_float</code>.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat16*</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat16*</code> - shared memory</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Second, it adds <code>float</code> support for <code>min</code> and <code>max</code> atomic operations (<code>OpAtomicFMinEXT</code> and <code>OpAtomicFMaxEXT</code>)</p>
</div>
<div class="paragraph">
<p>For <code>16-bit float</code> support (with <code>AtomicFloat16MinMaxEXT</code> capability):</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat16AtomicMinMax</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat16AtomicMinMax</code> - shared memory</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>For <code>32-bit float</code> support (with <code>AtomicFloat32MinMaxEXT</code> capability):</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat32AtomicMinMax</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat32AtomicMinMax</code> - shared memory</p>
</li>
<li>
<p><code>shaderImageFloat32AtomicMinMax</code> - images</p>
</li>
<li>
<p><code>sparseImageFloat32AtomicMinMax</code> - sparse images</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>For <code>64-bit float</code> support (with <code>AtomicFloat64MinMaxEXT</code> capability):</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>shaderBufferFloat64AtomicMinMax</code> - buffers</p>
</li>
<li>
<p><code>shaderSharedFloat64AtomicMinMax</code> - shared memory</p>
</li>
</ul>
</div>
</div>
</div>