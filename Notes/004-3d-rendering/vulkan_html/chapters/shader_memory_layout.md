<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#shader-memory-layout">Shader Memory Layout</a>
<ul class="sectlevel1">
<li><a href="#alignment-requirements">1. Alignment Requirements</a></li>
<li><a href="#VK_KHR_uniform_buffer_standard_layout">2. VK_KHR_uniform_buffer_standard_layout</a></li>
<li><a href="#VK_KHR_relaxed_block_layout">3. VK_KHR_relaxed_block_layout</a></li>
<li><a href="#VK_EXT_scalar_block_layout">4. VK_EXT_scalar_block_layout</a></li>
<li><a href="#alignment-examples">5. Alignment Examples</a>
<ul class="sectlevel2">
<li><a href="#_alignment_example_1">5.1. Alignment Example 1</a></li>
<li><a href="#_alignment_example_2">5.2. Alignment Example 2</a></li>
<li><a href="#_alignment_example_3">5.3. Alignment Example 3</a></li>
<li><a href="#_alignment_example_4">5.4. Alignment Example 4</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/shader_memory_layout.html
layout: default
---</p>
</div>
<h1 id="shader-memory-layout" class="sect0">Shader Memory Layout</h1>
<div class="paragraph">
<p>When an implementation accesses memory from an interface, it needs to know how the <strong>memory layout</strong>. This includes things such as <strong>offsets</strong>, <strong>stride</strong>, and <strong>alignments</strong>. While the Vulkan Spec has a <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-resources-layout">section dedicated to this</a>, it can be hard to parse due to the various extensions that add extra complexity to the spec language. This chapter aims to help explain all the memory layout concepts Vulkan uses with some high level shading language (GLSL) examples.</p>
</div>
<div class="sect1">
<h2 id="alignment-requirements">1. Alignment Requirements</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Vulkan has 3 <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-alignment-requirements">alignment requirements</a> that interface objects can be laid out in.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>extended alignment (also know as <code>std140</code>)</p>
</li>
<li>
<p>base alignment (also know as <code>std430</code>)</p>
</li>
<li>
<p>scalar alignment</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>The spec language for alignment breaks down the rule for each of the following block member types.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>scalar (<code>float</code>, <code>int</code>, <code>char</code>, etc)</p>
</li>
<li>
<p>vector (<code>float2</code>, <code>vec3</code>, &#8217;uvec4' etc)</p>
</li>
<li>
<p>matrix</p>
</li>
<li>
<p>array</p>
</li>
<li>
<p>struct</p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_uniform_buffer_standard_layout">2. VK_KHR_uniform_buffer_standard_layout</h2>
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
<p>This extension allows the use of <code>std430</code> memory layout in UBOs. <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-resources-standard-layout">Vulkan Standard Buffer Layout Interface</a> can be found outside this guide. These memory layout changes are only applied to <code>Uniforms</code> as other storage items such as Push Constants and SSBO already allow for std430 style layouts.</p>
</div>
<div class="paragraph">
<p>One example of when the <code>uniformBufferStandardLayout</code> feature is needed is when an application doesn&#8217;t want the array stride for a UBO to be restricted to <code>extended alignment</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(std140, binding = 0) uniform ubo140 {
   float array140[8];
};

layout(std430, binding = 1) uniform ubo430 {
   float array430[8];
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>Which translates in SPIR-V to</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">// extended alignment for array is rounded up to multiple of 16
OpDecorate %array140 ArrayStride 16

// base alignment is 4 bytes (OpTypeFloat 32)
// only valid with uniformBufferStandardLayout feature enabled
OpDecorate %array430 ArrayStride 4</code></pre>
</div>
</div>
<div class="paragraph">
<p>Make sure to set <code>--uniform-buffer-standard-layout</code> when running the SPIR-V Validator.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_relaxed_block_layout">3. VK_KHR_relaxed_block_layout</h2>
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
<p>There was never a feature bit added for this extension, so all Vulkan 1.1+ devices support relaxed block layout.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows implementations to indicate they can support more variation in block <code>Offset</code> decorations. This comes up when using <code>std430</code> memory layout where a <code>vec3</code> (which is 12 bytes) is still defined as a 16 byte alignment. With relaxed block layout an application can fit a <code>float</code> on either side of the <code>vec3</code> and maintain the 16 byte alignment between them.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// SPIR-V offsets WITHOUT relaxed block layout
layout (set = 0, binding = 0) buffer block {
    float b; // Offset: 0
    vec3 a;  // Offset: 16
} ssbo;

// SPIR-V offsets WITH relaxed block layout
layout (set = 0, binding = 0) buffer block {
    float b; // Offset: 0
    vec3 a;  // Offset: 4
} ssbo;</code></pre>
</div>
</div>
<div class="paragraph">
<p><code>VK_KHR_relaxed_block_layout</code> can also be seen as a subset of <code>VK_EXT_scalar_block_layout</code></p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Make sure to set <code>--relax-block-layout</code> when running the SPIR-V Validator and using a Vulkan 1.0 environment.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Currently there is no way in GLSL to legally express relaxed block layout, but an developer can use the <code>--hlsl-offsets</code> with <code>glslang</code> to produce the desired offsets.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_scalar_block_layout">4. VK_EXT_scalar_block_layout</h2>
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
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_scalar_block_layout.txt">GLSL - GL_EXT_scalar_block_layout</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows most storage types to be aligned in <code>scalar alignment</code>. A big difference is being able to straddle the 16-byte boundary.</p>
</div>
<div class="paragraph">
<p>In GLSL this can be used with <code>scalar</code> keyword and extension</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#extension GL_EXT_scalar_block_layout : enable
layout (scalar, binding = 0) buffer block { }</code></pre>
</div>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Make sure to set <code>--scalar-block-layout</code> when running the SPIR-V Validator.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The <code>Workgroup</code> storage class is not supported with <code>VK_EXT_scalar_block_layout</code> and the <code>workgroupMemoryExplicitLayoutScalarBlockLayout</code> in <a href="extensions/shader_features.html#VK_KHR_workgroup_memory_explicit_layout">VK_KHR_workgroup_memory_explicit_layout</a> is needed to enabled scalar support.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="alignment-examples">5. Alignment Examples</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The following are some GLSL to SPIR-V examples to help better understand the difference in the alignments supported.</p>
</div>
<div class="sect2">
<h3 id="_alignment_example_1">5.1. Alignment Example 1</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(binding = 0) buffer block {
    vec2 a[4];
    vec4 b;
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>Which translates in SPIR-V to</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">// extended alignment (std140)
OpDecorate %vec2array ArrayStride 16
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 64

// scalar alignment and base alignment (std430)
OpDecorate %vec2array ArrayStride 8
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 32</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_alignment_example_2">5.2. Alignment Example 2</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(binding = 0) buffer block {
    float a;
    vec2 b;
    vec2 c;
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>Which translates in SPIR-V to</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">// extended alignment (std140) and base alignment (std430)
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 8
OpMemberDecorate %block 2 Offset 16

// scalar alignment
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 4
OpMemberDecorate %block 2 Offset 12</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_alignment_example_3">5.3. Alignment Example 3</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(binding = 0) buffer block {
    vec3 a;
    vec2 b;
    vec4 c;
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>Which translates in SPIR-V to</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">// extended alignment (std140) and base alignment (std430)
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 16
OpMemberDecorate %block 2 Offset 32

// scalar alignment
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 12
OpMemberDecorate %block 2 Offset 20</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_alignment_example_4">5.4. Alignment Example 4</h3>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout (binding = 0) buffer block {
    vec3 a;
    vec2 b;
    vec2 c;
    vec3 d;
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>Which translates in SPIR-V to</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">// extended alignment (std140) and base alignment (std430)
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 16
OpMemberDecorate %block 2 Offset 24
OpMemberDecorate %block 3 Offset 32

// scalar alignment
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 12
OpMemberDecorate %block 2 Offset 20
OpMemberDecorate %block 3 Offset 28</code></pre>
</div>
</div>
</div>
</div>
</div>