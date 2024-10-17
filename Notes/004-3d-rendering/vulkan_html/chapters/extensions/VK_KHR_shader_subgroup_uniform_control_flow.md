<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_shader_subgroup_uniform_control_flow">VK_KHR_shader_subgroup_uniform_control_flow</a>
<ul class="sectlevel1">
<li><a href="#_overview">1. Overview</a></li>
<li><a href="#_example">2. Example</a></li>
<li><a href="#_related_extensions">3. Related Extensions</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_shader_subgroup_uniform_control_flow.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_shader_subgroup_uniform_control_flow" class="sect0">VK_KHR_shader_subgroup_uniform_control_flow</h1>
<div class="sect1">
<h2 id="_overview">1. Overview</h2>
<div class="sectionbody">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_subgroup_uniform_control_flow.html">VK_KHR_shader_subgroup_uniform_control_flow</a>
provides stronger guarantees for reconvergence of invocations in a shader. If
the extension is supported, shaders can be modified to include a new attribute
that provides the stronger guarantees (see
<a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_subgroup_uniform_control_flow.txt">GL_EXT_subgroup_uniform_control_flow</a>).
This attribute can only be applied to shader stages that support subgroup
operations (check <code>VkPhysicalDeviceSubgroupProperties::supportedStages</code> or
<code>VkPhysicalDeviceVulkan11Properties::subgroupSupportedStages</code>).</p>
</div>
<div class="paragraph">
<p>The stronger guarantees cause the uniform control flow rules in the SPIR-V
specification to also apply to individual subgroups. The most important part of
those rules is the requirement to reconverge at a merge block if the all
invocations were converged upon entry to the header block. This is often
implicitly relied upon by shader authors, but not actually guaranteed by the
core Vulkan specification.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_example">2. Example</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Consider the following GLSL snippet of a compute shader that attempts to reduce
the number of atomic operations from one per invocation to one per subgroup:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// Free should be initialized to 0.
layout(set=0, binding=0) buffer BUFFER { uint free; uint data[]; } b;
void main() {
  bool needs_space = false;
  ...
  if (needs_space) {
    // gl_SubgroupSize may be larger than the actual subgroup size so
    // calculate the actual subgroup size.
    uvec4 mask = subgroupBallot(needs_space);
    uint size = subgroupBallotBitCount(mask);
    uint base = 0;
    if (subgroupElect()) {
      // "free" tracks the next free slot for writes.
      // The first invocation in the subgroup allocates space
      // for each invocation in the subgroup that requires it.
      base = atomicAdd(b.free, size);
    }

    // Broadcast the base index to other invocations in the subgroup.
    base = subgroupBroadcastFirst(base);
    // Calculate the offset from "base" for each invocation.
    uint offset = subgroupBallotExclusiveBitCount(mask);

    // Write the data in the allocated slot for each invocation that
    // requested space.
    b.data[base + offset] = ...;
  }
  ...
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>There is a problem with the code that might lead to unexpected results. Vulkan
only requires invocations to reconverge after the if statement that performs
the subgroup election if all the invocations in the <em>workgroup</em> are converged at
that if statement. If the invocations don&#8217;t reconverge then the broadcast and
offset calculations will be incorrect. Not all invocations would write their
results to the correct index.</p>
</div>
<div class="paragraph">
<p><code>VK_KHR_shader_subgroup_uniform_control_flow</code> can be utilized to make the shader
behave as expected in most cases. Consider the following rewritten version of
the example:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">// Free should be initialized to 0.
layout(set=0, binding=0) buffer BUFFER { uint free; uint data[]; } b;
// Note the addition of a new attribute.
void main() [[subroup_uniform_control_flow]] {
  bool needs_space = false;
  ...
  // Note the change of the condition.
  if (subgroupAny(needs_space)) {
    // gl_SubgroupSize may be larger than the actual subgroup size so
    // calculate the actual subgroup size.
    uvec4 mask = subgroupBallot(needs_space);
    uint size = subgroupBallotBitCount(mask);
    uint base = 0;
    if (subgroupElect()) {
      // "free" tracks the next free slot for writes.
      // The first invocation in the subgroup allocates space
      // for each invocation in the subgroup that requires it.
      base = atomicAdd(b.free, size);
    }

    // Broadcast the base index to other invocations in the subgroup.
    base = subgroupBroadcastFirst(base);
    // Calculate the offset from "base" for each invocation.
    uint offset = subgroupBallotExclusiveBitCount(mask);

    if (needs_space) {
      // Write the data in the allocated slot for each invocation that
      // requested space.
      b.data[base + offset] = ...;
    }
  }
  ...
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>The differences from the original shader are relatively minor. First, the
addition of the <code>subgroup_uniform_control_flow</code> attribute informs the
implementation that stronger guarantees are required by this shader. Second,
the first if statement no longer tests needs_space. Instead, all invocations in
the subgroup enter the if statement if any invocation in the subgroup needs to
write data. This keeps the subgroup uniform to utilize the enhanced guarantees
for the inner subgroup election.</p>
</div>
<div class="paragraph">
<p>There is a final caveat with this example. In order for the shader to operate
correctly in all circumstances, the subgroup must be uniform (converged) prior
to the first if statement.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_related_extensions">3. Related Extensions</h2>
<div class="sectionbody">
<div class="ulist">
<ul>
<li>
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_subgroup_uniform_control_flow.txt">GL_EXT_subgroup_uniform_control_flow</a> - adds a GLSL attribute for entry points
to notify implementations that stronger guarantees for convergence are
required. This translates to a new execution mode in the SPIR-V entry point.</p>
</li>
<li>
<p><a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_subgroup_uniform_control_flow.html">SPV_KHR_subgroup_uniform_control_flow</a> - adds an execution mode for entry
points to indicate the requirement for stronger reconvergence guarantees.</p>
</li>
</ul>
</div>
</div>
</div>