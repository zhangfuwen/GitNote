<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#dynamic-state">Pipeline Dynamic State</a>
<ul class="sectlevel1">
<li><a href="#_overview">1. Overview</a></li>
<li><a href="#_when_to_use_dynamic_state">2. When to use dynamic state</a></li>
<li><a href="#states-that-are-dynamic">3. What states are dynamic</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/dynamic_state.html
layout: default
---</p>
</div>
<h1 id="dynamic-state" class="sect0">Pipeline Dynamic State</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#pipelines-dynamic-state">Vulkan Spec chapter</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="_overview">1. Overview</h2>
<div class="sectionbody">
<div class="paragraph">
<p>When creating a graphics <code>VkPipeline</code> object the logical flow for setting state is:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Using viewport state as an example
VkViewport viewport = {0.0, 0.0, 32.0, 32.0, 0.0, 1.0};

// Set value of state
VkPipelineViewportStateCreateInfo viewportStateCreateInfo;
viewportStateCreateInfo.pViewports = &amp;viewport;
viewportStateCreateInfo.viewportCount = 1;

// Create the pipeline with the state value set
VkGraphicsPipelineCreateInfo pipelineCreateInfo;
pipelineCreateInfo.pViewportState = &amp;viewportStateCreateInfo;
vkCreateGraphicsPipelines(pipelineCreateInfo, &amp;pipeline);

vkBeginCommandBuffer();
// Select the pipeline and draw with the state's static value
vkCmdBindPipeline(pipeline);
vkCmdDraw();
vkEndCommandBuffer();</code></pre>
</div>
</div>
<div class="paragraph">
<p>When the <code>VkPipeline</code> uses <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#pipelines-dynamic-state">dynamic state</a>, some pipeline information can be omitted at creation time and instead set during recording of the command buffer. The new logical flow is:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Using viewport state as an example
VkViewport viewport = {0.0, 0.0, 32.0, 32.0, 0.0, 1.0};
VkDynamicState dynamicState = VK_DYNAMIC_STATE_VIEWPORT;

// not used now
VkPipelineViewportStateCreateInfo viewportStateCreateInfo;
viewportStateCreateInfo.pViewports = nullptr;
// still need to say how many viewports will be used here
viewportStateCreateInfo.viewportCount = 1;

// Set the state as being dynamic
VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo;
dynamicStateCreateInfo.dynamicStateCount = 1;
dynamicStateCreateInfo.pDynamicStates = &amp;dynamicState;

// Create the pipeline with state value not known
VkGraphicsPipelineCreateInfo pipelineCreateInfo;
pipelineCreateInfo.pViewportState = &amp;viewportStateCreateInfo;
pipelineCreateInfo.pDynamicState = &amp;dynamicStateCreateInfo;
vkCreateGraphicsPipelines(pipelineCreateInfo, &amp;pipeline);

vkBeginCommandBuffer();
vkCmdBindPipeline(pipeline);
// Set the state for the pipeline at recording time
vkCmdSetViewport(viewport);
vkCmdDraw();
viewport.height = 64.0;
// set a new state value between draws
vkCmdSetViewport(viewport);
vkCmdDraw();
vkEndCommandBuffer();</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_when_to_use_dynamic_state">2. When to use dynamic state</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Vulkan is a tool, so as with most things, and there is no single answer for this.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Some implementations might have a performance loss using some certain <code>VkDynamicState</code> state over a static value, but dynamic states might prevent an application from having to create many permutations of pipeline objects which might be a bigger desire for the application.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="states-that-are-dynamic">3. What states are dynamic</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The full list of possible dynamic states can be found in <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkDynamicState">VkDynamicState</a>.</p>
</div>
<div class="paragraph">
<p>The <code>VK_EXT_extended_dynamic_state</code>, <code>VK_EXT_extended_dynamic_state2</code>, <code>VK_EXT_vertex_input_dynamic_state</code>, and <code>VK_EXT_color_write_enable</code> extensions were added with the goal to support applications that need to reduce the number of pipeline state objects they compile and bind.</p>
</div>
</div>
</div>