<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_EXT_memory_priority">VK_EXT_memory_priority</a>
<ul class="sectlevel1">
<li><a href="#_suggestions">1. Suggestions</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_EXT_memory_priority.html
layout: default
---</p>
</div>
<h1 id="VK_EXT_memory_priority" class="sect0">VK_EXT_memory_priority</h1>
<div class="paragraph">
<p>Memory management is an important part of Vulkan. The <code>VK_EXT_memory_priority</code> extension was designed to allow an application to prevent important allocations from being moved to slower memory.</p>
</div>
<div class="paragraph">
<p>This extension can be explained with an example of two applications (the main application and another process on the host machine). Over time the applications both attempt to consume the limited device heap memory.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_EXT_memory_priority_overview.png" alt="VK_EXT_memory_priority_overview">
</div>
</div>
<div class="paragraph">
<p>In this situation, the allocation from the main application is still present, just possibly on slower memory (implementation might have moved it to host visible memory until it is needed again).</p>
</div>
<div class="paragraph">
<p>The decision of <strong>what</strong> memory will get moved is implementation defined. Let&#8217;s now imagine this is the main application&#8217;s memory usage</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_EXT_memory_priority_app.png" alt="VK_EXT_memory_priority_app">
</div>
</div>
<div class="paragraph">
<p>As we can see, there was some memory the application felt was more important to always attempt to keep in fast memory.</p>
</div>
<div class="paragraph">
<p>The <code>VK_EXT_memory_priority</code> extension makes this very easy. When allocating memory, an application just needs to add <code>VkMemoryPriorityAllocateInfoEXT</code> to <code>VkMemoryAllocateInfo::pNext</code>. From here the <code>VkMemoryPriorityAllocateInfoEXT::priority</code> value can be set with a value between <code>0.0</code> and <code>1.0</code> (where <code>0.5</code>) is the default. This allows the application to help the implementation make a better guess if the above situation occurs.</p>
</div>
<div class="sect1">
<h2 id="_suggestions">1. Suggestions</h2>
<div class="sectionbody">
<div class="ulist">
<ul>
<li>
<p>Make sure the extension is supported.</p>
</li>
<li>
<p>Remember this is a <strong>hint</strong> to the implementation and an application should still try to budget properly prior to using this.</p>
</li>
<li>
<p>Always measure memory bottlenecks instead of making assumptions when possible.</p>
</li>
<li>
<p>Any memory being written to will have a good chance of being a high priority.</p>
<div class="ulist">
<ul>
<li>
<p>Render targets (Ex: Framebuffer&#8217;s output attachments) are usually important to set to high priority</p>
</li>
</ul>
</div>
</li>
<li>
<p>View high priority memory as having &#8220;high frequency access&#8221; and &#8220;low latency tolerance&#8221;</p>
<div class="ulist">
<ul>
<li>
<p>Ex: Vertex buffers, which remain stable across multiple frames, have each value accessed only once, and typically are forgiving for access latency, are usually a good candidate for lower priorities.</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>