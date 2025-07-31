<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#external-memory">External Memory and Synchronization</a>
<ul class="sectlevel1">
<li><a href="#_capabilities">1. Capabilities</a></li>
<li><a href="#_memory_vs_synchronization">2. Memory vs Synchronization</a>
<ul class="sectlevel2">
<li><a href="#_memory">2.1. Memory</a></li>
<li><a href="#_synchronization">2.2. Synchronization</a></li>
</ul>
</li>
<li><a href="#_example">3. Example</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/external.html
layout: default
---</p>
</div>
<h1 id="external-memory" class="sect0">External Memory and Synchronization</h1>
<div class="paragraph">
<p>Sometimes not everything an application does related to the GPU is done in Vulkan. There are various situations where memory is written or read outside the scope of Vulkan. To support these use cases a set of external memory and synchronization functions was created</p>
</div>
<div class="paragraph">
<p>The list of extensions involved are:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_external_fence</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_fence_capabilities</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_memory</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_memory_capabilities</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_semaphore</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_semaphore_capabilities</code></p>
<div class="ulist">
<ul>
<li>
<p>Promoted to core in 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_external_fence_fd</code></p>
</li>
<li>
<p><code>VK_KHR_external_fence_win32</code></p>
</li>
<li>
<p><code>VK_KHR_external_memory_fd</code></p>
</li>
<li>
<p><code>VK_KHR_external_memory_win32</code></p>
</li>
<li>
<p><code>VK_KHR_external_semaphore_fd</code></p>
</li>
<li>
<p><code>VK_KHR_external_semaphore_win32</code></p>
</li>
<li>
<p><code>VK_ANDROID_external_memory_android_hardware_buffer</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>This seems like a lot so let&#8217;s break it down little by little.</p>
</div>
<div class="sect1">
<h2 id="_capabilities">1. Capabilities</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VK_KHR_external_fence_capabilities</code>, <code>VK_KHR_external_semaphore_capabilities</code>, and <code>VK_KHR_external_memory_capabilities</code> are simply just ways to query information about what external support an implementation provides.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_memory_vs_synchronization">2. Memory vs Synchronization</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There is a set of extensions to handle the importing/exporting of just the memory itself. The other set extensions are for the synchronization primitives (<code>VkFence</code> and <code>VkSemaphore</code>) used to control internal Vulkan commands. It is common practice that for each piece of memory imported/exported there is also a matching fence/semaphore to manage the memory access.</p>
</div>
<div class="sect2">
<h3 id="_memory">2.1. Memory</h3>
<div class="paragraph">
<p>The <code>VK_KHR_external_memory</code> extension is mainly to provide the <code>VkExternalMemoryHandleTypeFlagBits</code> enum which describes the type of memory being used externally.</p>
</div>
<div class="paragraph">
<p>There are currently 3 supported ways to import/export memory</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_external_memory_fd</code> for memory in a POSIX file descriptor</p>
</li>
<li>
<p><code>VK_KHR_external_memory_win32</code> for memory in a Windows handle</p>
</li>
<li>
<p><code>VK_ANDROID_external_memory_android_hardware_buffer</code> for memory in a AHardwareBuffer</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Each of these methods has their own detailed descriptions about limitations, requirements, ownership, etc.</p>
</div>
<div class="sect3">
<h4 id="_importing_memory">2.1.1. Importing Memory</h4>
<div class="paragraph">
<p>To import memory, there is a <code>VkImport*Info</code> struct provided by the given external memory extension. This is passed into <code>vkAllocateMemory</code> where Vulkan will now have a <code>VkDeviceMemory</code> handle that maps to the imported memory.</p>
</div>
</div>
<div class="sect3">
<h4 id="_exporting_memory">2.1.2. Exporting Memory</h4>
<div class="paragraph">
<p>To export memory, there is a <code>VkGetMemory*</code> function provided by the given external memory extension. This function will take in a <code>VkDeviceMemory</code> handle and then map that to the extension exposed object.</p>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_synchronization">2.2. Synchronization</h3>
<div class="paragraph">
<p>External synchronization can be used in Vulkan for both <code>VkFence</code> and <code>VkSemaphores</code>. There is almost no difference between the two with regards to how it is used to import and export them.</p>
</div>
<div class="paragraph">
<p>The <code>VK_KHR_external_fence</code> and <code>VK_KHR_external_semaphore</code> extension both expose a <code>Vk*ImportFlagBits</code> enum and <code>VkExport*CreateInfo</code> struct to describe the type a synchronization being imported/exported.</p>
</div>
<div class="paragraph">
<p>There are currently 2 supported ways to import/export synchronization</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_external_fence_fd</code> / <code>VK_KHR_external_semaphore_fd</code></p>
</li>
<li>
<p><code>VK_KHR_external_fence_win32</code> / <code>VK_KHR_external_semaphore_win32</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Each extension explains how it manages ownership of the synchronization primitives.</p>
</div>
<div class="sect3">
<h4 id="_importing_and_exporting_synchronization_primitives">2.2.1. Importing and Exporting Synchronization Primitives</h4>
<div class="paragraph">
<p>There is a <code>VkImport*</code> function for importing and a <code>VkGet*</code> function for exporting. These both take the <code>VkFence</code>/<code>VkSemaphores</code> handle passed in along with the extension&#8217;s method of defining the external synchronization object.</p>
</div>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_example">3. Example</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Here is a simple diagram showing the timeline of events between Vulkan and some other API talking to the GPU. This is used to represent a common use case for these external memory and synchronization extensions.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/external_example.png" alt="external_example.png">
</div>
</div>
</div>
</div>