<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#vulkan-spec">Vulkan Specification</a>
<ul class="sectlevel1">
<li><a href="#_vulkan_spec_variations">1. Vulkan Spec Variations</a></li>
<li><a href="#_vulkan_spec_format">2. Vulkan Spec Format</a>
<ul class="sectlevel2">
<li><a href="#_html_chunked">2.1. HTML Chunked</a></li>
<li><a href="#_html_full">2.2. HTML Full</a></li>
<li><a href="#_pdf">2.3. PDF</a></li>
<li><a href="#_man_pages">2.4. Man pages</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/vulkan_spec.html
layout: default
---</p>
</div>
<h1 id="vulkan-spec" class="sect0">Vulkan Specification</h1>
<div class="paragraph">
<p>The Vulkan Specification (usually referred to as the <em>Vulkan Spec</em>) is the official description of how the Vulkan API works and is ultimately used to decide what is and is not valid Vulkan usage. At first glance, the Vulkan Spec seems like an incredibly huge and dry chunk of text, but it is usually the most useful item to have open when developing.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Reference the Vulkan Spec early and often.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="_vulkan_spec_variations">1. Vulkan Spec Variations</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The Vulkan Spec can be built for any version and with any permutation of extensions. The Khronos Group hosts the <a href="https://www.khronos.org/registry/vulkan/specs/">Vulkan Spec Registry</a> which contains a few publicly available variations that most developers will find sufficient. Anyone can build their own variation of the Vulkan Spec from <a href="https://github.com/KhronosGroup/Vulkan-Docs/blob/main/BUILD.adoc">Vulkan-Docs</a>.</p>
</div>
<div class="paragraph">
<p>When building the Vulkan Spec, you pass in what version of Vulkan to build for as well as what extensions to include. A Vulkan Spec without any extensions is also referred to as the <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-coreversions">core version</a> as it is the minimal amount of Vulkan an implementation needs to support in order to be <a href="vulkan_cts.html#vulkan-cts">conformant</a>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vulkan_spec_format">2. Vulkan Spec Format</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The Vulkan Spec can be built into different formats.</p>
</div>
<div class="sect2">
<h3 id="_html_chunked">2.1. HTML Chunked</h3>
<div class="paragraph">
<p>Due to the size of the Vulkan Spec, a chunked version is the default when you visit the default <code>index.html</code> page.</p>
</div>
<div class="paragraph">
<p>Example: <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/">https://www.khronos.org/registry/vulkan/specs/1.3/html/</a></p>
</div>
<div class="paragraph">
<p>Prebuilt HTML Chunked Vulkan Spec</p>
</div>
<div class="ulist">
<ul>
<li>
<p>The Vulkan SDK comes packaged with the chunked version of the spec. Each Vulkan SDK version includes the corresponding spec version. See the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/chunked_spec/index.html">Chunked Specification</a> for the latest Vulkan SDK.</p>
</li>
<li>
<p>Vulkan 1.0 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0/html/">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-wsi_extensions/html/">Core with WSI Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.1 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1/html/">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-khr-extensions/html/">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.2 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2/html/">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/html/">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.3 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-khr-extensions/html/">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_html_full">2.2. HTML Full</h3>
<div class="paragraph">
<p>If you want to view the Vulkan Spec in its entirety as HTML, you just need to view the <code>vkspec.html</code> file.</p>
</div>
<div class="paragraph">
<p>Example: <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html" class="bare">https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html</a></p>
</div>
<div class="paragraph">
<p>Prebuilt HTML Full Vulkan Spec</p>
</div>
<div class="ulist">
<ul>
<li>
<p>The Vulkan SDK comes packaged with Vulkan Spec in its entirety as HTML for the version corresponding to the Vulkan SDK version. See the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vkspec.html">HTML version of the Specification</a> for the latest Vulkan SDK. (Note: Slow to load. The advantage of the full HTML version is its searching capability).</p>
</li>
<li>
<p>Vulkan 1.0 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html">Core with Extensions </a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-wsi_extensions/html/vkspec.html">Core with WSI Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.1 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1/html/vkspec.html">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-khr-extensions/html/vkspec.html">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.2 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/html/vkspec.html">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.3 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-khr-extensions/html/vkspec.html">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_pdf">2.3. PDF</h3>
<div class="paragraph">
<p>To view the PDF format, visit the <code>pdf/vkspec.pdf</code> file.</p>
</div>
<div class="paragraph">
<p>Example: <a href="https://www.khronos.org/registry/vulkan/specs/1.3/pdf/vkspec.pdf" class="bare">https://www.khronos.org/registry/vulkan/specs/1.3/pdf/vkspec.pdf</a></p>
</div>
<div class="paragraph">
<p>Prebuilt PDF Vulkan Spec</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Vulkan 1.0 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0/pdf/vkspec.pdf">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-extensions/pdf/vkspec.pdf">Core with Extensions </a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.0-wsi_extensions/pdf/vkspec.pdf">Core with WSI Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.1 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1/pdf/vkspec.pdf">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-extensions/pdf/vkspec.pdf">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.1-khr-extensions/pdf/vkspec.pdf">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.2 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2/pdf/vkspec.pdf">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-extensions/pdf/vkspec.pdf">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/pdf/vkspec.pdf">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Vulkan 1.3 Specification</p>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3/pdf/vkspec.pdf">Core</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/pdf/vkspec.pdf">Core with Extensions</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-khr-extensions/pdf/vkspec.pdf">Core with KHR Extensions</a></p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_man_pages">2.4. Man pages</h3>
<div class="paragraph">
<p>The Khronos Group currently only host the Vulkan Man Pages for the latest version of the 1.3 spec, with all extensions, on the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/">online registry</a>.</p>
</div>
<div class="paragraph">
<p>The Vulkan Man Pages can also be found in the VulkanSDK for each SDK version. See the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/apispec.html">Man Pages</a> for the latest Vulkan SDK.</p>
</div>
</div>
</div>
</div>