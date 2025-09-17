<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#spirv-extensions">Using SPIR-V Extensions</a>
<ul class="sectlevel1">
<li><a href="#_spir_v_extension_example">1. SPIR-V Extension Example</a>
<ul class="sectlevel2">
<li><a href="#steps-for-using-spriv-features">1.1. Steps for using SPIR-V features:</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/spirv_extensions.html
layout: default
---</p>
</div>
<h1 id="spirv-extensions" class="sect0">Using SPIR-V Extensions</h1>
<div class="paragraph">
<p><a href="../what_is_spirv.html">SPIR-V</a> is the shader representation used at <code>vkCreateShaderModule</code> time. Just like Vulkan, <a href="https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/extension_overview.md">SPIR-V also has extensions</a> and a <a href="https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/capabilities.md">capabilities system</a>.</p>
</div>
<div class="paragraph">
<p>It is important to remember that SPIR-V is an intermediate language and not an API, it relies on an API, such as Vulkan, to expose what features are available to the application at runtime. This chapter aims to explain how Vulkan, as a SPIR-V client API, interacts with the SPIR-V extensions and capabilities.</p>
</div>
<div class="sect1">
<h2 id="_spir_v_extension_example">1. SPIR-V Extension Example</h2>
<div class="sectionbody">
<div class="paragraph">
<p>For this example, the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_atomic_int64.html">VK_KHR_8bit_storage</a> and <a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_8bit_storage.html">SPV_KHR_8bit_storage</a> will be used to expose the <code>UniformAndStorageBuffer8BitAccess</code> capability. The following is what the SPIR-V disassembled looks like:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-swift" data-lang="swift">OpCapability Shader
OpCapability UniformAndStorageBuffer8BitAccess
OpExtension  "SPV_KHR_8bit_storage"</code></pre>
</div>
</div>
<div class="sect2">
<h3 id="steps-for-using-spriv-features">1.1. Steps for using SPIR-V features:</h3>
<div class="olist arabic">
<ol class="arabic">
<li>
<p>Make sure the SPIR-V extension and capability are available in Vulkan.</p>
</li>
<li>
<p>Check if the required Vulkan extension, features or version are supported.</p>
</li>
<li>
<p>If needed, enable the Vulkan extension and features.</p>
</li>
<li>
<p>If needed, see if there is a matching extension for the high-level shading language (ex. GLSL or HLSL) being used.</p>
</li>
</ol>
</div>
<div class="paragraph">
<p>Breaking down each step in more detail:</p>
</div>
<div class="sect3">
<h4 id="_check_if_spir_v_feature_is_supported">1.1.1. Check if SPIR-V feature is supported</h4>
<div class="paragraph">
<p>Depending on the shader feature there might only be a <code>OpExtension</code> or <code>OpCapability</code> that is needed. For this example, the <code>UniformAndStorageBuffer8BitAccess</code> is part of the <a href="http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_8bit_storage.html">SPV_KHR_8bit_storage</a> extension.</p>
</div>
<div class="paragraph">
<p>To check if the SPIR-V extension is supported take a look at the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#spirvenv-extensions">Supported SPIR-V Extension Table</a> in the Vulkan Spec.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/spirv_extensions_8bit_extension.png" alt="spirv_extensions_8bit_extension">
</div>
</div>
<div class="paragraph">
<p>Also, take a look at the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#spirvenv-capabilities">Supported SPIR-V Capabilities Table</a> in the Vulkan Spec.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/spirv_extensions_8bit_capability.png" alt="spirv_extensions_8bit_capability">
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
<p>while it says <code>VkPhysicalDeviceVulkan12Features::uniformAndStorageBuffer8BitAccess</code> in the table, the <code>VkPhysicalDevice8BitStorageFeatures::uniformAndStorageBuffer8BitAccess</code> is an alias can be considered the same here.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Luckily if you forget to check, the Vulkan Validation Layers has an <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/layers/generated/spirv_validation_helper.cpp">auto-generated validation</a> in place. Both the Validation Layers and the Vulkan Spec table are all based on the <a href="https://github.com/KhronosGroup/Vulkan-Docs/blob/main/xml/vk.xml">./xml/vk.xml</a> file.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-xml" data-lang="xml">&lt;spirvcapability name="UniformAndStorageBuffer8BitAccess"&gt;
    &lt;enable struct="VkPhysicalDeviceVulkan12Features" feature="uniformAndStorageBuffer8BitAccess" requires="VK_VERSION_1_2,VK_KHR_8bit_storage"/&gt;
&lt;/spirvcapability&gt;

&lt;spirvextension name="SPV_KHR_8bit_storage"&gt;
    &lt;enable version="VK_VERSION_1_2"/&gt;
    &lt;enable extension="VK_KHR_8bit_storage"/&gt;
&lt;/spirvextension&gt;</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="_check_for_support_then_enable_if_needed">1.1.2. Check for support then enable if needed</h4>
<div class="paragraph">
<p>In this example, either <code>VK_KHR_8bit_storage</code> or a Vulkan 1.2 device is required.</p>
</div>
<div class="paragraph">
<p>If using a Vulkan 1.0 or 1.1 device, the <code>VK_KHR_8bit_storage</code> extension will need to be <a href="../enabling_extensions.html#enabling-extensions">supported and enabled</a> at device creation time.</p>
</div>
<div class="paragraph">
<p>Regardless of using the Vulkan extension or version, if required, an app still <strong>must</strong> make sure any matching Vulkan feature needed is <a href="../enabling_features.html#enabling-extensions">supported and enabled</a> at device creation time. Some SPIR-V extensions and capabilities don&#8217;t require a Vulkan feature, but this is all listed in the tables in the spec.</p>
</div>
<div class="paragraph">
<p>For this example, either the <code>VkPhysicalDeviceVulkan12Features::uniformAndStorageBuffer8BitAccess</code> or <code>VkPhysicalDevice8BitStorageFeatures::uniformAndStorageBuffer8BitAccess</code> feature must be supported and enabled.</p>
</div>
</div>
<div class="sect3">
<h4 id="_using_high_level_shading_language_extensions">1.1.3. Using high level shading language extensions</h4>
<div class="paragraph">
<p>For this example, GLSL has a <a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt">GL_EXT_shader_16bit_storage</a> extension that includes the match <code>GL_EXT_shader_8bit_storage</code> extension in it.</p>
</div>
<div class="paragraph">
<p>Tools such as <code>glslang</code> and <code>SPIRV-Tools</code> will handle to make sure the matching <code>OpExtension</code> and <code>OpCapability</code> are used.</p>
</div>
</div>
</div>
</div>
</div>