<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#versions">Versions</a>
<ul class="sectlevel1">
<li><a href="#_instance_and_device">1. Instance and Device</a></li>
<li><a href="#_header">2. Header</a></li>
<li><a href="#_extensions">3. Extensions</a></li>
<li><a href="#_structs_and_enums">4. Structs and enums</a></li>
<li><a href="#_functions">5. Functions</a></li>
<li><a href="#_features">6. Features</a></li>
<li><a href="#_limits">7. Limits</a></li>
<li><a href="#_spir_v">8. SPIR-V</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/versions.html
layout: default
---</p>
</div>
<h1 id="versions" class="sect0">Versions</h1>
<div class="paragraph">
<p>Vulkan works on a <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-coreversions-versionnumbers">major, minor, patch</a> versioning system. Currently, there are 3 minor version releases of Vulkan (1.0, 1.1, 1.2 and 1.3) which are backward compatible with each other. An application can use <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#vkEnumerateInstanceVersion">vkEnumerateInstanceVersion</a> to check what version of a Vulkan instance is supported. There is also a <a href="https://www.lunarg.com/wp-content/uploads/2019/02/Vulkan-1.1-Compatibility-Statement_01_19.pdf">white paper</a> by LunarG on how to query and check for the supported version. While working across minor versions, there are some subtle things to be aware of.</p>
</div>
<div class="sect1">
<h2 id="_instance_and_device">1. Instance and Device</h2>
<div class="sectionbody">
<div class="paragraph">
<p>It is important to remember there is a difference between the instance-level version and device-level version. It is possible that the loader and implementations will support different versions.</p>
</div>
<div class="paragraph">
<p>The <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-coreversions-queryingversionsupport">Querying Version Support</a> section in the Vulkan Spec goes into details on how to query for supported versions at both the instance and device level.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_header">2. Header</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There is only one supported header for all major releases of Vulkan. This means that there is no such thing as &#8220;Vulkan 1.0 headers&#8221; as all headers for a minor and patch version are unified. This should not be confused with the ability to generate a 1.0 version of the <a href="vulkan_spec.html#vulkan-spec">Vulkan Spec</a>, as the Vulkan Spec and header of the same patch version will match. An example would be that the generated 1.0.42 Vulkan Spec will match the 1.x.42 header.</p>
</div>
<div class="paragraph">
<p>It is highly recommended that developers try to keep up to date with the latest header files released. The Vulkan SDK comes in many versions which map to the header version it will have been packaged for.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_extensions">3. Extensions</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Between minor versions of Vulkan, <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#versions-1.1">some extensions</a> get <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-compatibility-promotions">promoted</a> to the <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#extendingvulkan-coreversions">core version</a>. When targeting a newer minor version of Vulkan, an application will not need to enable the newly promoted extensions at the instance and device creation. However, if an application wants to keep backward compatibility, it will need to enable the extensions.</p>
</div>
<div class="paragraph">
<p>For a summary of what is new in each version, check out the <a href="vulkan_release_summary.html#vulkan-release-summary">Vulkan Release Summary</a></p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_structs_and_enums">4. Structs and enums</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Structs and enums are dependent on the header file being used and not the version of the instance or device queried. For example, the struct <code>VkPhysicalDeviceFeatures2</code> used to be <code>VkPhysicalDeviceFeatures2KHR</code> before Vulkan 1.1 was released. Regardless of the 1.x version of Vulkan being used, an application should use <code>VkPhysicalDeviceFeatures2</code> in its code as it matches the newest header version. For applications that did have <code>VkPhysicalDeviceFeatures2KHR</code> in the code, there is no need to worry as the Vulkan header also aliases any promoted structs and enums (<code>typedef VkPhysicalDeviceFeatures2 VkPhysicalDeviceFeatures2KHR;</code>).</p>
</div>
<div class="paragraph">
<p>The reason for using the newer naming is that the Vulkan Spec itself will only refer to <code>VkPhysicalDeviceFeatures2</code> regardless of what version of the Vulkan Spec is generated. Using the newer naming makes it easier to quickly search for where the structure is used.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_functions">5. Functions</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Since functions are used to interact with the loader and implementations, there needs to be a little more care when working between minor versions. As an example, let&#8217;s look at <code>vkGetPhysicalDeviceFeatures2KHR</code> which was promoted to core as <code>vkGetPhysicalDeviceFeatures2</code> from Vulkan 1.0 to Vulkan 1.1. Looking at the Vulkan header both are declared.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceFeatures2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures);
// ...
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceFeatures2KHR)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures);</code></pre>
</div>
</div>
<div class="paragraph">
<p>The main difference is when calling <code>vkGetInstanceProcAddr(instance, &#8220;vkGetPhysicalDeviceFeatures2&#8221;);</code> a Vulkan 1.0 implementation may not be aware of <code>vkGetPhysicalDeviceFeatures2</code> existence and <code>vkGetInstanceProcAddr</code> will return <code>NULL</code>. To be backward compatible with Vulkan 1.0 in this situation, the application should query for <code>vkGetPhysicalDeviceFeatures2KHR</code> as a 1.1 Vulkan implementation will likely have the function directly pointed to the <code>vkGetPhysicalDeviceFeatures2</code> function pointer internally.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The <code>vkGetPhysicalDeviceFeatures2KHR</code> function will only exist in a Vulkan 1.0 implementation if it is supported as an extension.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_features">6. Features</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Between minor versions, it is possible that some feature bits are added, removed, made optional, or made mandatory. All details of features that have changed are described in the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#versions">Core Revisions</a> section.</p>
</div>
<div class="paragraph">
<p>The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#features-requirements">Feature Requirements</a> section in the Vulkan Spec can be used to view the list of features that are required from implementations across minor versions.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_limits">7. Limits</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Currently, all versions of Vulkan share the same minimum/maximum limit requirements, but any changes would be listed in the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#limits-minmax">Limit Requirements</a> section of the Vulkan Spec.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_spir_v">8. SPIR-V</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Every minor version of Vulkan maps to a version of <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#spirvenv">SPIR-V that must be supported</a>.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Vulkan 1.0 supports SPIR-V 1.0</p>
</li>
<li>
<p>Vulkan 1.1 supports SPIR-V 1.3 and below</p>
</li>
<li>
<p>Vulkan 1.2 supports SPIR-V 1.5 and below</p>
</li>
<li>
<p>Vulkan 1.3 supports SPIR-V 1.6 and below</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>It is up to the application to make sure that the SPIR-V in <code>VkShaderModule</code> is of a valid version to the corresponding Vulkan version.</p>
</div>
</div>
</div>