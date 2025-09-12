<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#device-groups">Device Groups</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/device_groups.html
layout: default
---</p>
</div>
<h1 id="device-groups" class="sect0">Device Groups</h1>
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
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_device_group.html">SPV_KHR_device_group</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_device_group.txt">GLSL - GL_EXT_device_group</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Device groups are a way to have multiple physical devices (single-vendor) represented as a single logical device. If for example, an application have two of the same GPU, connected by some vendor-provided bridge interface, in a single system, one approach is to create two logical devices in Vulkan. The issue here is that there are limitations on what can be shared and synchronized between two <code>VkDevice</code> objects which is not a bad thing, but there are use cases where an application might want to combine the memory between two GPUs. Device Groups were designed for this use case by having an application create &#8220;sub-devices&#8221; to a single <code>VkDevice</code>. With device groups, objects like <code>VkCommandBuffers</code> and <code>VkQueue</code> are not tied to a single &#8220;sub-device&#8221; but instead, the driver will manage which physical device to run it on. Another usage of device groups is an alternative frame presenting system where every frame is displayed by a different &#8220;sub-device&#8221;.</p>
</div>
<div class="paragraph">
<p>There are two extensions, <code>VK_KHR_device_group</code> and <code>VK_KHR_device_group_creation</code>. The reason for two separate extensions is that extensions are either &#8220;instance level extensions&#8221; or &#8220;device level extensions&#8221;. Since device groups need to interact with instance level calls as well as device level calls, two extensions were created. There is also a matching <a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_device_group.html">SPV_KHR_device_group</a> extension adding the <code>DeviceGroup</code> scope and a new <code>DeviceIndex</code> built-in type to shaders that allow shaders to control what to do for each logical device. If using GLSL there is also a <a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_device_group.txt">GL_EXT_device_group</a> extension that introduces a <code>highp int gl_DeviceIndex;</code> built-in variable for all shader types.</p>
</div>