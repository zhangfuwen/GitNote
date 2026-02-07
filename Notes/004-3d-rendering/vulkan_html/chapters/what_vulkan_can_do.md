<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#what-vulkan-can-do">What Vulkan Can Do</a>
<ul class="sectlevel1">
<li><a href="#_graphics">1. Graphics</a></li>
<li><a href="#_compute">2. Compute</a></li>
<li><a href="#_ray_tracing">3. Ray Tracing</a></li>
<li><a href="#_video">4. Video</a></li>
<li><a href="#_machine_learning">5. Machine Learning</a></li>
<li><a href="#_safety_critical">6. Safety Critical</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/what_vulkan_can_do.html
---</p>
</div>
<h1 id="what-vulkan-can-do" class="sect0">What Vulkan Can Do</h1>
<div class="paragraph">
<p>Vulkan can be used to develop applications for many use cases. While Vulkan applications can choose to use a subset of the functionality described below, it was designed so a developer could use all of them in a single API.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>It is important to understand Vulkan is a box of tools and there are multiple ways of doing a task.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="_graphics">1. Graphics</h2>
<div class="sectionbody">
<div class="paragraph">
<p>2D and 3D graphics are primarily what the Vulkan API is designed for. Vulkan is designed to allow developers to create hardware accelerated graphical applications.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>All Vulkan implementations are required to support Graphics, but the <a href="wsi.html#wsi">WSI</a> system is not required.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_compute">2. Compute</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Due to the parallel nature of GPUs, a new style of programming referred to as <a href="https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units">GPGPU</a> can be used to exploit a GPU for computational tasks. Vulkan supports compute variations of <code>VkQueues</code>, <code>VkPipelines</code>, and more which allow Vulkan to be used for general computation.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>All Vulkan implementations are required to support Compute.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_ray_tracing">3. Ray Tracing</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Ray tracing is an alternative rendering technique, based around the concept of simulating the physical behavior of light.</p>
</div>
<div class="paragraph">
<p>Cross-vendor API support for ray tracing was added to Vulkan as a set of extensions in the 1.2.162 specification.
These are primarily <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_ray_tracing_pipeline"><code>VK_KHR_ray_tracing_pipeline</code></a>, <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_ray_query"><code>VK_KHR_ray_query</code></a>, and <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_acceleration_structure"><code>VK_KHR_acceleration_structure</code></a>.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>There is also an older <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_NV_ray_tracing">NVIDIA vendor extension</a> exposing an implementation of ray tracing on Vulkan. This extension preceded the cross-vendor extensions. For new development, applications are recommended to prefer the more recent KHR extensions.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_video">4. Video</h2>
<div class="sectionbody">
<div class="paragraph">
<p><a href="https://www.khronos.org/blog/an-introduction-to-vulkan-video?mc_cid=8052312abe&amp;mc_eid=64241dfcfa">Vulkan Video</a> has release a provisional specification as of the 1.2.175 spec release.</p>
</div>
<div class="paragraph">
<p>Vulkan Video adheres to the Vulkan philosophy of providing flexible, fine-grained control over video processing scheduling, synchronization, and memory utilization to the application.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/Vulkan-Docs/issues/1497">feedback</a> for the provisional specification is welcomed</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_machine_learning">5. Machine Learning</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Currently, the Vulkan Working Group is looking into how to make Vulkan a first class API for exposing ML compute capabilities of modern GPUs. More information was announced at <a href="https://www.youtube.com/watch?v=_57aiwJISCI&amp;feature=youtu.be&amp;t=5007">Siggraph 2019</a>.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>As of now, there exists no public Vulkan API for machine learning.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_safety_critical">6. Safety Critical</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Vulkan SC ("Safety Critical") aims to bring the graphics and compute capabilities of modern GPUs to safety-critical systems in the automotive, avionics, industrial and medical space. It was publicly <a href="https://www.khronos.org/news/press/khronos-releases-vulkan-safety-critical-1.0-specification-to-deliver-safety-critical-graphics-compute">launched on March 1st 2022</a> and the specification is available <a href="https://www.khronos.org/vulkansc/">here</a>.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Vulkan SC is based on Vulkan 1.2, but removed functionality that is not needed for safety-critical markets, increases the robustness of the specification by eliminating ignored parameters and undefined behaviors, and enables enhanced detection, reporting, and correction of run-time faults.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>